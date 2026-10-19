"""execution-v1 路由：respond、respond/stream、reset、cancel、capabilities、health。

契约（技术方案 §6.1/§7.1，与旧服务兼容）：
- respond：执行失败映射非 2xx + error 信封；业务成功 200 常规返回体。
- respond/stream：SSE meta(seq=0) → delta* → done（唯一终态，含 metadata）；
  错误映射 error_code/error_message + metadata；SSE 建立后不改 HTTP 状态。
- cancel：只取消本进程活动任务；不存在/已完成返回明确状态。
- reset：V3 无服务端会话状态，幂等返回 ok。
- capabilities/v2：framework_id=workflow、runtime_mode=workflow_managed、
  snapshot.versions=[3]、respond/stream/config_receipt/memory_context_v1/
  execution_scoped_mcp/skills/model_pre_call_authorization。
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from flowagent.adapters.platform import context as platform_context
from flowagent.adapters.platform.capabilities_runtime import ModelAuthorizer
from flowagent.adapters.platform.protocol import ExecutionRequest
from flowagent.adapters.platform.receipts import build_response, error_envelope
from flowagent.core.contracts import ErrorCode, FlowAgentCoreError, RunInput, Status
from flowagent.core.runner import Runner

logger = logging.getLogger("flowagent.platform.routes")

# 能力清单：与平台 FrameworkCapabilitiesV1/V2（strict, extra=forbid）逐字段对齐。
_APPLIED_FIELDS = [
    "workflow.yaml", "system_prompt", "skills", "schema_version",
    "agent_id", "revision_id", "runtime_id", "config_digest",
    "model", "tools", "memory",
]
_EDITABLE_FIELDS = [
    "adapter.endpoint", "adapter.timeout_ms", "model", "system_prompt",
    "workflow", "mcp_tools", "skills", "memory", "runtime.history_limit",
]


def _capabilities_v1(adapter_version: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "framework_id": "workflow",
        "adapter_version": adapter_version,
        "runtime_mode": "workflow_managed",
        "basic": {"respond": True, "stream": True, "config_receipt": True},
        "context": {"history_import": "messages", "memory_context_versions": [1]},
        "control": {"versions": [], "hitl_modes": [], "steering_modes": [],
                    "cancel": "implemented", "reset": "implemented"},
        "tools": {"managed_mcp": True},
        "applied_config_fields": ["workflow.yaml", "system_prompt", "skills"],
    }


def _capabilities_v2(adapter_version: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "framework_id": "workflow",
        "adapter_version": adapter_version,
        "runtime_mode": "workflow_managed",
        "snapshot": {"versions": [3]},
        "basic": {"respond": True, "stream": True, "config_receipt": True},
        "context": {"history_import": "messages", "memory_context_versions": [1]},
        "control": {"versions": [], "hitl_modes": [], "steering_modes": [],
                    "cancel": "implemented", "reset": "implemented"},
        "tools": {"managed_mcp": True, "execution_scoped_mcp": True},
        "skills": {"catalog_versions": [1], "load_versions": [1], "session_pinning": True},
        "references": {"uuid": True},
        "models": {"pre_call_authorization": True},
        "applied_config_fields": _APPLIED_FIELDS,
        "editable_fields": _EDITABLE_FIELDS,
    }


def build_platform_app(
    *,
    runner: Runner,
    internal_token: str = "",
    platform_policy: dict[str, Any] | None = None,
    build_id: str = "dev",
) -> Any:
    """组装 execution-v1 FastAPI 应用（api extra 提供 fastapi）。"""
    app = FastAPI(title="FlowAgent execution-v1", version="3")

    def _bearer_ok(request: Request) -> bool:
        authorization = request.headers.get("authorization") or ""
        if not internal_token:
            return True
        return authorization == f"Bearer {internal_token}"

    async def _execute(request_model: ExecutionRequest) -> Any:
        """通用执行：能力组装 → ContextVar 注入 → Runner.execute。"""
        caps = platform_context.build_capabilities(request_model)
        token = platform_context.set_current_capabilities(caps)
        try:
            result = await runner.execute(
                platform_context.to_run_input(request_model))
        finally:
            platform_context.reset_capabilities(token)
        return caps, result

    def _respond_body(caps: Any, result: Any) -> JSONResponse:
        if result.status is Status.FAILED and result.error is not None:
            envelope, status_code = error_envelope(
                result.error.code.value, str(result.error))
            envelope["execution_id"] = caps.request.execution_id
            envelope.setdefault("metadata", {})["recovery"] = {
                k: v for k, v in result.recovery.items() if k != "_evidence_tool_calls"}
            return JSONResponse(envelope, status_code=status_code)
        return JSONResponse(build_response(caps.request, result, caps=caps))

    @app.post("/execution-v1/respond")
    async def respond(payload: dict[str, Any], request: Request) -> Any:
        if not _bearer_ok(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        try:
            model = ExecutionRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - 信封非法统一 422
            raise HTTPException(
                status_code=422, detail=f"invalid envelope: {str(exc)[:400]}") from exc
        _reject_stream_only(model)
        caps, result = await _execute(model)
        return _respond_body(caps, result)

    @app.post("/execution-v1/respond/stream")
    async def respond_stream(payload: dict[str, Any], request: Request) -> Any:
        if not _bearer_ok(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        try:
            model = ExecutionRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422, detail=f"invalid envelope: {str(exc)[:400]}") from exc
        if model.function_task is not None:
            raise HTTPException(
                status_code=422, detail="function_task_requires_respond")

        async def event_source() -> AsyncIterator[bytes]:
            # SSE 建立后不再改变 HTTP 状态；错误以 error 事件 + metadata 呈现。
            seq = 0
            yield _sse({"seq": seq, "kind": "meta", "trace_id": model.execution_id,
                        "framework": "workflow"})
            caps = platform_context.build_capabilities(model)
            token = platform_context.set_current_capabilities(caps)
            result_obj: Any = None
            try:
                async for event in runner.stream(
                    platform_context.to_run_input(model)
                ):
                    if event.kind.value == "text.delta" and event.data.get("text"):
                        seq += 1
                        yield _sse({"seq": seq, "kind": "delta",
                                    "trace_id": model.execution_id,
                                    "content": event.data["text"]})
                    elif event.kind.value == "run.finished":
                        result_obj = event.data.get("_result")
                        break
                if result_obj is None:
                    result_obj = _BareResult()
                if result_obj.status is Status.FAILED and result_obj.error is not None:
                    envelope, _ = error_envelope(result_obj.error.code.value, str(result_obj.error))
                    seq += 1
                    yield _sse({"seq": seq, "kind": "error",
                                "trace_id": model.execution_id,
                                "error_code": envelope["error"]["code"],
                                "error_message": envelope["error"]["message"],
                                "metadata": {"recovery": {
                                    k: v for k, v in result_obj.recovery.items()
                                    if k != "_evidence_tool_calls"}}})
                else:
                    seq += 1
                    yield _sse({"seq": seq, "kind": "done",
                                "trace_id": model.execution_id,
                                "content": result_obj.reply,
                                "framework": "workflow",
                                "metadata": build_response(
                                    caps.request, result_obj, caps=caps)["metadata"]})
            finally:
                platform_context.reset_capabilities(token)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.post("/execution-v1/reset")
    async def reset(payload: dict[str, Any], request: Request) -> Any:
        if not _bearer_ok(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        continuity = str((payload or {}).get("continuity_key") or "")
        generation = int((payload or {}).get("generation") or 0)
        # V3 无服务端会话状态：幂等确认（平台仍负责持久历史重置）。
        return JSONResponse({"ok": True, "continuity_key": continuity, "generation": generation})

    @app.post("/execution-v1/cancel")
    async def cancel(payload: dict[str, Any], request: Request) -> Any:
        if not _bearer_ok(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        execution_id = str((payload or {}).get("execution_id") or "")
        outcome = runner.cancel(execution_id)
        return JSONResponse(outcome)

    async def _capabilities_body() -> dict[str, Any]:
        return _capabilities_v1(build_id)

    async def _capabilities_v2_body() -> dict[str, Any]:
        return _capabilities_v2(build_id)

    @app.get("/capabilities")
    async def capabilities(request: Request) -> Any:
        if not _bearer_ok(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        return JSONResponse(await _capabilities_body())

    @app.get("/capabilities/v2")
    async def capabilities_v2(request: Request) -> Any:
        if not _bearer_ok(request):
            raise HTTPException(status_code=401, detail="unauthorized")
        return JSONResponse(await _capabilities_v2_body())

    @app.get("/health")
    async def health() -> Any:
        return JSONResponse({"ok": True, "version": "3"})

    return app


def _reject_stream_only(model: ExecutionRequest) -> None:
    """respond 支持 function_task（会商走同步返回）；无额外限制。"""
    return None


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n".encode("utf-8")


class _BareResult:
    """流式终态的最小结果壳（回执映射只需这些字段）。"""

    def __init__(self) -> None:
        self.run_id = ""
        self.status = Status.FAILED
        self.reply = ""
        self.output: dict | None = None
        self.execution_path: str | None = None
        self.recovery: dict[str, Any] = {}
        self.error: FlowAgentCoreError | None = None

