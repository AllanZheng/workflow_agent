"""面向 Demo 的受管 execution-v1 同步服务。"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .engine import build_graph
from .errors import ConfigError
from .llm import MockLLM
from .managed_config import load_managed_spec
from .agent_runtime import ManagedAgentRuntime
from .mcp_edge_client import McpEdgeClient


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Trace(StrictModel):
    trace_id: str


class ManagedAgentConfig(StrictModel):
    agent_id: uuid.UUID
    agent_type: Literal["workflow"]
    revision_id: uuid.UUID
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: dict[str, Any]
    model_api_key: SecretStr | None = Field(default=None, repr=False)
    mcp_endpoint: str | None = None


class ExecutionRequest(StrictModel):
    protocol_version: Literal["execution-v1"]
    execution_id: str
    idempotency_key: str | None = None
    deadline_ms: int | None = None
    trace: Trace
    binding_generation: int
    continuity_key: str
    message: str
    identity: dict[str, Any]
    scene: dict[str, Any]
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    model_revision: int | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    runtime_injection: None = None
    unified_runtime_injection: None = None
    agent_config: ManagedAgentConfig


def _digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_llm(spec, config: ManagedAgentConfig):
    model = config.snapshot.get("model")
    needs_model = any(node.type in {"llm", "agent"} for node in spec.nodes.values())
    if not needs_model:
        return MockLLM()
    if not isinstance(model, dict) or config.model_api_key is None:
        raise ConfigError("managed_model_config_missing")
    from langchain_openai import ChatOpenAI

    parameters = dict(model.get("parameters") or {})
    return ChatOpenAI(
        model=model["name"], base_url=str(model["base_url"]),
        api_key=config.model_api_key.get_secret_value(), **parameters,
    )


def _managed_system_prompt(snapshot: dict[str, Any]) -> str:
    sections = [str(snapshot.get("system_prompt") or "").strip()]
    for skill in snapshot.get("skills") or []:
        content = skill.get("content") if isinstance(skill, dict) else None
        if content:
            sections.append(str(content).strip())
    return "\n\n".join(section for section in sections if section)


async def execute_managed(request: ExecutionRequest, mcp_workload_token: str = "") -> dict[str, Any]:
    config = request.agent_config
    if _digest(config.snapshot) != config.config_digest:
        raise ValueError("config_digest_mismatch")
    agent = config.snapshot.get("agent") or {}
    if str(agent.get("id")) != str(config.agent_id) or agent.get("type") != "workflow":
        raise ValueError("config_identity_mismatch")
    workflow = config.snapshot.get("workflow") or {}
    yaml_text = workflow.get("yaml")
    if not isinstance(yaml_text, str) or not yaml_text:
        raise ConfigError("managed_yaml_missing")
    spec = load_managed_spec(yaml_text)
    llm = _request_llm(spec, config)
    agent_runtime = None
    if any(node.type == "agent" for node in spec.nodes.values()):
        if not config.mcp_endpoint:
            raise ConfigError("managed_mcp_endpoint_missing")
        if not mcp_workload_token:
            raise ConfigError("managed_mcp_workload_token_missing")
        mcp_client = McpEdgeClient(str(config.mcp_endpoint), mcp_workload_token)
        agent_runtime = ManagedAgentRuntime(llm, mcp_client)
    # 每次请求创建独立模型客户端，不进入缓存；需要模型时禁止 auto->Mock 回退。
    graph = build_graph(
        spec, llm=llm, checkpointer=False, agent_runtime=agent_runtime,
    )
    final = await graph.ainvoke({
        "vars": {
            "message": request.message, "history": request.history,
            "system_prompt": _managed_system_prompt(config.snapshot),
            "skills": config.snapshot.get("skills", []),
        },
        "trace": [], "loop_counts": {}, "error": None,
    })
    reply = str((final.get("vars") or {}).get("reply", ""))
    return {
        "execution_id": request.execution_id,
        "reply": reply,
        "used_context_refs": [],
        "metadata": {"agent_config_applied": _receipt(config)},
    }


def _receipt(config: ManagedAgentConfig) -> dict[str, Any]:
    return {
        "revision_id": str(config.revision_id),
        "config_digest": config.config_digest,
        "applied_fields": ["workflow.yaml", "system_prompt", "skills"],
    }


async def stream_managed(request: ExecutionRequest, mcp_workload_token: str = ""):
    """执行受管图，并把终态 reply 节点的真实模型增量转为事件。"""
    seq = 0
    yield {"seq": seq, "kind": "meta", "trace_id": request.execution_id, "framework": "workflow"}
    seq += 1
    config = request.agent_config
    if _digest(config.snapshot) != config.config_digest:
        raise ValueError("config_digest_mismatch")
    agent = config.snapshot.get("agent") or {}
    if str(agent.get("id")) != str(config.agent_id) or agent.get("type") != "workflow":
        raise ValueError("config_identity_mismatch")
    workflow = config.snapshot.get("workflow") or {}
    yaml_text = workflow.get("yaml")
    if not isinstance(yaml_text, str) or not yaml_text:
        raise ConfigError("managed_yaml_missing")
    spec = load_managed_spec(yaml_text)
    llm = _request_llm(spec, config)
    agent_runtime = None
    if any(node.type == "agent" for node in spec.nodes.values()):
        if not config.mcp_endpoint:
            raise ConfigError("managed_mcp_endpoint_missing")
        if not mcp_workload_token:
            raise ConfigError("managed_mcp_workload_token_missing")
        agent_runtime = ManagedAgentRuntime(
            llm, McpEdgeClient(str(config.mcp_endpoint), mcp_workload_token)
        )
    graph = build_graph(
        spec, llm=llm, checkpointer=False, agent_runtime=agent_runtime,
        stream_final_reply=True,
    )
    initial = {
        "vars": {
            "message": request.message, "history": request.history,
            "system_prompt": _managed_system_prompt(config.snapshot),
            "skills": config.snapshot.get("skills", []),
        },
        "trace": [], "loop_counts": {}, "error": None,
    }
    final: dict[str, Any] = {}
    async for mode, event in graph.astream(initial, stream_mode=["custom", "values"]):
        if mode == "custom" and isinstance(event, dict) and event.get("kind") == "delta":
            content = event.get("content")
            if isinstance(content, str) and content:
                yield {"seq": seq, "kind": "delta", "trace_id": request.execution_id, "content": content}
                seq += 1
        elif mode == "values" and isinstance(event, dict):
            final = event
    reply = str((final.get("vars") or {}).get("reply", ""))
    yield {
        "seq": seq, "kind": "done", "trace_id": request.execution_id,
        "content": reply, "framework": "workflow",
        "metadata": {"agent_config_applied": _receipt(config)},
    }


def build_execution_app(internal_token: str, mcp_workload_token: str = "") -> FastAPI:
    if not internal_token:
        raise ValueError("FRAMEWORK_INTERNAL_TOKEN is required")
    app = FastAPI(title="FlowAgent execution-v1")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/execution-v1/respond")
    async def respond(request: Request):
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {internal_token}"
        if not hmac.compare_digest(authorization, expected):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        try:
            payload = ExecutionRequest.model_validate(await request.json())
            return await execute_managed(payload, mcp_workload_token)
        except ValueError as exc:
            code = str(exc)
            status = 409 if code in {"config_digest_mismatch", "config_identity_mismatch"} else 422
            return JSONResponse(status_code=status, content={"error": {"code": code}})
        except ConfigError as exc:
            return JSONResponse(status_code=422, content={"error": {"code": "managed_config_invalid", "message": str(exc)}})

    @app.post("/execution-v1/respond/stream")
    async def respond_stream(request: Request):
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {internal_token}"
        if not hmac.compare_digest(authorization, expected):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        try:
            payload = ExecutionRequest.model_validate(await request.json())
        except ValueError as exc:
            code = str(exc)
            status = 409 if code in {"config_digest_mismatch", "config_identity_mismatch"} else 422
            return JSONResponse(status_code=status, content={"error": {"code": code}})

        async def events():
            emitted = 0
            try:
                async for event in stream_managed(payload, mcp_workload_token):
                    emitted = event["seq"] + 1
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception:  # 对外只给稳定错误码，不能泄露配置、工具或密钥。
                event = {
                    "seq": emitted, "kind": "error", "trace_id": payload.execution_id,
                    "error_code": "execution_failed", "error_message": "managed execution failed",
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/execution-v1/reset")
    async def reset(request: Request):
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {internal_token}"
        if not hmac.compare_digest(authorization, expected):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        body = await request.json()
        continuity_key = body.get("continuity_key")
        generation = body.get("generation")
        if not isinstance(continuity_key, str) or not continuity_key or not isinstance(generation, int):
            return JSONResponse(status_code=422, content={"error": {"code": "invalid_reset"}})
        # 受管 Workflow 首期显式关闭 checkpoint，无服务端会话状态需要清理。
        return {"ok": True, "continuity_key": continuity_key, "generation": generation}

    return app
