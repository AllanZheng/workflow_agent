"""平台适配上下文：V3 请求 → 原生内核能力组装与 history 重建。

要点（技术方案 §7.1）：
- 请求级凭据经 ContextVar 注入进程级 Runner 的 context_factory（cancel 仍全局有效）。
- 模型：快照配置 + 每次请求预授权（AuthorizedModel）。
- 工具：快照固定工具 → 执行期 MCP（别名/版本头）。
- Skill：目录不算使用；正文按固定版本装载；不执行本地脚本。
- memory.read=false 时不注入记忆；有 memory_context 需快照 memory.read=true。
- request_metadata 只读取约定白名单（original_user_message/timezone/continuation/new_request）。
- history 澄清重建：显式 continuation 优先；否则尾扫"代码澄清 assistant → user 补充"对
  并解析 v1 机器标记（§4.3）；截断/不识别按新请求并记诊断。
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Callable

from flowagent.adapters.platform.capabilities_runtime import (
    AuthorizedModel,
    DemoModelAuthorizer,
    ExecutionMcpTools,
    PlatformSkills,
    http_skill_loader,
)
from flowagent.adapters.platform.protocol import ExecutionRequest, MemoryContextV1
from flowagent.clients.model import OpenAICompatibleModel
from flowagent.core.context import Budget, RunContext
from flowagent.core.contracts import RunInput, strip_clarification_marker

# 进程级：当前请求的能力束（由 respond/stream 处理器设置）。
_CURRENT: contextvars.ContextVar["RequestCapabilities | None"] = contextvars.ContextVar(
    "flowagent_platform_request", default=None
)

DEFAULT_MAIN_RESERVE_MS = 5_000  # 平台预留兜底时间（Sample 只拿主阶段）


@dataclass
class RequestCapabilities:
    """一次 V3 请求的全部执行期能力（不跨请求共享）。"""

    request: ExecutionRequest
    model: Any
    tools: Any = None
    skills: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_capabilities(
    request: ExecutionRequest, *,
    model_transport: Any | None = None,
    skill_transport: Any | None = None,
    mcp_session_factory: Any | None = None,
) -> RequestCapabilities:
    """V3 快照 + 信封凭据 → 内核能力束；协议失败在进入本函数前已拒绝。"""
    snapshot = request.agent_config.snapshot_v3
    assert snapshot is not None  # 协议层已保证
    config = request.agent_config

    inner_model = OpenAICompatibleModel(
        base_url=snapshot.model.base_url,
        api_key=config.model_api_key.get_secret_value() if config.model_api_key else "EMPTY",
        model_name=snapshot.model.model_name,
        temperature=(snapshot.model.parameters.temperature
                     if snapshot.model.parameters.temperature is not None else 0.0),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        client=model_transport,
    )
    model = AuthorizedModel(
        inner_model, DemoModelAuthorizer(
            config.model_authorization_url or "",
            config.execution_token.get_secret_value() if config.execution_token else "",
            str(snapshot.runtime_id),
            credential_generation=config.model_credential_generation,
        ),
        execution_id=request.execution_id,
        model_version_id=str(snapshot.model.version_id),
        generation=request.binding_generation,
    )

    tools = None
    if snapshot.tools:
        tools = ExecutionMcpTools(
            endpoint=config.mcp_endpoint or "",
            workload_token=config.execution_token.get_secret_value()
            if config.execution_token else "",
            fixed_tools=[tool.model_dump(mode="json") for tool in snapshot.tools],
            execution_id=request.execution_id,
            agent_id=str(snapshot.agent_id),
            revision_id=str(snapshot.revision_id),
            runtime_id=str(snapshot.runtime_id),
            generation=request.binding_generation,
            session_factory=mcp_session_factory,
        )

    skills = None
    if snapshot.skills:
        loader = _BoundSkillLoader(
            url=config.model_authorization_url or "",
            token=config.execution_token.get_secret_value() if config.execution_token else "",
            runtime_id=str(snapshot.runtime_id), transport=skill_transport,
        )
        skills = PlatformSkills(
            skills=list(snapshot.skills),
            loader=lambda **kw: loader.run(**kw),
        )
        skills.bind_execution(request.execution_id, request.binding_generation)

    return RequestCapabilities(request=request, model=model, tools=tools, skills=skills)


class _BoundSkillLoader:
    """把授权 URL/令牌绑进 http_skill_loader 的关键字参数。"""

    def __init__(self, *, url: str, token: str, runtime_id: str, transport: Any | None) -> None:
        self._url, self._token, self._runtime_id, self._transport = url, token, runtime_id, transport

    async def run(self, **kwargs: Any) -> Any:
        return await http_skill_loader(
            model_authorization_url=self._url, execution_token=self._token,
            runtime_id=self._runtime_id, transport=self._transport, **kwargs)


def context_factory(platform_policy: dict[str, Any] | None = None) -> Callable[[], RunContext]:
    """进程级 Runner 的 context_factory：能力与预算均取自当前请求（ContextVar）。"""

    def factory() -> RunContext:
        caps = _CURRENT.get()
        if caps is None:
            raise RuntimeError("platform execution context missing (no active request)")
        request = caps.request
        ctx = RunContext(
            model=caps.model, tools=caps.tools, skills=caps.skills,
            budget=budget_from_request(request, platform_policy),
        )
        ctx.reference_time = _reference_time(request)
        ctx.original_question = str(
            _metadata(request).get("original_user_message") or request.message)
        return ctx

    return factory


def set_current_capabilities(caps: RequestCapabilities | None) -> Any:
    return _CURRENT.set(caps)


def reset_capabilities(token: Any) -> None:
    _CURRENT.reset(token)


# ---------------------------------------------------------------- 预算

def budget_from_request(
    request: ExecutionRequest, platform_policy: dict[str, Any] | None = None,
) -> Budget:
    """请求总时限 - 兜底预留 = Sample 主阶段时间；额度取 platform_policy 与默认较小值。"""
    policy = dict(platform_policy or {}).get("limits") or {}
    deadline_ms = float(request.deadline_ms or 90_000)
    main_ms = max(1_000.0, deadline_ms - DEFAULT_MAIN_RESERVE_MS)
    model_total = min(int(policy.get("model_calls", 24)), 24)
    mcp_total = min(int(policy.get("tool_calls", 32)), 32)
    autonomous = (policy.get("autonomous") or {})
    budget = Budget(
        model_total=model_total, mcp_total=mcp_total,
        deadline_ms=deadline_ms, main_deadline_ms=main_ms,
        fallback_timeout_ms=DEFAULT_MAIN_RESERVE_MS,
        context_bytes=int(policy.get("context_bytes", 131_072)),
    )
    budget._platform_autonomous = {
        "model_calls": min(int(autonomous.get("model_calls", 12)), 12),
        "tool_calls": min(int(autonomous.get("tool_calls", 16)), 16),
    }
    return budget


# ---------------------------------------------------------------- 输入映射

def to_run_input(request: ExecutionRequest) -> RunInput:
    """V3 信封 → 内核 RunInput；含 continuation/history 重建与元数据白名单。"""
    workflow = "consultation_stage" if request.function_task is not None else "chat"
    metadata = _metadata(request)
    rebuilt, rebuild_diag = rebuild_continuation_from_history(request)
    continuation = None
    diagnostics: dict[str, Any] = {}
    if isinstance(metadata.get("continuation"), dict):
        continuation = dict(metadata["continuation"])  # 显式载荷优先
        diagnostics["continuation_source"] = "explicit"
    elif rebuilt is not None:
        continuation = rebuilt
        diagnostics["continuation_source"] = "history_rebuild"
    else:
        diagnostics["continuation_source"] = "new_request"
    diagnostics.update(rebuild_diag)
    if metadata.get("new_request") is True:
        continuation = None
        diagnostics["continuation_source"] = "new_request_explicit"

    # history 进模型前剥离机器标记（存储保留原文）。
    history = [
        {"role": str(item.get("role")), "content": strip_clarification_marker(str(item.get("content") or ""))}
        for item in request.history
        if isinstance(item, dict) and item.get("role") in ("user", "assistant")
        and str(item.get("content") or "").strip()
    ]
    inputs: dict[str, Any] = {
        "message": request.message,
        "history": history,
        "request_metadata": {
            key: metadata[key] for key in
            ("original_user_message", "timezone", "new_request") if key in metadata
        },
        "_platform_diagnostics": diagnostics,
    }
    if request.analysis_target is not None:
        inputs["analysis_target"] = request.analysis_target
    if request.function_task is not None:
        inputs["function_task"] = request.function_task.model_dump()
    if request.memory_context is not None:
        inputs["memory_rendered_text"] = request.memory_context.rendered_text
        inputs["memory_included_ids"] = list(request.memory_context.included_ids)
    if continuation is not None:
        inputs["continuation"] = continuation
    return RunInput(
        workflow=workflow, inputs=inputs,
        history=history, correlation_id=request.execution_id,
        continuation=continuation,
    )


def _metadata(request: ExecutionRequest) -> dict[str, Any]:
    metadata = request.request_metadata
    return metadata if isinstance(metadata, dict) else {}


def _reference_time(request: ExecutionRequest) -> str:
    tz_name = str(_metadata(request).get("timezone") or "Asia/Shanghai")
    offset = 8 if "Shanghai" in tz_name else 0
    now = datetime.now(dt_timezone(timedelta(hours=offset)))
    return now.isoformat()


# ---------------------------------------------------------------- history 澄清重建

def rebuild_continuation_from_history(
    request: ExecutionRequest,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """§4.3：无显式 continuation 时，从 history 尾部识别澄清链并重建。

    规则：连续"代码澄清 assistant（含 v1 标记）→ user 补充"对；首个澄清前的
    user 是 original_request；链条不完整/标记不识别/计数无法验证 → 新请求并记诊断。
    """
    from flowagent.core.contracts import parse_clarification_marker

    history = [item for item in request.history
               if isinstance(item, dict) and item.get("role") in ("user", "assistant")]
    if not history:
        return None, {}
    tail: list[tuple[str, dict[str, Any]]] = []
    for item in reversed(history):
        content = str(item.get("content") or "")
        if item.get("role") == "assistant" and parse_clarification_marker(content):
            tail.append(("assistant", item))
            continue
        if item.get("role") == "user" and tail and tail[-1][0] == "assistant":
            tail.append(("user", item))
            continue
        if item.get("role") == "user" and tail and tail[-1][0] == "user":
            # 连续 user：取最后一条为补充，其余并入原请求上下文。
            continue
        break
    if not tail or tail[-1][0] != "user":
        return None, {"history_incomplete": True}
    tail.reverse()
    clarify_count = sum(1 for role, _ in tail if role == "assistant")
    tail.reverse()
    # original_request：链条之前的最后一条 user（无则用当前 message）。
    chain_start_index = len(history) - len(tail)
    original = ""
    for item in reversed(history[:chain_start_index]):
        if item.get("role") == "user":
            original = str(item.get("content") or "")
            break
    marker = parse_clarification_marker(str(tail[0][1].get("content") or "")) or {}
    if marker.get("clarify_count") != clarify_count:
        return None, {"template_invalid": True, "expected": clarify_count,
                      "marker": marker.get("clarify_count")}
    supplements = [str(item.get("content") or "") for role, item in tail if role == "user"]
    return {
        "version": 1,
        "workflow": marker.get("workflow") or "group_analysis",
        "original_request": original or request.message,
        "reference_time": marker.get("reference_time") or _reference_time(request),
        "timezone": marker.get("timezone") or "Asia/Shanghai",
        "resolved_fields": {},
        "clarify_count": clarify_count,
        "_supplements": supplements,
    }, {"rebuilt_pairs": len(tail) // 2}
