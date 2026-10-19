"""共享 Agent 循环：模型—工具迭代、批次校验、重复调用守卫与单轮恢复。

规则落点（技术方案 §3.2/§3.4）：
- 每轮先校验整个批次再执行；非法批次（未知工具/混合终端/参数非法）零部分执行，
  以工具错误消息反馈模型修正（普通工具参数错误受总额度约束）。
- 终端业务工具独占一轮：返回已校验 RunResult 后直接结束，不追加模型改写。
- 允许工具的轮次正文一律缓冲，仅在无待执行工具、完整结束、正文有效时发布。
- 单轮恢复：可恢复通信错误或无效/空响应（该响应未执行任何工具）最多重发一次；
  重发计入模型额度；身份拒绝不重试。
- 自主路径重复守卫：同一工具+相同有效参数第二次调用前终止（DUPLICATE_TOOL_CALL）；
  load_skill 缓存命中与安全读重试不算重复。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from flowagent.core.context import BudgetExceeded
from flowagent.core.contracts import (
    ErrorCode,
    EventKind,
    FlowAgentCoreError,
    RunResult,
    Skills,
    Status,
    ToolCallSpec,
    ToolResultData,
    Tools,
)
from flowagent.clients.model import parse_tool_call_arguments

logger = logging.getLogger("flowagent.agent")

_RECOVERABLE_CODES = {
    ErrorCode.MODEL_TIMEOUT,
    ErrorCode.MODEL_UPSTREAM_ERROR,
    ErrorCode.MODEL_INVALID_RESPONSE,
}
_LOAD_SKILL = "load_skill"


@dataclass
class LocalTool:
    """本地工具：进程内处理器；terminal=True 时其 RunResult 直接成为终态。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    handler: Callable[[dict[str, Any]], Awaitable[ToolResultData]] | None = None
    terminal: bool = False


class ToolRegistry:
    """本轮执行的合并工具目录：本地工具（含终端）+ 远程 MCP + load_skill。"""

    def __init__(
        self, *, remote: Tools | None = None, local: list[LocalTool] | None = None,
        skills: Skills | None = None,
    ) -> None:
        self.remote = remote
        self.local = {tool.name: tool for tool in (local or [])}
        self.skills = skills
        self._remote_specs: list[ToolCallSpec] = []
        self._skill_bodies: dict[str, str] = {}

    async def list(self) -> list[ToolCallSpec]:
        specs: list[ToolCallSpec] = []
        if self.remote is not None:
            self._remote_specs = await self.remote.list()
            specs.extend(self._remote_specs)
        if self.local:
            specs.extend(
                ToolCallSpec(name=t.name, description=t.description, input_schema=t.input_schema)
                for t in self.local.values())
        if self.skills is not None and self.skills.catalog():
            catalog = self.skills.catalog()
            enum_ids = sorted(str(item["skill_id"]) for item in catalog)
            specs.append(ToolCallSpec(
                name=_LOAD_SKILL,
                description="加载指定 Skill 的正文（目录不算使用；正文中的工具引用已解析）",
                input_schema={
                    "type": "object",
                    "properties": {"skill_id": {"type": "string", "enum": enum_ids}},
                    "required": ["skill_id"],
                },
            ))
        return specs

    async def call_local(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        if name == _LOAD_SKILL and self.skills is not None:
            skill_id = str(arguments.get("skill_id") or "")
            try:
                if skill_id not in self._skill_bodies:
                    self._skill_bodies[skill_id] = await self.skills.load(skill_id)
                return ToolResultData.success(self._skill_bodies[skill_id])
            except FlowAgentCoreError as exc:
                return ToolResultData.error(str(exc), ErrorCode.SKILL_LOAD_FAILED.value)
            except Exception as exc:  # noqa: BLE001 - Skill 失败不终止循环
                return ToolResultData.error(f"skill load failed: {exc}", ErrorCode.SKILL_LOAD_FAILED.value)
        tool = self.local.get(name)
        if tool is None or tool.handler is None:
            return ToolResultData.error(f"unknown local tool: {name}", ErrorCode.TOOL_ARGUMENT_INVALID.value)
        return await tool.handler(arguments)

    @property
    def loaded_skills(self) -> list[str]:
        return sorted(self._skill_bodies)


@dataclass
class AgentOutcome:
    """循环出口：发布正文，或终端工具的已校验 RunResult。"""

    text: str | None = None
    terminal: RunResult | None = None
    rounds: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.terminal is not None


def _effective_key(name: str, arguments: dict[str, Any]) -> str:
    """重复守卫键：工具身份 + 规范化有效 JSON（键序无关）。"""
    return name + "|" + json.dumps(arguments, ensure_ascii=False, sort_keys=True)


async def run_agent(
    ctx: Any,  # RunContext（避免循环导入用 Any；结构由 core.context 定义）
    *,
    system_prompt: str,
    user_message: str,
    history: list[dict[str, Any]],
    registry: ToolRegistry,
    duplicate_guard: bool = True,
    limits: dict[str, Any] | None = None,
) -> AgentOutcome:
    """执行模型—工具循环至发布正文或终端工具结果。

    错误经 FlowAgentCoreError 抛出（含 MODEL_BUDGET_EXCEEDED / DUPLICATE_TOOL_CALL /
    CONTEXT_LIMIT_EXCEEDED / MODEL_INVALID_RESPONSE 等），由 Runner 统一进入兜底。
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(dict(item) for item in history)
    messages.append({"role": "user", "content": user_message})

    seen_effective: set[str] = set()
    schemas = _tool_schemas(await registry.list())
    rounds = 0

    while True:
        ctx.check_cancel()
        ctx.step = f"agent-round-{rounds}"
        ctx.emit(EventKind.STEP_STARTED, {"round": rounds})
        rounds += 1

        # ---- 一次模型请求（含单轮恢复）----
        try:
            await ctx.budget.reserve_model()
        except BudgetExceeded as exc:
            raise FlowAgentCoreError(
                ErrorCode.MODEL_BUDGET_EXCEEDED, str(exc)) from exc
        ctx.usage.model_requests += 1
        _ensure_context_budget(ctx, messages, schemas)
        completed = None
        retried = False
        while True:
            try:
                completed = await _consume_stream(ctx, messages, schemas, limits)
                if _invalid_response(completed):
                    # 无效/空响应在未执行任何工具时同样享有一次重发（§3.3）。
                    raise FlowAgentCoreError(
                        ErrorCode.MODEL_INVALID_RESPONSE,
                        f"empty or truncated response: finish={completed.finish_reason}",
                    )
            except FlowAgentCoreError as exc:
                if exc.code is ErrorCode.AUTHORIZATION_DENIED or retried:
                    raise
                if exc.code not in _RECOVERABLE_CODES:
                    raise
                # 单轮恢复：未发布正文且该响应未执行任何工具，重发一次。
                retried = True
                ctx.usage.model_retries += 1
                logger.warning(
                    "model_round_retry run_id=%s round=%s code=%s",
                    ctx.run_id, rounds - 1, exc.code.value,
                )
                try:
                    await ctx.budget.reserve_model()
                except BudgetExceeded as exc2:
                    raise FlowAgentCoreError(
                        ErrorCode.MODEL_BUDGET_EXCEEDED, str(exc2)) from exc2
                ctx.usage.model_requests += 1
                continue
            break

        tool_calls = completed.tool_calls or []
        if not tool_calls:
            text = (completed.text or "").strip()
            ctx.emit(EventKind.STEP_FINISHED, {"round": rounds - 1, "published": True})
            return AgentOutcome(text=text, rounds=rounds)

        # ---- 批次校验：先整批后执行；非法批次零部分执行 ----
        batch_error = _validate_batch(tool_calls, schemas, registry)
        if batch_error is not None:
            logger.warning(
                "tool_batch_rejected run_id=%s round=%s reason=%s",
                ctx.run_id, rounds - 1, batch_error[:200],
            )
            for call in tool_calls:
                messages.append({
                    "role": "tool", "tool_call_id": call.get("id") or "",
                    "content": f"[TOOL ERROR] {batch_error}",
                })
            continue

        # ---- 执行：终端工具独占一轮 ----
        first = tool_calls[0]
        if _is_terminal(registry, first["name"]):
            ctx.emit(EventKind.STEP_STARTED, {"terminal_tool": first["name"]})
            result = await registry.call_local(first["name"], parse_tool_call_arguments(first))
            ctx.record_tool(first["name"], {}, result, remote=False)
            ctx.emit(EventKind.TOOL_FINISHED, {"name": first["name"], "status": result.status})
            terminal = result.terminal
            if terminal is None:
                # 终端工具实现异常：未产出已校验结果，按内部错误处理。
                raise FlowAgentCoreError(
                    ErrorCode.INTERNAL_ERROR, f"terminal tool returned no result: {first['name']}")
            ctx.emit(EventKind.STEP_FINISHED, {"terminal_tool": first["name"], "status": terminal.status.value})
            return AgentOutcome(terminal=terminal, rounds=rounds)

        # ---- 普通工具按序执行 ----
        assistant_message = {
            "role": "assistant", "content": completed.text or "",
            "tool_calls": [
                {"id": call.get("id"), "type": "function",
                 "function": {"name": call.get("name"), "arguments": call.get("arguments") or ""}}
                for call in tool_calls
            ],
        }
        messages.append(assistant_message)
        for call in tool_calls:
            name = call["name"]
            arguments = parse_tool_call_arguments(call)
            if duplicate_guard and name != _LOAD_SKILL:
                key = _effective_key(name, arguments)
                if key in seen_effective:
                    raise FlowAgentCoreError(
                        ErrorCode.DUPLICATE_TOOL_CALL,
                        f"duplicate effective call: {name} {arguments}",
                    )
                seen_effective.add(key)

            result = await _execute_remote(ctx, registry, name, arguments)
            ctx.record_tool(name, arguments, result)
            logger.info(
                "tool_finished run_id=%s step=%s call=%s status=%s code=%s",
                ctx.run_id, ctx.step, name, result.status, result.error_code,
            )
            ctx.emit(EventKind.TOOL_FINISHED, {"name": name, "status": result.status})
            messages.append({
                "role": "tool", "tool_call_id": call.get("id") or "",
                "content": _tool_message_content(result),
            })


def _invalid_response(completed: Any) -> bool:
    """无效响应判定：截断（finish=length）、空文本且无工具调用。"""
    if (completed.finish_reason or "stop") == "length":
        return True
    if not (completed.tool_calls or []) and not (completed.text or "").strip():
        return True
    return False


def _tool_schemas(specs: list[ToolCallSpec]) -> list[dict[str, Any]]:
    """ToolCallSpec → OpenAI functions schema。"""
    return [
        {"type": "function", "function": {
            "name": spec.name, "description": spec.description,
            "parameters": spec.input_schema or {"type": "object"},
        }}
        for spec in specs
    ]


def _known_names(schemas: list[dict[str, Any]]) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def _validate_batch(
    tool_calls: list[dict[str, Any]], schemas: list[dict[str, Any]], registry: ToolRegistry
) -> str | None:
    """整批校验：未知工具、混合终端、参数非法 → 返回批次错误（零执行）。"""
    known = _known_names(schemas)
    names = [call.get("name") or "" for call in tool_calls]
    unknown = [name for name in names if name not in known]
    if unknown:
        return f"未知工具：{', '.join(unknown)}；请只使用提供的三可用工具列表"
    terminals = [name for name in names if _is_terminal(registry, name)]
    if terminals and len(tool_calls) > 1:
        return f"终端工具 {terminals[0]} 必须单独调用，不能与其他工具混合"
    for call in tool_calls:
        try:
            parse_tool_call_arguments(call)
        except FlowAgentCoreError:
            return f"工具 {call.get('name')} 参数不是合法 JSON 对象"
    return None


def _is_terminal(registry: ToolRegistry, name: str) -> bool:
    tool = registry.local.get(name)
    return bool(tool and tool.terminal)


async def _execute_remote(
    ctx: Any, registry: ToolRegistry, name: str, arguments: dict[str, Any]
) -> ToolResultData:
    """远程/本地普通工具执行：额度预占 + 结果四态（load_skill 走本地）。"""
    if name == _LOAD_SKILL or name in registry.local:
        result = await registry.call_local(name, arguments)
        if name == _LOAD_SKILL and result.status == "success":
            ctx.usage.skill_loads += 1
        return result
    if registry.remote is None:
        return ToolResultData.error("no remote tools available", ErrorCode.CAPABILITY_UNAVAILABLE.value)
    ctx.check_cancel()
    try:
        await ctx.budget.reserve_tool()
    except BudgetExceeded as exc:
        raise FlowAgentCoreError(
            ErrorCode.TOOL_BUDGET_EXCEEDED, str(exc)) from exc
    ctx.usage.mcp_attempts += 1
    return await registry.remote.call(name, arguments)


async def _consume_stream(
    ctx: Any, messages: list[dict[str, Any]], schemas: list[dict[str, Any]],
    limits: dict[str, Any] | None,
) -> Any:
    """消费一次模型流：缓冲正文增量（进度事件），返回 completed ModelEvent。"""
    completed = None
    async for event in ctx.model.stream(messages, schemas, limits or {}):
        if event.kind == "text_delta":
            # 进度可实时显示；发布与否由最终轮判定（缓冲策略）。
            ctx.emit(EventKind.TEXT_DELTA, {"text": event.text})
        elif event.kind == "completed":
            completed = event
    if completed is None:
        raise FlowAgentCoreError(ErrorCode.MODEL_INVALID_RESPONSE, "stream ended without completed")
    # usage 记账（不可得为 None）。
    usage = completed.usage or {}
    if ctx.usage.tokens_in is None and usage.get("input_tokens") is None:
        ctx.usage.tokens_in = ctx.usage.tokens_in  # 保持 None
    else:
        ctx.usage.tokens_in = (ctx.usage.tokens_in or 0) + (usage.get("input_tokens") or 0)
        ctx.usage.tokens_out = (ctx.usage.tokens_out or 0) + (usage.get("output_tokens") or 0)
    return completed


def _ensure_context_budget(ctx: Any, messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> None:
    """每次请求的 UTF-8 字节预算（提示/历史/schema/工具结果）。"""
    payload = json.dumps({"messages": messages, "tools": schemas}, ensure_ascii=False)
    if not ctx.budget.fits_context(payload):
        raise FlowAgentCoreError(
            ErrorCode.CONTEXT_LIMIT_EXCEEDED,
            f"context bytes exceeded: {len(payload.encode('utf-8'))} > {ctx.budget.context_bytes}",
        )


def _tool_message_content(result: ToolResultData) -> str:
    """工具结果 → 模型可见文本；unknown 显式声明结果不明，防模型虚构成功。"""
    if result.status == "success":
        value = result.value
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)
    prefix = {"error": "[TOOL ERROR]", "unknown": "[TOOL UNKNOWN] 结果不明，请勿假设成功"}
    text = result.message or result.error_code or result.status
    return f"{prefix.get(result.status, '[TOOL ERROR]')} {text}"


def make_terminal_tool_result(
    *, run_id: str, status: Status, reply: str, output: dict[str, Any] | None,
) -> ToolResultData:
    """终端业务包装器产出已校验 RunResult 的便捷构造（业务侧使用）。"""
    return ToolResultData.terminal_result(RunResult(
        run_id=run_id, status=status, reply=reply, output=output,
    ))
