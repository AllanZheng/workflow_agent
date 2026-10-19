"""Runner：路由、唯一执行、终态与本地一次兜底。

执行骨架（技术方案 §3.2）：
- 未注册 / disabled / dependency_missing → 共享 Agent 自主规划（route_reason 记录）。
- 可用工作流执行一次；completed/partial/needs_input → 校验 → 发布。
- 未恢复错误 → 停止主阶段 → 唯一直连所有者（入口决定，用户不可覆盖）。
- 取消 / 断连 / 总时限 / 身份归属拒绝 → 停止，无新模型调用，不进入兜底。
- finally 统一回收子任务与 MCP 会话。

execute(input) 是唯一执行实现；run() 消费它取得最终结果，
stream() 转发同次事件（seq 连续、唯一 run.finished 携带完整 RunResult）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable

from flowagent.core.agent import AgentOutcome, ToolRegistry, run_agent
from flowagent.core.contracts import (
    ErrorCode,
    EventKind,
    FlowAgentCoreError,
    RunEvent,
    RunInput,
    RunResult,
    Status,
    WorkflowSpec,
)
from flowagent.core.context import RunContext

logger = logging.getLogger("flowagent.runner")

AUTONOMOUS_SYSTEM_PROMPT = (
    "你是通用业务助手。可使用当前可用的工具与 Skill 完成用户请求；"
    "工具参数缺失或含混时优先向用户澄清；不得虚构工具结果或统计数据。"
)


class RuntimePolicy:
    """运行策略：工作流启用开关与准入上限（不携带模型/凭据）。"""

    def __init__(
        self, *, workflow_enabled: dict[str, bool] | None = None,
        max_active_runs: int = 8,
    ) -> None:
        self.workflow_enabled = dict(workflow_enabled or {})
        self.max_active_runs = max_active_runs

    def is_enabled(self, name: str) -> bool:
        return self.workflow_enabled.get(name, True)


class Runner:
    """注册表 + 上下文工厂 + 兜底所有者构成的单进程执行器。"""

    def __init__(
        self,
        *,
        workflows: dict[str, WorkflowSpec],
        context_factory: Callable[[], RunContext],
        fallback_owner: str = "local",  # "local" | "caller"
        policy: RuntimePolicy | None = None,
    ) -> None:
        if fallback_owner not in ("local", "caller"):
            raise ValueError(f"unknown fallback owner: {fallback_owner}")
        self.workflows = dict(workflows)
        self.context_factory = context_factory
        self.fallback_owner = fallback_owner
        self.policy = policy or RuntimePolicy()
        self._active: dict[str, RunContext] = {}
        self._active_lock = asyncio.Lock()

    # ---- 对外接口 ----
    async def run(self, request: RunInput) -> RunResult:
        """消费 execute 取得最终结果（不订阅事件）。"""
        return await self.execute(request)

    async def stream(self, request: RunInput) -> AsyncIterator[RunEvent]:
        """转发同次执行事件；唯一 run.finished 携带完整 RunResult。

        队列有界：慢消费者时丢弃最旧事件为新事件腾位——run.finished 必须送达，
        否则消费者永久等待；丢弃的只能是中间进度事件。
        """
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue(maxsize=1024)

        def sink(event: RunEvent) -> None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()  # 丢弃最旧事件，保证终态不被挤掉
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        task = asyncio.create_task(self._execute_and_signal(queue, sink, request))
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break  # 执行任务已结束（含异常/取消路径）且无 run.finished
                yield event
                if event.kind is EventKind.RUN_FINISHED:
                    break
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _execute_and_signal(
        self, queue: asyncio.Queue, sink: Callable[[RunEvent], None], request: RunInput
    ) -> RunResult:
        """执行并入队结束哨兵：消费者在无 run.finished 的路径（异常/取消）也能终止。"""
        try:
            return await self.execute(request, emit=sink)
        finally:
            for _attempt in range(2):
                try:
                    queue.put_nowait(None)
                    break
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

    def cancel(self, run_id: str) -> dict[str, Any]:
        """取消本进程活动任务；不存在/已完成返回明确状态，不声称撤销远端工具。"""
        ctx = self._active.get(run_id)
        if ctx is None:
            return {"run_id": run_id, "cancelled": False, "reason": "not_found_or_finished"}
        ctx.cancel()
        return {"run_id": run_id, "cancelled": True, "reason": "cancelled"}

    # ---- 唯一执行实现 ----
    async def execute(
        self, request: RunInput, *, emit: Callable[[RunEvent], None] | None = None
    ) -> RunResult:
        ctx = self.context_factory()
        ctx._emit_sink = emit or (lambda event: None)
        # 关联调用方请求号（仅诊断，不进入模型上下文）。
        ctx.diagnostics["correlation_id"] = request.correlation_id
        ctx.original_question = str(request.inputs.get("message") or request.inputs.get("question") or "")
        async with self._active_lock:
            self._active[ctx.run_id] = ctx
        result: RunResult | None = None
        try:
            result = await self._execute_once(ctx, request)
        finally:
            ctx.usage.elapsed_ms = ctx.budget.elapsed_ms()
            # 证据随结果传递（回执/知识状态消费；对外序列化时剥离）。
            # 取消/异常路径 result 可能未产出——证据随上下文一起丢弃。
            if result is not None:
                result.recovery["_evidence_tool_calls"] = list(ctx.evidence.get("tool_calls") or [])
            async with self._active_lock:
                self._active.pop(ctx.run_id, None)
            await ctx.reclaim_children()
            await ctx.aclose()
        assert result is not None
        result.usage = ctx.usage
        # 唯一 run.finished：经 ctx.emit 保证 seq 连续递增，携带完整 RunResult。
        # "_result" 携带原对象供流式适配层取证据；不会进入对外序列化。
        ctx.emit(EventKind.RUN_FINISHED, {"result": result.to_public_dict(), "_result": result})
        return result

    # ---- 内部：一次执行 ----
    async def _execute_once(self, ctx: RunContext, request: RunInput) -> RunResult:
        ctx.emit(EventKind.RUN_STARTED, {"workflow": request.workflow})
        route_reason = "registered"
        spec = self.workflows.get(request.workflow)
        if spec is None:
            route_reason = "unknown_workflow"
        elif not self.policy.is_enabled(request.workflow):
            route_reason = "disabled"
        else:
            preflight_reason = await self._preflight(spec, request, ctx)
            if preflight_reason is not None:
                route_reason = preflight_reason  # dependency_missing
        path = "fixed" if route_reason == "registered" else "autonomous"
        ctx.budget.switch_path(path)
        ctx.diagnostics["route_reason"] = route_reason
        logger.info(
            "run_route run_id=%s workflow=%s path=%s reason=%s",
            ctx.run_id, request.workflow, path, route_reason,
        )

        primary_error: FlowAgentCoreError | None = None
        result: RunResult | None = None
        try:
            if path == "fixed":
                result = await spec.run(request.inputs, ctx)  # type: ignore[union-attr]
                result.run_id = ctx.run_id
                result.execution_path = "workflow"
                if not self._validate(spec, result, ctx):  # type: ignore[arg-type]
                    raise FlowAgentCoreError(
                        ErrorCode.OUTPUT_INVALID,
                        f"workflow output invalid: {request.workflow}",
                    )
            else:
                result = await self._run_autonomous(ctx, request)
                result.run_id = ctx.run_id
                result.execution_path = "autonomous"
        except FlowAgentCoreError as exc:
            primary_error = exc
        except asyncio.CancelledError:
            # Task 级取消：标记后继续传播，清理在 execute() 的 finally 完成（S03）。
            ctx.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - 未知异常保留堆栈，进入统一错误边界
            logger.exception(
                "run_unexpected_error run_id=%s workflow=%s cause=%s",
                ctx.run_id, request.workflow, type(exc).__name__,
            )
            primary_error = FlowAgentCoreError(ErrorCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        if primary_error is not None:
            return await self._handle_primary_error(ctx, request, primary_error, route_reason)

        assert result is not None
        result.recovery.setdefault("requested_workflow", request.workflow)
        result.recovery.setdefault("route_reason", route_reason)
        return result

    async def _preflight(
        self, spec: WorkflowSpec, request: RunInput, ctx: RunContext
    ) -> str | None:
        """preflight：返回 dependency_missing 或 None；连接失败以异常形式=执行错误。"""
        if spec.preflight is None:
            return None
        try:
            return await spec.preflight(request.inputs, ctx)
        except FlowAgentCoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FlowAgentCoreError(
                ErrorCode.CAPABILITY_UNAVAILABLE, f"preflight failed: {exc}") from exc

    def _validate(self, spec: WorkflowSpec, result: RunResult, ctx: RunContext) -> bool:
        """工作流输出校验：completed/partial 需通过；needs_input 交澄清模板规则。"""
        if result.status in (Status.NEEDS_INPUT,):
            return result.clarification is not None
        if spec.validate is None:
            return True
        return bool(spec.validate(result, ctx))

    async def _run_autonomous(self, ctx: RunContext, request: RunInput) -> RunResult:
        """共享 Agent 自主规划：不可用流程的终端工具不暴露（由注册表裁剪）。"""
        message = ctx.original_question or str(request.inputs.get("message", ""))
        registry = ToolRegistry(remote=ctx.tools, skills=ctx.skills)
        outcome: AgentOutcome = await run_agent(
            ctx, system_prompt=AUTONOMOUS_SYSTEM_PROMPT, user_message=message,
            history=request.history, registry=registry, duplicate_guard=True,
        )
        return self._outcome_to_result(ctx, outcome, request)

    def _outcome_to_result(self, ctx: RunContext, outcome: AgentOutcome, request: RunInput) -> RunResult:
        if outcome.is_terminal:
            result = outcome.terminal
            result.recovery.setdefault("requested_workflow", request.workflow)
            result.recovery.setdefault("route_reason", ctx.diagnostics.get("route_reason"))
            return result
        return RunResult(
            run_id=ctx.run_id, status=Status.COMPLETED, reply=outcome.text or "",
            execution_path="autonomous",
            recovery={
                "requested_workflow": request.workflow,
                "route_reason": ctx.diagnostics.get("route_reason"),
            },
        )

    # ---- 错误边界与兜底 ----
    async def _handle_primary_error(
        self, ctx: RunContext, request: RunInput, error: FlowAgentCoreError, route_reason: str
    ) -> RunResult:
        """未恢复错误 → 唯一直连所有者；取消/授权/时限不进入兜底。"""
        recovery: dict[str, Any] = {
            "requested_workflow": request.workflow,
            "route_reason": route_reason,
            "primary_error": {"code": error.code.value, "message": str(error)},
            "fallback_attempted": False,
            "fallback_error": None,
            "final_source": "error",
        }
        logger.warning(
            "run_primary_error run_id=%s workflow=%s code=%s detail=%s",
            ctx.run_id, request.workflow, error.code.value, str(error)[:300],
        )
        # 兜底前置检查：取消态与无兜底窗口时不再发起任何调用。
        if ctx.cancelled:
            recovery["final_source"] = "cancelled"
            recovery["cancelled_after"] = error.code.value
            return RunResult(
                run_id=ctx.run_id, status=Status.CANCELLED,
                error=FlowAgentCoreError(
                    ErrorCode.CANCELLED_BY_USER,
                    f"cancelled (primary: {error.code.value})"),
                execution_path="direct_llm", recovery=recovery,
            )
        if ctx.budget.fallback_window_ms() <= 0:
            recovery["final_source"] = "error"
            return RunResult(
                run_id=ctx.run_id, status=Status.FAILED,
                error=FlowAgentCoreError(ErrorCode.DEADLINE_EXCEEDED, "no window for fallback"),
                execution_path="direct_llm", recovery=recovery,
            )
        exhausted = error.code is ErrorCode.CLARIFICATION_EXHAUSTED
        if error.code is ErrorCode.CANCELLED_BY_USER:
            recovery["final_source"] = "cancelled"
            return RunResult(
                run_id=ctx.run_id, status=Status.CANCELLED, error=error,
                execution_path="direct_llm", recovery=recovery,
            )
        if error.code in (ErrorCode.AUTHORIZATION_DENIED, ErrorCode.DEADLINE_EXCEEDED):
            return RunResult(
                run_id=ctx.run_id, status=Status.FAILED, error=error,
                execution_path="direct_llm", recovery=recovery,
            )
        if self.fallback_owner == "caller":
            # 平台调用：交还错误，由平台一次直连；本地零直连。
            recovery["final_source"] = "caller"
            return RunResult(
                run_id=ctx.run_id, status=Status.FAILED, error=error,
                execution_path="direct_llm", recovery=recovery,
            )

        # 独立调用：本地一次无工具直连。
        recovery["fallback_attempted"] = True
        try:
            reply = await self._local_fallback(ctx, request, error, exhausted=exhausted)
        except FlowAgentCoreError as fallback_error:
            recovery["fallback_error"] = {
                "code": fallback_error.code.value, "message": str(fallback_error),
            }
            recovery["final_source"] = "fallback_failed"
            logger.error(
                "run_fallback_failed run_id=%s workflow=%s primary=%s fallback=%s",
                ctx.run_id, request.workflow, error.code.value, fallback_error.code.value,
            )
            return RunResult(
                run_id=ctx.run_id, status=Status.FAILED, error=FlowAgentCoreError(
                    ErrorCode.FALLBACK_FAILED,
                    f"primary={error.code.value}; fallback={fallback_error.code.value}",
                ),
                execution_path="direct_llm", recovery=recovery,
            )
        recovery["final_source"] = "direct_llm"
        logger.info(
            "run_fallback_success run_id=%s workflow=%s primary=%s elapsed_ms=%s",
            ctx.run_id, request.workflow, error.code.value, ctx.budget.elapsed_ms(),
        )
        output = None
        if exhausted:
            output = {"data_unavailable": True}
        return RunResult(
            run_id=ctx.run_id, status=Status.DEGRADED, reply=reply, output=output,
            error=error, execution_path="direct_llm", recovery=recovery,
        )

    async def _local_fallback(
        self, ctx: RunContext, request: RunInput, error: FlowAgentCoreError, *, exhausted: bool
    ) -> str:
        """本地直连：只带原问题、限长近期文本与错误摘要；无 tools、无 Skill、不重放工具。"""
        ctx.check_cancel()
        window = ctx.budget.fallback_window_ms()
        if window <= 0:
            raise FlowAgentCoreError(ErrorCode.DEADLINE_EXCEEDED, "no time for fallback")
        await ctx.budget.reserve_model(fallback=True)
        ctx.usage.model_requests += 1
        recent = "\n".join(
            f"{item.get('role')}: {item.get('content', '')}" for item in request.history[-4:]
        )
        if exhausted:
            prompt = (
                f"原始请求：{ctx.original_question}\n\n"
                "多次补充后仍缺少必要条件，任务未能完成。请只用两三句话向用户说明"
                "未完成的原因和需要补充的条件，不要给出任何统计或结论。"
            )
        else:
            prompt = (
                f"原始请求：{ctx.original_question}\n"
                f"近期对话：\n{recent or '（无）'}\n"
                f"执行中出现的错误：{error.code.value}: {error}\n\n"
                "请基于以上信息给出一段诚实的通用说明：明确标注原任务未能完成及原因；"
                "如能给出安全的一般性建议可以给出，但不得编造数据或声称任务已完成。"
            )
        messages = [
            {"role": "system", "content": "你是业务助手；如实说明任务状态，绝不虚构结果。"},
            {"role": "user", "content": prompt[:6000]},
        ]
        completed = None
        try:
            from flowagent.core.context import aiter_with_deadline

            stream = aiter_with_deadline(
                ctx.model.stream(messages, [], {"timeout_ms": window}), window)
            async for event in stream:
                if event.kind == "completed":
                    completed = event
        except FlowAgentCoreError:
            raise
        except asyncio.TimeoutError as exc:
            raise FlowAgentCoreError(ErrorCode.MODEL_TIMEOUT, "fallback timeout") from exc
        if completed is None or not (completed.text or "").strip():
            raise FlowAgentCoreError(ErrorCode.MODEL_INVALID_RESPONSE, "fallback empty response")
        return (completed.text or "").strip()

