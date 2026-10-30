"""执行上下文：预算、事件、取消、usage 与证据。

预算规则（技术方案 §3.3）：
- 固定路径模型总额度 24（主阶段最多 23，兜底保留 1）；自主路径 12（主 11 + 兜底 1）。
- MCP 尝试：固定 32 / 自主 16；失败与安全读重试都计数。
- 时间：总 90s，主阶段 60s，兜底 30s；清理计入总时限。
- 切换路径额度不清零：以新路径上限扣去已消耗量，并与全局剩余取较小值。
- 额度预占原子化：两个协程争最后一个名额，只有一个获准。
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time as _time
import uuid
from typing import Any, Callable

from flowagent.core.contracts import (
    EventKind,
    Model,
    RunEvent,
    Skills,
    ToolResultData,
    Tools,
    Usage,
)

ContextBytes = 131_072  # 每次模型请求的 UTF-8 字节预算（非性能承诺）
EvidenceValueLimit = 4_000  # 单条工具结果入证据的字符上限（防超长材料膨胀）
_CLEANUP_GRACE_SECONDS = 0.05
_DETACHED_TASKS: set[asyncio.Future] = set()
logger = logging.getLogger("flowagent.context")


def _observe_detached(task: asyncio.Future) -> None:
    """延迟消费后台任务异常，避免清理超时后出现无主异常。"""
    _DETACHED_TASKS.discard(task)
    if not task.cancelled():
        try:
            error = task.exception()
            if error is not None:
                logger.warning("detached capability failed: %s", type(error).__name__)
        except (asyncio.CancelledError, Exception):
            logger.exception("detached capability completion could not be inspected")


async def _cancel_and_reap(tasks: set[asyncio.Future]) -> set[asyncio.Future]:
    """只等待有限清理时间；拒绝取消的任务继续跟踪至最终完成。"""
    for task in tasks:
        if not task.done():
            task.cancel()
    try:
        if tasks:
            await asyncio.wait(tasks, timeout=_CLEANUP_GRACE_SECONDS)
    finally:
        pending = {task for task in tasks if not task.done()}
        if pending:
            logger.warning("capability cleanup timed out; remote outcome is unknown")
        for task in pending:
            if task not in _DETACHED_TASKS:
                _DETACHED_TASKS.add(task)
                task.add_done_callback(_observe_detached)
        for task in tasks - pending:
            if not task.cancelled():
                task.exception()  # 消费异常，不改变外层已有异常。
    return pending


def _bounded(value: Any) -> Any:
    """证据值有界化：超长文本截断并标注。"""
    if isinstance(value, str) and len(value) > EvidenceValueLimit:
        return value[:EvidenceValueLimit] + f"…(truncated {len(value)} chars)"
    return value


class BudgetExceeded(PermissionError):
    """额度或时限不足（原子预占失败时抛出；不与业务错误混用）。"""

    def __init__(self, message: str, *, resource: str) -> None:
        self.resource = resource
        super().__init__(message)


class Budget:
    """一次执行的额度账本：预占原子化，路径切换不清零。"""

    def __init__(
        self,
        *,
        model_total: int,
        mcp_total: int,
        deadline_ms: float,
        main_deadline_ms: float,
        fallback_timeout_ms: float,
        context_bytes: int = ContextBytes,
        autonomous_model_total: int = 12,
        autonomous_tool_total: int = 16,
        clock: Callable[[], float] = _time.monotonic,
    ) -> None:
        self._lock = asyncio.Lock()
        self._clock = clock
        self._started = clock()
        self.model_total = model_total
        self.mcp_total = mcp_total
        # 自主路径独立上限含一次兜底，仍不得突破全局与默认硬上限。
        self.autonomous_model_total = max(0, min(autonomous_model_total, 12))
        self.autonomous_tool_total = max(0, min(autonomous_tool_total, 16))
        self.deadline_ms = deadline_ms
        self.main_deadline_ms = main_deadline_ms
        self.fallback_timeout_ms = fallback_timeout_ms
        self.context_bytes = context_bytes
        self.model_spent = 0
        self.mcp_spent = 0
        self._path: str | None = None  # None=未定 / "fixed" / "autonomous"

    # ---- 时间 ----
    def elapsed_ms(self) -> float:
        return round((self._clock() - self._started) * 1000, 1)

    def remaining_total_ms(self) -> float:
        return self.deadline_ms - self.elapsed_ms()

    def remaining_main_ms(self) -> float:
        """主阶段剩余 = min(总剩余, 主阶段时限-已耗)。"""
        return min(self.remaining_total_ms(), self.main_deadline_ms - self.elapsed_ms())

    def fallback_window_ms(self) -> float:
        """兜底可用窗口 = min(兜底时限, 总剩余)。总时限耗尽则不再兜底。"""
        return min(self.fallback_timeout_ms, self.remaining_total_ms())

    # ---- 路径 ----
    def switch_path(self, path: str) -> None:
        """切换路径：额度不清零；后续预占取新路径上限与全局剩余的较小值。"""
        if path not in ("fixed", "autonomous"):
            raise ValueError(f"unknown budget path: {path}")
        self._path = path

    def _model_cap(self) -> int:
        """当前路径的主阶段模型上限（兜底 1 次已在总额度中预留）。"""
        if self._path == "autonomous":
            return max(0, min(self.model_total - 1, self.autonomous_model_total - 1))
        return max(0, min(self.model_total - 1, 23))

    def _mcp_cap(self) -> int:
        # 平台硬额度始终约束路径上限，包括显式的零额度。
        return min(self.mcp_total, self.autonomous_tool_total if self._path == "autonomous" else 32)

    def _fallback_model_cap(self) -> int:
        if self._path == "autonomous":
            return min(self.model_total, self.autonomous_model_total)
        return self.model_total

    # ---- 原子预占 ----
    async def reserve_model(self, *, fallback: bool = False) -> None:
        """预占模型调用；主阶段与兜底均受对应路径和全局额度约束。"""
        async with self._lock:
            cap = self._fallback_model_cap() if fallback else self._model_cap()
            if self.model_spent >= cap:
                raise BudgetExceeded(
                    f"model budget: spent={self.model_spent} cap={cap} path={self._path} fallback={fallback}",
                    resource="model")
            self.model_spent += 1

    async def reserve_tool(self) -> None:
        async with self._lock:
            if self.mcp_spent >= self._mcp_cap():
                raise BudgetExceeded(f"mcp budget: spent={self.mcp_spent} cap={self._mcp_cap()} path={self._path}",
                                     resource="tool")
            self.mcp_spent += 1

    def model_available(self, *, fallback: bool = False) -> bool:
        cap = self._fallback_model_cap() if fallback else self._model_cap()
        return self.model_spent < cap

    def fits_context(self, payload: str | bytes) -> bool:
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        return len(data) <= self.context_bytes


class RunContext:
    """一次执行的运行时上下文：能力、预算、事件、取消、证据与活动任务登记。"""

    def __init__(
        self,
        *,
        model: Model,
        tools: Tools | None = None,
        skills: Skills | None = None,
        budget: Budget,
        run_id: str | None = None,
        emit: Callable[[RunEvent], None] | None = None,
        clock: Callable[[], float] = _time.monotonic,
        wall_clock: Callable[[], float] = _time.time,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.model = model
        self.tools = tools
        self.skills = skills
        self.budget = budget
        self.usage = Usage()
        # 平台模型在 SDK 发起时记账；其他模型在原生流入口记账。
        self.model_request_evidence: Any = None
        self.evidence: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}  # 有界诊断（参数来源等；不含凭据/群材料）
        self._emit_sink = emit
        self._async_emit_sink: Any = None
        self._seq = 0
        self._cancelled = asyncio.Event()
        self._children: set[asyncio.Task] = set()
        self.step: str | None = None
        # 执行级资源回收钩子（MCP 会话等）：Runner 在 finally 中统一 await。
        self.closers: list[Any] = []
        # 代码提供的原问题与参考时钟（不来自模型参数）。
        self.original_question: str = ""
        self.reference_time: str = ""

    async def stream_model(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                           limits: Any, *, fallback: bool = False):
        from flowagent.core.model_context import native_request_usage
        import sys

        native_observed = self.model_request_evidence is None and getattr(
            self.model, "records_actual_requests", False)
        count_generic = self.model_request_evidence is None and not native_observed
        started = False
        iterator = self.model.stream(messages, tools, limits).__aiter__()
        async def next_event() -> Any:
            nonlocal started
            self.check_cancel()
            remaining = self.budget.fallback_window_ms() if fallback else self.budget.remaining_main_ms()
            if remaining <= 0:
                raise _deadline_error(self, fallback=fallback)
            if count_generic and not started:
                # 非 SDK 模型在实际进入流时计数，预占不等于请求已发起。
                self.usage.model_requests += 1
                started = True
            return await iterator.__anext__()
        try:
            while True:
                # 每次迭代在同一 task 设置和重置；截止时间包装可能切换 task。
                token = native_request_usage.set(self.usage) if native_observed else None
                try:
                    event = await await_capability(self, next_event(), fallback=fallback)
                except StopAsyncIteration:
                    return
                finally:
                    if token is not None:
                        native_request_usage.reset(token)
                if event.kind == "completed" and event.usage:
                    # 输入和输出分别累加，单侧缺失时保留未知状态。
                    for key, source in (("tokens_in", "input_tokens"), ("tokens_out", "output_tokens")):
                        value = event.usage.get(source)
                        if value is not None:
                            setattr(self.usage, key, (getattr(self.usage, key) or 0) + value)
                yield event
        finally:
            # 外层提前关闭时仍尝试回收，但不可让顽固的 aclose 绕过截止时间。
            await _close_model_iterator(self, iterator, sys.exc_info()[1], fallback=fallback)

    async def aclose(self) -> None:
        """按逆序关闭执行资源，清理等待受剩余时限与有限宽限约束。"""
        closers, self.closers = self.closers, []
        for closer in reversed(closers):
            try:
                result = closer()
                if not inspect.isawaitable(result):
                    continue
                task = self.track(asyncio.ensure_future(result))
                cancel_waiter = asyncio.create_task(self.wait_cancelled())
                try:
                    timeout = min(1.0, max(0.0, self.budget.remaining_total_ms() / 1000.0))
                    # 接收取消后不再等待完整清理窗口。
                    await asyncio.wait({task, cancel_waiter}, timeout=timeout,
                                       return_when=asyncio.FIRST_COMPLETED)
                finally:
                    pending = await _cancel_and_reap({task, cancel_waiter})
                    if pending:
                        self.diagnostics["capability_cleanup_pending"] = True
                        self._children.difference_update(pending)
                if task.done() and not task.cancelled() and task.exception() is not None:
                    logger.warning("resource cleanup failed: %s", type(task.exception()).__name__)
            except Exception:  # noqa: BLE001 - 清理失败不覆盖运行结果。
                logger.exception("resource cleanup failed")

    # ---- 事件 ----
    def emit(self, kind: EventKind, data: dict[str, Any] | None = None) -> RunEvent:
        event = RunEvent(
            run_id=self.run_id, seq=self._seq, kind=kind, step=self.step,
            data=dict(data or {}),
        )
        self._seq += 1
        if self._emit_sink is not None:
            try:
                self._emit_sink(event)
            except Exception:  # noqa: BLE001 - 订阅方异常不得中断执行
                pass
        return event

    async def emit_async(self, kind: EventKind, data: dict[str, Any] | None = None) -> RunEvent:
        """最终正文和终态等待消费者接收，不能因进度队列满而丢失。"""
        if self._async_emit_sink is None:
            event = self.emit(kind, data)
            if kind is EventKind.RUN_FINISHED:
                self._terminal_committed = True
            return event
        event = RunEvent(run_id=self.run_id, seq=self._seq, kind=kind, step=self.step, data=dict(data or {}))
        await self._async_emit_sink(event)
        self._seq += 1
        if kind is EventKind.RUN_FINISHED:
            self._terminal_committed = True
        return event

    # ---- 取消 ----
    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    async def wait_cancelled(self) -> None:
        await self._cancelled.wait()

    def check_cancel(self) -> None:
        """在能力调用前检查：取消/断连/总时限耗尽后不再发起新调用。"""
        if self.cancelled:
            from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
            raise FlowAgentCoreError(ErrorCode.CANCELLED_BY_USER, "run cancelled by user")
        if self.budget.remaining_total_ms() <= 0:
            from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
            raise FlowAgentCoreError(ErrorCode.DEADLINE_EXCEEDED, "total deadline exceeded")

    # ---- 活动任务登记（finally 统一回收）----
    def track(self, task: asyncio.Task) -> asyncio.Task:
        self._children.add(task)
        task.add_done_callback(self._children.discard)
        return task

    async def reclaim_children(self) -> None:
        """回收所有子任务；已被远端接收的写操作不得声称撤销。"""
        pending = await _cancel_and_reap(set(self._children))
        if pending:
            self.diagnostics["capability_cleanup_pending"] = True
            self._children.difference_update(pending)

    # ---- 证据 ----
    def record_tool(
        self, name: str, arguments: dict[str, Any], result: ToolResultData, *,
        remote: bool = True, remote_name: str | None = None,
    ) -> None:
        self.evidence.setdefault("tool_calls", []).append({
            "name": name, "remote_name": remote_name or name,
            "arguments": arguments, "status": result.status,
            "error_code": result.error_code, "remote": remote,
            # 结果值入证据（知识状态/引用过滤依据；有界截断防超长材料）。
            "result_value": _bounded(result.value),
        })


async def _close_model_iterator(ctx: RunContext, iterator: Any, active_error: BaseException | None,
                                *, fallback: bool) -> None:
    from flowagent.core.stream import close_async_resource
    task = asyncio.create_task(close_async_resource(iterator))
    cancel_waiter = asyncio.create_task(ctx.wait_cancelled())
    interrupted = False
    try:
        remaining = ctx.budget.fallback_window_ms() if fallback else ctx.budget.remaining_main_ms()
        done, _ = await asyncio.wait({task, cancel_waiter}, timeout=min(1.0, max(0.0, remaining / 1000)),
                                     return_when=asyncio.FIRST_COMPLETED)
        interrupted = task not in done or ctx.cancelled
    except asyncio.CancelledError:
        # 正常关闭时新收到的取消须传播；已有真实错误不能被清理异常覆盖。
        if active_error is None or isinstance(active_error, GeneratorExit):
            raise
        logger.warning("model cleanup cancelled while handling an existing error")
    finally:
        pending = await _cancel_and_reap({task, cancel_waiter})
        if pending:
            ctx.diagnostics["capability_cleanup_pending"] = True
            ctx.diagnostics["capability_outcome"] = "unknown"
    if interrupted and (active_error is None or isinstance(active_error, GeneratorExit)):
        ctx.check_cancel()
        remaining = ctx.budget.fallback_window_ms() if fallback else ctx.budget.remaining_main_ms()
        if remaining <= 0:
            raise _deadline_error(ctx, fallback=fallback)


def _deadline_error(ctx: RunContext, *, fallback: bool = False):
    """主阶段耗时不能占用保留给唯一兜底所有者的总时限。"""
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    ctx.check_cancel()
    # 限时器精度可能比时钟读数提前不到一毫秒，等长窗口仍属于总时限。
    if not fallback and ctx.budget.main_deadline_ms >= ctx.budget.deadline_ms:
        return FlowAgentCoreError(ErrorCode.DEADLINE_EXCEEDED, "total deadline exceeded")
    if fallback:
        return FlowAgentCoreError(ErrorCode.MODEL_TIMEOUT, "fallback window exceeded")
    return FlowAgentCoreError(ErrorCode.MAIN_DEADLINE_EXCEEDED, "main deadline exceeded")


async def await_capability(ctx: RunContext, awaitable: Any, *, fallback: bool = False,
                           remote: bool = False) -> Any:
    """在主阶段剩余时间内竞争能力结果与取消，失败时回收本地任务。"""
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError

    try:
        ctx.check_cancel()
        remaining = ctx.budget.fallback_window_ms() if fallback else ctx.budget.remaining_main_ms()
        if remaining <= 0:
            raise _deadline_error(ctx, fallback=fallback)
    except BaseException:
        # 协程尚未调度，关闭它以免零时限产生无主协程或实际请求。
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise
    operation = ctx.track(asyncio.ensure_future(awaitable))
    cancelled = asyncio.create_task(ctx.wait_cancelled())
    try:
        done, _ = await asyncio.wait(
            {operation, cancelled}, timeout=remaining / 1000.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # 同时完成时取消优先，避免发布已接收取消的结果。
        if ctx.cancelled:
            ctx.check_cancel()
        if operation not in done:
            raise _deadline_error(ctx, fallback=fallback)
        ctx.check_cancel()
        remaining_after = ctx.budget.fallback_window_ms() if fallback else ctx.budget.remaining_main_ms()
        if remaining_after <= 0:
            raise _deadline_error(ctx, fallback=fallback)
        return operation.result()
    except FlowAgentCoreError as exc:
        if remote and exc.code in (ErrorCode.CANCELLED_BY_USER, ErrorCode.DEADLINE_EXCEEDED,
                                   ErrorCode.MAIN_DEADLINE_EXCEEDED):
            ctx.diagnostics["capability_outcome"] = "unknown"
        raise
    except asyncio.CancelledError:
        if remote:
            ctx.diagnostics["capability_outcome"] = "unknown"
        raise
    finally:
        pending = await _cancel_and_reap({operation, cancelled})
        if pending:
            # 本地等待已结束不等于远端操作已取消，结果须标记未知。
            ctx.diagnostics["capability_cleanup_pending"] = True
            ctx._children.difference_update(pending)


async def await_remote_tool(ctx: RunContext, call: Callable[[], Any]) -> Any:
    """在实际调度时记录工具尝试；超时/取消后远端结果不能推断。"""
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    started = False

    async def dispatch() -> Any:
        nonlocal started
        ctx.check_cancel()
        if ctx.budget.remaining_main_ms() <= 0:
            raise _deadline_error(ctx)
        started = True
        ctx.usage.mcp_attempts += 1
        return await call()

    try:
        return await await_capability(ctx, dispatch())
    except FlowAgentCoreError as exc:
        if started and exc.code in (ErrorCode.CANCELLED_BY_USER, ErrorCode.DEADLINE_EXCEEDED,
                                    ErrorCode.MAIN_DEADLINE_EXCEEDED):
            ctx.diagnostics["capability_outcome"] = "unknown"
        raise
    except asyncio.CancelledError:
        if started:
            ctx.diagnostics["capability_outcome"] = "unknown"
        raise


def default_budget(clock: Callable[[], float] = _time.monotonic) -> Budget:
    """独立模式默认预算：总 90s / 主 60s / 兜底 30s；固定路径 24/32。

    路由决定后调用 budget.switch_path("fixed"|"autonomous") 收紧上限；
    切换在同一账本上进行，已消耗量不清零（技术方案 §3.3）。
    """
    return Budget(
        model_total=24, mcp_total=32,
        deadline_ms=90_000, main_deadline_ms=60_000, fallback_timeout_ms=30_000,
        clock=clock,
    )


async def aiter_with_deadline(iterator: Any, deadline_ms: float, *, code: str = "MODEL_TIMEOUT") -> Any:
    """给异步迭代加总墙钟：超时抛 FlowAgentCoreError（默认 MODEL_TIMEOUT）。

    用途：模型流式响应可能持续到达分块导致读超时永不触发（超长生成）；
    预算的主阶段时限必须约束到单次调用的整体耗时。
    """
    import asyncio

    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    from flowagent.core.stream import close_async_resource

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.001, deadline_ms / 1000)

    async def _next(item: Any):
        return await item.__anext__()

    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise FlowAgentCoreError(code, f"stream deadline exceeded ({deadline_ms:.0f}ms)")
            try:
                value = await asyncio.wait_for(_next(iterator), remaining)
            except asyncio.TimeoutError as exc:
                raise FlowAgentCoreError(
                    code, f"stream deadline exceeded ({deadline_ms:.0f}ms)") from exc
            except StopAsyncIteration:
                return
            yield value
    finally:
        await close_async_resource(iterator)
