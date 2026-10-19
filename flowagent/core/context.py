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


def _bounded(value: Any) -> Any:
    """证据值有界化：超长文本截断并标注。"""
    if isinstance(value, str) and len(value) > EvidenceValueLimit:
        return value[:EvidenceValueLimit] + f"…(truncated {len(value)} chars)"
    return value


class BudgetExceeded(PermissionError):
    """额度或时限不足（原子预占失败时抛出；不与业务错误混用）。"""


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
        clock: Callable[[], float] = _time.monotonic,
    ) -> None:
        self._lock = asyncio.Lock()
        self._clock = clock
        self._started = clock()
        self.model_total = model_total
        self.mcp_total = mcp_total
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
            return min(self.model_total - 1, 11)
        return min(self.model_total - 1, 23)

    def _mcp_cap(self) -> int:
        return 16 if self._path == "autonomous" else 32

    # ---- 原子预占 ----
    async def reserve_model(self, *, fallback: bool = False) -> None:
        """预占一次模型调用；主阶段受路径上限约束，兜底受总额度约束。"""
        async with self._lock:
            cap = self.model_total if fallback else self._model_cap()
            if self.model_spent >= cap:
                raise BudgetExceeded(
                    f"model budget: spent={self.model_spent} cap={cap} fallback={fallback}")
            self.model_spent += 1

    async def reserve_tool(self) -> None:
        async with self._lock:
            if self.mcp_spent >= self._mcp_cap():
                raise BudgetExceeded(f"mcp budget: spent={self.mcp_spent} cap={self._mcp_cap()}")
            self.mcp_spent += 1

    def model_available(self, *, fallback: bool = False) -> bool:
        cap = self.model_total if fallback else self._model_cap()
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
        self.evidence: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}  # 有界诊断（参数来源等；不含凭据/群材料）
        self._emit_sink = emit
        self._seq = 0
        self._cancelled = asyncio.Event()
        self._children: set[asyncio.Task] = set()
        self.step: str | None = None
        # 执行级资源回收钩子（MCP 会话等）：Runner 在 finally 中统一 await。
        self.closers: list[Any] = []
        # 代码提供的原问题与参考时钟（不来自模型参数）。
        self.original_question: str = ""
        self.reference_time: str = ""

    async def aclose(self) -> None:
        """执行资源回收：MCP 会话等按注册逆序关闭；关闭不抛出。"""
        for closer in reversed(self.closers):
            try:
                result = closer()
                if hasattr(result, "__await__"):
                    await result
            except Exception:  # noqa: BLE001 - 回收不抛出
                pass
        self.closers.clear()

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
        for task in list(self._children):
            if not task.done():
                task.cancel()
        for task in list(self._children):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - 回收不抛出
                pass

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
