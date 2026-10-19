"""原生 V3 内核预算账本测试：原子预占、路径上限、时限与上下文字节。

覆盖行为编号：
- B01 原子预占：模型/工具最后一个名额并发争抢只有一个获准，失败方不增额。
- B02 路径上限：固定路径模型主阶段 23、MCP 32；自主路径模型主阶段 11、MCP 16；
  兜底保留 1 次模型调用；中途切换路径额度不清零，全局总额度永不超过。
- B03 时限与上下文：主阶段 60s / 总 90s / 兜底 30s 窗口；总时限耗尽后不再兜底；
  check_cancel 区分 DEADLINE_EXCEEDED 与 CANCELLED_BY_USER；context 按 UTF-8
  字节计；switch_path 非法路径抛 ValueError。

测试全程离线：模型/工具替身与受控时钟来自 tests.native_helpers，无网络、无真实等待。
"""
from __future__ import annotations

import asyncio

import pytest

from flowagent.core.context import Budget, BudgetExceeded, default_budget
from flowagent.core.contracts import ErrorCode, FlowAgentCoreError

# HELPER-BUG：tests/native_helpers.py 仍从 flowagent.core.contracts 导入 RunContext，
# 而内核已把它定义在 flowagent.core.context。为遵守“只写本文件”的约束，导入前补别名；
# 若共享替身模块被修复（contracts 已有该名），此兼容垫片自动跳过。
import flowagent.core.contracts as _contracts
from flowagent.core.context import RunContext as _RunContext

if not hasattr(_contracts, "RunContext"):
    _contracts.RunContext = _RunContext  # type: ignore[attr-defined]

from tests.native_helpers import ScriptedClock, make_context  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _budget(clock: ScriptedClock | None = None, **overrides) -> Budget:
    """构造默认口径账本（模型 24 / MCP 32 / 90-60-30s），可注入受控时钟与覆盖项。"""
    clock = clock or ScriptedClock()
    params = dict(
        model_total=24,
        mcp_total=32,
        deadline_ms=90_000,
        main_deadline_ms=60_000,
        fallback_timeout_ms=30_000,
        clock=clock,
    )
    params.update(overrides)
    return Budget(**params)


async def _race_reserve(budget: Budget, kind: str, *, fallback: bool = False) -> tuple[int, int]:
    """两个协程用 Barrier 对齐起跑，争抢同一账本的最后一个名额。

    返回 (成功数, 被拒数)；被拒方抛 BudgetExceeded 且不得增加已消耗量。
    """
    barrier = asyncio.Barrier(2)

    async def attempt() -> bool:
        await barrier.wait()  # 同一时刻发起预占，制造真实竞争
        try:
            if kind == "model":
                await budget.reserve_model(fallback=fallback)
            else:
                await budget.reserve_tool()
        except BudgetExceeded:
            return False
        return True

    results = await asyncio.gather(attempt(), attempt())
    successes = sum(results)
    return successes, len(results) - successes


# ---------------------------------------------------------------- B01 原子预占


async def test_b01_model_last_slot_race_is_atomic():
    """B01：固定路径已花 22 次，最后 1 个模型名额并发争抢，只有一个成功。"""
    budget = _budget()
    budget.switch_path("fixed")
    for _ in range(22):
        await budget.reserve_model()
    assert budget.model_spent == 22

    ok, rejected = await _race_reserve(budget, "model")
    assert (ok, rejected) == (1, 1)  # 账本锁保证预占原子化
    assert budget.model_spent == 23  # 失败方的尝试不计数（不清零、不超支）


async def test_b01_tool_last_slot_race_is_atomic():
    """B01：固定路径 MCP 上限 32，已花 31 次后并发争抢最后一个尝试名额。"""
    budget = _budget()
    budget.switch_path("fixed")
    for _ in range(31):
        await budget.reserve_tool()
    assert budget.mcp_spent == 31

    ok, rejected = await _race_reserve(budget, "tool")
    assert (ok, rejected) == (1, 1)
    assert budget.mcp_spent == 32  # 只有获准的一次计入消耗


# ---------------------------------------------------------------- B02 路径上限


def test_b02_default_budget_numbers():
    """B02：default_budget 的账面口径为 24/32 次、90/60/30 秒。"""
    budget = default_budget(ScriptedClock())
    assert (budget.model_total, budget.mcp_total) == (24, 32)
    assert (budget.deadline_ms, budget.main_deadline_ms, budget.fallback_timeout_ms) == (
        90_000,
        60_000,
        30_000,
    )


async def test_b02_fixed_model_cap_23_with_one_fallback_reserved():
    """B02：固定路径主阶段最多 23 次；兜底仍保留 1 次（第 24 次用 fallback=True）。"""
    budget = _budget()
    budget.switch_path("fixed")
    for _ in range(23):
        await budget.reserve_model()
    assert budget.model_spent == 23

    with pytest.raises(BudgetExceeded):
        await budget.reserve_model()  # 主阶段第 24 次被拒（保留兜底名额）

    await budget.reserve_model(fallback=True)  # 兜底预留的 1 次可用
    assert budget.model_spent == 24
    with pytest.raises(BudgetExceeded):
        await budget.reserve_model(fallback=True)  # 总额度耗尽后兜底同样拒绝


async def test_b02_autonomous_switch_keeps_spend_and_global_cap():
    """B02：固定路径花 3 次后切自主（上限 11），只能再花 8 次；总额度 24 不被突破。"""
    budget = _budget()
    budget.switch_path("fixed")
    for _ in range(3):
        await budget.reserve_model()
    budget.switch_path("autonomous")
    assert budget.model_spent == 3  # 切换路径不重置已消耗量

    for _ in range(8):
        await budget.reserve_model()
    assert budget.model_spent == 11  # 自主路径主阶段上限 min(24-1, 11)
    with pytest.raises(BudgetExceeded):
        await budget.reserve_model()

    # 兜底仍可继续，但全局总额度 24 是硬上限
    for _ in range(13):
        await budget.reserve_model(fallback=True)
    assert budget.model_spent == 24
    with pytest.raises(BudgetExceeded):
        await budget.reserve_model(fallback=True)


async def test_b02_mcp_caps_fixed_autonomous_and_midrun_switch():
    """B02：MCP 固定路径 32、自主 16；中途切换按新路径上限继续扣减。"""
    fixed = _budget()
    fixed.switch_path("fixed")
    for _ in range(32):
        await fixed.reserve_tool()
    with pytest.raises(BudgetExceeded):
        await fixed.reserve_tool()

    auto = _budget()
    auto.switch_path("autonomous")
    for _ in range(16):
        await auto.reserve_tool()
    with pytest.raises(BudgetExceeded):
        await auto.reserve_tool()

    # 中途切换：固定已花 10 次，切自主后上限 16，只能再花 6 次
    mixed = _budget()
    mixed.switch_path("fixed")
    for _ in range(10):
        await mixed.reserve_tool()
    mixed.switch_path("autonomous")
    for _ in range(6):
        await mixed.reserve_tool()
    assert mixed.mcp_spent == 16
    with pytest.raises(BudgetExceeded):
        await mixed.reserve_tool()


# ---------------------------------------------------------------- B03 时限与上下文


def test_b03_main_deadline_expires_before_total():
    """B03：越过主阶段 60s 后主阶段剩余 <= 0，但总剩余仍为正。"""
    clock = ScriptedClock()
    budget = _budget(clock)
    clock.advance_ms(60_500)

    assert budget.remaining_main_ms() <= 0  # 主阶段已耗尽
    assert budget.remaining_total_ms() > 0  # 总时限尚有 29.5s


def test_b03_total_deadline_zeroes_fallback_window():
    """B03：越过总时限 90s 后总剩余 <= 0，兜底窗口随之为 0（不再兜底）。"""
    clock = ScriptedClock()
    budget = _budget(clock)
    clock.advance_ms(90_500)

    assert budget.remaining_total_ms() <= 0
    assert budget.fallback_window_ms() <= 0


def test_b03_fallback_window_is_min_of_timeout_and_remaining():
    """B03：兜底窗口 = min(兜底 30s, 总剩余)。"""
    clock = ScriptedClock()
    budget = _budget(clock)

    clock.advance_ms(10_000)  # 总剩余 80s，窗口被 30s 兜底时限封顶
    assert budget.fallback_window_ms() == pytest.approx(30_000)

    clock.advance_ms(60_000)  # 累计 70s，总剩余 20s，窗口收缩到 20s
    assert budget.fallback_window_ms() == pytest.approx(20_000)


def test_b03_check_cancel_distinguishes_deadline_and_user_cancel():
    """B03：总时限耗尽 -> DEADLINE_EXCEEDED；用户取消 -> CANCELLED_BY_USER。"""
    clock = ScriptedClock()
    ctx = make_context(clock=clock, budget=_budget(clock))
    assert ctx.check_cancel() is None  # 时限充足且未取消时不抛错

    clock.advance_ms(90_500)
    with pytest.raises(FlowAgentCoreError) as deadline_error:
        ctx.check_cancel()
    assert deadline_error.value.code == ErrorCode.DEADLINE_EXCEEDED

    clock2 = ScriptedClock()
    ctx2 = make_context(clock=clock2, budget=_budget(clock2))
    ctx2.cancel()
    with pytest.raises(FlowAgentCoreError) as cancel_error:
        ctx2.check_cancel()
    assert cancel_error.value.code == ErrorCode.CANCELLED_BY_USER


def test_b03_context_bytes_counted_as_utf8():
    """B03：context 预算按 UTF-8 字节计算，中文每字占 3 字节（不是字符数）。"""
    budget = _budget(context_bytes=10)

    assert budget.fits_context("x" * 10) is True
    assert budget.fits_context("x" * 11) is False
    assert budget.fits_context("中" * 3) is True  # 3 个汉字 = 9 字节
    assert budget.fits_context("中" * 4) is False  # 4 个汉字 = 12 字节 > 10
    assert budget.fits_context(("x" * 4).encode("utf-8")) is True  # bytes 直接计量


def test_b03_switch_path_rejects_unknown_path():
    """B03：路径只允许 fixed/autonomous，非法值抛 ValueError。"""
    budget = _budget()
    budget.switch_path("fixed")
    budget.switch_path("autonomous")

    with pytest.raises(ValueError):
        budget.switch_path("hybrid")
