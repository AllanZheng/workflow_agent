"""原生内核流式契约测试：事件序列、终态映射、消费端取消与子任务回收。

覆盖行为 ID：
- S01：stream() 事件转发（text.delta / step.* / tool.finished）；慢消费端取消后
  底层 execute 被取消、活动运行表不泄漏；RunContext.reclaim_children 回收登记子任务。
- S02：五种终态（completed / needs_input / partial / degraded / failed）下
  seq 从 0 连续、唯一 run.finished、其 data.result.status 正确；
  completed 时 stream 的 reply 与 run() 完全一致（每次执行使用新上下文）。

全部离线：模型/工具使用 tests/native_helpers 替身；取消与回收用 Event 同步，
不使用基于时长的 sleep（仅保留防死锁的 wait_for 兜底）。
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

import flowagent.core.contracts as _contracts
from flowagent.core.context import RunContext
from flowagent.core.contracts import (
    Clarification,
    ErrorCode,
    EventKind,
    FlowAgentCoreError,
    RunEvent,
    RunInput,
    RunResult,
    Status,
    WorkflowSpec,
)
from flowagent.core.runner import Runner
from flowagent.workflows.chat import build_chat_workflow

# 兼容垫片：tests/native_helpers.py 从 flowagent.core.contracts 导入 RunContext，
# 但该类实际定义在 flowagent.core.context。该替身文件属其他测试资产，按约定不在
# 本任务中改动，因此在导入前把名字回填到 contracts 命名空间，使 from-import 成功。
if not hasattr(_contracts, "RunContext"):  # pragma: no cover - 兼容旧替身
    _contracts.RunContext = RunContext  # type: ignore[attr-defined]

from tests.native_helpers import FakeModel, FakeTools, make_context

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------- 公共工具


def _context_factory(
    model: Any,
    *,
    tools: Any = None,
    run_id: str | None = None,
    contexts: list[RunContext] | None = None,
) -> Callable[[], RunContext]:
    """每次执行新建上下文（工厂语义）；可选固定 run_id 与上下文登记。"""

    def factory() -> RunContext:
        ctx = make_context(model=model, tools=tools)
        if run_id is not None:
            ctx.run_id = run_id
        if contexts is not None:
            contexts.append(ctx)
        return ctx

    return factory


async def _stream(runner: Runner, request: RunInput) -> list[RunEvent]:
    """完整消费一次流（测试内一次性读完，不做慢消费）。"""
    return [event async for event in runner.stream(request)]


def _indices(events: list[RunEvent], kind: EventKind) -> list[int]:
    """返回某类事件在序列中的全部下标。"""
    return [index for index, event in enumerate(events) if event.kind is kind]


def _assert_stream_contract(events: list[RunEvent], status: Status) -> dict[str, Any]:
    """S02 公共断言：seq 连续、唯一 run.finished 且位于末尾、终态匹配。"""
    assert events, "stream 至少应产出 run.started 与 run.finished"
    # seq 从 0 开始逐 1 递增，不得跳号/重号（有界队列静默丢事件会立刻暴露）
    assert [event.seq for event in events] == list(range(len(events)))
    assert events[0].kind is EventKind.RUN_STARTED
    finished = [event for event in events if event.kind is EventKind.RUN_FINISHED]
    assert len(finished) == 1, "一次执行只允许一个 run.finished"
    assert finished[0] is events[-1], "run.finished 必须是最后一个事件"
    public = finished[0].data["result"]
    assert public["status"] == status.value
    return public


# ---------------------------------------------------------------- S02：终态


async def test_stream_completed_matches_run_reply() -> None:
    """S02：completed 终态；同输入跑 run() 与 stream()，reply 必须一致。"""
    model = FakeModel(default_text="完成答复")
    contexts: list[RunContext] = []
    runner = Runner(
        workflows={"chat": build_chat_workflow()},
        context_factory=_context_factory(model, contexts=contexts),
    )
    request = RunInput(workflow="chat", inputs={"message": "你好"})

    run_result = await runner.run(request)
    events = await _stream(runner, request)

    public = _assert_stream_contract(events, Status.COMPLETED)
    assert public["reply"] == run_result.reply == "完成答复"
    assert public["execution_path"] == "workflow"
    # 两次执行必须各自新建上下文，不得复用上一次的 seq/取消状态
    assert len(contexts) == 2
    assert contexts[0] is not contexts[1]


async def test_stream_needs_input_carries_clarification() -> None:
    """S02：needs_input 终态；run.finished 携带完整澄清载荷。"""

    async def run(inputs: dict[str, Any], ctx: Any) -> RunResult:
        return RunResult(
            run_id="",
            status=Status.NEEDS_INPUT,
            reply="请补充时间范围",
            clarification=Clarification(
                reason="缺少必要参数",
                missing_fields=["时间范围"],
                confirmed_fields={"范围": "全网"},
                hints=["例如：最近 7 天"],
                continuation={"version": 1, "workflow": "clarify", "clarify_count": 1},
            ),
        )

    runner = Runner(
        workflows={"clarify": WorkflowSpec(name="clarify", run=run)},
        context_factory=_context_factory(FakeModel()),
    )
    events = await _stream(runner, RunInput(workflow="clarify", inputs={}))

    public = _assert_stream_contract(events, Status.NEEDS_INPUT)
    clarification = public["clarification"]
    assert clarification["missing_fields"] == ["时间范围"]
    assert clarification["confirmed_fields"] == {"范围": "全网"}
    assert public["execution_path"] == "workflow"


async def test_stream_partial_keeps_output() -> None:
    """S02：partial 终态；output 原样透传到 run.finished。"""

    async def run(inputs: dict[str, Any], ctx: Any) -> RunResult:
        return RunResult(
            run_id="", status=Status.PARTIAL, reply="已完成一半",
            output={"finished": 1, "total": 2},
        )

    runner = Runner(
        workflows={"batch": WorkflowSpec(name="batch", run=run)},
        context_factory=_context_factory(FakeModel()),
    )
    events = await _stream(runner, RunInput(workflow="batch", inputs={}))

    public = _assert_stream_contract(events, Status.PARTIAL)
    assert public["output"] == {"finished": 1, "total": 2}
    assert public["reply"] == "已完成一半"


async def test_stream_degraded_after_model_failure_falls_back_locally() -> None:
    """S02：模型失败后本地一次直连兜底 → degraded；recovery 保留主错误。"""
    model = FakeModel(turns=[
        FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "上游 503"),
        FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "上游仍不可用"),
        FakeModel.text_turn("任务未完成：模型服务暂不可用。"),
    ])
    runner = Runner(
        workflows={"chat": build_chat_workflow()},
        context_factory=_context_factory(model),
    )
    events = await _stream(
        runner, RunInput(workflow="chat", inputs={"message": "查数据"}))

    public = _assert_stream_contract(events, Status.DEGRADED)
    assert public["reply"] == "任务未完成：模型服务暂不可用。"
    recovery = public["recovery"]
    assert recovery["fallback_attempted"] is True
    assert recovery["final_source"] == "direct_llm"
    assert recovery["primary_error"]["code"] == ErrorCode.MODEL_UPSTREAM_ERROR.value
    # 1 次主调用 + 1 次单轮重发 + 1 次本地兜底，之后不得再发起请求
    assert model.call_count == 3


async def test_stream_failed_when_fallback_owner_is_caller() -> None:
    """S02：fallback_owner=caller 时本地零直连，失败交还平台并保留主错误。"""
    model = FakeModel(turns=[
        FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "上游 503"),
        FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "上游仍不可用"),
    ])
    runner = Runner(
        workflows={"chat": build_chat_workflow()},
        context_factory=_context_factory(model),
        fallback_owner="caller",
    )
    events = await _stream(
        runner, RunInput(workflow="chat", inputs={"message": "查数据"}))

    public = _assert_stream_contract(events, Status.FAILED)
    assert public["error"]["code"] == ErrorCode.MODEL_UPSTREAM_ERROR.value
    assert public["recovery"]["fallback_attempted"] is False
    assert public["recovery"]["final_source"] == "caller"
    assert model.call_count == 2  # 只有主调用与单轮重发，无本地兜底


# ---------------------------------------------------------------- S01：事件


async def test_stream_emits_text_step_and_tool_events() -> None:
    """S01：agent 轮次的 text.delta、step.* 与每次工具调用的 tool.finished。"""
    model = FakeModel(turns=[
        FakeModel.tool_turn(
            {"id": "call-echo", "name": "echo", "arguments": {"text": "巡检"}},
        ),
        FakeModel.text_turn("最终答复：已回显巡检"),
    ])
    tools = FakeTools(results={"echo": {"echoed": "巡检"}})
    runner = Runner(
        workflows={"chat": build_chat_workflow()},
        context_factory=_context_factory(model, tools=tools),
    )
    events = await _stream(
        runner, RunInput(workflow="chat", inputs={"message": "回显"}))

    _assert_stream_contract(events, Status.COMPLETED)
    started = _indices(events, EventKind.STEP_STARTED)
    # 两轮 agent：第 0 轮发起工具调用，第 1 轮发布正文
    assert [events[index].data["round"] for index in started] == [0, 1]
    finished = _indices(events, EventKind.STEP_FINISHED)
    assert finished, "发布正文的轮次必须有 step.finished"
    assert finished[-1] > started[-1], "step.finished 必须收束最后一轮"
    assert events[finished[-1]].data["published"] is True

    tool_events = _indices(events, EventKind.TOOL_FINISHED)
    assert len(tool_events) == 1, "每次工具调用恰好一个 tool.finished"
    assert events[tool_events[0]].data == {"name": "echo", "status": "success"}
    # 工具在第 0 轮内执行：位于第 0 轮开始之后、第 1 轮开始之前
    assert started[0] < tool_events[0] < started[1]
    assert tools.calls == [("echo", {"text": "巡检"})]

    delta_indices = _indices(events, EventKind.TEXT_DELTA)
    assert delta_indices, "模型正文增量应以 text.delta 事件下发"
    assert "".join(
        events[index].data["text"] for index in delta_indices
    ) == "最终答复：已回显巡检"
    finished_index = _indices(events, EventKind.RUN_FINISHED)[0]
    assert all(index < finished_index for index in delta_indices)


# ---------------------------------------------------------------- S01：取消


class _StallingModel:
    """挂起模型替身：首个流请求进入后永久等待，只有取消能结束（无 sleep）。"""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.entered = asyncio.Event()

    async def stream(self, messages, tools, limits):
        self.requests.append({"messages": messages, "tools": list(tools)})
        self.entered.set()
        await asyncio.Event().wait()  # 模拟上游流卡住，直到被取消
        yield  # pragma: no cover - 保持异步生成器形态


async def test_cancel_consumer_stops_execution_and_releases_active_run() -> None:
    """S01：慢消费端取消 → stream() 取消并等待 execute，活动运行表为空。"""
    model = _StallingModel()
    contexts: list[RunContext] = []
    runner = Runner(
        workflows={"chat": build_chat_workflow()},
        context_factory=_context_factory(model, contexts=contexts),
    )
    agen = runner.stream(RunInput(workflow="chat", inputs={"message": "长任务"}))
    received: list[RunEvent] = []
    first_event = asyncio.Event()

    async def consume() -> None:
        async for event in agen:
            received.append(event)
            if len(received) == 1:
                first_event.set()
                await asyncio.Event().wait()  # 慢消费者：读完首事件后停住

    consumer = asyncio.create_task(consume())
    await first_event.wait()
    await model.entered.wait()  # 保证 execute 已卡在模型流中

    consumer.cancel()
    outcome = await asyncio.gather(consumer, return_exceptions=True)
    assert isinstance(outcome[0], asyncio.CancelledError)
    # 显式关闭生成器触发 stream() 的 finally：取消 execute 并等待其回收完成
    # （wait_for 仅作防死锁兜底，不参与任何时序断言）
    await asyncio.wait_for(agen.aclose(), timeout=5)

    assert [event.kind for event in received] == [EventKind.RUN_STARTED]
    assert contexts[0].cancelled is True  # 取消已传导到执行上下文
    assert runner._active == {}  # 无泄漏的活动运行
    assert len(model.requests) == 1  # 取消后不再发起任何模型请求


async def test_run_cancel_reclaims_tracked_child_tasks() -> None:
    """S01：取消执行后，ctx.track 登记的子任务被取消并回收，run 正常返回。"""
    child_started = asyncio.Event()
    child_cancelled = asyncio.Event()
    child_handles: list[asyncio.Task[None]] = []

    async def watch_run(inputs: dict[str, Any], ctx: Any) -> RunResult:
        async def forever() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                child_cancelled.set()

        child = asyncio.create_task(forever())
        child_handles.append(ctx.track(child))
        child_started.set()
        await ctx.wait_cancelled()  # Runner.cancel() 置位后继续
        ctx.check_cancel()  # 抛出 CANCELLED_BY_USER，进入取消终态
        return RunResult(run_id="", status=Status.COMPLETED)  # 正常路径不可达

    runner = Runner(
        workflows={"watch": WorkflowSpec(name="watch", run=watch_run)},
        context_factory=_context_factory(FakeModel(), run_id="run-reclaim"),
    )
    run_task = asyncio.create_task(runner.run(RunInput(workflow="watch", inputs={})))
    await child_started.wait()

    info = runner.cancel("run-reclaim")
    assert info == {"run_id": "run-reclaim", "cancelled": True, "reason": "cancelled"}

    # wait_for 仅作防死锁兜底，不参与任何时序断言
    result = await asyncio.wait_for(run_task, timeout=5)
    assert result.status is Status.CANCELLED
    assert result.error is not None
    assert result.error.code is ErrorCode.CANCELLED_BY_USER
    assert result.recovery["final_source"] == "cancelled"
    # 子任务已被 cancel 回收，且其 finally 已执行
    assert child_handles[0].done()
    assert child_handles[0].cancelled()
    assert child_cancelled.is_set()
    assert runner._active == {}
