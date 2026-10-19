"""原生 V3 内核 Runner 测试：唯一执行、路由、唯一兜底、并发隔离与取消。

覆盖行为编号（测试方案 §4）：
- R01 注册流程每次请求只执行一次；run()/stream() 的终态、正文、usage 完全一致。
- R02 未注册/disabled/dependency_missing 分别记对应 route_reason，转共享 Agent
  自主规划，且流程的 run() 绝不执行。
- R03 已开始流程抛错只走唯一本地直连兜底（DEGRADED，主错误保留），绝不重新自主规划。
- R04 两请求并行：run_id/usage/证据/模型实例互不串用；一个失败不影响另一个完成。
- S03 合作式取消（Runner.cancel）得到 CANCELLED 终态且无后续能力请求；
  Python Task 取消在清理完成后向调用方传播 CancelledError。

全程离线：模型替身与受控上下文来自 tests.native_helpers，无网络、无真实等待。
"""
from __future__ import annotations

import asyncio

import pytest

from flowagent.core.contracts import (
    ErrorCode,
    EventKind,
    FlowAgentCoreError,
    RunInput,
    RunResult,
    Status,
    WorkflowSpec,
)
from flowagent.core.runner import AUTONOMOUS_SYSTEM_PROMPT, Runner, RuntimePolicy
from tests.native_helpers import FakeModel, completed_event, make_context

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------- 测试脚手架


async def _model_once(ctx, *, prompt: str = "ping") -> str:
    """工作流侧的最小模型调用：检查取消、预占额度、消费脚本流并返回正文。

    节奏与 run_agent 每轮一致（check → reserve → 计数 → stream），便于用
    FakeModel.call_count 观察真实请求次数。
    """
    ctx.check_cancel()
    await ctx.budget.reserve_model()
    ctx.usage.model_requests += 1
    completed = None
    async for event in ctx.model.stream([{"role": "user", "content": prompt}], [], {}):
        if event.kind == "completed":
            completed = event
    assert completed is not None, "替身模型必须产出 completed 事件"
    return completed.text or ""


async def _hang() -> None:
    """永不就绪的挂起协程：充当需要被回收的子任务。"""
    await asyncio.Event().wait()


def _chat_spec(calls: dict, *, preflight_reason: str | None = None) -> WorkflowSpec:
    """chat 替身流程：run/preflight 调用次数记入 calls，preflight 可指定返回值。"""

    async def run(inputs, ctx):
        calls["run"] = calls.get("run", 0) + 1
        return RunResult(run_id="", status=Status.COMPLETED, reply="流程正文")

    async def preflight(inputs, ctx):
        calls["preflight"] = calls.get("preflight", 0) + 1
        return preflight_reason

    return WorkflowSpec(name="chat", run=run, preflight=preflight)


# ---------------------------------------------------------------- R01 唯一执行


async def test_r01_workflow_runs_once_and_run_stream_agree():
    """R01：注册流程每次请求恰好执行一次；run 与 stream 的终态完全一致。"""
    calls = {"job": 0}

    async def run(inputs, ctx):
        calls["job"] += 1
        text = await _model_once(ctx)
        return RunResult(run_id="", status=Status.COMPLETED,
                         reply=f"wf:{text}", output={"echo": text})

    def factory():
        ctx = make_context(model=FakeModel(default_text="ok"))
        ctx.run_id = "run-r01"  # 固定 run_id，两次执行的公开结果才能逐字段对比
        return ctx

    runner = Runner(workflows={"job": WorkflowSpec(name="job", run=run)},
                    context_factory=factory)
    request = RunInput(workflow="job", inputs={"message": "开始"})

    by_run = await runner.run(request)
    assert calls["job"] == 1  # 每次请求只执行一次
    assert by_run.status is Status.COMPLETED
    assert by_run.execution_path == "workflow"

    events = [event async for event in runner.stream(request)]
    assert calls["job"] == 2  # stream 是第二次独立请求，流程同样只执行一次
    assert [event.seq for event in events] == list(range(len(events)))  # seq 连续
    assert events[0].kind is EventKind.RUN_STARTED
    finished = completed_event(events)  # 唯一 run.finished
    by_stream = finished.data["result"]
    assert by_stream == by_run.to_public_dict()  # 终态/正文/output/usage 全一致
    assert by_stream["usage"]["model_requests"] == 1
    assert by_stream["run_id"] == finished.run_id == "run-r01"


# ---------------------------------------------------------------- R02 不可用路由


async def test_r02_unknown_workflow_routes_autonomous():
    """R02：未注册流程记 unknown_workflow 并转自主规划，注册表中的流程不执行。"""
    calls: dict = {}
    model = FakeModel(default_text="自主回答")
    runner = Runner(workflows={"chat": _chat_spec(calls)},
                    context_factory=lambda: make_context(model=model))

    result = await runner.run(RunInput(workflow="ghost", inputs={"message": "帮我看下"}))

    assert calls.get("run", 0) == 0  # 未注册流程的 run 绝不执行
    assert result.status is Status.COMPLETED
    assert result.execution_path == "autonomous"
    assert result.reply == "自主回答"
    assert result.recovery["route_reason"] == "unknown_workflow"
    assert result.recovery["requested_workflow"] == "ghost"
    assert model.requests[0]["messages"][0]["content"] == AUTONOMOUS_SYSTEM_PROMPT


async def test_r02_disabled_workflow_routes_autonomous():
    """R02：策略开关关闭记 disabled，同样转自主规划且 run 不执行。"""
    calls: dict = {}
    model = FakeModel(default_text="自主回答")
    runner = Runner(
        workflows={"chat": _chat_spec(calls)},
        context_factory=lambda: make_context(model=model),
        policy=RuntimePolicy(workflow_enabled={"chat": False}),
    )

    result = await runner.run(RunInput(workflow="chat", inputs={"message": "在吗"}))

    assert calls.get("run", 0) == 0
    assert result.execution_path == "autonomous"
    assert result.recovery["route_reason"] == "disabled"


async def test_r02_dependency_missing_routes_autonomous():
    """R02：preflight 返回 dependency_missing 记该原因，run 不执行。"""
    calls: dict = {}
    model = FakeModel(default_text="自主回答")
    runner = Runner(
        workflows={"chat": _chat_spec(calls, preflight_reason="dependency_missing")},
        context_factory=lambda: make_context(model=model),
    )

    result = await runner.run(RunInput(workflow="chat", inputs={"message": "分析"}))

    assert calls.get("preflight", 0) == 1  # preflight 执行了一次
    assert calls.get("run", 0) == 0  # 缺依赖时不执行 run
    assert result.execution_path == "autonomous"
    assert result.recovery["route_reason"] == "dependency_missing"
    assert model.requests[0]["messages"][0]["content"] == AUTONOMOUS_SYSTEM_PROMPT


# ---------------------------------------------------------------- R03 唯一兜底


async def test_r03_workflow_error_single_fallback_never_autonomous():
    """R03：已开始流程抛错 → 唯一本地直连兜底（DEGRADED），绝不重新自主规划。"""
    calls = {"wf": 0}

    async def run(inputs, ctx):
        calls["wf"] += 1
        await _model_once(ctx)  # 流程已开始：先有一次真实模型调用
        raise FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "上游模型不可用")

    model = FakeModel(turns=[FakeModel.text_turn("阶段进展")],
                      default_text="任务未完成：上游模型暂不可用。")
    runner = Runner(workflows={"job": WorkflowSpec(name="job", run=run)},
                    context_factory=lambda: make_context(model=model))

    result = await runner.run(RunInput(workflow="job", inputs={"message": "查数据"}))

    assert calls["wf"] == 1  # 流程只执行一次：不重试、不重新自主规划
    assert model.call_count == 2  # 流程 1 次 + 兜底 1 次（唯一兜底）
    assert result.status is Status.DEGRADED
    assert result.reply == "任务未完成：上游模型暂不可用。"
    assert result.output is None  # 非澄清耗尽，无 data_unavailable
    assert result.error.code is ErrorCode.MODEL_UPSTREAM_ERROR  # 主错误原样保留
    assert result.execution_path == "direct_llm"
    assert result.recovery["fallback_attempted"] is True
    assert result.recovery["final_source"] == "direct_llm"
    assert result.recovery["primary_error"]["code"] == "MODEL_UPSTREAM_ERROR"
    # 兜底绝不改走自主规划：两次请求都不应出现自主 system 提示
    for request in model.requests:
        for message in request["messages"]:
            assert message["content"] != AUTONOMOUS_SYSTEM_PROMPT
    # 第二次请求是直连兜底：固定 system、零 tools
    fallback_request = model.requests[1]
    assert fallback_request["messages"][0]["role"] == "system"
    assert fallback_request["tools"] == []


# ---------------------------------------------------------------- R04 并发隔离


async def test_r04_concurrent_runs_are_isolated():
    """R04：两请求并行，一个失败一个完成；上下文/run_id/usage/证据互不串用。"""
    barrier = asyncio.Barrier(2)  # 同步点：保证两请求真实并行重叠
    made: list = []
    calls = {"job": 0}

    async def run(inputs, ctx):
        calls["job"] += 1
        ctx.evidence["tag"] = inputs["name"]
        await _model_once(ctx)
        await barrier.wait()
        if inputs.get("fail"):
            raise FlowAgentCoreError(ErrorCode.AUTHORIZATION_DENIED, "执行归属拒绝")
        return RunResult(run_id="", status=Status.COMPLETED, reply=f"done:{inputs['name']}")

    def factory():
        ctx = make_context(model=FakeModel(default_text="ok"))
        made.append(ctx)
        return ctx

    runner = Runner(workflows={"job": WorkflowSpec(name="job", run=run)},
                    context_factory=factory)
    result_a, result_b = await asyncio.gather(
        runner.run(RunInput(workflow="job", inputs={"name": "alpha"})),
        runner.run(RunInput(workflow="job", inputs={"name": "beta", "fail": True})),
    )

    assert calls["job"] == 2
    assert len(made) == 2  # 每次执行各自领取独立上下文
    ctx_a = next(c for c in made if c.evidence.get("tag") == "alpha")
    ctx_b = next(c for c in made if c.evidence.get("tag") == "beta")
    assert ctx_a.run_id != ctx_b.run_id  # run_id 隔离
    assert result_a.run_id == ctx_a.run_id
    assert result_b.run_id == ctx_b.run_id
    assert ctx_a.model is not ctx_b.model  # 模型实例互不串用
    assert ctx_a.model.call_count == 1 and ctx_b.model.call_count == 1
    assert ctx_a.usage.model_requests == 1 and ctx_b.usage.model_requests == 1
    assert ctx_a.evidence == {"tag": "alpha"} and ctx_b.evidence == {"tag": "beta"}
    # 一个完成、一个失败，互不影响
    assert result_a.status is Status.COMPLETED and result_a.reply == "done:alpha"
    assert result_b.status is Status.FAILED
    assert result_b.error.code is ErrorCode.AUTHORIZATION_DENIED
    assert result_b.recovery["fallback_attempted"] is False  # 终止码不进兜底


# ---------------------------------------------------------------- S03 取消


async def test_s03_cooperative_cancel_stops_without_new_model_calls():
    """S03：合作式取消（Runner.cancel）→ CANCELLED 终态、子任务回收、无新能力请求。"""
    gate = asyncio.Event()
    holder: dict = {}
    calls = {"wf": 0}
    model = FakeModel(default_text="too-late")

    async def run(inputs, ctx):
        calls["wf"] += 1
        await _model_once(ctx)  # 取消前的一次模型调用
        holder["child"] = ctx.track(asyncio.create_task(_hang()))
        await gate.wait()  # 测试在流式侧完成取消
        ctx.check_cancel()  # 取消后这里抛 CANCELLED_BY_USER
        await _model_once(ctx)  # 不应到达：取消后不再发新请求
        return RunResult(run_id="", status=Status.COMPLETED, reply="never")

    runner = Runner(workflows={"job": WorkflowSpec(name="job", run=run)},
                    context_factory=lambda: make_context(model=model))
    agen = runner.stream(RunInput(workflow="job"))
    started = await agen.__anext__()
    assert started.kind is EventKind.RUN_STARTED  # 消费到 run.started，工作流已停在 gate

    response = runner.cancel(started.run_id)
    assert response["cancelled"] is True and response["reason"] == "cancelled"
    gate.set()  # 放行：取消在下一个能力调用前生效

    events = [event async for event in agen]
    result = completed_event([started, *events]).data["result"]
    assert result["status"] == "cancelled"
    assert result["error"]["code"] == "CANCELLED_BY_USER"
    assert result["recovery"]["final_source"] == "cancelled"
    assert result["recovery"]["fallback_attempted"] is False  # 取消不进入兜底
    assert model.call_count == 1  # 取消后没有新的模型请求
    assert holder["child"].cancelled()  # 挂起子任务已被回收
    assert calls["wf"] == 1
    # 活动登记已移除：重复取消返回明确未找到
    assert runner.cancel(started.run_id) == {
        "run_id": started.run_id, "cancelled": False, "reason": "not_found_or_finished"}


async def test_s03_task_cancel_propagates_after_cleanup():
    """S03：Python Task 取消在清理完成后向调用方传播 CancelledError。"""
    parked = asyncio.Event()
    gate = asyncio.Event()
    holder: dict = {}
    model = FakeModel(default_text="too-late")

    async def run(inputs, ctx):
        holder["ctx"] = ctx
        await _model_once(ctx)
        holder["child"] = ctx.track(asyncio.create_task(_hang()))
        parked.set()
        await gate.wait()  # execute 任务在此被 task.cancel() 打断
        return RunResult(run_id="", status=Status.COMPLETED, reply="never")

    runner = Runner(workflows={"job": WorkflowSpec(name="job", run=run)},
                    context_factory=lambda: make_context(model=model))
    task = asyncio.create_task(runner.execute(RunInput(workflow="job")))
    await parked.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task  # 清理（回收子任务/移除活动登记）完成后，取消传播给调用方

    ctx = holder["ctx"]
    assert ctx.cancelled is True  # 取消标记同步到上下文
    assert holder["child"].cancelled()  # 挂起子任务已被 await 回收
    assert model.call_count == 1  # 取消后没有新的能力请求
    assert runner.cancel(ctx.run_id)["reason"] == "not_found_or_finished"  # 活动登记已移除


# ---------------------------------------------------------------- Runner 入口守卫


def test_cancel_unknown_run_returns_not_found():
    """S03 边界：取消不存在的 run 返回明确未找到，不误报已撤销。"""
    runner = Runner(workflows={}, context_factory=make_context)
    assert runner.cancel("no-such-run") == {
        "run_id": "no-such-run", "cancelled": False, "reason": "not_found_or_finished"}


def test_unknown_fallback_owner_rejected_at_construction():
    """兜底所有者只允许 local/caller；非法值在构造期报 ValueError。"""
    with pytest.raises(ValueError):
        Runner(workflows={}, context_factory=make_context, fallback_owner="platform")
