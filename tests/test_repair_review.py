"""复审问题：批次副作用、主阶段错误与流清理。"""
import asyncio
from types import SimpleNamespace

import pytest

from flowagent.core.context import Budget
from flowagent.core.contracts import (ErrorCode, EventKind, FlowAgentCoreError, ModelEvent,
                                      RunInput, RunResult, Status, WorkflowSpec)
from flowagent.core.runner import Runner
from flowagent.workflows.chat import build_chat_workflow
from flowagent.workflows.group_analysis import _build_report, _classify_batch
from tests.native_helpers import FakeModel, FakeTools, make_context


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('prior', [False, True])
async def test_duplicate_batch_is_rejected_before_any_effect_in_that_batch(prior):
    repeated = {'name': 'echo', 'arguments': {'x': 1}}
    first = {'id': 'first', **repeated}
    duplicate = {'id': 'duplicate', **repeated}
    fresh = {'id': 'fresh', 'name': 'echo', 'arguments': {'x': 2}}
    turns = ([FakeModel.tool_turn(first), FakeModel.tool_turn(fresh, duplicate)] if prior
             else [FakeModel.tool_turn(first, duplicate)])
    tools = FakeTools(catalog=[{'name': 'echo'}])
    ctx = make_context(model=FakeModel(turns), tools=tools)
    result = await Runner(workflows={}, context_factory=lambda: ctx, fallback_owner='caller').run(
        RunInput(workflow='unknown', inputs={'message': 'hi'}))
    assert result.status is Status.FAILED and result.error.code is ErrorCode.DUPLICATE_TOOL_CALL
    assert tools.calls == ([('echo', {'x': 1})] if prior else [])


@pytest.mark.anyio
async def test_group_classifier_propagates_main_deadline_without_retry():
    model = FakeModel([FlowAgentCoreError(ErrorCode.MAIN_DEADLINE_EXCEEDED, 'main expired')])
    ctx = make_context(model=model)
    with pytest.raises(FlowAgentCoreError) as exc:
        await _classify_batch(ctx, [{'message_ref': 'm1', 'text': 'hello'}], ('x',))
    assert exc.value.code is ErrorCode.MAIN_DEADLINE_EXCEEDED
    assert model.call_count == 1 and ctx.budget.model_spent == 1


@pytest.mark.anyio
@pytest.mark.parametrize('repair', [False, True])
async def test_report_budget_exhaustion_is_terminal(repair):
    model = FakeModel([FakeModel.text_turn('报告\n引用: [invalid]')])
    budget = Budget(model_total=2 if repair else 1, mcp_total=32, deadline_ms=5000,
                    main_deadline_ms=5000, fallback_timeout_ms=100)
    ctx = make_context(model=model, budget=budget)
    control = SimpleNamespace(start='start', end_exclusive='end')
    with pytest.raises(FlowAgentCoreError) as exc:
        await _build_report(ctx, control, {'categories': []}, [], ['m1'])
    assert exc.value.code is ErrorCode.MODEL_BUDGET_EXCEEDED
    assert model.call_count == int(repair)


@pytest.mark.anyio
async def test_slow_stream_keeps_id_active_and_accepts_cancel_without_success_done():
    reply = 'x' * 2_300_000
    async def work(inputs, ctx):
        return RunResult(run_id='', status=Status.COMPLETED, reply=reply)
    def factory():
        ctx = make_context()
        ctx.run_id = 'slow-run'
        return ctx
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=factory)
    stream = runner.stream(RunInput(workflow='job'))
    try:
        consumed = []
        while not any(event.kind is EventKind.TEXT_DELTA for event in consumed):
            consumed.append(await stream.__anext__())
        await asyncio.sleep(0.05)  # 填满有界队列并让生产者等待背压。
        duplicate = await runner.run(RunInput(workflow='job'))
        assert duplicate.status is Status.FAILED and duplicate.error.code is ErrorCode.INVALID_INPUT
        assert runner.cancel('slow-run')['cancelled'] is True
        events = consumed + [event async for event in stream]
        result = [event.data['_result'] for event in events if event.kind is EventKind.RUN_FINISHED][0]
        partial = ''.join(event.data['text'] for event in events if event.kind is EventKind.TEXT_DELTA)
        assert partial == consumed[-1].data['text']  # 只保留已交付的前缀，不发送队列中的剩余部分。
        assert result.status is Status.CANCELLED and result.error.code is ErrorCode.CANCELLED_BY_USER
        assert not result.reply and not runner._active
    finally:
        await stream.aclose()


@pytest.mark.anyio
async def test_cancel_when_terminal_waits_on_full_queue_cannot_commit_success():
    release = asyncio.Event()
    async def work(inputs, ctx):
        await release.wait()
        for _ in range(1200):
            ctx.emit(EventKind.STEP_STARTED, {})
        return RunResult(run_id='', status=Status.COMPLETED)
    def factory():
        ctx = make_context()
        ctx.run_id = 'terminal-waits'
        return ctx
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=factory)
    stream = runner.stream(RunInput(workflow='job'))
    try:
        await stream.__anext__()
        release.set()
        await asyncio.sleep(0.05)
        assert runner.cancel('terminal-waits')['cancelled'] is True
        events = [event async for event in stream]
        result = [event.data['_result'] for event in events if event.kind is EventKind.RUN_FINISHED][0]
        assert result.status is Status.CANCELLED and result.error.code is ErrorCode.CANCELLED_BY_USER
    finally:
        await stream.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('trigger', ['cancel', 'deadline'])
async def test_stalled_consumer_releases_run_and_discards_unread_output(trigger):
    reply = 'x' * 2_300_000
    async def work(inputs, ctx):
        return RunResult(run_id='', status=Status.COMPLETED, reply=reply)
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=1500 if trigger == 'deadline' else 5000,
                    main_deadline_ms=1500 if trigger == 'deadline' else 5000, fallback_timeout_ms=100)
    contexts = []
    def factory():
        ctx = make_context(budget=budget)
        ctx.run_id = 'stalled-consumer'
        contexts.append(ctx)
        return ctx
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=factory)
    stream = runner.stream(RunInput(workflow='job'))
    try:
        seen = []
        while not any(event.kind is EventKind.TEXT_DELTA for event in seen):
            seen.append(await stream.__anext__())
        async def filled():
            while contexts[0]._seq < 1026:
                await asyncio.sleep(0.001)
        await asyncio.wait_for(filled(), 1)  # 两个事件已消费，队列中还有 1024 个事件。
        if trigger == 'cancel':
            assert runner.cancel('stalled-consumer')['cancelled'] is True
        async def released():
            while 'stalled-consumer' in runner._active:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(released(), 2)
        remaining = [event async for event in stream]
        assert not any(event.kind is EventKind.TEXT_DELTA for event in remaining)
        result = [event.data['_result'] for event in remaining if event.kind is EventKind.RUN_FINISHED][0]
        assert result.status is (Status.CANCELLED if trigger == 'cancel' else Status.FAILED)
        assert result.error.code is (ErrorCode.CANCELLED_BY_USER if trigger == 'cancel'
                                     else ErrorCode.DEADLINE_EXCEEDED)
        assert not result.reply
    finally:
        await stream.aclose()


@pytest.mark.anyio
async def test_hung_model_iterator_close_is_bounded():
    entered, release = asyncio.Event(), asyncio.Event()
    class Iterator:
        def __init__(self): self.sent = False
        def __aiter__(self): return self
        async def __anext__(self):
            if self.sent: raise StopAsyncIteration
            self.sent = True
            return ModelEvent(kind='completed', text='done', finish_reason='stop')
        async def aclose(self):
            entered.set()
            try: await release.wait()
            except asyncio.CancelledError: await release.wait()
    class Model:
        def stream(self, *args): return Iterator()
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=120, main_deadline_ms=120,
                    fallback_timeout_ms=100)
    ctx = make_context(model=Model(), budget=budget)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx, fallback_owner='caller')
    task = asyncio.create_task(runner.run(RunInput(workflow='chat', inputs={'message': 'hi'})))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        done, _ = await asyncio.wait({task}, timeout=0.5)
        assert task in done, 'model iterator cleanup blocked the deadline'
        result = task.result()
        assert result.status is Status.FAILED and result.error.code is ErrorCode.DEADLINE_EXCEEDED
        assert result.recovery['capability_cleanup_pending'] is True
    finally:
        release.set()
        await asyncio.wait_for(task, 1)
