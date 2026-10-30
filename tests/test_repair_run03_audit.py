"""RUN-03：以事件控制终态竞争，检查取消边界与执行资源回收。"""
import asyncio

import pytest

from flowagent.core.contracts import ErrorCode, EventKind, ModelEvent, RunInput, Status
from flowagent.core.runner import Runner
from flowagent.workflows.chat import build_chat_workflow
from tests.native_helpers import make_context


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('winner', ['completion', 'cancel'])
async def test_run03_completion_and_cancellation_have_one_terminal_and_cleanup(winner):
    model_ready, permit_answer = asyncio.Event(), asyncio.Event()
    model_closed, child_started, child_closed, resource_closed = (asyncio.Event() for _ in range(4))
    terminal_seen, release_consumer = asyncio.Event(), asyncio.Event()
    child_tasks = []
    events = []
    model_calls = 0
    ctx = None

    async def child():
        child_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            child_closed.set()

    class Model:
        async def stream(self, messages, tools, limits):
            nonlocal model_calls
            model_calls += 1
            child_tasks.append(ctx.track(asyncio.create_task(child())))
            await child_started.wait()
            model_ready.set()
            try:
                await permit_answer.wait()
                yield ModelEvent(kind='completed', text='FINAL', finish_reason='stop')
            finally:
                model_closed.set()

    async def close_resource():
        resource_closed.set()

    ctx = make_context(model=Model())
    ctx.run_id = 'terminal-race'
    ctx.closers.append(close_resource)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx, fallback_owner='local')

    async def consume():
        async for event in runner.stream(RunInput(workflow='chat', inputs={'message': 'hello'})):
            events.append(event)
            if event.kind is EventKind.RUN_FINISHED:
                terminal_seen.set()
                await release_consumer.wait()

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(model_ready.wait(), 2)
        if winner == 'cancel':
            # 先明确接受取消，再释放迟到的模型结果。
            outcome = runner.cancel(ctx.run_id)
            assert outcome['cancelled'] is True
            permit_answer.set()
            await asyncio.wait_for(terminal_seen.wait(), 2)
        else:
            # 等成功终态已提交后再取消，不能改变已完成结果。
            permit_answer.set()
            await asyncio.wait_for(terminal_seen.wait(), 2)
            outcome = runner.cancel(ctx.run_id)
            assert outcome == {'run_id': ctx.run_id, 'cancelled': False, 'reason': 'not_found_or_finished'}
        release_consumer.set()
        await asyncio.wait_for(consumer, 2)
    finally:
        permit_answer.set()
        release_consumer.set()
        if not consumer.done():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)

    finished = [event for event in events if event.kind is EventKind.RUN_FINISHED]
    assert len(finished) == 1
    result = finished[0].data['_result']
    public = finished[0].data['result']
    deltas = ''.join(event.data['text'] for event in events if event.kind is EventKind.TEXT_DELTA)
    if winner == 'cancel':
        assert result.status is Status.CANCELLED and result.error.code is ErrorCode.CANCELLED_BY_USER
        assert public['status'] == 'cancelled' and not result.reply and not deltas
        assert not result.recovery['fallback_attempted']
    else:
        assert result.status is Status.COMPLETED and public['status'] == 'completed'
        assert deltas == result.reply == public['reply'] == 'FINAL'
    assert model_calls == result.usage.model_requests == 1
    assert model_closed.is_set() and child_closed.is_set() and resource_closed.is_set()
    assert child_tasks[0].done() and not ctx._children and not runner._active
