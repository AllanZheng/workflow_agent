"""RUN-02：真实聊天流程在取消后的迟到模型结果不可发布。"""

import asyncio

import pytest

from flowagent.core.contracts import ErrorCode, EventKind, ModelEvent, RunInput, Status
from flowagent.core.runner import Runner
from flowagent.workflows.chat import build_chat_workflow
from tests.native_helpers import FakeTools, make_context


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('fallback_owner', ['local', 'caller'])
@pytest.mark.parametrize('answer_kind', ['text', 'tool_call'])
async def test_cancelled_chat_discards_model_result_that_arrives_late(fallback_owner, answer_kind):
    waiting_for_final = asyncio.Event()
    release_final = asyncio.Event()
    final_produced = asyncio.Event()
    model_closed = asyncio.Event()
    events = []
    model_calls = []
    tools = FakeTools()

    class Model:
        async def stream(self, messages, schemas, limits):
            model_calls.append((messages, schemas, limits))
            try:
                yield ModelEvent(kind='text_delta', text='UNVERIFIED_DRAFT')
                waiting_for_final.set()
                try:
                    await release_final.wait()
                except asyncio.CancelledError:
                    # 模拟模型侧已生成结果、不能保证上游取消会撤回它。
                    await release_final.wait()
                final_produced.set()
                if answer_kind == 'text':
                    yield ModelEvent(kind='completed', text='LATE_ANSWER', finish_reason='stop')
                else:
                    # 迟到结果若要求调用工具，也不能继续发起新的能力请求。
                    yield ModelEvent(kind='completed', finish_reason='tool_calls', tool_calls=[{
                        'id': 'late-call', 'name': 'echo', 'arguments': '{}',
                    }])
            finally:
                model_closed.set()

    ctx = make_context(model=Model(), tools=tools)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx,
                    fallback_owner=fallback_owner)

    async def consume():
        async for event in runner.stream(RunInput(workflow='chat', inputs={'message': 'hello'})):
            events.append(event)

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(waiting_for_final.wait(), 2)
        cancelled = runner.cancel(ctx.run_id)
        assert cancelled == {'run_id': ctx.run_id, 'cancelled': True, 'reason': 'cancelled'}
        release_final.set()
        await asyncio.wait_for(task, 2)
    finally:
        release_final.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # 验证上游确实释放了迟到结果，而非只测一个挂起流被取消。
    assert final_produced.is_set() and model_closed.is_set()
    final = [event for event in events if event.kind is EventKind.RUN_FINISHED]
    assert len(final) == 1
    result = final[0].data['_result']
    assert result.status is Status.CANCELLED and result.error.code is ErrorCode.CANCELLED_BY_USER
    assert result.recovery['final_source'] == 'cancelled'
    assert result.recovery['fallback_attempted'] is False
    assert not result.reply and result.output is None
    assert not [event for event in events if event.kind is EventKind.TEXT_DELTA]
    assert len(model_calls) == result.usage.model_requests == 1
    assert tools.calls == [] and not ctx._children and not runner._active
