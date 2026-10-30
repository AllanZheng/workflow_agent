"""执行身份、取消、正文与工具协议的离线回归。"""
import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.platform import context as platform_context
from flowagent.adapters.platform.protocol import ExecutionRequest
from flowagent.app import build_platform_service
from flowagent.core.context import Budget
from flowagent.core.contracts import EventKind, ModelEvent, RunInput, RunResult, Status, WorkflowSpec
from flowagent.core.runner import Runner
from flowagent.workflows.chat import build_chat_workflow
from tests.native_helpers import FakeModel, FakeTools, make_context
from tests.test_managed_config_v3 import _envelope
from tests.test_platform_v3_adapter import _install_offline_stack, _text_chunk, _final_chunk
from tests.test_repair_context import _envelope_with_context


@pytest.fixture
def anyio_backend():
    return 'asyncio'


def test_platform_run_id_is_canonical_execution_uuid():
    envelope = _envelope()
    request = ExecutionRequest.model_validate(envelope)
    caps = platform_context.build_capabilities(request)
    token = platform_context.set_current_capabilities(caps)
    try:
        ctx = platform_context.context_factory()()
    finally:
        platform_context.reset_capabilities(token)
    assert ctx.run_id == str(uuid.UUID(request.execution_id))


def test_main_deadline_maps_to_platform_recoverable_error():
    from flowagent.adapters.platform.receipts import error_envelope
    body, status = error_envelope('MAIN_DEADLINE_EXCEEDED', 'main deadline exceeded')
    assert status == 502 and body['error']['code'] == 'main_deadline_exceeded'


@pytest.mark.anyio
async def test_duplicate_active_run_id_cannot_replace_first_run():
    entered = asyncio.Event()
    async def work(inputs, ctx):
        entered.set()
        await ctx.wait_cancelled()
        ctx.check_cancel()
    def factory():
        ctx = make_context()
        ctx.run_id = 'same-id'
        return ctx
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=factory)
    first = asyncio.create_task(runner.run(RunInput(workflow='job')))
    await asyncio.wait_for(entered.wait(), 1)
    duplicate = await asyncio.wait_for(runner.run(RunInput(workflow='job')), 0.4)
    assert duplicate.status is Status.FAILED
    assert runner.cancel('same-id')['cancelled'] is True
    result = await asyncio.wait_for(first, 0.4)
    assert result.status is Status.CANCELLED and not runner._active


@pytest.mark.anyio
@pytest.mark.parametrize('trigger', ['deadline', 'cancel'])
async def test_hung_model_obeys_deadline_and_cancel(trigger):
    entered, stopped = asyncio.Event(), asyncio.Event()
    class Model:
        async def stream(self, messages, tools, limits):
            entered.set()
            try:
                await asyncio.Event().wait()
                yield ModelEvent(kind='completed', text='never')
            finally:
                stopped.set()
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=50 if trigger == 'deadline' else 5000,
                    main_deadline_ms=50 if trigger == 'deadline' else 5000, fallback_timeout_ms=100)
    ctx = make_context(model=Model(), budget=budget)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx, fallback_owner='caller')
    task = asyncio.create_task(runner.run(RunInput(workflow='chat', inputs={'message': 'hi'})))
    await asyncio.wait_for(entered.wait(), 1)
    if trigger == 'cancel':
        runner.cancel(ctx.run_id)
    result = await asyncio.wait_for(task, 0.4)
    assert stopped.is_set() and result.status is (Status.CANCELLED if trigger == 'cancel' else Status.FAILED)
    assert result.error.code.value == ('CANCELLED_BY_USER' if trigger == 'cancel' else 'DEADLINE_EXCEEDED')
    assert not result.recovery['fallback_attempted']


@pytest.mark.anyio
@pytest.mark.parametrize('owner', ['local', 'caller'])
async def test_main_deadline_preserves_remaining_budget_for_unique_fallback_owner(owner):
    class Model:
        def __init__(self): self.calls = 0
        async def stream(self, messages, tools, limits):
            self.calls += 1
            if self.calls == 1:
                await asyncio.Event().wait()
            yield ModelEvent(kind='completed', text='fallback', finish_reason='stop')
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=1500, main_deadline_ms=80,
                    fallback_timeout_ms=1000)
    model = Model()
    ctx = make_context(model=model, budget=budget)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx, fallback_owner=owner)
    result = await asyncio.wait_for(runner.run(RunInput(workflow='chat', inputs={'message': 'hi'})), 1.4)
    assert result.recovery['primary_error']['code'] == 'MAIN_DEADLINE_EXCEEDED'
    if owner == 'local':
        assert result.status is Status.DEGRADED and result.reply == 'fallback' and model.calls == 2
        assert result.recovery['fallback_attempted'] is True
    else:
        assert result.status is Status.FAILED and result.error.code.value == 'MAIN_DEADLINE_EXCEEDED'
        assert result.recovery['final_source'] == 'caller' and model.calls == 1
        assert result.recovery['fallback_attempted'] is False


@pytest.mark.anyio
async def test_total_deadline_expires_during_fallback_without_restarting_clock():
    entered = asyncio.Event()
    class Model:
        def __init__(self): self.calls = 0
        async def stream(self, messages, tools, limits):
            self.calls += 1
            if self.calls == 2: entered.set()
            await asyncio.Event().wait()
            yield ModelEvent(kind='completed', text='never')
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=250, main_deadline_ms=70,
                    fallback_timeout_ms=5000)
    model = Model()
    ctx = make_context(model=model, budget=budget)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx, fallback_owner='local')
    result = await asyncio.wait_for(runner.run(RunInput(workflow='chat', inputs={'message': 'hi'})), 0.6)
    assert entered.is_set() and model.calls == 2
    assert result.status is Status.FAILED and result.error.code.value == 'DEADLINE_EXCEEDED'


@pytest.mark.anyio
async def test_cancel_during_local_fallback_stays_cancelled():
    entered = asyncio.Event()
    class Model:
        async def stream(self, messages, tools, limits):
            entered.set()
            await asyncio.Event().wait()
            yield ModelEvent(kind='completed', text='never')
    async def work(inputs, ctx):
        from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
        raise FlowAgentCoreError(ErrorCode.CAPABILITY_UNAVAILABLE, 'failed')
    ctx = make_context(model=Model())
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=lambda: ctx)
    task = asyncio.create_task(runner.run(RunInput(workflow='job', inputs={'message': 'hi'})))
    await asyncio.wait_for(entered.wait(), 1)
    runner.cancel(ctx.run_id)
    result = await asyncio.wait_for(task, 0.4)
    assert result.status is Status.CANCELLED and result.error.code.value == 'CANCELLED_BY_USER'
    assert result.output is None and not result.reply


@pytest.mark.anyio
@pytest.mark.parametrize('stage', ['preflight', 'discovery', 'tool', 'skill'])
@pytest.mark.parametrize('trigger', ['cancel', 'deadline'])
async def test_capability_cancel_or_deadline_reclaims_without_fallback(stage, trigger):
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def hang():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    class Tools(FakeTools):
        async def list(self):
            if stage == 'discovery': await hang()
            return await super().list()
        async def call(self, name, arguments):
            if stage == 'tool': await hang()
            return await super().call(name, arguments)
    class Skills:
        def catalog(self): return [{'skill_id': 's1', 'name': 'skill', 'summary': 'summary'}]
        async def load(self, skill_id): await hang()
    tools = Tools(catalog=[{'name': 'echo'}])
    if stage == 'tool':
        model = FakeModel([FakeModel.tool_turn({'id': 'c1', 'name': 'echo', 'arguments': {}})])
    elif stage == 'skill':
        model = FakeModel([FakeModel.tool_turn({'id': 'c1', 'name': 'load_skill', 'arguments': {'skill_id': 's1'}})])
    else:
        model = FakeModel(default_text='never')
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=200 if trigger == 'deadline' else 5000,
                    main_deadline_ms=200 if trigger == 'deadline' else 5000, fallback_timeout_ms=100)
    ctx = make_context(model=model, tools=tools, skills=Skills() if stage == 'skill' else None, budget=budget)
    if stage == 'preflight':
        async def preflight(inputs, ctx): await hang()
        async def work(inputs, ctx): return RunResult(run_id='', status=Status.COMPLETED, reply='never')
        workflow = WorkflowSpec(name='job', run=work, preflight=preflight)
    else:
        workflow = build_chat_workflow()
    runner = Runner(workflows={workflow.name: workflow}, context_factory=lambda: ctx, fallback_owner='caller')
    task = asyncio.create_task(runner.run(RunInput(workflow=workflow.name, inputs={'message': 'hi'})))
    await asyncio.wait_for(entered.wait(), 1)
    if trigger == 'cancel':
        runner.cancel(ctx.run_id)
    result = await asyncio.wait_for(task, 0.6)
    assert stopped.is_set() and result.status is (Status.CANCELLED if trigger == 'cancel' else Status.FAILED)
    assert result.error.code.value == ('CANCELLED_BY_USER' if trigger == 'cancel' else 'DEADLINE_EXCEEDED')
    assert not result.recovery['fallback_attempted']
    if stage == 'tool':
        assert result.recovery['capability_outcome'] == 'unknown'


@pytest.mark.anyio
@pytest.mark.parametrize('workflow_kind', ['group', 'knowledge', 'consultation'])
@pytest.mark.parametrize('trigger', ['cancel', 'deadline'])
async def test_fixed_workflow_remote_call_obeys_cancel_and_deadline(workflow_kind, trigger):
    from types import SimpleNamespace
    from flowagent.workflows.consultation import build_consultation_workflow
    from flowagent.workflows.group_analysis import build_group_analysis_workflow, FETCH_TOOL
    from flowagent.workflows.knowledge import build_knowledge_workflow
    entered, stopped = asyncio.Event(), asyncio.Event()
    name = FETCH_TOOL if workflow_kind == 'group' else 'wiki__hybrid_search'
    class Tools(FakeTools):
        async def call(self, name, arguments):
            entered.set()
            try: await asyncio.Event().wait()
            finally: stopped.set()
    tools = Tools(catalog=[{'name': name}])
    if workflow_kind == 'group':
        workflow = build_group_analysis_workflow()
        inputs = {'message': '分析本群', 'analysis_target': SimpleNamespace(conversation_id=42)}
    elif workflow_kind == 'knowledge':
        workflow = build_knowledge_workflow(bindings={'search': name})
        inputs = {'message': 'question'}
    else:
        workflow = build_consultation_workflow()
        inputs = {'message': 'materials', 'function_task': {'stage': 'draft',
                  'output_schema': 'consultation_document_v1', 'inputs': {'knowledge_mode': 'required'}}}
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=200 if trigger == 'deadline' else 5000,
                    main_deadline_ms=200 if trigger == 'deadline' else 5000, fallback_timeout_ms=100)
    model = FakeModel(default_text='never')
    ctx = make_context(model=model, tools=tools, budget=budget)
    runner = Runner(workflows={workflow.name: workflow}, context_factory=lambda: ctx, fallback_owner='caller')
    task = asyncio.create_task(runner.run(RunInput(workflow=workflow.name, inputs=inputs)))
    await asyncio.wait_for(entered.wait(), 1)
    if trigger == 'cancel': runner.cancel(ctx.run_id)
    result = await asyncio.wait_for(task, 0.6)
    assert stopped.is_set() and result.status is (Status.CANCELLED if trigger == 'cancel' else Status.FAILED)
    assert result.error.code.value == ('CANCELLED_BY_USER' if trigger == 'cancel' else 'DEADLINE_EXCEEDED')
    assert result.usage.mcp_attempts == 1 and model.call_count == 0
    assert result.recovery['capability_outcome'] == 'unknown' and not result.recovery['fallback_attempted']


@pytest.mark.anyio
@pytest.mark.parametrize('turns,expected', [
    ([[ModelEvent(kind='text_delta', text='discard prelude'),
       ModelEvent(kind='completed', text='discard prelude', finish_reason='tool_calls',
                  tool_calls=[{'id': 'c1', 'name': 'echo', 'arguments': '{}'}])],
      FakeModel.text_turn('final answer')], 'final answer'),
    ([FakeModel.text_turn('discard truncated', finish_reason='length'), FakeModel.text_turn('retry answer')],
     'retry answer'),
])
async def test_stream_only_publishes_validated_final_reply(turns, expected):
    model = FakeModel(turns)
    ctx = make_context(model=model, tools=FakeTools(catalog=[{'name': 'echo'}]))
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx, fallback_owner='caller')
    events = [event async for event in runner.stream(RunInput(workflow='chat', inputs={'message': 'hi'}))]
    result = [event for event in events if event.kind is EventKind.RUN_FINISHED][0].data['_result']
    assert result.reply == expected
    assert ''.join(event.data['text'] for event in events if event.kind is EventKind.TEXT_DELTA) == expected


@pytest.mark.anyio
async def test_slow_consumer_keeps_all_final_text_and_terminal_event():
    reply = 'x' * 2_200_000  # 正文切块超过 1024 条，必须经过队列背压。
    async def work(inputs, ctx):
        for i in range(1500): ctx.emit(EventKind.STEP_STARTED, {'index': i})
        return RunResult(run_id='', status=Status.COMPLETED, reply=reply)
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=make_context)
    stream = runner.stream(RunInput(workflow='job'))
    first = await stream.__anext__()
    await asyncio.sleep(0)  # 消费者暂停，允许进度队列充满。
    events = [first] + [event async for event in stream]
    deltas = [event.data['text'] for event in events if event.kind is EventKind.TEXT_DELTA]
    assert ''.join(deltas) == reply and all(len(text) <= 4096 for text in deltas)
    assert len([event for event in events if event.kind is EventKind.RUN_FINISHED]) == 1


@pytest.mark.anyio
async def test_failed_validation_stream_publishes_only_local_fallback():
    workflow = build_chat_workflow()
    workflow.validate = lambda result, ctx: False
    model = FakeModel([FakeModel.text_turn('candidate'), FakeModel.text_turn('fallback')])
    ctx = make_context(model=model)
    runner = Runner(workflows={'chat': workflow}, context_factory=lambda: ctx, fallback_owner='local')
    events = [event async for event in runner.stream(RunInput(workflow='chat', inputs={'message': 'hi'}))]
    result = [event for event in events if event.kind is EventKind.RUN_FINISHED][0].data['_result']
    assert result.status is Status.DEGRADED and result.reply == 'fallback'
    assert ''.join(event.data['text'] for event in events if event.kind is EventKind.TEXT_DELTA) == 'fallback'


@pytest.mark.anyio
@pytest.mark.parametrize('kind,expected', [('model', 'MODEL_BUDGET_EXCEEDED'), ('tool', 'TOOL_BUDGET_EXCEEDED')])
async def test_fixed_workflow_budget_exceptions_keep_resource_specific_error(kind, expected):
    async def work(inputs, ctx):
        if kind == 'model': await ctx.budget.reserve_model()
        else: await ctx.budget.reserve_tool()
        return RunResult(run_id='', status=Status.COMPLETED)
    budget = Budget(model_total=0, mcp_total=0, deadline_ms=5000, main_deadline_ms=5000,
                    fallback_timeout_ms=100)
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)},
                    context_factory=lambda: make_context(budget=budget), fallback_owner='caller')
    result = await runner.run(RunInput(workflow='job'))
    assert result.status is Status.FAILED and result.error.code.value == expected


@pytest.mark.anyio
async def test_model_attempt_is_not_counted_when_main_deadline_prevents_dispatch():
    async def work(inputs, ctx):
        async for _ in ctx.stream_model([{'role': 'user', 'content': 'hi'}], [], {}):
            pass
        return RunResult(run_id='', status=Status.COMPLETED)
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=5000, main_deadline_ms=0,
                    fallback_timeout_ms=100)
    model = FakeModel(default_text='never')
    ctx = make_context(model=model, budget=budget)
    result = await Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=lambda: ctx,
                          fallback_owner='caller').run(RunInput(workflow='job'))
    assert result.status is Status.FAILED and result.error.code.value == 'MAIN_DEADLINE_EXCEEDED'
    assert model.call_count == result.usage.model_requests == 0


@pytest.mark.anyio
async def test_tool_attempt_is_not_counted_when_main_deadline_prevents_dispatch():
    from flowagent.core.agent import ToolRegistry, _execute_remote
    from flowagent.core.contracts import FlowAgentCoreError
    tools = FakeTools(catalog=[{'name': 'echo'}])
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=5000, main_deadline_ms=0,
                    fallback_timeout_ms=100)
    ctx = make_context(tools=tools, budget=budget)
    with pytest.raises(FlowAgentCoreError) as exc:
        await _execute_remote(ctx, ToolRegistry(remote=tools), 'echo', {})
    assert exc.value.code.value == 'MAIN_DEADLINE_EXCEEDED'
    assert not tools.calls and ctx.usage.mcp_attempts == 0


@pytest.mark.anyio
async def test_skill_load_consumes_tool_budget_once_and_cache_is_free():
    from tests.native_helpers import FakeSkills
    skills = FakeSkills()
    model = FakeModel([FakeModel.tool_turn({'id': 'c1', 'name': 'load_skill', 'arguments': {'skill_id': 's1'}}),
                       FakeModel.tool_turn({'id': 'c2', 'name': 'load_skill', 'arguments': {'skill_id': 's1'}}),
                       FakeModel.text_turn('done')])
    budget = Budget(model_total=24, mcp_total=1, deadline_ms=5000, main_deadline_ms=5000,
                    fallback_timeout_ms=100)
    ctx = make_context(model=model, skills=skills, budget=budget)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx, fallback_owner='caller')
    result = await runner.run(RunInput(workflow='chat', inputs={'message': 'hi'}))
    assert result.status is Status.COMPLETED and skills.load_calls == ['s1']
    assert (budget.mcp_spent, result.usage.skill_loads, result.usage.mcp_attempts) == (1, 1, 0)


@pytest.mark.anyio
async def test_skill_load_with_zero_tool_budget_never_starts():
    from tests.native_helpers import FakeSkills
    skills = FakeSkills()
    model = FakeModel([FakeModel.tool_turn({'id': 'c1', 'name': 'load_skill', 'arguments': {'skill_id': 's1'}})])
    budget = Budget(model_total=24, mcp_total=0, deadline_ms=5000, main_deadline_ms=5000,
                    fallback_timeout_ms=100)
    ctx = make_context(model=model, skills=skills, budget=budget)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx, fallback_owner='caller')
    result = await runner.run(RunInput(workflow='chat', inputs={'message': 'hi'}))
    assert result.status is Status.FAILED and result.error.code.value == 'TOOL_BUDGET_EXCEEDED'
    assert not skills.load_calls and result.usage.skill_loads == result.usage.mcp_attempts == 0


@pytest.mark.anyio
async def test_failed_skill_load_is_counted_as_an_attempt():
    from tests.native_helpers import FakeSkills
    skills = FakeSkills()
    skills.bodies.clear()
    model = FakeModel([FakeModel.tool_turn({'id': 'c1', 'name': 'load_skill', 'arguments': {'skill_id': 's1'}}),
                       FakeModel.text_turn('failed to load')])
    budget = Budget(model_total=24, mcp_total=1, deadline_ms=5000, main_deadline_ms=5000,
                    fallback_timeout_ms=100)
    ctx = make_context(model=model, skills=skills, budget=budget)
    result = await Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx,
                          fallback_owner='caller').run(RunInput(workflow='chat', inputs={'message': 'hi'}))
    assert result.status is Status.COMPLETED and skills.load_calls == ['s1']
    assert (budget.mcp_spent, result.usage.skill_loads, result.usage.mcp_attempts) == (1, 1, 0)


@pytest.mark.anyio
async def test_cancel_accepted_before_publish_discards_completed_output():
    async def work(inputs, ctx):
        ctx.cancel()
        return RunResult(run_id='', status=Status.COMPLETED, reply='must not publish', output={'result': 'fake'})
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=make_context)
    events = [event async for event in runner.stream(RunInput(workflow='job'))]
    result = [event for event in events if event.kind is EventKind.RUN_FINISHED][0].data['_result']
    assert result.status is Status.CANCELLED and not result.reply and result.output is None
    assert not [event for event in events if event.kind is EventKind.TEXT_DELTA]


@pytest.mark.anyio
async def test_failed_workflow_does_not_publish_unvalidated_body():
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    async def work(inputs, ctx):
        return RunResult(run_id='', status=Status.FAILED, reply='unvalidated',
                         output={'business': 'invalid'}, error=FlowAgentCoreError(ErrorCode.OUTPUT_INVALID, 'bad'))
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=make_context,
                    fallback_owner='caller')
    events = [event async for event in runner.stream(RunInput(workflow='job'))]
    result = [event for event in events if event.kind is EventKind.RUN_FINISHED][0].data['_result']
    assert result.status is Status.FAILED and result.reply == '' and result.output is None
    assert not [event for event in events if event.kind is EventKind.TEXT_DELTA]


@pytest.mark.anyio
async def test_hung_cleanup_does_not_publish_past_deadline():
    entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def closer():
        entered.set()
        try:
            while not release.is_set():
                try: await release.wait()
                except asyncio.CancelledError: continue
        finally:
            stopped.set()
    async def work(inputs, ctx):
        ctx.closers.append(closer)
        return RunResult(run_id='', status=Status.COMPLETED, reply='should not publish')
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=50, main_deadline_ms=50,
                    fallback_timeout_ms=10)
    ctx = make_context(budget=budget)
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)}, context_factory=lambda: ctx)
    task = asyncio.create_task(runner.run(RunInput(workflow='job')))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        result = await asyncio.wait_for(asyncio.shield(task), 0.4)
        assert result.status is Status.FAILED and result.error.code.value == 'DEADLINE_EXCEEDED'
        assert not result.reply and result.recovery['capability_cleanup_pending'] is True
    finally:
        release.set()
        await asyncio.wait_for(stopped.wait(), 1)
        await asyncio.wait_for(task, 1)


@pytest.mark.anyio
async def test_cancel_during_cleanup_wins_over_unpublished_failure():
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def closer():
        entered.set()
        try: await asyncio.Event().wait()
        finally: stopped.set()
    async def work(inputs, ctx):
        ctx.closers.append(closer)
        raise FlowAgentCoreError(ErrorCode.OUTPUT_INVALID, 'failed before cleanup')
    ctx = make_context()
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=work)},
                    context_factory=lambda: ctx, fallback_owner='caller')
    task = asyncio.create_task(runner.run(RunInput(workflow='job')))
    await asyncio.wait_for(entered.wait(), 1)
    assert runner.cancel(ctx.run_id)['cancelled'] is True
    result = await asyncio.wait_for(task, 0.4)
    assert stopped.is_set() and result.status is Status.CANCELLED
    assert result.error.code.value == 'CANCELLED_BY_USER' and not result.reply


@pytest.mark.anyio
async def test_invalid_tool_batch_has_assistant_call_before_each_error():
    model = FakeModel([FakeModel.tool_turn({'id': 'c1', 'name': 'unknown', 'arguments': {}},
                                           {'id': 'c2', 'name': 'echo', 'arguments': {}}),
                       FakeModel.text_turn('fixed')])
    tools = FakeTools(catalog=[{'name': 'echo'}])
    ctx = make_context(model=model, tools=tools)
    result = await Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx,
                          fallback_owner='caller').run(RunInput(workflow='chat', inputs={'message': 'hi'}))
    assert result.status is Status.COMPLETED and not tools.calls
    messages = model.requests[1]['messages']
    assert messages[-3]['role'] == 'assistant' and [call['id'] for call in messages[-3]['tool_calls']] == ['c1', 'c2']
    assert [m['tool_call_id'] for m in messages[-2:]] == ['c1', 'c2']


@pytest.mark.anyio
@pytest.mark.parametrize('mode', ['bad_json', 'mixed_terminal'])
async def test_other_invalid_batches_keep_protocol_pairing_and_execute_nothing(mode):
    from flowagent.core.agent import LocalTool
    terminal_calls = []
    async def terminal(arguments, ctx):
        terminal_calls.append(arguments)
        raise AssertionError('terminal must not execute in mixed batch')
    local = LocalTool(name='finish', terminal=True, handler=terminal)
    calls = ([{'id': 'c1', 'name': 'echo', 'arguments': '{'},
              {'id': 'c2', 'name': 'echo', 'arguments': '{}'}] if mode == 'bad_json' else
             [{'id': 'c1', 'name': 'finish', 'arguments': '{}'},
              {'id': 'c2', 'name': 'echo', 'arguments': '{}'}])
    model = FakeModel([[ModelEvent(kind='completed', tool_calls=calls, finish_reason='tool_calls')],
                       FakeModel.text_turn('corrected')])
    tools = FakeTools(catalog=[{'name': 'echo'}])
    ctx = make_context(model=model, tools=tools)
    result = await Runner(workflows={'chat': build_chat_workflow(terminal_tools=[local])}, context_factory=lambda: ctx,
                          fallback_owner='caller').run(RunInput(workflow='chat', inputs={'message': 'hi'}))
    assert result.status is Status.COMPLETED and not tools.calls and not terminal_calls
    messages = model.requests[1]['messages']
    assert [call['id'] for call in messages[-3]['tool_calls']] == ['c1', 'c2']
    assert [item['tool_call_id'] for item in messages[-2:]] == ['c1', 'c2']


@pytest.mark.anyio
@pytest.mark.parametrize('ids', [[None], ['dup', 'dup']])
async def test_missing_or_duplicate_tool_ids_fail_without_executing_tools(ids):
    calls = [{'id': value, 'name': 'echo', 'arguments': '{}'} for value in ids]
    model = FakeModel([[ModelEvent(kind='completed', tool_calls=calls, finish_reason='tool_calls')]])
    tools = FakeTools(catalog=[{'name': 'echo'}])
    ctx = make_context(model=model, tools=tools)
    result = await Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: ctx,
                          fallback_owner='caller').run(RunInput(workflow='chat', inputs={'message': 'hi'}))
    assert result.status is Status.FAILED and result.error.code.value == 'MODEL_INVALID_RESPONSE'
    assert model.call_count == 1 and not tools.calls


def test_sse_done_includes_actual_used_context_refs(monkeypatch):
    _install_offline_stack(monkeypatch, turns=[[_text_chunk('ok'), _final_chunk()]])
    response = TestClient(build_platform_service()).post('/execution-v1/respond/stream', json=_envelope_with_context())
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    done = [event for event in events if event['kind'] == 'done'][0]
    assert done['metadata']['used_context_refs'] == ['mem-1']
    assert done['metadata']['used_context_refs'] == done['metadata']['agent_config_applied']['memory_included_ids']


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_platform_cancel_uses_canonical_execution_id_end_to_end(monkeypatch, stream):
    import httpx
    from types import SimpleNamespace
    entered, stopped = asyncio.Event(), asyncio.Event()
    class Allow:
        def __init__(self, *args, **kwargs): pass
        async def authorize(self, **kwargs): pass
    async def model_stream():
        entered.set()
        try:
            await asyncio.Event().wait()
            yield _text_chunk('never')
        finally:
            stopped.set()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: model_stream())))
    real_build = platform_context.build_capabilities
    monkeypatch.setattr(platform_context, 'DemoModelAuthorizer', Allow)
    monkeypatch.setattr(platform_context, 'build_capabilities',
                        lambda request: real_build(request, model_transport=sdk))
    envelope = _envelope()
    app = build_platform_service()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test.local') as client:
        path = '/execution-v1/respond/stream' if stream else '/execution-v1/respond'
        run = asyncio.create_task(client.post(path, json=envelope))
        await asyncio.wait_for(entered.wait(), 1)
        cancel = await client.post('/execution-v1/cancel', json={'execution_id': envelope['execution_id'].upper()})
        assert cancel.json()['cancelled'] is True
        response = await asyncio.wait_for(run, 0.5)
    assert stopped.is_set()
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        assert response.status_code == 200 and [event['kind'] for event in events] == ['meta', 'error']
        assert events[-1]['error_code'] == 'cancelled'
    else:
        assert response.status_code == 499 and response.json()['error']['code'] == 'cancelled'
