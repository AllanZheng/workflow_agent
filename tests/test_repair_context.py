import copy

import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.platform import context as platform_context
from flowagent.adapters.platform.protocol import ExecutionRequest
from flowagent.app import build_platform_service
from flowagent.core.contracts import RunInput, render_clarification_marker
from flowagent.core.runner import Runner
from flowagent.workflows.chat import build_chat_workflow
from tests.native_helpers import FakeModel, make_context
from tests.test_managed_config_v3 import _digest, _envelope, _snapshot
from tests.test_platform_v3_adapter import _install_offline_stack, _text_chunk, _final_chunk


@pytest.fixture
def anyio_backend():
    return 'asyncio'


def _envelope_with_context():
    snapshot = _snapshot()
    snapshot['system_prompt'] = snapshot['config']['system_prompt'] = '冻结系统提示'
    snapshot['model']['parameters'] = {'temperature': 0.3, 'top_p': 0.8, 'max_tokens': 99,
                                      'presence_penalty': 0.1, 'frequency_penalty': -0.2, 'stop': ['END']}
    snapshot['memory']['read'] = snapshot['config']['memory_read'] = True
    snapshot['config_digest'] = _digest(snapshot)
    envelope = _envelope(snapshot)
    envelope['memory_context'] = {'schema_version': 1, 'status': 'ok', 'rendered_text': '记忆内容',
                                  'included_ids': ['mem-1'], 'items': []}
    return envelope


def test_platform_model_request_applies_snapshot_and_records_actual_usage(monkeypatch):
    transport, _ = _install_offline_stack(monkeypatch, turns=[[_text_chunk('好'), _final_chunk()]])
    client = TestClient(build_platform_service())
    response = client.post('/execution-v1/respond', json=_envelope_with_context())
    assert response.status_code == 200, response.text
    sent = transport.requests[0]
    assert '冻结系统提示' in sent['messages'][0]['content']
    assert '记忆内容' in sent['messages'][0]['content']
    assert {k: sent[k] for k in ('temperature', 'top_p', 'max_tokens', 'presence_penalty', 'frequency_penalty', 'stop')} == {
        'temperature': 0.3, 'top_p': 0.8, 'max_tokens': 99, 'presence_penalty': 0.1,
        'frequency_penalty': -0.2, 'stop': ['END']}
    body = response.json()
    receipt = body['metadata']['agent_config_applied']
    assert receipt['model_request_count'] == 1
    assert receipt['applied_model_parameters'] == _envelope_with_context()['agent_config']['snapshot']['model']['parameters']
    assert receipt['memory_injected'] is True and receipt['memory_included_ids'] == ['mem-1']
    assert body['used_context_refs'] == ['mem-1']


@pytest.mark.anyio
async def test_native_top_level_history_and_continuation_win_without_mutation():
    model = FakeModel(default_text='ok')
    seen = {}
    workflow = build_chat_workflow()
    original_run = workflow.run
    async def capture(inputs, ctx):
        seen.update(copy.deepcopy(inputs))
        return await original_run(inputs, ctx)
    workflow.run = capture
    runner = Runner(workflows={'chat': workflow}, context_factory=lambda: make_context(model=model))
    request = RunInput(workflow='chat', inputs={'message': 'now', 'history': [{'role': 'user', 'content': 'old'}],
                                               'continuation': {'old': 1}}, history=[], continuation={'new': 2})
    before = copy.deepcopy(request)
    await runner.run(request)
    assert seen['history'] == [] and seen['continuation'] == {'new': 2}
    assert request == before
    assert not any(m.get('content') == 'old' for m in model.requests[0]['messages'])


@pytest.mark.anyio
async def test_native_legacy_history_remains_usable_when_top_level_omitted():
    model = FakeModel(default_text='ok')
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: make_context(model=model))
    await runner.run(RunInput(workflow='chat', inputs={'message': 'now', 'history': [{'role': 'user', 'content': 'legacy'}]}))
    assert any(m.get('content') == 'legacy' for m in model.requests[0]['messages'])


def _marker(count):
    return render_clarification_marker({'workflow': 'group_analysis', 'reference_time': '2026-10-20T10:00:00+08:00',
                                        'timezone': 'Asia/Shanghai', 'clarify_count': count})


def test_clarification_rebuild_keeps_original_and_ordered_supplements():
    envelope = _envelope()
    envelope['message'] = '第二次补充'
    envelope['history'] = [{'role': 'user', 'content': '原始问题'},
                           {'role': 'assistant', 'content': '询问一' + _marker(1)},
                           {'role': 'user', 'content': '第一次补充'},
                           {'role': 'assistant', 'content': '询问二' + _marker(2)}]
    continuation = platform_context.to_run_input(ExecutionRequest.model_validate(envelope)).continuation
    assert continuation['original_request'] == '原始问题'
    assert continuation['_supplements'] == ['第一次补充']
    assert continuation['clarify_count'] == 2


def test_clarification_truncated_history_does_not_invent_original():
    envelope = _envelope()
    envelope['history'] = [{'role': 'assistant', 'content': '询问' + _marker(1)}]
    mapped = platform_context.to_run_input(ExecutionRequest.model_validate(envelope))
    assert mapped.continuation is None


def test_latest_clarification_time_takes_precedence_over_older_answers():
    from datetime import datetime, timedelta, timezone
    from flowagent.workflows.group_rules import resolve_window
    reference = datetime(2026, 10, 20, 10, tzinfo=timezone(timedelta(hours=8)))
    start, _, _ = resolve_window(original_question='分析本群', reference_time=reference,
                                 supplement={'time': ['昨天', '今天']})
    assert start.startswith('2026-10-19')


def test_no_model_request_does_not_claim_application(monkeypatch):
    _install_offline_stack(monkeypatch, turns=[])
    from flowagent.core.contracts import RunResult, Status, WorkflowSpec
    async def no_model(inputs, ctx):
        return RunResult(run_id='', status=Status.COMPLETED, reply='ok')
    from flowagent.adapters.platform.routes import build_platform_app
    runner = Runner(workflows={'chat': WorkflowSpec(name='chat', run=no_model)},
                    context_factory=platform_context.context_factory(), fallback_owner='caller')
    app = build_platform_app(runner=runner)
    body = TestClient(app).post('/execution-v1/respond', json=_envelope_with_context()).json()
    receipt = body['metadata']['agent_config_applied']
    assert receipt['model_request_count'] == 0
    assert receipt['applied_model_parameters'] == {}
    assert receipt['memory_injected'] is False and receipt['memory_included_ids'] == []
    assert not {'system_prompt', 'model', 'memory'} & set(receipt['applied_fields'])
    assert body['used_context_refs'] == []


@pytest.mark.anyio
async def test_frozen_parameters_override_limits_except_max_tokens_can_shrink():
    from flowagent.clients.model import OpenAICompatibleModel
    from flowagent.core.model_context import ContextualModel, ModelRequestContext
    from tests.test_platform_v3_adapter import _FakeChatTransport
    transport = _FakeChatTransport([[_final_chunk()], [_final_chunk()]])
    state = ModelRequestContext(system_prompt='snapshot', parameters={
        'temperature': 0.3, 'max_tokens': 100, 'top_p': 0.8})
    inner = OpenAICompatibleModel(base_url='https://fake.test', api_key='secret',
                                  model_name='m', client=transport.sdk, on_request=state.record_request)
    model = ContextualModel(inner, state)
    for maximum in (50, 200):
        async for _ in model.stream([{'role': 'system', 'content': 'workflow'}], [],
                                    {'max_tokens': maximum, 'temperature': 1.9, 'top_p': 0.1}):
            pass
    assert [r['max_tokens'] for r in transport.requests] == [50, 100]
    assert all(r['temperature'] == 0.3 and r['top_p'] == 0.8 for r in transport.requests)
    assert state.request_count == 2 and state.applied_parameters['max_tokens'] == 100


@pytest.mark.anyio
@pytest.mark.parametrize('component', ['system_prompt', 'memory_text'])
async def test_injected_context_counts_toward_model_budget(component):
    from flowagent.clients.model import OpenAICompatibleModel
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    from flowagent.core.model_context import ContextualModel, ModelRequestContext
    from tests.test_platform_v3_adapter import _FakeChatTransport
    transport = _FakeChatTransport([[_text_chunk('ok'), _final_chunk()]])
    state = ModelRequestContext(**{component: 'x' * 500})
    state.context_bytes = 200
    inner = OpenAICompatibleModel(base_url='https://fake.test', api_key='secret', model_name='m',
                                  client=transport.sdk, on_request=state.record_request)
    with pytest.raises(FlowAgentCoreError) as exc:
        async for _ in ContextualModel(inner, state).stream(
                [{'role': 'user', 'content': 'hi'}], [{'name': 'tool'}], {}):
            pass
    assert exc.value.code is ErrorCode.CONTEXT_LIMIT_EXCEEDED
    assert transport.requests == [] and state.request_count == 0


def test_empty_memory_is_not_reported(monkeypatch):
    _install_offline_stack(monkeypatch, turns=[[_text_chunk('ok'), _final_chunk()]])
    envelope = _envelope_with_context()
    envelope['memory_context']['rendered_text'] = '  '
    body = TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope).json()
    receipt = body['metadata']['agent_config_applied']
    assert receipt['memory_injected'] is False
    assert receipt['memory_included_ids'] == [] and body['used_context_refs'] == []
    assert 'memory' not in receipt['applied_fields']


@pytest.mark.anyio
async def test_authorization_failure_records_no_model_application(monkeypatch):
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    from flowagent.adapters.platform.receipts import build_receipt
    class Deny:
        def __init__(self, *args, **kwargs): pass
        async def authorize(self, **kwargs):
            raise FlowAgentCoreError(ErrorCode.AUTHORIZATION_DENIED, 'denied')
    monkeypatch.setattr(platform_context, 'DemoModelAuthorizer', Deny)
    request = ExecutionRequest.model_validate(_envelope_with_context())
    caps = platform_context.build_capabilities(request)
    token = platform_context.set_current_capabilities(caps)
    try:
        runner = Runner(workflows={'chat': build_chat_workflow()},
                        context_factory=platform_context.context_factory(), fallback_owner='caller')
        result = await runner.run(platform_context.to_run_input(request))
    finally:
        platform_context.reset_capabilities(token)
    receipt = build_receipt(request, result, caps=caps)
    assert receipt['model_request_count'] == 0
    assert receipt['applied_model_parameters'] == {} and receipt['memory_injected'] is False
    assert result.usage.model_requests == 0


@pytest.mark.anyio
async def test_explicit_continuation_retains_prior_valid_supplement():
    from tests.test_native_workflows import _run_group, _group_inputs, _payload, TARGET
    first, _, _ = await _run_group([], _group_inputs('分析最近3天的消息'), payload=_payload([]))
    second, _, _ = await _run_group([], _group_inputs('昨天', continuation=first.clarification.continuation),
                                     payload=_payload([]))
    third, _, tools = await _run_group([], _group_inputs('其他说明', target=TARGET,
                                        continuation=second.clarification.continuation), payload=_payload([]))
    assert third.status.value == 'completed'
    assert tools.calls[0][1]['start'].startswith('2026-10-18')


@pytest.mark.anyio
@pytest.mark.parametrize('instant,zone,start,end', [
    ('2026-10-19T18:30:00+00:00', 'Asia/Shanghai',
     '2026-10-19T00:00:00+08:00', '2026-10-20T00:00:00+08:00'),
    ('2026-03-09T05:30:00+00:00', 'America/New_York',
     '2026-03-08T00:00:00-05:00', '2026-03-09T00:00:00-04:00'),
    ('2026-11-02T05:30:00+00:00', 'America/New_York',
     '2026-11-01T00:00:00-04:00', '2026-11-02T00:00:00-05:00'),
])
async def test_continuation_yesterday_uses_declared_iana_timezone(instant, zone, start, end):
    from tests.test_native_workflows import _run_group, _group_inputs, _payload, TARGET, _CONT_V1
    continuation = dict(_CONT_V1, original_request='分析本群消息', reference_time=instant, timezone=zone)
    result, _, tools = await _run_group([], _group_inputs('昨天', target=TARGET, continuation=continuation),
                                        payload=_payload([]))
    assert result.status.value == 'completed'
    assert tools.calls[0][1]['start'] == start
    assert tools.calls[0][1]['end'] == end


@pytest.mark.anyio
async def test_continuation_keeps_original_time_representation_for_history_chain():
    from tests.test_native_workflows import _run_group, _group_inputs, _payload, _CONT_V1
    instant = '2026-03-09T05:30:00+00:00'
    continuation = dict(_CONT_V1, original_request='分析消息', reference_time=instant, timezone='America/New_York')
    result, _, _ = await _run_group([], _group_inputs('昨天', continuation=continuation), payload=_payload([]))
    assert result.clarification.continuation['reference_time'] == instant
    assert result.clarification.continuation['timezone'] == 'America/New_York'
    envelope = _envelope()
    envelope['history'] = [
        {'role': 'user', 'content': '分析消息'},
        {'role': 'assistant', 'content': render_clarification_marker(continuation)},
        {'role': 'user', 'content': '昨天'},
        {'role': 'assistant', 'content': result.reply},
    ]
    rebuilt = platform_context.to_run_input(ExecutionRequest.model_validate(envelope))
    assert rebuilt.continuation['reference_time'] == instant
    assert rebuilt.continuation['_supplements'] == ['昨天']


@pytest.mark.parametrize('field,value', [('workflow', None), ('workflow', 'invalid'),
                                         ('reference_time', None), ('reference_time', 'invalid'),
                                         ('timezone', None), ('timezone', 'invalid/zone')])
def test_corrupt_clarification_marker_cannot_be_reconstructed(field, value):
    from flowagent.core.contracts import render_clarification_marker
    marker = {'workflow': 'group_analysis', 'reference_time': '2026-10-20T10:00:00+08:00',
              'timezone': 'Asia/Shanghai', 'clarify_count': 1}
    marker[field] = value
    envelope = _envelope()
    envelope['history'] = [{'role': 'user', 'content': 'original'},
                           {'role': 'assistant', 'content': render_clarification_marker(marker)}]
    run_input = platform_context.to_run_input(ExecutionRequest.model_validate(envelope))
    assert run_input.continuation is None
    assert run_input.inputs['_platform_diagnostics']['template_invalid'] is True


def test_inconsistent_clarification_markers_cannot_be_reconstructed():
    from flowagent.core.contracts import render_clarification_marker
    first = {'workflow': 'group_analysis', 'reference_time': '2026-10-20T10:00:00+08:00',
             'timezone': 'Asia/Shanghai', 'clarify_count': 1}
    second = dict(first, clarify_count=2, timezone='UTC')
    envelope = _envelope()
    envelope['history'] = [{'role': 'user', 'content': 'original'},
                           {'role': 'assistant', 'content': render_clarification_marker(first)},
                           {'role': 'user', 'content': 'supplement'},
                           {'role': 'assistant', 'content': render_clarification_marker(second)}]
    run_input = platform_context.to_run_input(ExecutionRequest.model_validate(envelope))
    assert run_input.continuation is None
    assert run_input.inputs['_platform_diagnostics']['template_invalid'] is True


@pytest.mark.anyio
async def test_invalid_explicit_continuation_time_is_rejected():
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    from tests.test_native_workflows import _run_group, _group_inputs, _CONT_V1
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_group([], _group_inputs('supplement', continuation=dict(_CONT_V1, reference_time='bad')))
    assert exc.value.code is ErrorCode.INVALID_INPUT


@pytest.mark.anyio
async def test_group_model_requests_and_tokens_include_report_repair():
    from tests.test_native_workflows import _ctx, _group_inputs, _payload, _messages, TARGET, Q_3D
    from flowagent.workflows.group_analysis import build_group_analysis_workflow, FETCH_TOOL
    from tests.native_helpers import FakeTools
    scripts = [FakeModel.text_turn('{"assignments":[{"message_ref":"m1","category":"风险类"}]}', usage={'input_tokens': 1, 'output_tokens': 2}),
               FakeModel.text_turn('引用: [m9]', usage={'input_tokens': 3, 'output_tokens': 4}),
               FakeModel.text_turn('引用: [m1]', usage={'input_tokens': 5, 'output_tokens': 6})]
    model = FakeModel(scripts)
    tools = FakeTools(catalog=[{'name': FETCH_TOOL}], results={FETCH_TOOL: _payload(_messages(('m1', 'risk')))})
    ctx = _ctx(model, tools)
    result = await build_group_analysis_workflow().run(_group_inputs(Q_3D, target=TARGET), ctx)
    assert result.status.value == 'completed'
    assert ctx.budget.model_spent == ctx.usage.model_requests == model.call_count == 3
    assert (ctx.usage.tokens_in, ctx.usage.tokens_out) == (9, 12)


@pytest.mark.anyio
async def test_group_report_repair_obeys_model_budget():
    from tests.test_native_workflows import _ctx, _group_inputs, _payload, _messages, TARGET, Q_3D, _cls, _rep
    from flowagent.workflows.group_analysis import build_group_analysis_workflow, FETCH_TOOL
    from tests.native_helpers import FakeTools
    model = FakeModel([_cls({'m1': '风险类'}), _rep(['m9']), _rep(['m1'])])
    ctx = _ctx(model, FakeTools(catalog=[{'name': FETCH_TOOL}], results={FETCH_TOOL: _payload(_messages(('m1', 'risk')))}))
    ctx.budget.model_total = 3  # 主阶段最多两次，第三次必须被拒绝。
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    with pytest.raises(FlowAgentCoreError) as exc:
        await build_group_analysis_workflow().run(_group_inputs(Q_3D, target=TARGET), ctx)
    assert model.call_count == 2 and ctx.budget.model_spent == 2
    assert exc.value.code is ErrorCode.MODEL_BUDGET_EXCEEDED


@pytest.mark.anyio
async def test_group_authorization_denial_is_terminal():
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    from flowagent.workflows.group_analysis import build_group_analysis_workflow, FETCH_TOOL
    from tests.native_helpers import FakeTools
    from tests.test_native_workflows import _ctx, _group_inputs, _payload, _messages, TARGET, Q_3D
    model = FakeModel([FlowAgentCoreError(ErrorCode.AUTHORIZATION_DENIED, 'denied')])
    ctx = _ctx(model, FakeTools(catalog=[{'name': FETCH_TOOL}], results={FETCH_TOOL: _payload(_messages(('m1', 'risk')))}))
    with pytest.raises(FlowAgentCoreError) as exc:
        await build_group_analysis_workflow().run(_group_inputs(Q_3D, target=TARGET), ctx)
    assert exc.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert model.call_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize('path', ['consultation_repair', 'knowledge', 'fallback'])
async def test_direct_model_paths_record_tokens(path):
    from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
    from flowagent.workflows.consultation import _validate_with_repair
    from flowagent.workflows.knowledge import build_knowledge_workflow
    from tests.native_helpers import FakeTools
    from tests.test_native_workflows import _doc_text
    usage = {'input_tokens': 7, 'output_tokens': 4}
    text = _doc_text()[-1].text if path == 'consultation_repair' else 'answer'
    model = FakeModel([FakeModel.text_turn(text, usage=usage)])
    if path == 'consultation_repair':
        ctx = make_context(model=model)
        await _validate_with_repair(ctx, 'draft', 'invalid')
    elif path == 'knowledge':
        ctx = make_context(model=model, tools=FakeTools(results={'search': {'data': [{'text': 'x'}]}}))
        await build_knowledge_workflow(bindings={'search': 'search'}).run({'message': 'question'}, ctx)
    else:
        ctx = make_context(model=model)
        runner = Runner(workflows={}, context_factory=lambda: ctx)
        await runner._local_fallback(ctx, RunInput(workflow='chat', inputs={'message': 'hi'}),
                                     FlowAgentCoreError(ErrorCode.OUTPUT_INVALID, 'bad'), exhausted=False)
    assert ctx.usage.model_requests == 1
    assert (ctx.usage.tokens_in, ctx.usage.tokens_out) == (7, 4)


@pytest.mark.anyio
async def test_native_sdk_setup_failure_does_not_count_as_actual_request():
    from flowagent.clients.model import OpenAICompatibleModel
    from flowagent.core.contracts import FlowAgentCoreError
    model = OpenAICompatibleModel(base_url='https://fake.test', api_key='x', model_name='m', client=object())
    ctx = make_context(model=model)
    with pytest.raises(FlowAgentCoreError):
        async for _ in ctx.stream_model([{'role': 'user', 'content': 'hi'}], [], {}):
            pass
    assert ctx.usage.model_requests == 0


@pytest.mark.anyio
async def test_closing_model_stream_closes_underlying_generator():
    from flowagent.core.contracts import ModelEvent
    closed = []
    class Model:
        async def stream(self, messages, tools, limits):
            try:
                yield ModelEvent(kind='text_delta', text='first')
                yield ModelEvent(kind='completed', text='done')
            finally:
                closed.append(True)
    ctx = make_context(model=Model())
    stream = ctx.stream_model([], [], {})
    assert (await stream.__anext__()).text == 'first'
    await stream.aclose()
    assert closed == [True]


@pytest.mark.anyio
async def test_deadline_stream_close_propagates_to_model():
    from flowagent.core.context import aiter_with_deadline
    from flowagent.core.contracts import ModelEvent
    closed = []
    class Model:
        async def stream(self, messages, tools, limits):
            try:
                yield ModelEvent(kind='text_delta', text='first')
                yield ModelEvent(kind='completed', text='done')
            finally:
                closed.append(True)
    stream = aiter_with_deadline(make_context(model=Model()).stream_model([], [], {}), 5000)
    assert (await stream.__anext__()).text == 'first'
    await stream.aclose()
    assert closed == [True]


@pytest.mark.anyio
async def test_model_iterator_close_error_does_not_hide_original_error():
    closed = []
    class Iterator:
        def __aiter__(self): return self
        async def __anext__(self): raise ValueError('original failure')
        async def aclose(self):
            closed.append(True)
            raise RuntimeError('close failure')
    class Model:
        def stream(self, messages, tools, limits): return Iterator()
    with pytest.raises(ValueError, match='original failure'):
        async for _ in make_context(model=Model()).stream_model([], [], {}):
            pass
    assert closed == [True]


@pytest.mark.anyio
async def test_new_cancellation_during_generator_close_propagates():
    import asyncio
    from flowagent.core.contracts import ModelEvent
    cleanup_started = asyncio.Event()
    class Model:
        async def stream(self, messages, tools, limits):
            try:
                yield ModelEvent(kind='text_delta', text='first')
            finally:
                cleanup_started.set()
                await asyncio.Event().wait()
    outer = make_context(model=Model()).stream_model([], [], {})
    await outer.__anext__()
    close_task = asyncio.create_task(outer.aclose())
    await cleanup_started.wait()
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task


@pytest.mark.anyio
async def test_cancellation_during_cleanup_keeps_original_execution_error():
    import asyncio
    cleanup_started = asyncio.Event()
    class Iterator:
        def __aiter__(self): return self
        async def __anext__(self): raise ValueError('original failure')
        async def aclose(self):
            cleanup_started.set()
            await asyncio.Event().wait()
    class Model:
        def stream(self, messages, tools, limits): return Iterator()
    async def consume():
        async for _ in make_context(model=Model()).stream_model([], [], {}):
            pass
    task = asyncio.create_task(consume())
    await cleanup_started.wait()
    task.cancel()
    with pytest.raises(ValueError, match='original failure'):
        await task


@pytest.mark.anyio
async def test_cancelling_model_consumption_closes_underlying_iterator():
    import asyncio
    entered, closed = asyncio.Event(), asyncio.Event()
    class Iterator:
        def __aiter__(self): return self
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()
        async def aclose(self): closed.set()
    class Model:
        def stream(self, messages, tools, limits): return Iterator()
    async def consume():
        async for _ in make_context(model=Model()).stream_model([], [], {}):
            pass
    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


@pytest.mark.anyio
async def test_close_propagates_through_platform_model_to_sdk_stream():
    from types import SimpleNamespace
    from flowagent.adapters.platform.capabilities_runtime import AuthorizedModel
    from flowagent.clients.model import OpenAICompatibleModel
    from flowagent.core.model_context import ContextualModel, ModelRequestContext
    closed = []
    async def sdk_stream():
        try:
            yield _text_chunk('first')
            yield _final_chunk()
        finally:
            closed.append(True)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: sdk_stream())))
    class Allow:
        async def authorize(self, **kwargs): pass
    inner = OpenAICompatibleModel(base_url='https://fake.test', api_key='x', model_name='m', client=client)
    wrapped = AuthorizedModel(ContextualModel(inner, ModelRequestContext()), Allow(),
                              execution_id='x', model_version_id='y', generation=0)
    stream = make_context(model=wrapped).stream_model([], [], {})
    assert (await stream.__anext__()).text == 'first'
    await stream.aclose()
    assert closed == [True]


@pytest.mark.anyio
async def test_noncallable_model_create_has_no_actual_usage_or_receipt(monkeypatch):
    from types import SimpleNamespace
    from flowagent.adapters.platform.receipts import build_receipt
    class Allow:
        def __init__(self, *args, **kwargs): pass
        async def authorize(self, **kwargs): pass
    monkeypatch.setattr(platform_context, 'DemoModelAuthorizer', Allow)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=42)))
    request = ExecutionRequest.model_validate(_envelope_with_context())
    caps = platform_context.build_capabilities(request, model_transport=client)
    token = platform_context.set_current_capabilities(caps)
    try:
        runner = Runner(workflows={'chat': build_chat_workflow()},
                        context_factory=platform_context.context_factory(), fallback_owner='caller')
        result = await runner.run(platform_context.to_run_input(request))
    finally:
        platform_context.reset_capabilities(token)
    receipt = build_receipt(request, result, caps=caps)
    assert result.usage.model_requests == receipt['model_request_count'] == 0
    assert receipt['applied_model_parameters'] == {} and receipt['memory_injected'] is False
    assert not {'system_prompt', 'model', 'memory'} & set(receipt['applied_fields'])


@pytest.mark.anyio
async def test_create_signature_mismatch_has_no_actual_usage_or_receipt(monkeypatch):
    from types import SimpleNamespace
    from flowagent.adapters.platform.receipts import build_receipt
    class Allow:
        def __init__(self, *args, **kwargs): pass
        async def authorize(self, **kwargs): pass
    monkeypatch.setattr(platform_context, 'DemoModelAuthorizer', Allow)
    def create_without_kwargs():
        raise AssertionError('function body must never be entered')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_without_kwargs)))
    request = ExecutionRequest.model_validate(_envelope_with_context())
    caps = platform_context.build_capabilities(request, model_transport=client)
    token = platform_context.set_current_capabilities(caps)
    try:
        runner = Runner(workflows={'chat': build_chat_workflow()},
                        context_factory=platform_context.context_factory(), fallback_owner='caller')
        result = await runner.run(platform_context.to_run_input(request))
    finally:
        platform_context.reset_capabilities(token)
    receipt = build_receipt(request, result, caps=caps)
    assert result.usage.model_requests == receipt['model_request_count'] == 0
    assert receipt['applied_model_parameters'] == {} and receipt['memory_injected'] is False
    assert not {'system_prompt', 'model', 'memory'} & set(receipt['applied_fields'])


@pytest.mark.anyio
async def test_real_async_openai_sdk_is_counted_with_offline_transport():
    import httpx
    import openai
    from flowagent.clients.model import OpenAICompatibleModel
    observed = []
    async def respond(request):
        observed.append(request)
        body = ('data: {"id":"r1","object":"chat.completion.chunk","created":1,'
                '"model":"test","choices":[{"index":0,"delta":{"content":"ok"},'
                '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        sdk = openai.AsyncOpenAI(base_url='https://fake.test/v1', api_key='secret', http_client=http_client)
        ctx = make_context(model=OpenAICompatibleModel(base_url='https://fake.test/v1', api_key='secret',
                                                       model_name='test', client=sdk))
        events = [event async for event in ctx.stream_model([{'role': 'user', 'content': 'hi'}], [], {})]
    assert len(observed) == ctx.usage.model_requests == 1
    assert events[-1].kind == 'completed' and events[-1].text == 'ok'


@pytest.mark.anyio
async def test_async_generator_create_is_counted_after_call_returns():
    from types import SimpleNamespace
    from flowagent.clients.model import OpenAICompatibleModel
    from flowagent.core.model_context import ModelRequestContext
    state = ModelRequestContext(system_prompt='frozen', memory_text='memory', parameters={'top_p': 0.8})
    async def create(**request):
        yield _text_chunk('ok')
        yield _final_chunk()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = OpenAICompatibleModel(base_url='https://fake.test/v1', api_key='secret', model_name='test',
                                  client=sdk, on_request=state.record_request)
    events = [event async for event in model.stream([{'role': 'system', 'content': 'memory'}], [], {'top_p': 0.8})]
    assert events[-1].kind == 'completed' and state.request_count == 1
    assert state.memory_injected is True and state.applied_parameters['top_p'] == 0.8


@pytest.mark.anyio
async def test_create_returning_invalid_object_is_not_counted():
    from types import SimpleNamespace
    from flowagent.clients.model import OpenAICompatibleModel
    from flowagent.core.contracts import FlowAgentCoreError
    from flowagent.core.model_context import ModelRequestContext
    state = ModelRequestContext(system_prompt='frozen', memory_text='memory')
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **request: 42)))
    model = OpenAICompatibleModel(base_url='https://fake.test/v1', api_key='secret', model_name='test',
                                  client=sdk, on_request=state.record_request)
    ctx = make_context(model=model)
    with pytest.raises(FlowAgentCoreError):
        async for _ in ctx.stream_model([{'role': 'system', 'content': 'memory'}], [], {}):
            pass
    assert ctx.usage.model_requests == state.request_count == 0
    assert state.memory_injected is False and state.applied_parameters == {}


@pytest.mark.anyio
async def test_native_sdk_stream_counts_actual_call_across_deadline_tasks():
    from flowagent.clients.model import OpenAICompatibleModel
    from tests.test_platform_v3_adapter import _FakeChatTransport
    transport = _FakeChatTransport([[_text_chunk('ok'), _final_chunk(7, 4)]])
    model = OpenAICompatibleModel(base_url='https://fake.test', api_key='x', model_name='m', client=transport.sdk)
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: make_context(model=model),
                    fallback_owner='caller')
    result = await runner.run(RunInput(workflow='chat', inputs={'message': 'hi'}))
    assert result.status.value == 'completed'
    assert result.usage.model_requests == 1
    assert (result.usage.tokens_in, result.usage.tokens_out) == (7, 4)


@pytest.mark.anyio
async def test_concurrent_platform_requests_isolate_context(monkeypatch):
    import asyncio
    from tests.test_platform_v3_adapter import _FakeChatTransport
    class Allow:
        def __init__(self, *args, **kwargs): pass
        async def authorize(self, **kwargs):
            await asyncio.sleep(0)
    monkeypatch.setattr(platform_context, 'DemoModelAuthorizer', Allow)
    runner = Runner(workflows={'chat': build_chat_workflow()},
                    context_factory=platform_context.context_factory(), fallback_owner='caller')
    async def invoke(label):
        envelope = _envelope_with_context()
        snapshot = envelope['agent_config']['snapshot']
        snapshot['system_prompt'] = snapshot['config']['system_prompt'] = 'prompt-' + label
        snapshot['config_digest'] = _digest(snapshot)
        envelope['agent_config']['config_digest'] = snapshot['config_digest']
        envelope['memory_context']['rendered_text'] = 'memory-' + label
        envelope['memory_context']['included_ids'] = [label]
        envelope['execution_id'] = envelope['trace']['trace_id'] = (
            'f04b6209-30da-4e57-bf93-73c6867cc10' + ('1' if label == 'a' else '2'))
        request = ExecutionRequest.model_validate(envelope)
        transport = _FakeChatTransport([[_text_chunk('ok'), _final_chunk()]])
        caps = platform_context.build_capabilities(request, model_transport=transport.sdk)
        token = platform_context.set_current_capabilities(caps)
        try:
            await runner.run(platform_context.to_run_input(request))
        finally:
            platform_context.reset_capabilities(token)
        return transport.requests[0], caps.model_context
    results = await asyncio.gather(invoke('a'), invoke('b'))
    for label, (sent, state) in zip(('a', 'b'), results):
        system = sent['messages'][0]['content']
        assert 'prompt-' + label in system and 'memory-' + label in system
        assert 'prompt-' + ('b' if label == 'a' else 'a') not in system
        assert state.memory_ids == [label] and state.request_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize('path', ['group', 'consultation', 'autonomous'])
async def test_business_and_autonomous_paths_apply_model_context(path):
    from types import SimpleNamespace
    from flowagent.clients.model import OpenAICompatibleModel
    from flowagent.core.model_context import ContextualModel, ModelRequestContext
    from flowagent.workflows.group_analysis import build_group_analysis_workflow, FETCH_TOOL
    from flowagent.workflows.consultation import build_consultation_workflow
    from tests.native_helpers import FakeTools
    from tests.test_native_workflows import _doc_text, _payload, _messages
    from tests.test_platform_v3_adapter import _FakeChatTransport
    if path == 'group':
        turns = [[_text_chunk('{"assignments":[{"message_ref":"m1","category":"问题"}]}'), _final_chunk()],
                 [_text_chunk('## 总结\nok\n## 分类统计\nok\n## 汇报要点\nok\n引用: [m1]'), _final_chunk()]]
        workflow = build_group_analysis_workflow()
        tools = FakeTools(catalog=[{'name': FETCH_TOOL}], results={FETCH_TOOL: _payload(_messages(('m1', 'risk')))})
        request = RunInput(workflow='group_analysis', inputs={
            'message': '分析本群最近3天', 'analysis_target': SimpleNamespace(conversation_id=42)})
    elif path == 'consultation':
        turns = [[_text_chunk(_doc_text()[-1].text), _final_chunk()]]
        workflow = build_consultation_workflow()
        tools = None
        request = RunInput(workflow='consultation_stage', inputs={'message': 'materials',
            'function_task': {'stage': 'draft', 'output_schema': 'consultation_document_v1', 'inputs': {}}})
    else:
        turns = [[_text_chunk('ok'), _final_chunk()]]
        workflow, tools = None, None
        request = RunInput(workflow='unknown', inputs={'message': 'hello'})
    transport = _FakeChatTransport(turns)
    state = ModelRequestContext(system_prompt='snapshot-prompt', memory_text='memory-context',
                                memory_ids=['m1'], parameters={'top_p': 0.6, 'max_tokens': 75})
    inner = OpenAICompatibleModel(base_url='https://fake.test', api_key='secret', model_name='model',
                                  client=transport.sdk, on_request=state.record_request)
    model = ContextualModel(inner, state)
    runner = Runner(workflows={workflow.name: workflow} if workflow else {},
                    context_factory=lambda: make_context(model=model, tools=tools), fallback_owner='caller')
    result = await runner.run(request)
    assert result.status.value == 'completed'
    assert transport.requests
    for sent in transport.requests:
        assert 'snapshot-prompt' in sent['messages'][0]['content']
        assert 'memory-context' in sent['messages'][0]['content']
        assert sent['top_p'] == 0.6 and sent['max_tokens'] == 75
    assert state.request_count == len(transport.requests) and state.memory_injected
