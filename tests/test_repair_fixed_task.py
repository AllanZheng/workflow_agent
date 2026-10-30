"""固定群任务、冻结工具解析和预算的离线真实入口回归。"""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.platform.capabilities_runtime import tool_alias
from flowagent.adapters.platform.context import budget_from_request, to_run_input
from flowagent.adapters.platform.protocol import ExecutionRequest
from flowagent.app import build_platform_service
from flowagent.core.context import BudgetExceeded
from tests.test_managed_config_v3 import _envelope
from tests.test_platform_v3_adapter import (
    TOOL1_ID, TOOL2_ID, _tools_envelope, _reseal, _make_mcp_session_factory,
    _install_offline_stack, _text_chunk, _final_chunk,
)


def _group_envelope(*, tools=True):
    envelope = _tools_envelope() if tools else _envelope()
    if tools:
        snapshot = envelope['agent_config']['snapshot']
        snapshot['tools'] = snapshot['tools'][:1]
        snapshot['tools'][0]['remote_name'] = 'fetch_group_analysis_input'
        snapshot['config']['tools'] = snapshot['config']['tools'][:1]
        snapshot['skills'] = []
        snapshot['config']['skills'] = []
        _reseal(envelope)
    envelope['message'] = '分析本群最近3天'
    envelope['analysis_target'] = {'conversation_id': 42}
    envelope['function_task'] = {'function_key': 'group_analysis', 'stage': 'analyze',
                                 'output_schema': 'group_analysis_result_v1', 'inputs': {}}
    return envelope


def _offline_group(monkeypatch, *, visible=None, data=None):
    alias = tool_alias(TOOL1_ID)
    calls, sessions = [], []
    payload = data if data is not None else {
        'messages': [{'message_ref': 'm1', 'text': '设备告警'}],
        'window': {}, 'counts': {'fetched': 1, 'included': 1}}
    factory = _make_mcp_session_factory([alias] if visible is None else visible,
                                        structured=payload, cm_calls=sessions, tool_calls=calls)
    turns = [[_text_chunk('{"assignments":[{"message_ref":"m1","category":"问题"}]}'), _final_chunk()],
             [_text_chunk('## 总结\n告警\n## 分类统计\n问题\n## 汇报要点\n需处理\n引用: [m1]'), _final_chunk()]]
    transport, counter = _install_offline_stack(monkeypatch, turns=turns, session_factory=factory)
    return alias, calls, sessions, transport, counter


def test_group_task_union_routes_and_manifest_is_versioned():
    request = ExecutionRequest.model_validate(_group_envelope(tools=False))
    assert to_run_input(request).workflow == 'group_analysis'
    manifest = TestClient(build_platform_service()).get('/capabilities/v2').json()
    assert manifest['tasks'] == {'group_analysis_versions': [1]}
    assert manifest['execution_limits_versions'] == [1]
    assert manifest['application_receipt_versions'] == [1]


@pytest.mark.parametrize('mutation', [
    {'function_key': 'group_analysis', 'stage': 'draft', 'output_schema': 'group_analysis_result_v1'},
    {'function_key': 'group_analysis', 'stage': 'analyze', 'output_schema': 'consultation_document_v1'},
    {'function_key': 'risk_consultation', 'stage': 'analyze', 'output_schema': 'group_analysis_result_v1'},
    {'function_key': 'unknown', 'stage': 'analyze', 'output_schema': 'group_analysis_result_v1'},
])
def test_task_union_rejects_invalid_combinations(mutation, monkeypatch):
    _install_offline_stack(monkeypatch, turns=[])
    envelope = _group_envelope(tools=False)
    envelope['function_task'] = {**mutation, 'inputs': {}}
    assert TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope).status_code == 422


@pytest.mark.parametrize('limits', [
    {'model_total': -1, 'tool_total': 1}, {'model_total': True, 'tool_total': 1},
    {'model_total': '1', 'tool_total': 1}, {'model_total': 1, 'tool_total': -1},
    {'model_total': 1}, {'model_total': 1, 'tool_total': 1, 'extra': 1}, None,
])
def test_execution_limits_reject_invalid_values(limits, monkeypatch):
    _install_offline_stack(monkeypatch, turns=[])
    envelope = _envelope()
    envelope['execution_limits'] = limits
    assert TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope).status_code == 422


@pytest.mark.anyio
async def test_execution_limits_are_hard_caps_with_platform_fallback_reserved():
    envelope = _envelope()
    envelope['execution_limits'] = {'model_total': 2, 'tool_total': 1}
    budget = budget_from_request(ExecutionRequest.model_validate(envelope))
    budget.switch_path('fixed')
    await budget.reserve_model()
    with pytest.raises(BudgetExceeded):
        await budget.reserve_model()
    await budget.reserve_tool()
    with pytest.raises(BudgetExceeded):
        await budget.reserve_tool()


@pytest.mark.parametrize('model_total,expected_main', [(0, 0), (1, 0), (23, 22), (24, 23)])
def test_platform_model_limit_reserves_one_for_platform(model_total, expected_main):
    envelope = _envelope()
    envelope['execution_limits'] = {'model_total': model_total, 'tool_total': 32}
    budget = budget_from_request(ExecutionRequest.model_validate(envelope))
    budget.switch_path('fixed')
    assert budget._model_cap() == expected_main


def test_execution_limits_are_clamped_by_stricter_local_policy():
    envelope = _envelope()
    envelope['execution_limits'] = {'model_total': 23, 'tool_total': 32}
    budget = budget_from_request(ExecutionRequest.model_validate(envelope),
                                 {'limits': {'model_calls': 5, 'tool_calls': 3}})
    budget.switch_path('fixed')
    assert (budget.model_total, budget.mcp_total, budget._model_cap(), budget._mcp_cap()) == (5, 3, 4, 3)


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.parametrize('stream', [False, True])
def test_group_task_real_route_discovers_and_calls_frozen_alias(monkeypatch, stream):
    alias, calls, sessions, transport, _ = _offline_group(monkeypatch)
    envelope = _group_envelope()
    envelope['execution_limits'] = {'model_total': 3, 'tool_total': 1}
    path = '/execution-v1/respond/stream' if stream else '/execution-v1/respond'
    response = TestClient(build_platform_service()).post(path, json=envelope)
    assert response.status_code == 200, response.text
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        assert [e['kind'] for e in events] == ['meta', 'done']
        metadata = events[-1]['metadata']
    else:
        metadata = response.json()['metadata']
    assert metadata['status'] == 'completed'
    assert metadata['analysis_result']['status'] == 'complete'
    assert metadata['analysis_result']['statistics']['total_messages'] == 1
    assert metadata['agent_config_applied']['called_tool_versions'][0]['tool_id'] == TOOL1_ID
    assert [call['name'] for call in calls] == [alias]
    assert len(transport.requests) == 2
    assert any('X-Agent-Tool-ID' in session['headers'] for session in sessions)


@pytest.mark.parametrize('kind', ['complete', 'partial', 'empty'])
def test_group_result_matches_platform_persistence_contract(monkeypatch, kind):
    start, end = '2026-10-16T10:00:00+08:00', '2026-10-19T10:00:00+08:00'
    payload = {'window': {'start': start, 'end_exclusive': end},
               'messages': [] if kind == 'empty' else [{'message_ref': 'm1', 'text': '设备告警'}],
               'status': 'partial' if kind == 'partial' else 'complete',
               'partial_reasons': ['page_limit'] if kind == 'partial' else [], 'counts': {}}
    _offline_group(monkeypatch, data=payload)
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=_group_envelope())
    assert response.status_code == 200, response.text
    result = response.json()['metadata']['analysis_result']
    assert result['conversation_id'] == 42
    assert result['status'] == ('partial' if kind == 'partial' else 'complete')
    assert result['window_start'] == start and result['window_end_exclusive'] == end
    for key in ('statistics', 'coverage', 'requested_taxonomy', 'effective_taxonomy'):
        assert isinstance(result[key], dict)
    assert isinstance(result['partial_reasons'], list)


@pytest.mark.parametrize('mode', ['missing', 'invisible', 'duplicate'])
def test_group_task_missing_or_ambiguous_tool_fails_without_fake_result(monkeypatch, mode):
    alias, calls, sessions, transport, _ = _offline_group(monkeypatch, visible=[] if mode == 'invisible' else None)
    envelope = _group_envelope(tools=mode != 'missing')
    if mode == 'duplicate':
        snapshot = envelope['agent_config']['snapshot']
        duplicate = copy.deepcopy(snapshot['tools'][0])
        duplicate['tool_id'] = TOOL2_ID
        duplicate['tool_version_id'] = '12000000-0000-0000-0000-000000000002'
        snapshot['tools'].append(duplicate)
        snapshot['config']['tools'].append({'tool_id': duplicate['tool_id'],
                                            'tool_version_id': duplicate['tool_version_id']})
        _reseal(envelope)
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope)
    assert response.status_code != 200
    assert response.json()['error']['code'] == 'capability_unavailable'
    assert calls == [] and transport.requests == []


@pytest.mark.anyio
async def test_native_profile_group_fetch_alias_is_resolved_and_discovered(monkeypatch):
    import flowagent.app as app_module
    from flowagent.core.contracts import RunInput
    from tests.native_helpers import FakeTools
    alias = 'server__fetch_group_analysis_input'
    tools = FakeTools(catalog=[{'name': alias}], results={alias: {'messages': [], 'counts': {}}})
    monkeypatch.setattr(app_module, '_attach_lazy_tools', lambda ctx, configs: setattr(ctx, 'tools', tools))
    profile = {'model': {'name': 'local', 'base_url': 'https://fake.test', 'api_key': 'x'},
               'mcp_servers': [{'name': 'server', 'url': 'https://fake.test'}],
               'workflow_bindings': {'group_analysis': {'fetch_tool': alias}}}
    runner = app_module.build_runner(profile, fallback_owner='caller')
    result = await runner.run(RunInput(workflow='group_analysis', inputs={
        'message': '分析本群最近3天', 'analysis_target': type('Target', (), {'conversation_id': 42})()}))
    assert result.status.value == 'completed'
    assert tools.calls[0][0] == alias


@pytest.mark.anyio
async def test_native_group_missing_tool_uses_one_local_degraded_fallback(monkeypatch):
    import flowagent.app as app_module
    from flowagent.core.contracts import RunInput
    from tests.native_helpers import FakeModel
    model = FakeModel([FakeModel.text_turn('任务未完成，请检查工具。')])
    monkeypatch.setattr(app_module, 'OpenAICompatibleModel', lambda **kwargs: model)
    profile = {'model': {'name': 'local', 'base_url': 'https://fake.test', 'api_key': 'x'}}
    result = await app_module.build_runner(profile, fallback_owner='local').run(RunInput(
        workflow='group_analysis', inputs={'message': '分析本群最近3天'}))
    assert result.status.value == 'degraded'
    assert result.error.code.value == 'CAPABILITY_UNAVAILABLE'
    assert result.recovery['fallback_attempted'] is True and model.call_count == 1
    assert result.output is None and result.execution_path == 'direct_llm'


def test_explicit_group_task_disabled_returns_failure_not_autonomous(monkeypatch):
    _, calls, _, transport, _ = _offline_group(monkeypatch)
    client = TestClient(build_platform_service(platform_policy={'workflow_enabled_map': {'group_analysis': False}}))
    response = client.post('/execution-v1/respond', json=_group_envelope())
    assert response.status_code != 200 and response.json()['error']['code'] == 'capability_unavailable'
    assert calls == [] and transport.requests == []


@pytest.mark.parametrize('limits,error_code,expected_tool_calls', [
    ({'model_total': 3, 'tool_total': 0}, 'tool_budget_exceeded', 0),
    ({'model_total': 1, 'tool_total': 1}, 'model_budget_exceeded', 1),
])
def test_group_task_zero_budget_fails_before_exceeding_limit(monkeypatch, limits, error_code, expected_tool_calls):
    _, calls, _, transport, _ = _offline_group(monkeypatch)
    envelope = _group_envelope()
    envelope['execution_limits'] = limits
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope)
    assert response.status_code != 200 and response.json()['error']['code'] == error_code
    assert len(calls) == expected_tool_calls and not transport.requests


def test_group_stream_missing_binding_emits_error_without_business_result(monkeypatch):
    _, calls, _, transport, _ = _offline_group(monkeypatch)
    response = TestClient(build_platform_service()).post('/execution-v1/respond/stream',
                                                         json=_group_envelope(tools=False))
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert [event['kind'] for event in events] == ['meta', 'error']
    assert events[-1]['error_code'] == 'capability_unavailable'
    assert events[-1]['metadata']['recovery']['final_source'] == 'caller'
    assert not calls and not transport.requests


def test_zero_remaining_deadline_does_not_start_any_capability(monkeypatch):
    _, calls, sessions, transport, counter = _offline_group(monkeypatch)
    envelope = _group_envelope()
    envelope['deadline_ms'] = 0
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope)
    assert response.status_code != 200 and response.json()['error']['code'] == 'deadline_exceeded'
    assert not calls and not sessions and not transport.requests and counter['authorize'] == 0


@pytest.mark.anyio
@pytest.mark.parametrize('source', ['platform', 'profile'])
async def test_autonomous_budget_intersects_configured_and_global_limits(source):
    from flowagent.config import build_budget
    if source == 'platform':
        envelope = _envelope()
        envelope['execution_limits'] = {'model_total': 10, 'tool_total': 5}
        budget = budget_from_request(ExecutionRequest.model_validate(envelope),
                                     {'limits': {'model_calls': 9, 'tool_calls': 4,
                                                 'autonomous': {'model_calls': 3, 'tool_calls': 1}}})
    else:
        budget = build_budget({'limits': {'model_calls': 9, 'tool_calls': 4,
                                         'autonomous': {'model_calls': 3, 'tool_calls': 1}}})
    budget.switch_path('autonomous')
    await budget.reserve_model()
    await budget.reserve_model()
    with pytest.raises(BudgetExceeded):
        await budget.reserve_model()
    await budget.reserve_model(fallback=True)
    with pytest.raises(BudgetExceeded):
        await budget.reserve_model(fallback=True)
    await budget.reserve_tool()
    with pytest.raises(BudgetExceeded):
        await budget.reserve_tool()


@pytest.mark.anyio
async def test_autonomous_zero_cap_and_path_switch_do_not_restore_spent_budget():
    from flowagent.config import build_budget
    budget = build_budget({'limits': {'autonomous': {'model_calls': 2, 'tool_calls': 0}}})
    budget.switch_path('fixed')
    await budget.reserve_model()
    budget.switch_path('autonomous')
    with pytest.raises(BudgetExceeded):
        await budget.reserve_model()
    with pytest.raises(BudgetExceeded):
        await budget.reserve_tool()
    await budget.reserve_model(fallback=True)
    with pytest.raises(BudgetExceeded):
        await budget.reserve_model(fallback=True)


@pytest.mark.anyio
async def test_autonomous_explicit_zero_and_downstream_limits_are_hard_caps():
    from flowagent.config import build_budget
    budget = build_budget({'limits': {'autonomous': {'model_calls': 0, 'tool_calls': 0}}})
    budget.switch_path('autonomous')
    for fallback in (False, True):
        with pytest.raises(BudgetExceeded):
            await budget.reserve_model(fallback=fallback)
    with pytest.raises(BudgetExceeded):
        await budget.reserve_tool()
    envelope = _envelope()
    envelope['execution_limits'] = {'model_total': 2, 'tool_total': 0}
    limited = budget_from_request(ExecutionRequest.model_validate(envelope),
                                  {'limits': {'autonomous': {'model_calls': 99, 'tool_calls': 99}}})
    limited.switch_path('autonomous')
    await limited.reserve_model()
    with pytest.raises(BudgetExceeded):
        await limited.reserve_model()
    await limited.reserve_model(fallback=True)
    with pytest.raises(BudgetExceeded):
        await limited.reserve_model(fallback=True)
    with pytest.raises(BudgetExceeded):
        await limited.reserve_tool()


@pytest.mark.anyio
async def test_autonomous_zero_limit_fallback_returns_stable_failure_not_exception():
    from flowagent.config import build_budget
    from flowagent.core.contracts import RunInput
    from flowagent.core.runner import Runner
    from tests.native_helpers import FakeModel, make_context
    model = FakeModel(default_text='not called')
    budget = build_budget({'limits': {'autonomous': {'model_calls': 0}}})
    runner = Runner(workflows={}, context_factory=lambda: make_context(model=model, budget=budget))
    result = await runner.run(RunInput(workflow='unknown', inputs={'message': 'hello'}))
    assert result.status.value == 'failed'
    assert result.recovery['primary_error']['code'] == 'MODEL_BUDGET_EXCEEDED'
    assert result.recovery['fallback_error']['code'] == 'MODEL_BUDGET_EXCEEDED'
    assert model.call_count == 0


@pytest.mark.anyio
async def test_group_discovery_respects_remaining_deadline_and_reclaims_task():
    import asyncio
    from flowagent.core.context import Budget
    from flowagent.core.contracts import RunInput
    from flowagent.core.runner import Runner
    from flowagent.workflows.group_analysis import build_group_analysis_workflow
    from tests.native_helpers import FakeModel, make_context
    entered, stopped = asyncio.Event(), asyncio.Event()
    class HangingTools:
        async def list(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        async def call(self, name, arguments):
            raise AssertionError('fetch must not begin')
    budget = Budget(model_total=24, mcp_total=32, deadline_ms=50,
                    main_deadline_ms=50, fallback_timeout_ms=10)
    model = FakeModel(default_text='not called')
    ctx = make_context(model=model, tools=HangingTools(), budget=budget)
    runner = Runner(workflows={'group_analysis': build_group_analysis_workflow()},
                    context_factory=lambda: ctx, fallback_owner='caller')
    result = await asyncio.wait_for(runner.run(RunInput(workflow='group_analysis',
                                                       inputs={'message': '分析本群'})), timeout=0.35)
    assert entered.is_set() and stopped.is_set()
    assert result.status.value == 'failed' and result.error.code.value == 'DEADLINE_EXCEEDED'
    assert model.call_count == 0 and ctx.usage.mcp_attempts == 0 and not runner._active


@pytest.mark.anyio
async def test_group_discovery_cancel_reclaims_task_without_model_or_fetch():
    import asyncio
    from flowagent.core.contracts import RunInput
    from flowagent.core.runner import Runner
    from flowagent.workflows.group_analysis import build_group_analysis_workflow
    from tests.native_helpers import FakeModel, make_context
    entered, stopped = asyncio.Event(), asyncio.Event()
    class HangingTools:
        async def list(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        async def call(self, name, arguments):
            raise AssertionError('fetch must not begin')
    model = FakeModel(default_text='not called')
    ctx = make_context(model=model, tools=HangingTools())
    runner = Runner(workflows={'group_analysis': build_group_analysis_workflow()},
                    context_factory=lambda: ctx, fallback_owner='caller')
    run = asyncio.create_task(runner.run(RunInput(workflow='group_analysis', inputs={'message': '分析本群'})))
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert runner.cancel(ctx.run_id)['cancelled'] is True
    result = await asyncio.wait_for(run, timeout=0.35)
    assert stopped.is_set() and result.status.value == 'cancelled'
    assert result.error.code.value == 'CANCELLED_BY_USER'
    assert model.call_count == 0 and ctx.usage.mcp_attempts == 0 and not runner._active


@pytest.mark.anyio
@pytest.mark.parametrize('trigger', ['deadline', 'cancel'])
async def test_group_discovery_ignoring_cancellation_has_bounded_cleanup(trigger):
    import asyncio
    from flowagent.core.context import Budget
    from flowagent.core.contracts import RunInput
    from flowagent.core.runner import Runner
    from flowagent.workflows.group_analysis import build_group_analysis_workflow
    from tests.native_helpers import FakeModel, make_context
    entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class StubbornTools:
        async def list(self):
            entered.set()
            try:
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue  # 模拟远端库拒绝本地取消，直到测试主动释放。
                return []
            finally:
                stopped.set()
        async def call(self, name, arguments):
            raise AssertionError('fetch must not begin')
    budget = Budget(model_total=24, mcp_total=32,
                    deadline_ms=50 if trigger == 'deadline' else 5000,
                    main_deadline_ms=50 if trigger == 'deadline' else 5000,
                    fallback_timeout_ms=100)
    model = FakeModel(default_text='not called')
    ctx = make_context(model=model, tools=StubbornTools(), budget=budget)
    runner = Runner(workflows={'group_analysis': build_group_analysis_workflow()},
                    context_factory=lambda: ctx, fallback_owner='caller')
    run = asyncio.create_task(runner.run(RunInput(workflow='group_analysis', inputs={'message': '分析本群'})))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        if trigger == 'cancel':
            assert runner.cancel(ctx.run_id)['cancelled'] is True
        result = await asyncio.wait_for(asyncio.shield(run), timeout=0.4)
        assert result.status.value == ('cancelled' if trigger == 'cancel' else 'failed')
        assert result.error.code.value == ('CANCELLED_BY_USER' if trigger == 'cancel' else 'DEADLINE_EXCEEDED')
        assert result.recovery['capability_cleanup_pending'] is True
        assert result.recovery['capability_outcome'] == 'unknown'
        assert not runner._active and model.call_count == 0
    finally:
        release.set()
        await asyncio.wait_for(stopped.wait(), timeout=1)
        await asyncio.wait_for(run, timeout=1)


def test_risk_consultation_stream_remains_unsupported():
    envelope = _envelope()
    envelope['function_task'] = {'function_key': 'risk_consultation', 'stage': 'draft',
                                 'output_schema': 'consultation_document_v1', 'inputs': {}}
    response = TestClient(build_platform_service()).post('/execution-v1/respond/stream', json=envelope)
    assert response.status_code == 422 and response.json()['detail'] == 'function_task_requires_respond'
