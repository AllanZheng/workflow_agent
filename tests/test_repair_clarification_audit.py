"""群分析澄清历史：经平台适配与固定工作流验证真实多轮行为。"""
import copy
import json

from fastapi.testclient import TestClient

from flowagent.adapters.platform import context as platform_context
from flowagent.adapters.platform.capabilities_runtime import tool_alias
from flowagent.adapters.platform.protocol import ExecutionRequest
from flowagent.app import build_platform_service
from tests.test_platform_v3_adapter import (TOOL1_ID, _final_chunk, _install_offline_stack,
                                           _make_mcp_session_factory, _text_chunk)
from tests.test_repair_fixed_task import _group_envelope


def _setup(monkeypatch, turns):
    calls, sessions, requests = [], [], []
    factory = _make_mcp_session_factory([tool_alias(TOOL1_ID)], structured={
        'messages': [{'message_ref': 'm1', 'text': '设备告警'}],
        'window': {}, 'counts': {'fetched': 1, 'included': 1}}, cm_calls=sessions, tool_calls=calls)
    transport, counter = _install_offline_stack(monkeypatch, turns=turns, session_factory=factory)
    create = transport.sdk.chat.completions.create
    def capture(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return create(**kwargs)
    transport.sdk.chat.completions.create = capture
    return TestClient(build_platform_service()), calls, requests, counter


def _turn(category):
    classification = json.dumps({'assignments': [{'message_ref': 'm1', 'category': category}]}, ensure_ascii=False)
    return [[_text_chunk(classification), _final_chunk()],
            [_text_chunk('## 总结\n一条消息\n## 分类统计\n1条\n## 汇报要点\n跟进\n引用: [m1]'), _final_chunk()]]


def _send(client, envelope):
    response = client.post('/execution-v1/respond', json=envelope)
    assert response.status_code == 200, response.text
    return response.json()


def test_hist03_real_clarification_then_target_preserves_original_window(monkeypatch):
    clock = {'value': '2026-10-20T10:00:00+08:00'}
    monkeypatch.setattr(platform_context, '_reference_time', lambda request: clock['value'])
    client, calls, requests, counter = _setup(monkeypatch, _turn('风险类'))
    original = '分析昨天的消息，按风险类、建议类统计'
    first = _group_envelope()
    first['message'] = original
    first['analysis_target'] = None
    first_result = _send(client, first)
    assert first_result['metadata']['status'] == 'needs_input'
    clarification = first_result['metadata']['clarification']
    assert clarification['continuation']['original_request'] == original
    assert clarification['continuation']['reference_time'] == clock['value']
    assert clarification['continuation']['clarify_count'] == 1
    assert not calls and not requests and counter['authorize'] == 0

    clock['value'] = '2026-11-20T10:00:00+08:00'  # 续接不能换用当前请求时钟。
    second = _group_envelope()
    second['message'] = '目标是42群'
    second['history'] = [{'role': 'user', 'content': original},
                         {'role': 'assistant', 'content': first_result['reply']}]
    normalized = platform_context.to_run_input(ExecutionRequest.model_validate(second))
    assert normalized.continuation['original_request'] == original
    assert normalized.continuation['reference_time'] == '2026-10-20T10:00:00+08:00'
    result = _send(client, second)
    analysis = result['metadata']['analysis_result']
    assert analysis['status'] == 'complete' and analysis['requested_taxonomy']['text'] == original
    assert analysis['effective_taxonomy']['categories'] == ['风险类', '建议类', '未归类']
    assert calls[0]['arguments'] == {'conversation_id': 42, 'start': '2026-10-19T00:00:00+08:00',
                                     'end': '2026-10-20T00:00:00+08:00'}
    assert len(requests) == 2 and all('2026-11-20' not in str(request['messages']) for request in requests)
    assert '风险类' in requests[0]['messages'][1]['content'] and '建议类' in requests[0]['messages'][1]['content']
    assert '2026-10-19T00:00:00+08:00' in requests[1]['messages'][1]['content']


def test_hist04_two_clarifications_explicit_wins_then_new_request_resets(monkeypatch):
    clock = {'value': '2026-10-20T10:00:00+08:00'}
    monkeypatch.setattr(platform_context, '_reference_time', lambda request: clock['value'])
    client, calls, requests, _ = _setup(monkeypatch, _turn('风险类') + _turn('问题'))
    original = '分析最近3天的消息，按风险类、建议类统计'
    first = _group_envelope()
    first.update(message=original, analysis_target=None)
    one = _send(client, first)
    first_continuation = one['metadata']['clarification']['continuation']
    assert one['metadata']['status'] == 'needs_input' and first_continuation['clarify_count'] == 1

    second = _group_envelope()
    second.update(message='昨天', analysis_target=None,
                  request_metadata={'continuation': first_continuation})
    two = _send(client, second)
    explicit = two['metadata']['clarification']['continuation']
    assert two['metadata']['status'] == 'needs_input' and explicit['clarify_count'] == 2
    assert explicit['original_request'] == original and explicit['_supplements'] == ['昨天']
    assert not calls and not requests

    # 合法历史故意携带另一原问题，验证显式续接优先于历史重建。
    conflicting_history = [{'role': 'user', 'content': '分析最近30天消息，按问题、通知统计'},
                           {'role': 'assistant', 'content': one['reply']},
                           {'role': 'user', 'content': '昨天'},
                           {'role': 'assistant', 'content': two['reply']}]
    third = _group_envelope()
    third.update(message='目标是42群', history=conflicting_history,
                 request_metadata={'continuation': explicit})
    normalized = platform_context.to_run_input(ExecutionRequest.model_validate(third))
    assert normalized.inputs['_platform_diagnostics']['continuation_source'] == 'explicit'
    assert normalized.continuation['original_request'] == original
    old_result = _send(client, third)
    old_analysis = old_result['metadata']['analysis_result']
    assert old_analysis['requested_taxonomy']['text'] == original
    assert old_analysis['effective_taxonomy']['categories'] == ['风险类', '建议类', '未归类']
    assert '风险类' in requests[0]['messages'][1]['content'] and '建议类' in requests[0]['messages'][1]['content']
    assert calls[0]['arguments'] == {'conversation_id': 42, 'start': '2026-10-19T00:00:00+08:00',
                                     'end': '2026-10-20T00:00:00+08:00'}

    clock['value'] = '2026-11-05T10:00:00+08:00'
    fresh = _group_envelope()
    fresh.update(message='分析本群最近2天，按问题、通知统计', history=conflicting_history,
                 request_metadata={'continuation': explicit, 'new_request': True})
    normalized_new = platform_context.to_run_input(ExecutionRequest.model_validate(fresh))
    assert normalized_new.continuation is None
    assert normalized_new.inputs['_platform_diagnostics']['continuation_source'] == 'new_request_explicit'
    result = _send(client, fresh)
    analysis = result['metadata']['analysis_result']
    assert analysis['requested_taxonomy']['text'] == fresh['message']
    assert analysis['effective_taxonomy']['categories'] == ['问题', '通知', '未归类']
    assert calls[1]['arguments'] == {'conversation_id': 42, 'start': '2026-11-03T10:00:00+08:00',
                                     'end': '2026-11-05T10:00:00+08:00'}
    assert len(requests) == 4
    assert '问题' in requests[2]['messages'][1]['content'] and '通知' in requests[2]['messages'][1]['content']
    assert all(all(stale not in str(request['messages']) for stale in ('昨天', '最近30天', '风险类', '建议类'))
               for request in requests[2:])
