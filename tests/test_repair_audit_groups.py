"""CFG-04/07、TOOL-02：平台适配入口与实际请求/回执的离线组合证据。"""
import copy
import json

import httpx
import openai
from fastapi.testclient import TestClient

from flowagent.adapters.platform import context as platform_context
from flowagent.adapters.platform.capabilities_runtime import tool_alias
from flowagent.app import build_platform_service
from tests.test_repair_context import _envelope_with_context
from tests.test_repair_fixed_task import _group_envelope, _offline_group
from tests.test_platform_v3_adapter import (
    TOOL1_ID, TOOL1_VER, SRV1_VER, TOOL2_ID, _final_chunk, _install_offline_stack,
    _make_mcp_session_factory, _reseal, _text_chunk, _tool_chunk, _tools_envelope,
)


def test_cfg04_frozen_endpoint_key_and_zero_parameters_reach_sdk_request(monkeypatch):
    envelope = _envelope_with_context()
    snapshot = envelope['agent_config']['snapshot']
    parameters = {'temperature': 0, 'top_p': 0.75, 'max_tokens': 37,
                  'presence_penalty': 0, 'frequency_penalty': 0, 'stop': []}
    snapshot['model'].update(base_url='https://frozen.invalid/v1', model_name='frozen-model', parameters=parameters)
    envelope['agent_config']['model_api_key'] = 'frozen-execution-key'
    _reseal(envelope)
    observed = []
    def respond(request):
        observed.append(request)
        body = ('data: {"id":"r1","object":"chat.completion.chunk","created":1,'
                '"model":"frozen-model","choices":[{"index":0,"delta":{"content":"ok"},'
                '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=body)
    actual_sdk = openai.AsyncOpenAI
    def offline_sdk(**kwargs):
        return actual_sdk(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    class Allow:
        def __init__(self, *args, **kwargs): pass
        async def authorize(self, **kwargs): return {'credential_generation': 7}
    monkeypatch.setattr(openai, 'AsyncOpenAI', offline_sdk)
    monkeypatch.setattr(platform_context, 'DemoModelAuthorizer', Allow)
    monkeypatch.setenv('OPENAI_API_KEY', 'unrelated-local-key')
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope)
    assert response.status_code == 200, response.text
    assert len(observed) == 1
    sent = observed[0]
    assert str(sent.url) == 'https://frozen.invalid/v1/chat/completions'
    assert sent.headers['authorization'] == 'Bearer frozen-execution-key'
    payload = json.loads(sent.content)
    assert payload['model'] == 'frozen-model'
    assert {key: payload[key] for key in parameters} == parameters
    receipt = response.json()['metadata']['agent_config_applied']
    assert receipt['model_request_count'] == 1 and receipt['applied_model_parameters'] == parameters


def test_cfg07_group_clarification_has_no_model_or_false_context_receipt(monkeypatch):
    _, calls, _, transport, counter = _offline_group(monkeypatch)
    envelope = _group_envelope()
    envelope['analysis_target'] = None
    envelope['message'] = '分析最近3天的消息'
    snapshot = envelope['agent_config']['snapshot']
    snapshot['memory']['read'] = snapshot['config']['memory_read'] = True
    _reseal(envelope)
    envelope['memory_context'] = {'schema_version': 1, 'status': 'ok', 'rendered_text': '不应注入的记忆',
                                  'included_ids': ['memory-1'], 'items': []}
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['metadata']['status'] == 'needs_input'
    assert body['metadata']['clarification']['reason'] == 'group_unresolved'
    assert not transport.requests and counter['authorize'] == 0 and not calls
    receipt = body['metadata']['agent_config_applied']
    assert receipt['model_request_count'] == 0 and receipt['applied_model_parameters'] == {}
    assert receipt['memory_injected'] is False and receipt['memory_included_ids'] == []
    assert not {'system_prompt', 'model', 'memory'} & set(receipt['applied_fields'])
    assert body['used_context_refs'] == [] and receipt['called_tool_versions'] == []


def test_tool02_invalid_batch_then_corrected_call_has_only_successful_receipt(monkeypatch):
    envelope = _tools_envelope()
    alias1, alias2 = tool_alias(TOOL1_ID), tool_alias(TOOL2_ID)
    calls, sessions = [], []
    factory = _make_mcp_session_factory([alias1, alias2], structured={'result': '实际检索结果'},
                                        cm_calls=sessions, tool_calls=calls)
    invalid = _tool_chunk('bad', 'unknown', '{}')
    valid_but_rejected = _tool_chunk('rejected', alias2, '{}').choices[0].delta.tool_calls[0]
    valid_but_rejected.index = 1
    invalid.choices[0].delta.tool_calls.append(valid_but_rejected)
    transport, _ = _install_offline_stack(monkeypatch, session_factory=factory, turns=[
        [invalid], [_tool_chunk('corrected', alias1, '{"query":"设备"}')],
        [_text_chunk('已获得实际检索结果'), _final_chunk()],
    ])
    actual_create = transport.sdk.chat.completions.create
    requests = []
    def capture_request(**kwargs):
        # 模型消息列表后续会继续追加，证据必须在发出时冻结。
        requests.append(copy.deepcopy(kwargs))
        return actual_create(**kwargs)
    transport.sdk.chat.completions.create = capture_request
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['metadata']['status'] == 'completed' and body['reply'] == '已获得实际检索结果'
    assert calls == [{'name': alias1, 'arguments': {'query': '设备'}}]
    assert len(requests) == 3
    messages = requests[1]['messages']
    rejected = next(i for i, message in enumerate(messages) if message.get('role') == 'assistant'
                    and [item['id'] for item in message.get('tool_calls', [])] == ['bad', 'rejected'])
    assert [item['tool_call_id'] for item in messages[rejected + 1:rejected + 3]] == ['bad', 'rejected']
    assert all('[TOOL ERROR]' in item['content'] for item in messages[rejected + 1:rejected + 3])
    receipt = body['metadata']['agent_config_applied']
    assert receipt['called_tool_versions'] == [{'tool_id': TOOL1_ID, 'tool_version_id': TOOL1_VER,
                                                'server_version_id': SRV1_VER, 'remote_name': 'search_notes',
                                                'outcome': 'success'}]
    assert body['metadata']['usage']['mcp_attempts'] == 1
