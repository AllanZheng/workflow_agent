"""普通多工具批次：真实托管入口、逐 ID 结果及实际使用回执。"""
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.platform.capabilities_runtime import tool_alias
from flowagent.app import build_platform_service
from tests.test_platform_v3_adapter import (
    TOOL1_ID, TOOL1_VER, SRV1_VER, TOOL2_ID, TOOL2_VER, SRV2_VER,
    _final_chunk, _install_offline_stack, _text_chunk, _tool_chunk, _tools_envelope,
)


@pytest.mark.parametrize('second_error', [False, True])
def test_tool04_multiple_calls_pair_ids_and_results_without_duplicate_execution(monkeypatch, second_error):
    alias1, alias2 = tool_alias(TOOL1_ID), tool_alias(TOOL2_ID)
    executions, requests = [], []

    @asynccontextmanager
    async def session_factory(endpoint, headers, timeout):
        async def list_tools():
            return SimpleNamespace(tools=[
                SimpleNamespace(name=alias, description='tool', input_schema={'type': 'object'})
                for alias in (alias1, alias2)])
        async def call_tool(name, arguments):
            executions.append((name, copy.deepcopy(arguments)))
            if name == alias2 and second_error:
                return SimpleNamespace(is_error=True, structured_content={'error_code': 'calendar_unavailable'}, content=[])
            return SimpleNamespace(is_error=False,
                                   structured_content={'source': name, 'argument': arguments}, content=[])
        yield SimpleNamespace(list_tools=list_tools, call_tool=call_tool)

    first = _tool_chunk('call-notes', alias1, '{"query":"inspection"}')
    second = _tool_chunk('call-calendar', alias2, '{"day":"monday"}').choices[0].delta.tool_calls[0]
    second.index = 1
    first.choices[0].delta.tool_calls.append(second)
    transport, _ = _install_offline_stack(monkeypatch, session_factory=session_factory, turns=[
        [first], [_text_chunk('completed'), _final_chunk()]])
    original_create = transport.sdk.chat.completions.create
    def capture(**kwargs):
        # 每次模型请求按发送时冻结，避免后续追加消息污染断言。
        requests.append(copy.deepcopy(kwargs))
        return original_create(**kwargs)
    transport.sdk.chat.completions.create = capture
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=_tools_envelope())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['metadata']['status'] == 'completed' and body['reply'] == 'completed'
    assert executions == [(alias1, {'query': 'inspection'}), (alias2, {'day': 'monday'})]
    assert len(requests) == 2 and body['metadata']['usage']['mcp_attempts'] == 2

    messages = requests[1]['messages']
    assistant = next(index for index, item in enumerate(messages)
                     if item['role'] == 'assistant' and item.get('tool_calls'))
    dispatched = messages[assistant]['tool_calls']
    assert [(call['id'], call['function']['name'], json.loads(call['function']['arguments']))
            for call in dispatched] == [('call-notes', alias1, {'query': 'inspection'}),
                                        ('call-calendar', alias2, {'day': 'monday'})]
    results = messages[assistant + 1:]
    assert [item['role'] for item in results] == ['tool', 'tool']
    assert [item['tool_call_id'] for item in results] == ['call-notes', 'call-calendar']
    assert json.loads(results[0]['content']) == {'source': alias1, 'argument': {'query': 'inspection'}}
    if second_error:
        assert results[1]['content'].startswith('[TOOL ERROR]')
        assert 'calendar_unavailable' in results[1]['content']
    else:
        assert json.loads(results[1]['content']) == {'source': alias2, 'argument': {'day': 'monday'}}
    called = body['metadata']['agent_config_applied']['called_tool_versions']
    assert [(item['tool_id'], item['tool_version_id'], item['server_version_id']) for item in called] == (
        [(TOOL1_ID, TOOL1_VER, SRV1_VER)] if second_error else
        [(TOOL1_ID, TOOL1_VER, SRV1_VER), (TOOL2_ID, TOOL2_VER, SRV2_VER)])
    assert all(item['outcome'] == 'success' for item in called)
