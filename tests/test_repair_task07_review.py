"""TASK-07 fixed group binding and authorization, with offline Edge transport."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.platform.capabilities_runtime import tool_alias
from flowagent.app import build_platform_service
from tests.test_platform_v3_adapter import TOOL1_ID, TOOL1_VER, SRV1_VER, _final_chunk, _install_offline_stack, _text_chunk
from tests.test_repair_fixed_task import _group_envelope


@pytest.mark.parametrize('mode', ['allowed', 'invisible', 'edge_denied'])
def test_task07_group_pinned_headers_and_edge_authorization(monkeypatch, mode):
    alias = tool_alias(TOOL1_ID)
    sessions, calls = [], []
    @asynccontextmanager
    async def edge(endpoint, headers, timeout):
        sessions.append(dict(headers))
        async def list_tools():
            names = [] if mode == 'invisible' else [alias]
            return SimpleNamespace(tools=[SimpleNamespace(name=n, description='', input_schema={'type': 'object'}) for n in names])
        async def call_tool(name, arguments):
            calls.append((name, dict(arguments), dict(headers)))
            if mode == 'edge_denied':
                return SimpleNamespace(is_error=True, structured_content={'error_code': 'authorization_denied'}, content=[])
            return SimpleNamespace(is_error=False, structured_content={
                'messages': [{'message_ref': 'm1', 'text': 'risk'}], 'window': {}, 'counts': {}}, content=[])
        yield SimpleNamespace(list_tools=list_tools, call_tool=call_tool)
    transport, _ = _install_offline_stack(monkeypatch, session_factory=edge, turns=[
        [_text_chunk('{"assignments":[{"message_ref":"m1","category":"问题"}]}'), _final_chunk()],
        [_text_chunk('report\n引用: [m1]'), _final_chunk()],
    ])
    envelope = _group_envelope()
    response = TestClient(build_platform_service()).post('/execution-v1/respond', json=envelope)
    if mode == 'invisible':
        assert response.status_code != 200
        assert response.json()['error']['code'] == 'capability_unavailable'
        assert calls == [] and transport.requests == []
        return
    assert len(calls) == 1
    name, arguments, headers = calls[0]
    assert name == alias and arguments['conversation_id'] == 42
    assert headers['X-Agent-Tool-ID'] == TOOL1_ID
    assert headers['X-Agent-Tool-Version-ID'] == TOOL1_VER
    assert headers['Authorization'] == 'Bearer ' + envelope['agent_config']['tool_access_token']
    assert headers['X-Execution-ID'] == envelope['execution_id']
    if mode == 'edge_denied':
        assert response.status_code != 200
        assert response.json()['error']['code'] == 'capability_unavailable'
        assert response.json()['metadata']['recovery']['final_source'] == 'caller'
        assert 'analysis_result' not in response.json()['metadata']
        assert calls[0][0] == alias and transport.requests == []
        return
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['metadata']['status'] == 'completed'
    assert body['metadata']['agent_config_applied']['called_tool_versions'] == [
        {'tool_id': TOOL1_ID, 'tool_version_id': TOOL1_VER, 'server_version_id': SRV1_VER,
         'remote_name': 'fetch_group_analysis_input', 'outcome': 'success'}]
