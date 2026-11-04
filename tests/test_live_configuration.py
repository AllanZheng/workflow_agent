"""Live 驱动的离线自检，不连接任何外部服务。"""
import json
from types import SimpleNamespace

import httpx
import pytest

from tests import test_native_live as live


@pytest.fixture
def anyio_backend():
    return 'asyncio'


def test_live_profile_explicitly_disables_thinking_when_requested(tmp_path, monkeypatch):
    monkeypatch.setenv('BENCH_MODEL_BASE_URL', 'http://model.invalid/v1')
    monkeypatch.setenv('BENCH_MODEL_NAME', 'offline-model')
    monkeypatch.setenv('BENCH_MODEL_API_KEY', 'test-key')
    monkeypatch.setenv('BENCH_MODEL_DISABLE_THINKING', '1')
    profile = json.loads(live._profile(tmp_path, mcp_url=None).read_text(encoding='utf8'))
    model = profile['profiles']['default']['model']
    assert model['parameters']['extra_body'] == {'chat_template_kwargs': {'enable_thinking': False}}
    from flowagent.config import resolve_model_spec
    assert resolve_model_spec(profile['profiles']['default'])['extra_body'] == model['parameters']['extra_body']
    assert 'test-key' not in json.dumps(profile)
    assert model['api_key_env'] == 'BENCH_MODEL_API_KEY'
    assert resolve_model_spec(profile['profiles']['default'])['api_key'] == 'test-key'


def test_live_profile_keeps_default_when_not_requested(tmp_path, monkeypatch):
    monkeypatch.setenv('BENCH_MODEL_BASE_URL', 'http://model.invalid/v1')
    monkeypatch.setenv('BENCH_MODEL_NAME', 'offline-model')
    monkeypatch.setenv('BENCH_MODEL_API_KEY', 'test-key')
    monkeypatch.delenv('BENCH_MODEL_DISABLE_THINKING', raising=False)
    profile = json.loads(live._profile(tmp_path, mcp_url=None).read_text(encoding='utf8'))
    assert 'parameters' not in profile['profiles']['default']['model']


def test_live_profile_rejects_missing_key_instead_of_writing_a_default(tmp_path, monkeypatch):
    from flowagent.config import resolve_model_spec
    from flowagent.errors import ConfigError
    monkeypatch.setenv('BENCH_MODEL_BASE_URL', 'http://model.invalid/v1')
    monkeypatch.setenv('BENCH_MODEL_NAME', 'offline-model')
    monkeypatch.delenv('BENCH_MODEL_API_KEY', raising=False)
    profile = json.loads(live._profile(tmp_path, mcp_url=None).read_text(encoding='utf8'))
    with pytest.raises(ConfigError, match='BENCH_MODEL_API_KEY'):
        resolve_model_spec(profile['profiles']['default'])


@pytest.mark.parametrize('failure', [None, 'degraded', 'missing_calls', 'wrong_query', 'wrong_result', 'missing_attempt'])
def test_live_tool_evidence_requires_completed_real_call(failure):
    status = 'degraded' if failure == 'degraded' else 'completed'
    attempts = 0 if failure == 'missing_attempt' else 1
    result = SimpleNamespace(status=SimpleNamespace(value=status), usage=SimpleNamespace(mcp_attempts=attempts),
                             reply='核心机房每日巡检一次')
    calls = [{'query': '核心机房巡检频率', 'result': {'data': [{'text': '核心机房每日巡检一次'}]}}]
    if failure == 'missing_calls': calls = []
    if failure == 'wrong_query': calls[0]['query'] = '天气'
    if failure == 'wrong_result': calls[0]['result'] = {'data': [{'text': '无关结果'}]}
    if failure is None:
        live._assert_l01b_evidence(result, calls)
    else:
        with pytest.raises(AssertionError):
            live._assert_l01b_evidence(result, calls)


@pytest.mark.anyio
@pytest.mark.parametrize('failure', [None, 'fallback', 'skill', 'tool', 'revision', 'model'])
async def test_platform_live_driver_checks_real_response_and_receipt(failure):
    config = {'platform_url': 'http://platform.invalid', 'agent_id': 'agent',
              'desk_token': 'desk', 'management_token': 'manager', 'runtime_id': 'runtime',
              'revision_id': 'revision', 'config_digest': 'digest', 'skill_id': 'skill',
              'skill_body_version_id': 'body', 'tool_id': 'tool', 'tool_version_id': 'tool-version',
              'server_version_id': 'server-version', 'prompt': 'find marker', 'expected_text': 'marker'}
    details = {'schema_version': 3, 'runtime_id': 'runtime', 'model_request_count': 1,
               'loaded_skill_versions': [{'skill_id': 'skill', 'body_version_id': 'body', 'outcome': 'success'}],
               'called_tool_versions': [{'tool_id': 'tool', 'tool_version_id': 'tool-version',
                                         'server_version_id': 'server-version', 'outcome': 'success'}]}
    row = {'execution_id': 'execution', 'revision_id': 'revision', 'runtime_id': 'runtime',
           'config_digest': 'digest', 'status': 'loaded', 'details_json': details}
    if failure == 'skill': details['loaded_skill_versions'] = []
    if failure == 'tool': details['called_tool_versions'] = []
    if failure == 'revision': row['revision_id'] = 'other'
    if failure == 'model': details['model_request_count'] = 0
    def handler(request):
        if request.url.path == '/v1/agent/respond':
            assert request.headers['authorization'] == 'Bearer desk'
            body = json.loads(request.content)
            assert body['agent_id'] == 'agent' and body['message'] == 'find marker'
            source = 'direct_llm' if failure == 'fallback' else 'agent'
            return httpx.Response(200, json={'trace_id': 'execution', 'reply': 'marker',
                                            'metadata': {'response_source': source, 'status': 'completed'}})
        assert request.url.path == '/internal/management/v1/agents/agent/receipts'
        assert request.headers['authorization'] == 'Bearer manager'
        return httpx.Response(200, json={'data': [row]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if failure is None:
            evidence = await live._run_platform_v3_check(client, config)
            assert evidence['execution_id'] == 'execution' and 'desk_token' not in evidence
        else:
            with pytest.raises(AssertionError):
                await live._run_platform_v3_check(client, config)
