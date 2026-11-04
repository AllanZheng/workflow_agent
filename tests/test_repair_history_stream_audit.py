"""历史优先级和流式进度：通过真实 Runner 与原生 HTTP 入口离线验证。"""
import copy

import pytest
from fastapi.testclient import TestClient

from flowagent.core.contracts import EventKind, ModelEvent, RunInput, Status
from flowagent.core.runner import Runner
from flowagent.workflows.chat import build_chat_workflow
from tests.native_helpers import FakeModel, FakeTools, make_context
from tests.test_native_http import install_model, make_app, parse_sse, write_profile


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('top', [[{'role': 'user', 'content': 'TOP-U'},
                                 {'role': 'assistant', 'content': 'TOP-A'}], []])
async def test_hist02_runner_top_history_wins_without_mutating_caller(top):
    legacy = [{'role': 'user', 'content': 'LEGACY-U'}, {'role': 'assistant', 'content': 'LEGACY-A'}]
    request = RunInput(workflow='chat', inputs={'message': 'NOW', 'history': legacy}, history=top)
    before = copy.deepcopy(request)
    model = FakeModel(default_text='answer')
    runner = Runner(workflows={'chat': build_chat_workflow()}, context_factory=lambda: make_context(model=model))
    result = await runner.run(request)
    assert result.status is Status.COMPLETED and request == before
    messages = model.requests[0]['messages']
    assert messages[1:] == top + [{'role': 'user', 'content': 'NOW'}]
    assert all('LEGACY' not in str(item) for item in messages)


@pytest.mark.parametrize('top', [[{'role': 'user', 'content': 'TOP-U'},
                                 {'role': 'assistant', 'content': 'TOP-A'}], []])
def test_hist02_native_http_top_history_wins_without_mutating_payload(tmp_path, monkeypatch, top):
    app, created, _ = make_app(tmp_path, monkeypatch, lambda: FakeModel(default_text='answer'))
    payload = {'workflow': 'chat', 'history': top,
               'inputs': {'message': 'NOW', 'history': [{'role': 'user', 'content': 'LEGACY-U'}]}}
    before = copy.deepcopy(payload)
    response = TestClient(app).post('/runs', json=payload)
    assert response.status_code == 200 and response.json()['status'] == 'completed'
    assert payload == before
    messages = created[0].requests[0]['messages']
    assert messages[1:] == top + [{'role': 'user', 'content': 'NOW'}]
    assert all('LEGACY' not in str(item) for item in messages)


def test_stream01_native_http_exposes_progress_but_only_final_text(tmp_path, monkeypatch):
    calls = [{'id': 't1', 'name': 'echo', 'arguments': '{"query":"check"}'}]
    turns = [[ModelEvent(kind='text_delta', text='UNVERIFIED_DRAFT'),
              ModelEvent(kind='completed', text='UNVERIFIED_DRAFT', tool_calls=calls, finish_reason='tool_calls')],
             FakeModel.text_turn('FINAL')]
    created, _ = install_model(monkeypatch, lambda: FakeModel(turns))
    tools = FakeTools(catalog=[{'name': 'echo'}])
    # 经原生入口装配真实 Runner，替换的只有可控工具连接。
    monkeypatch.setattr('flowagent.app.build_mcp_configs', lambda profile: [object()])
    monkeypatch.setattr('flowagent.app._attach_lazy_tools', lambda ctx, configs: setattr(ctx, 'tools', tools))
    from flowagent.adapters.http import build_http_app
    app = build_http_app({'config_path': write_profile(tmp_path), 'fallback_owner': 'caller'})
    response = TestClient(app).post('/runs/stream', json={'workflow': 'chat', 'inputs': {'message': 'test'}})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert 'UNVERIFIED_DRAFT' not in response.text
    assert any(event['kind'] == EventKind.STEP_STARTED.value for event in events)
    assert any(event['kind'] == EventKind.TOOL_FINISHED.value and event['data']['name'] == 'echo'
               and event['data']['status'] == 'success' for event in events)
    assert tools.calls == [('echo', {'query': 'check'})] and len(created[0].requests) == 2
    deltas = ''.join(event['data']['text'] for event in events if event['kind'] == EventKind.TEXT_DELTA.value)
    finished = [event for event in events if event['kind'] == EventKind.RUN_FINISHED.value]
    assert len(finished) == 1 and finished[0]['data']['result']['status'] == 'completed'
    assert deltas == finished[0]['data']['result']['reply'] == 'FINAL'
