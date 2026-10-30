"""STREAM-02：恢复前草稿、重试预算及最终 SSE 终态的组合证据。"""
import pytest
from fastapi.testclient import TestClient

import flowagent.app as app_module
from flowagent.core.contracts import ErrorCode, FlowAgentCoreError, ModelEvent
from tests.test_native_http import make_app, parse_sse


@pytest.mark.parametrize('failure', ['length', 'upstream_error'])
def test_stream02_discards_draft_and_counts_single_recovery(tmp_path, monkeypatch, failure):
    class Model:
        def __init__(self): self.calls = 0
        async def stream(self, messages, tools, limits):
            self.calls += 1
            if self.calls == 1:
                yield ModelEvent(kind='text_delta', text='UNVERIFIED_DRAFT')
                if failure == 'upstream_error':
                    raise FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, 'temporary outage')
                yield ModelEvent(kind='completed', text='UNVERIFIED_DRAFT', finish_reason='length')
            else:
                assert self.calls == 2, '恢复只允许一次'
                yield ModelEvent(kind='text_delta', text='FINAL')
                yield ModelEvent(kind='completed', text='FINAL', finish_reason='stop')
    budgets = []
    build_budget = app_module.build_budget
    def capture_budget(profile):
        budget = build_budget(profile)
        budgets.append(budget)
        return budget
    monkeypatch.setattr(app_module, 'build_budget', capture_budget)
    app, models, _ = make_app(tmp_path, monkeypatch, Model, fallback_owner='local')
    response = TestClient(app).post('/runs/stream', json={'workflow': 'chat', 'inputs': {'message': 'hello'}})
    assert response.status_code == 200 and 'UNVERIFIED_DRAFT' not in response.text
    events = parse_sse(response.text)
    finished = [event for event in events if event['kind'] == 'run.finished']
    assert len(finished) == 1
    result = finished[0]['data']['result']
    deltas = ''.join(event['data']['text'] for event in events if event['kind'] == 'text.delta')
    assert result['status'] == 'completed' and deltas == result['reply'] == 'FINAL'
    assert not result['recovery'].get('fallback_attempted', False)
    assert models[0].calls == result['usage']['model_requests'] == budgets[0].model_spent == 2
    assert result['usage']['model_retries'] == 1
