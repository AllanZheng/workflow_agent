"""功能阶段经 /execution-v1/respond 的端到端行为（FST-01 路由、FST-03/04 错误外形）。"""
import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import flowagent.execution_service as execution_service
from flowagent.execution_service import build_execution_app
from tests.af_helpers import ScriptedLlm
from tests.test_managed_config_v3 import _envelope as _v3_envelope


class _Authorizer:
    async def authorize(self, *, execution_id, model_version_id, generation):
        return object()


def _payload(task):
    payload = _v3_envelope()
    payload["function_task"] = task
    return payload


_TASK = {"function_key": "risk_consultation", "stage": "draft",
         "output_schema": "consultation_document_v1", "inputs": {"knowledge_mode": "off"}}


def _post(monkeypatch, llm, payload):
    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: llm)
    return TestClient(build_execution_app("internal", model_authorizer=_Authorizer())).post(
        "/execution-v1/respond", json=payload, headers={"Authorization": "Bearer internal"})


# 带 function_task 的 V3 请求走 function_stage，返回结构化结果与稿件正文。
def test_respond_runs_function_stage(monkeypatch):
    document = {"title": "t", "summary": "s", "body_markdown": "## 稿件", "risks": []}
    response = _post(monkeypatch, ScriptedLlm([AIMessage(content=json.dumps(document))]),
                     _payload(_TASK))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reply"] == "## 稿件"
    result = body["metadata"]["function_result"]
    assert result["status"] == "ok" and result["knowledge"]["status"] == "not_used"
    assert body["metadata"]["agent_config_applied"]["schema_version"] == 3


# 两次结构化输出都非法时返回 422 invalid_structured_output，且标记可重试。
def test_respond_reports_invalid_structured_output_as_retryable(monkeypatch):
    llm = ScriptedLlm([AIMessage(content="坏"), AIMessage(content="还是坏")])
    response = _post(monkeypatch, llm, _payload(_TASK))
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_structured_output" and error["retryable"] is True


# 阶段与输出 schema 不对应的 function_task 在请求校验阶段被拒绝。
def test_mismatched_function_task_is_rejected(monkeypatch):
    bad = dict(_TASK, stage="tasks")
    response = _post(monkeypatch, ScriptedLlm([]), _payload(bad))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


# 不带 function_task 的 V3 请求仍走原 qa 流程（FST-01 回归）。
def test_v3_without_function_task_still_uses_qa(monkeypatch):
    llm = ScriptedLlm([AIMessage(content="普通回答")])
    response = _post(monkeypatch, llm, _v3_envelope())
    assert response.status_code == 200 and response.json()["reply"] == "普通回答"
    assert "function_result" not in response.json()["metadata"]
