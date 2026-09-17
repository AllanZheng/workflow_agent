import hashlib
import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import flowagent.execution_service as execution_service
from flowagent.execution_service import build_execution_app


def _envelope():
    snapshot = {
        "schema_version": 1,
        "agent": {"id": "604b6209-30da-4e57-bf93-73c6867cc101", "type": "workflow", "name": "Workflow"},
        "adapter": {"endpoint": "http://workflow:8080"},
        "model": None,
        "system_prompt": "answer",
        "workflow": {"yaml": "workflow:\n  name: t\n  start: answer\nnodes:\n  answer:\n    type: template\n    template: '{message}'\n    output_key: reply\n    next: END\n"},
    }
    digest = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "protocol_version": "execution-v1", "execution_id": "e1",
        "trace": {"trace_id": "e1"}, "binding_generation": 1,
        "continuity_key": "c1", "message": "hello",
        "identity": {"user_id": "u1"}, "scene": {"type": "chat"},
        "agent_config": {
            "agent_id": snapshot["agent"]["id"], "agent_type": "workflow",
            "revision_id": "704b6209-30da-4e57-bf93-73c6867cc101",
            "config_digest": digest, "snapshot": snapshot,
        },
    }


def test_execution_service_requires_internal_bearer():
    client = TestClient(build_execution_app("secret-token"))
    assert client.post("/execution-v1/respond", json=_envelope()).status_code == 401


def test_execution_service_accepts_authenticated_reset_without_checkpoint_state():
    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/reset",
        json={"continuity_key": "c1", "generation": 2},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "continuity_key": "c1", "generation": 2}


def test_execution_service_executes_managed_yaml_and_returns_receipt():
    client = TestClient(build_execution_app("secret-token"))
    response = client.post(
        "/execution-v1/respond", json=_envelope(),
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["execution_id"] == "e1"
    assert body["reply"] == "hello"
    assert body["metadata"]["agent_config_applied"]["revision_id"] == "704b6209-30da-4e57-bf93-73c6867cc101"


def test_execution_service_rejects_tampered_snapshot():
    payload = _envelope()
    payload["agent_config"]["snapshot"]["system_prompt"] = "tampered"
    client = TestClient(build_execution_app("secret-token"))
    response = client.post(
        "/execution-v1/respond", json=payload,
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "config_digest_mismatch"


def test_execution_service_wires_agent_node_to_demo_mcp(monkeypatch):
    payload = _envelope()
    snapshot = payload["agent_config"]["snapshot"]
    snapshot["workflow"]["yaml"] = """workflow:
  name: managed-agent
  start: main
nodes:
  main:
    type: agent
    output_key: reply
    max_tool_iterations: 2
    next: END
"""
    snapshot["model"] = {
        "provider": "openai_compatible", "name": "test-model",
        "base_url": "http://model.test/v1", "credential_id": "ignored",
    }
    payload["agent_config"]["model_api_key"] = "model-secret"
    payload["agent_config"]["mcp_endpoint"] = "http://demo.test/mcp/runtime/e1/"
    payload["agent_config"]["config_digest"] = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    captured = {}

    class FakeMcp:
        def __init__(self, endpoint, token): captured.update(endpoint=endpoint, token=token)
        async def list_tools(self): return []
        async def call_tool(self, name, arguments): raise AssertionError("unexpected tool call")

    class FakeBoundLlm:
        async def ainvoke(self, messages): return AIMessage(content="agent-answer")

    class FakeLlm:
        def bind_tools(self, tools): captured["tools"] = tools; return FakeBoundLlm()

    monkeypatch.setattr(execution_service, "McpEdgeClient", FakeMcp)
    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())

    response = TestClient(build_execution_app("internal", "workflow-mcp-token")).post(
        "/execution-v1/respond", json=payload,
        headers={"Authorization": "Bearer internal"},
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "agent-answer"
    assert captured == {
        "endpoint": "http://demo.test/mcp/runtime/e1/",
        "token": "workflow-mcp-token",
        "tools": [],
    }
