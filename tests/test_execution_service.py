import hashlib
import json

from fastapi.testclient import TestClient

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
