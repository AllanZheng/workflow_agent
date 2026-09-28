import hashlib
import json
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import flowagent.execution_service as execution_service
from flowagent.execution_service import _truncate_history, build_execution_app
from tests.test_managed_config_v3 import _digest, _envelope as _v3_envelope


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


def test_truncate_history_keeps_only_the_last_n_entries():
    """评审 7f：runtime.history_limit 真正生效。"""
    history = [{"role": "user", "content": str(i)} for i in range(5)]
    assert _truncate_history(history, 2) == history[-2:]
    assert _truncate_history(history, 100) == history
    # history_limit=0 表示不带历史；注意 Python 切片 history[-0:] 等价于整个
    # 列表，必须显式特殊处理（风险会商 YAML 正是用 0 表达"无状态"）。
    assert _truncate_history(history, 0) == []
    assert _truncate_history(history, -1) == []


def test_execution_service_requires_internal_bearer():
    client = TestClient(build_execution_app("secret-token"))
    assert client.post("/execution-v1/respond", json=_envelope()).status_code == 401


def test_execution_service_allows_all_requests_when_token_not_configured(caplog):
    """评审 D-11：FRAMEWORK_INTERNAL_TOKEN 为空时放行并打 WARNING，不再拒绝启动。"""
    import logging

    with caplog.at_level(logging.WARNING):
        client = TestClient(build_execution_app(""))
    assert any("execution-v1" in record.message for record in caplog.records)

    response = client.post("/execution-v1/respond", json=_envelope())
    assert response.status_code == 200
    assert response.json()["reply"] == "hello"

    # 依旧不需要任何 Authorization 头也能通过（无 token 场景不校验 Bearer）
    denied = client.get("/capabilities")
    assert denied.status_code == 200


def test_execution_service_reports_conservative_workflow_capabilities():
    client = TestClient(build_execution_app("secret-token"))
    denied = client.get("/capabilities")
    response = client.get("/capabilities", headers={"Authorization": "Bearer secret-token"})

    assert denied.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["framework_id"] == "workflow"
    assert payload["runtime_mode"] == "workflow_managed"
    assert payload["basic"] == {"respond": True, "stream": True, "config_receipt": True}
    assert payload["control"]["hitl_modes"] == []
    assert payload["tools"]["managed_mcp"] is False


def test_execution_app_opens_postgres_saver_during_lifespan(monkeypatch):
    saver = object()

    @asynccontextmanager
    async def fake_open_postgres_checkpointer():
        yield saver

    monkeypatch.setenv("FLOWAGENT_CHECKPOINT", "postgres")
    monkeypatch.setenv("FLOWAGENT_DATABASE_URL", "postgresql://configured-by-test")
    monkeypatch.setattr(execution_service, "open_postgres_checkpointer", fake_open_postgres_checkpointer)
    app = build_execution_app("secret-token")

    with TestClient(app):
        assert app.state.checkpointer is saver


def test_execution_app_defaults_checkpoint_to_disabled_without_env(monkeypatch):
    """评审问题 2：默认不再依赖 Postgres——未设置 FLOWAGENT_CHECKPOINT 时不建库连接。"""
    monkeypatch.delenv("FLOWAGENT_CHECKPOINT", raising=False)

    def fail_if_called():
        raise AssertionError("checkpoint 默认关闭时不应尝试打开 Postgres 连接")

    monkeypatch.setattr(execution_service, "open_postgres_checkpointer", fail_if_called)
    app = build_execution_app("secret-token")

    with TestClient(app):
        assert app.state.checkpointer is None


def test_execution_service_accepts_authenticated_reset_without_checkpoint_state():
    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/reset",
        json={"continuity_key": "c1", "generation": 2},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "continuity_key": "c1", "generation": 2}


def test_execution_service_reset_rejects_non_json_body():
    """评审测试计划第 6 项：非 JSON body 返回 422，不再是裸 500。"""
    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/reset",
        content=b"not json",
        headers={"Authorization": "Bearer secret-token", "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_reset"


def test_execution_service_reset_rejects_non_object_body():
    """非对象 body（如数组）不应触发 AttributeError -> 500。"""
    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/reset", json=["not", "an", "object"],
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_reset"


def test_execution_service_v1v2_execution_times_out_via_workflow_timeout_ms(monkeypatch):
    """评审 7f：workflow.timeout_ms 真正生效，而不是被忽略。"""
    import asyncio

    payload = _envelope()
    snapshot = payload["agent_config"]["snapshot"]
    snapshot["workflow"] = {
        "yaml": (
            "workflow:\n  name: t\n  start: slow\nnodes:\n"
            "  slow:\n    type: agent\n    output_key: reply\n    next: END\n"
        ),
        "timeout_ms": 1000,  # SnapshotWorkflow(V1) 的下限就是 1000ms
    }
    payload["agent_config"]["config_digest"] = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    class SlowBoundLlm:
        async def ainvoke(self, messages):
            await asyncio.sleep(3)
            return AIMessage(content="too late")

    class SlowLlm:
        def bind_tools(self, tools):
            return SlowBoundLlm()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: SlowLlm())

    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/respond", json=payload,
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "execution_timeout"


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


def _v3_skill_envelope():
    payload = _v3_envelope()
    snapshot = payload["agent_config"]["snapshot"]
    skill = {
        "skill_id": "604b6209-30da-4e57-bf93-73c6867cc101",
        "metadata_version_id": "704b6209-30da-4e57-bf93-73c6867cc101",
        "body_version_id": "804b6209-30da-4e57-bf93-73c6867cc101",
        "name": "Refund policy", "summary": "Published refund rules.",
    }
    snapshot["skills"] = [skill]
    snapshot["config"]["skills"] = [{key: skill[key] for key in (
        "skill_id", "metadata_version_id", "body_version_id",
    )}]
    snapshot["config_digest"] = _digest(snapshot)
    payload["agent_config"]["config_digest"] = snapshot["config_digest"]
    return payload


def test_v3_model_authorization_is_checked_before_every_model_call_and_receipt_is_versioned(monkeypatch):
    from langchain_core.messages import AIMessage

    model_calls = []
    auth_calls = []
    body = "Refunds are allowed for 30 days."

    class Authorizer:
        async def authorize(self, *, execution_id, model_version_id, generation):
            auth_calls.append((execution_id, model_version_id, generation))
            return object()

    class BoundLlm:
        async def ainvoke(self, messages):
            model_calls.append(messages)
            if len(model_calls) == 1:
                return AIMessage(content="", tool_calls=[{
                    "id": "load", "name": "load_skill",
                    "args": {"skill_id": "604b6209-30da-4e57-bf93-73c6867cc101"},
                }])
            return AIMessage(content="Refunds are allowed for 30 days.")

    class Llm:
        def bind_tools(self, tools):
            return BoundLlm()

    async def load_skill(*, execution_id, generation, skill):
        assert generation == 4
        assert skill["body_version_id"] == "804b6209-30da-4e57-bf93-73c6867cc101"
        return body

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: Llm())
    response = TestClient(build_execution_app(
        "internal", model_authorizer=Authorizer(), skill_loader=load_skill,
    )).post("/execution-v1/respond", json=_v3_skill_envelope(),
            headers={"Authorization": "Bearer internal"})

    assert response.status_code == 200
    assert response.json()["reply"] == "Refunds are allowed for 30 days."
    assert len(model_calls) == 2
    assert auth_calls == [
        ("f04b6209-30da-4e57-bf93-73c6867cc101", "a04b6209-30da-4e57-bf93-73c6867cc101", 4),
        ("f04b6209-30da-4e57-bf93-73c6867cc101", "a04b6209-30da-4e57-bf93-73c6867cc101", 4),
    ]
    receipt = response.json()["metadata"]["agent_config_applied"]
    assert receipt["execution_id"] == "f04b6209-30da-4e57-bf93-73c6867cc101"
    assert receipt["runtime_id"] == "804b6209-30da-4e57-bf93-73c6867cc101"
    assert receipt["schema_version"] == 3
    assert receipt["loaded_skill_versions"] == [{
        "skill_id": "604b6209-30da-4e57-bf93-73c6867cc101",
        "metadata_version_id": "704b6209-30da-4e57-bf93-73c6867cc101",
        "body_version_id": "804b6209-30da-4e57-bf93-73c6867cc101",
        "body_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "outcome": "success",
    }]
    assert body not in json.dumps(receipt)
    assert "model-secret" not in json.dumps(receipt)
    assert "execution-token-secret" not in json.dumps(receipt)


def test_v3_model_authorization_denial_prevents_any_model_request(monkeypatch):
    from flowagent.execution_authorization import AuthorizationError

    model_calls = []
    auth_calls = []

    class DenyingAuthorizer:
        async def authorize(self, **kwargs):
            auth_calls.append(kwargs)
            raise AuthorizationError("model_authorization_denied")

    class Llm:
        async def ainvoke(self, messages):
            model_calls.append(messages)
            raise AssertionError("model request must not be sent")

        def bind_tools(self, tools):
            return self

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: Llm())
    response = TestClient(build_execution_app(
        "internal", model_authorizer=DenyingAuthorizer(),
    )).post("/execution-v1/respond", json=_v3_envelope(),
            headers={"Authorization": "Bearer internal"})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "model_authorization_denied"
    assert len(auth_calls) == 1
    assert model_calls == []


def test_v3_injects_only_platform_memory_context_when_snapshot_enables_read(monkeypatch):
    from langchain_core.messages import AIMessage

    payload = _v3_envelope()
    snapshot = payload["agent_config"]["snapshot"]
    snapshot["memory"]["read"] = True
    snapshot["config"]["memory_read"] = True
    snapshot["config_digest"] = _digest(snapshot)
    payload["agent_config"]["config_digest"] = snapshot["config_digest"]
    payload["memory_context"] = {
        "schema_version": 1, "status": "ok",
        "rendered_text": "User prefers short answers.",
        "included_ids": ["memory-1"],
        "items": [{"id": "memory-1", "content": "User prefers short answers."}],
    }
    captured = {}

    class Llm:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return AIMessage(content="brief answer")

        def bind_tools(self, tools):
            return self

    class Authorizer:
        async def authorize(self, **kwargs):
            return object()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: Llm())
    response = TestClient(build_execution_app("internal", model_authorizer=Authorizer())).post(
        "/execution-v1/respond", json=payload,
        headers={"Authorization": "Bearer internal"},
    )

    assert response.status_code == 200
    assert "User prefers short answers." in captured["messages"][0].content
    assert response.json()["used_context_refs"] == ["memory-1"]
    receipt = response.json()["metadata"]["agent_config_applied"]
    assert receipt["memory_profile_version_id"] == "d04b6209-30da-4e57-bf93-73c6867cc101"
    assert receipt["memory_context_status"] == "ok"
    assert "User prefers short answers." not in json.dumps(receipt)


def test_v3_rejects_memory_context_when_frozen_policy_disables_read(monkeypatch):
    payload = _v3_envelope()
    payload["memory_context"] = {
        "schema_version": 1, "status": "ok", "rendered_text": "untrusted context",
        "included_ids": [], "items": [],
    }
    model_calls = []

    class Llm:
        async def ainvoke(self, messages):
            model_calls.append(messages)
            return AIMessage(content="should not run")

        def bind_tools(self, tools):
            return self

    class Authorizer:
        async def authorize(self, **kwargs):
            return object()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: Llm())
    response = TestClient(build_execution_app("internal", model_authorizer=Authorizer())).post(
        "/execution-v1/respond", json=payload,
        headers={"Authorization": "Bearer internal"},
    )

    assert response.status_code == 422
    assert model_calls == []
