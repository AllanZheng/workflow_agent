import sys
import types
import uuid

from flowagent.execution_service import ManagedAgentConfig, _request_llm
from flowagent.managed_config import load_managed_spec


# 生成当前状态的快照。
# English: Handle snapshot operations.
def _snapshot(yaml_text):
    from uuid import uuid5, NAMESPACE_URL
    return {
        "schema_version": 2,
        "agent": {"id": str(uuid5(NAMESPACE_URL, "risk-consultation:test")),
                  "type": "workflow", "name": "Risk consultation"},
        "adapter": {"endpoint": "http://flowagent:8080"},
        "model": {"provider": "openai_compatible", "name": "test-model",
                  "base_url": "http://model:8000/v1",
                  "credential_id": str(uuid5(NAMESPACE_URL, "risk-consultation:model")),
                  "parameters": {"temperature": 0.2}},
        "system_prompt": "",
        "workflow": {"yaml": yaml_text, "max_tool_iterations": 1, "timeout_ms": 180000},
        "mcp_tools": [], "skills": [],
        "memory": {"profile_id": "disabled", "read_enabled": False, "write_enabled": False},
        "runtime": {"history_limit": 0, "skill_load_mode": "on_demand",
                    "visible_text_policy": "incremental"},
    }


# 验证“风险咨询`managed`工作流判断是否模型仅与`does`不修改群组`mode`”场景下的预期行为。
# English: Verify risk consultation managed workflow is model only and does not change group mode.
def test_risk_consultation_managed_workflow_is_model_only_and_does_not_change_group_mode():
    from pathlib import Path

    raw = (Path(__file__).parents[1] / "configs" / "managed" / "risk_consultation.yaml").read_text(encoding="utf-8")
    spec = load_managed_spec(raw)

    assert spec.workflow.name == "risk-consultation"
    assert spec.workflow.group_analysis_mode == "legacy"
    assert {node.type for node in spec.nodes.values()} == {"llm"}
    assert spec.nodes["generate"].output_key == "reply"


# 验证“风险咨询提供方密钥判断是否已读来自`flowagent`环境”场景下的预期行为。
# English: Verify risk consultation provider key is read from flowagent environment.
def test_risk_consultation_provider_key_is_read_from_flowagent_environment(monkeypatch):
    import yaml

    raw = ("workflow:\n  name: risk-consultation\n  start: generate\nnodes:\n"
           "  generate:\n    type: llm\n    system: safe\n    prompt: '{message}'\n"
           "    output_key: reply\n    next: END\n")
    snapshot = _snapshot(raw)
    captured = {}

    class Model:
        pass

    # 构建模型。
    # English: Build model.
    def build_model(**kwargs):
        captured.update(kwargs)
        return Model()

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=build_model))
    monkeypatch.setenv("FLOWAGENT_RISK_CONSULTATION_MODEL_API_KEY", "service-held-test-key")
    digest = __import__("hashlib").sha256(
        __import__("json").dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode()
    ).hexdigest()
    config = ManagedAgentConfig.model_validate({
        "agent_id": snapshot["agent"]["id"], "agent_type": "workflow",
        "revision_id": str(uuid.uuid4()), "config_digest": digest, "snapshot": snapshot,
    })

    result = _request_llm(load_managed_spec(raw), config)

    assert isinstance(result, Model)
    assert captured["api_key"] == "service-held-test-key"
    assert config.model_api_key is None


# 验证“`non`咨询工作流`cannot``use`咨询服务凭据”场景下的预期行为。
# English: Verify non consultation workflow cannot use consultation service credential.
def test_non_consultation_workflow_cannot_use_consultation_service_credential(monkeypatch):
    from flowagent.errors import ConfigError

    raw = "workflow:\n  name: unrelated\n  start: generate\nnodes:\n  generate:\n    type: llm\n    system: safe\n    prompt: '{message}'\n    output_key: reply\n    next: END\n"
    snapshot = _snapshot(raw)
    snapshot["workflow"]["yaml"] = raw
    digest = __import__("hashlib").sha256(
        __import__("json").dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode()
    ).hexdigest()
    config = ManagedAgentConfig.model_validate({
        "agent_id": snapshot["agent"]["id"], "agent_type": "workflow",
        "revision_id": str(uuid.uuid4()), "config_digest": digest, "snapshot": snapshot,
    })
    monkeypatch.setenv("FLOWAGENT_RISK_CONSULTATION_MODEL_API_KEY", "service-held-test-key")

    try:
        _request_llm(load_managed_spec(raw), config)
    except ConfigError as exc:
        assert str(exc) == "managed_model_config_missing"
    else:
        raise AssertionError("unrelated workflow used the consultation-only credential")


# English: Verify flowagent model credentials env extends the default binding table.
def test_flowagent_model_credentials_env_extends_the_default_binding_table(monkeypatch):
    """评审 7e：额外的 credential_id/workflow_name 绑定可通过环境变量声明，
    不需要再为每个新场景写一条硬编码分支。"""
    import json as _json

    raw = ("workflow:\n  name: another-managed-agent\n  start: generate\nnodes:\n"
           "  generate:\n    type: llm\n    system: safe\n    prompt: '{message}'\n"
           "    output_key: reply\n    next: END\n")
    credential_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "another-agent:model"))
    snapshot = _snapshot(raw)
    snapshot["agent"]["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "another-agent:test"))
    snapshot["model"]["credential_id"] = credential_id
    captured = {}

    class Model:
        pass

    # 构建模型。
    # English: Build model.
    def build_model(**kwargs):
        captured.update(kwargs)
        return Model()

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=build_model))
    monkeypatch.setenv("FLOWAGENT_ANOTHER_AGENT_API_KEY", "another-service-key")
    monkeypatch.setenv("FLOWAGENT_MODEL_CREDENTIALS", _json.dumps([{
        "credential_id": credential_id, "workflow_name": "another-managed-agent",
        "env_var": "FLOWAGENT_ANOTHER_AGENT_API_KEY",
    }]))
    digest = __import__("hashlib").sha256(
        __import__("json").dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode()
    ).hexdigest()
    config = ManagedAgentConfig.model_validate({
        "agent_id": snapshot["agent"]["id"], "agent_type": "workflow",
        "revision_id": str(uuid.uuid4()), "config_digest": digest, "snapshot": snapshot,
    })

    result = _request_llm(load_managed_spec(raw), config)

    assert isinstance(result, Model)
    assert captured["api_key"] == "another-service-key"
    # 内置的风险会商绑定依旧生效，未被环境变量覆盖掉
    assert config.model_api_key is None


# English: Verify unreachable LLM node does not force a model credential.
def test_unreachable_llm_node_does_not_force_a_model_credential():
    """评审 7f：needs_model 只看可达节点——YAML 里存在一个不可达的 llm 节点
    不应该强制要求模型凭据。"""
    from flowagent.llm import MockLLM

    raw = ("workflow:\n  name: template-only\n  start: generate\nnodes:\n"
           "  generate:\n    type: template\n    template: '{message}'\n"
           "    output_key: reply\n    next: END\n"
           "  orphan:\n    type: llm\n    prompt: 'never reached'\n"
           "    output_key: unused\n    next: END\n")
    snapshot = _snapshot(raw)
    snapshot["model"] = None
    digest = __import__("hashlib").sha256(
        __import__("json").dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode()
    ).hexdigest()
    config = ManagedAgentConfig.model_validate({
        "agent_id": snapshot["agent"]["id"], "agent_type": "workflow",
        "revision_id": str(uuid.uuid4()), "config_digest": digest, "snapshot": snapshot,
    })

    result = _request_llm(load_managed_spec(raw), config)

    assert isinstance(result, MockLLM)

