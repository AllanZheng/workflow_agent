import hashlib
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from flowagent.execution_service import (
    ExecutionRequest,
    GroupAnalysisTarget,
    _with_analysis_target,
    build_execution_app,
)


# 处理`digest`相关逻辑。
# English: Handle digest operations.
def _digest(snapshot):
    content = {key: value for key, value in snapshot.items() if key != "config_digest"}
    return hashlib.sha256(json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")).hexdigest()


# 生成当前状态的快照。
# English: Handle snapshot operations.
def _snapshot():
    agent_id = "604b6209-30da-4e57-bf93-73c6867cc101"
    revision_id = "704b6209-30da-4e57-bf93-73c6867cc101"
    runtime_id = "804b6209-30da-4e57-bf93-73c6867cc101"
    model_id = "904b6209-30da-4e57-bf93-73c6867cc101"
    model_version_id = "a04b6209-30da-4e57-bf93-73c6867cc101"
    credential_id = "b04b6209-30da-4e57-bf93-73c6867cc101"
    memory_id = "c04b6209-30da-4e57-bf93-73c6867cc101"
    memory_version_id = "d04b6209-30da-4e57-bf93-73c6867cc101"
    snapshot = {
        "schema_version": 3,
        "agent_id": agent_id,
        "revision_id": revision_id,
        "runtime_id": runtime_id,
        "runtime_capability_digest": "e" * 64,
        "config": {
            "runtime_id": runtime_id,
            "enabled": True,
            "model_version_id": model_version_id,
            "memory_profile_version_id": memory_version_id,
            "memory_read": False,
            "memory_write": False,
            "system_prompt": "Answer concisely.",
            "tools": [],
            "skills": [],
        },
        "system_prompt": "Answer concisely.",
        "model": {
            "model_id": model_id,
            "version_id": model_version_id,
            "provider": "openai_compatible",
            "model_name": "model-test",
            "base_url": "https://model.test/v1",
            "credential_id": credential_id,
            "parameters": {"temperature": 0},
        },
        "tools": [],
        "skills": [],
        "memory": {
            "profile_id": memory_id,
            "profile_version_id": memory_version_id,
            "provider": "disabled",
            "credential_id": None,
            "namespace": "",
            "team_id": "",
            "logical_agent_id": "",
            "service_id": "",
            "read": False,
            "write": False,
        },
    }
    snapshot["config_digest"] = _digest(snapshot)
    return snapshot


# 处理`envelope`相关逻辑。
# English: Handle envelope operations.
def _envelope(snapshot=None):
    snapshot = snapshot or _snapshot()
    return {
        "protocol_version": "execution-v1",
        "execution_id": "f04b6209-30da-4e57-bf93-73c6867cc101",
        "trace": {"trace_id": "f04b6209-30da-4e57-bf93-73c6867cc101"},
        "binding_generation": 4,
        "continuity_key": "conversation-1",
        "message": "hello",
        "identity": {"user_id": "user-1"},
        "scene": {"type": "chat"},
        "agent_config": {
            "agent_id": snapshot["agent_id"],
            "agent_type": "workflow",
            "revision_id": snapshot["revision_id"],
            "config_digest": snapshot["config_digest"],
            "snapshot": snapshot,
            "model_api_key": "model-secret",
            "model_credential_generation": 7,
            "model_authorization_url": "https://demo.test/internal/executions/f04b6209-30da-4e57-bf93-73c6867cc101/model-authorization",
            "execution_token": "execution-token-secret",
        },
    }


# 验证“`v3`生成快照解析UUID 标识`references`与`verified``digest`”场景下的预期行为。
# English: Verify v3 snapshot parses UUID references and verified digest.
def test_v3_snapshot_parses_uuid_references_and_verified_digest():
    request = ExecutionRequest.model_validate(_envelope())

    assert isinstance(request.agent_config.agent_id, uuid.UUID)
    assert isinstance(request.agent_config.snapshot_v3.runtime_id, uuid.UUID)
    assert request.agent_config.snapshot_v3.schema_version == 3
    assert request.agent_config.snapshot_v3.model.version_id == uuid.UUID(
        "a04b6209-30da-4e57-bf93-73c6867cc101"
    )


# 验证“`v3`生成快照拒绝`tampering``even`当`outer``digest`判断是否`unchanged`”场景下的预期行为。
# English: Verify v3 snapshot rejects tampering even when outer digest is unchanged.
def test_v3_snapshot_rejects_tampering_even_when_outer_digest_is_unchanged():
    payload = _envelope()
    payload["agent_config"]["snapshot"]["model"]["parameters"]["temperature"] = 0.7

    with pytest.raises(ValidationError, match="digest"):
        ExecutionRequest.model_validate(payload)


# 验证“`v3`生成快照拒绝`non`UUID 标识`references`与`v2`结构”场景下的预期行为。
# English: Verify v3 snapshot rejects non UUID references and v2 schema.
def test_v3_snapshot_rejects_non_uuid_references_and_v2_schema():
    payload = _envelope()
    payload["agent_config"]["snapshot"]["runtime_id"] = "runtime-one"
    payload["agent_config"]["snapshot"]["config_digest"] = _digest(payload["agent_config"]["snapshot"])
    payload["agent_config"]["config_digest"] = payload["agent_config"]["snapshot"]["config_digest"]
    with pytest.raises(ValidationError):
        ExecutionRequest.model_validate(payload)

    payload = _envelope()
    payload["agent_config"]["snapshot"]["schema_version"] = 2
    payload["agent_config"]["snapshot"]["config_digest"] = _digest(payload["agent_config"]["snapshot"])
    payload["agent_config"]["config_digest"] = payload["agent_config"]["snapshot"]["config_digest"]
    with pytest.raises(ValidationError):
        ExecutionRequest.model_validate(payload)


# 验证“`v3`生成快照拒绝`reference``disagreement``between``sections`”场景下的预期行为。
# English: Verify v3 snapshot rejects reference disagreement between sections.
def test_v3_snapshot_rejects_reference_disagreement_between_sections():
    payload = _envelope()
    payload["agent_config"]["snapshot"]["model"]["version_id"] = "a14b6209-30da-4e57-bf93-73c6867cc101"
    payload["agent_config"]["snapshot"]["config_digest"] = _digest(payload["agent_config"]["snapshot"])
    payload["agent_config"]["config_digest"] = payload["agent_config"]["snapshot"]["config_digest"]

    with pytest.raises(ValidationError, match="version"):
        ExecutionRequest.model_validate(payload)


# 验证“`v3`模型校验必需的执行`scoped`授权字段”场景下的预期行为。
# English: Verify v3 model requires execution scoped authorization fields.
def test_v3_model_requires_execution_scoped_authorization_fields():
    payload = _envelope()
    payload["agent_config"].pop("model_authorization_url")
    payload["agent_config"].pop("execution_token")

    with pytest.raises(ValidationError, match="authorization"):
        ExecutionRequest.model_validate(payload)


# 验证“群组`analysis`目标判断是否`added``v3``system`提示词”场景下的预期行为。
# English: Verify group analysis target is added to v3 system prompt.
def test_group_analysis_target_is_added_to_v3_system_prompt():
    prompt = _with_analysis_target(
        "Answer concisely.", GroupAnalysisTarget(conversation_id=6),
    )

    assert "conversation_id: 6" in prompt
    assert "exact conversation_id" in prompt
    assert prompt.startswith("Answer concisely.")


# 验证“能力`advertise``v3`UUID 标识`lazy`技能与授权契约”场景下的预期行为。
# English: Verify capabilities advertise v3 UUID lazy skill and authorization contract.
def test_capabilities_advertise_v3_uuid_lazy_skill_and_authorization_contract():
    response = TestClient(build_execution_app("internal-token")).get(
        "/capabilities/v2", headers={"Authorization": "Bearer internal-token"},
    )

    assert response.status_code == 200
    body = response.json()
    assert 3 in body["snapshot"]["versions"]
    assert body["references"]["uuid"] is True
    assert body["skills"]["catalog_versions"] == [1]
    assert body["skills"]["load_versions"] == [1]
    assert body["skills"]["session_pinning"] is True
    assert body["tools"]["execution_scoped_mcp"] is True
    assert body["models"]["pre_call_authorization"] is True
