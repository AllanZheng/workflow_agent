import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from flowagent.execution_service import ManagedAgentConfig
from flowagent.execution_service import build_execution_app
from fastapi.testclient import TestClient


FIXTURES = Path(__file__).parent / "fixtures" / "agent_capabilities"


# 处理`fixture`相关逻辑。
# English: Handle fixture operations.
def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# 验证“`shared``v1``v2``fixture``digests``are``stable`”场景下的预期行为。
# English: Verify shared v1 v2 fixture digests are stable.
def test_shared_v1_v2_fixture_digests_are_stable():
    # 处理`digests`相关逻辑。
    # English: Handle digests operations.
    def digests(name):
        raw = (FIXTURES / name).read_bytes()
        return {
            hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        }

    assert "cc36e60c56c9c2a4fe0d66feae5b4ef74ae20dea66df919f63a7daf503f719b7" in digests("v1.json")
    assert "091e92e0660ea0e589d2346faa1c969450c346ee7120fad7cbdda15f3e3bc436" in digests("v2.json")


# 验证“`managed`代理配置接受`shared`生成快照”场景下的预期行为。
# English: Verify managed agent config accepts shared snapshot.
@pytest.mark.parametrize("name", ["v1.json", "v2.json"])
def test_managed_agent_config_accepts_shared_snapshot(name):
    raw = _fixture(name)
    result = ManagedAgentConfig.model_validate({
        "agent_id": raw["agent"]["id"], "agent_type": "workflow",
        "revision_id": "704b6209-30da-4e57-bf93-73c6867cc101",
        "config_digest": "a" * 64, "snapshot": raw,
    })
    assert result.snapshot.schema_version == raw["schema_version"]


# 验证“`managed`代理配置拒绝技能内容体未知字段与令牌”场景下的预期行为。
# English: Verify managed agent config rejects skill body unknown fields and token.
def test_managed_agent_config_rejects_skill_body_unknown_fields_and_token():
    for mutate in (
        lambda data: data["skills"][0].update(content="private body"),
        lambda data: data.update(unexpected=True),
        lambda data: data.update(tool_access_token="secret"),
    ):
        raw = _fixture("v2.json")
        mutate(raw)
        with pytest.raises(ValidationError):
            ManagedAgentConfig.model_validate({
                "agent_id": raw["agent"]["id"], "agent_type": "workflow",
                "revision_id": "704b6209-30da-4e57-bf93-73c6867cc101",
                "config_digest": "a" * 64, "snapshot": raw,
            })


# 验证“能力`v2``uses``separate``authenticated`接口”场景下的预期行为。
# English: Verify capabilities v2 uses a separate authenticated endpoint.
def test_capabilities_v2_uses_a_separate_authenticated_endpoint():
    client = TestClient(build_execution_app("internal", "legacy-workload"))

    old = client.get("/capabilities", headers={"Authorization": "Bearer internal"})
    new = client.get("/capabilities/v2", headers={"Authorization": "Bearer internal"})
    unauthorized = client.get("/capabilities/v2")

    assert old.status_code == new.status_code == 200
    assert old.json()["schema_version"] == 1
    assert new.json()["schema_version"] == 2
    assert new.json()["snapshot"]["versions"] == [1, 2, 3]
    assert new.json()["skills"]["load_versions"] == [1]
    assert new.json()["tools"]["execution_scoped_mcp"] is True
    assert unauthorized.status_code == 401
