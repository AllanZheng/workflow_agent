"""最小合法 V3 信封构造器（供平台适配/集成测试复用）。

纯数据构造：不依赖任何执行服务；快照变更后需重算 config_digest
（flowagent.adapters.platform.protocol.snapshot_digest 或本文件 _digest）。
"""
import hashlib
import json
import uuid

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


def test_digest_is_canonical():
    """摘要口径：去 config_digest、键序无关、紧凑分隔。"""
    snapshot = _snapshot()
    assert _digest(snapshot) == snapshot["config_digest"]
