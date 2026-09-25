"""群分析助手「发布脚本」（阶段 3）：向 Admin 配置库写入智能体发布内容。

对照 demo_platform seed_phase1_workflow.py 的写入契约（零改 demo_platform /
admin 代码），额外覆盖群分析助手特有面：
  - 两份凭据：模型 key（model_api_key）+ im 超管令牌（mcp_headers，AES-GCM
    契约与平台 PostgresCredentialResolver 一致：AAD = "<id>:mcp_headers:<ver>"）；
  - 快照 mcp_tools 注册 find_group / fetch_group_messages：url 指向 im-tools-mcp
    的 MCP 端点，input_schema 直接复用 services/im_tools_mcp/tools.py 的
    契约单一事实源（注册与实现永不漂移）；
  - 快照 workflow.yaml = configs/managed/group_analysis.yaml（改口径/改流程
    = 改 YAML 重跑本脚本）。

幂等：凭据/智能体/修订均 ON CONFLICT 更新；同一 digest 的修订不重复创建。

环境变量（缺失即拒绝启动，错误不携带任何敏感值）见 README.md。
运行（容器外，能连到 admin 库的环境）：
  python deploy/group-analysis/publish_group_analysis.py
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import sys
import uuid
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# 契约单一事实源：Admin 注册的 input_schema 与服务实现同源
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from services.im_tools_mcp.tools import (  # noqa: E402
    FETCH_GROUP_ANALYSIS_INPUT_SCHEMA,
    FETCH_GROUP_MESSAGES_INPUT_SCHEMA,
    FIND_GROUP_INPUT_SCHEMA,
)

from flowagent.managed_config import load_managed_spec  # noqa: E402
from flowagent.execution_service import ManagedSnapshotV1  # noqa: E402


class PublishConfigurationError(ValueError):
    """发布环境变量非法；消息只含变量名与期望，不含任何令牌。"""


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise PublishConfigurationError(f"{name} is required")
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip() or default


def _positive_int(name: str, default: int) -> int:
    try:
        result = int(_optional(name, str(default)))
    except ValueError as exc:
        raise PublishConfigurationError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise PublishConfigurationError(f"{name} must be a positive integer")
    return result


def _uuid(name: str) -> uuid.UUID:
    try:
        return uuid.UUID(_required(name))
    except ValueError as exc:
        raise PublishConfigurationError(f"{name} must be a UUID") from exc


def _master_key() -> bytes:
    try:
        key = base64.b64decode(_required("AGENT_CONFIG_MASTER_KEY_B64"), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise PublishConfigurationError(
            "AGENT_CONFIG_MASTER_KEY_B64 is invalid") from exc
    if len(key) != 32:
        raise PublishConfigurationError(
            "AGENT_CONFIG_MASTER_KEY_B64 must decode to 32 bytes")
    return key


def _is_true(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise PublishConfigurationError(f"{name} must be a boolean")


def _encrypt(key: bytes, credential_id: uuid.UUID, ctype: str,
             plaintext: str) -> tuple[bytes, bytes]:
    """Admin 凭据契约：AES-256-GCM，AAD = "<id>:<type>:<key_version>"。"""
    nonce = os.urandom(12)
    aad = f"{credential_id}:{ctype}:1".encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), aad)
    return ciphertext, nonce


def _workflow_yaml() -> str:
    """受管工作流正文；先过受管门禁再入库（坏 YAML 拒绝发布）。"""
    path = Path(_optional("GROUP_ANALYSIS_YAML_PATH",
                          str(REPO_ROOT / "configs" / "managed" / "group_analysis.yaml")))
    try:
        yaml_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PublishConfigurationError(
            f"GROUP_ANALYSIS_YAML_PATH 无法读取: {path}") from exc
    try:
        load_managed_spec(yaml_text)  # 受管门禁：仅 agent/llm/template 才允许入库
    except Exception as exc:  # noqa: BLE001 - 门禁拒绝信息直接回给操作者
        raise PublishConfigurationError(
            f"工作流 YAML 未通过受管门禁: {exc}") from exc
    return yaml_text


def _snapshot(agent_id: uuid.UUID, model_credential_id: uuid.UUID,
              im_credential_id: uuid.UUID) -> dict:
    im_tools_url = _optional(
        "GROUP_ANALYSIS_IM_TOOLS_URL", "http://im-tools-mcp:8090/mcp/"
    ).rstrip("/") + "/"
    model_name = _required("GROUP_ANALYSIS_MODEL_NAME")
    model_base_url = _required("GROUP_ANALYSIS_MODEL_BASE_URL").rstrip("/")
    agent_name = _optional("GROUP_ANALYSIS_AGENT_NAME", "群分析助手")

    server_id = _positive_int("GROUP_ANALYSIS_MCP_SERVER_ID", 0)

    def mcp_tool(exposed: str, schema: dict, description: str, tool_id: int) -> dict:
        return {
            "tool_id": tool_id,
            "server_id": server_id,
            "exposed_name": exposed,
            "remote_name": exposed,  # im-tools-mcp 侧同名暴露
            "transport": "streamable_http",
            "url": im_tools_url,
            "credential_id": str(im_credential_id),
            "description": description,
            "input_schema": schema,
        }

    return {
        "schema_version": 1,
        "agent": {"id": str(agent_id), "type": "workflow", "name": agent_name},
        "adapter": {
            "endpoint": _optional("GROUP_ANALYSIS_FLOWAGENT_ENDPOINT",
                                  "http://workflow:8080"),
            "timeout_ms": 300_000,
        },
        "model": {
            "provider": "openai_compatible",
            "name": model_name,
            "base_url": model_base_url,
            "credential_id": str(model_credential_id),
            "parameters": {"temperature": 0.2},
        },
        "system_prompt": (
            "你是群分析助手：根据用户请求对指定 im 群生成「总结 / 分类统计 / "
            "汇报要点」三段式中文 Markdown 报告；信息不足时反问澄清，绝不编造数据。"
        ),
        "workflow": {
            "yaml": _workflow_yaml(),
            "max_tool_iterations": 8,
            "timeout_ms": 300_000,
        },
        "mcp_tools": [
            mcp_tool(
                "fetch_group_analysis_input", FETCH_GROUP_ANALYSIS_INPUT_SCHEMA,
                "按固定群 ID 与时间窗读取去重消息，返回完整性状态",
                _positive_int("GROUP_ANALYSIS_INPUT_TOOL_ID", 0),
            ),
        ],
        "skills": [],
        "runtime": {"history_limit": 20},
        "compatibility": None,
    }


def _digest(snapshot: dict) -> str:
    """与 flowagent execution-v1 的 config_digest 同一 JSON 规则。"""
    import hashlib
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def publish() -> tuple[uuid.UUID, uuid.UUID, bool]:
    # DB 依赖惰性导入：本仓库 venv 无 sqlalchemy，发布须在带该依赖的环境运行
    # （demo_platform venv / 部署容器均满足）；--dry-run 则完全不需要
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import (async_sessionmaker, create_async_engine)
    except ImportError as exc:  # pragma: no cover - 环境问题，给操作者可读指引
        raise PublishConfigurationError(
            "发布需要 sqlalchemy（在 demo_platform venv 或部署容器内运行）") from exc

    database_url = _required("AGENT_PLATFORM_DATABASE_URL")
    agent_id = _uuid("GROUP_ANALYSIS_AGENT_ID")
    model_credential_id = _uuid("GROUP_ANALYSIS_MODEL_CREDENTIAL_ID")
    im_credential_id = _uuid("GROUP_ANALYSIS_IM_CREDENTIAL_ID")
    model_api_key = _required("GROUP_ANALYSIS_MODEL_API_KEY")
    im_access_token = _required("GROUP_ANALYSIS_IM_ACCESS_TOKEN")
    master_key = _master_key()
    set_default = _is_true("GROUP_ANALYSIS_SET_AS_DEFAULT", default=True)
    published_by = _optional("GROUP_ANALYSIS_PUBLISHED_BY", "group-analysis-mvp")

    snapshot = _snapshot(agent_id, model_credential_id, im_credential_id)
    ManagedSnapshotV1.model_validate(snapshot)
    digest = _digest(snapshot)
    # mcp_headers 凭据内容 = 注入 im-tools-mcp 的 HTTP 头（平台按原样透传）
    im_headers = json.dumps(
        {"Authorization": f"Bearer {im_access_token}"}, ensure_ascii=False)
    model_ct, model_nonce = _encrypt(master_key, model_credential_id,
                                     "model_api_key", model_api_key)
    im_ct, im_nonce = _encrypt(master_key, im_credential_id,
                               "mcp_headers", im_headers)

    engine = create_async_engine(database_url, pool_pre_ping=True)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions.begin() as session:
            for item in snapshot["mcp_tools"]:
                registered = (await session.execute(text(
                    "SELECT t.name, t.enabled, s.status, s.transport, s.url "
                    "FROM public.mcp_tools t JOIN public.mcp_servers s ON s.id=t.server_id "
                    "WHERE t.id=:tool_id AND s.id=:server_id"
                ), {"tool_id": item["tool_id"], "server_id": item["server_id"]})).mappings().one_or_none()
                if (registered is None or registered["name"] != item["remote_name"]
                        or not registered["enabled"]
                        or registered["status"] != "enabled"
                        or registered["transport"] != item["transport"]
                        or str(registered["url"]).rstrip("/") != item["url"].rstrip("/")):
                    raise PublishConfigurationError(
                        f"registered MCP tool mismatch: {item['exposed_name']}")
            existing_workflow_id = await session.scalar(text(
                "SELECT id FROM admin_system.agents WHERE agent_type='workflow' "
                "LIMIT 1"
            ))
            if existing_workflow_id is not None and existing_workflow_id != agent_id:
                raise PublishConfigurationError(
                    "a workflow Agent already exists; set GROUP_ANALYSIS_AGENT_ID "
                    "to its UUID（接管既有 Workflow 智能体）"
                )
            # 凭据：模型 key + im 令牌（mcp_headers）
            await session.execute(text(
                "INSERT INTO admin_system.credentials "
                "(id,name,type,ciphertext,nonce,key_version,metadata_json) "
                "VALUES (:id,:name,'model_api_key',:ciphertext,:nonce,1,'{}'::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET "
                "name=EXCLUDED.name,ciphertext=EXCLUDED.ciphertext,"
                "nonce=EXCLUDED.nonce,key_version=EXCLUDED.key_version,"
                "updated_at=now()"
            ), {
                "id": model_credential_id,
                "name": f"group-analysis-model-{agent_id}",
                "ciphertext": model_ct, "nonce": model_nonce,
            })
            await session.execute(text(
                "INSERT INTO admin_system.credentials "
                "(id,name,type,ciphertext,nonce,key_version,metadata_json) "
                "VALUES (:id,:name,'mcp_headers',:ciphertext,:nonce,1,"
                "'{\"purpose\":\"im-tools-mcp\"}'::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET "
                "name=EXCLUDED.name,ciphertext=EXCLUDED.ciphertext,"
                "nonce=EXCLUDED.nonce,key_version=EXCLUDED.key_version,"
                "updated_at=now()"
            ), {
                "id": im_credential_id,
                "name": f"group-analysis-im-{agent_id}",
                "ciphertext": im_ct, "nonce": im_nonce,
            })
            await session.execute(text(
                "INSERT INTO admin_system.agents (id,agent_key,agent_type,name,status) "
                "VALUES (:id,:agent_key,'workflow',:name,'disabled') "
                "ON CONFLICT (id) DO UPDATE SET updated_at=now()"
            ), {"id": agent_id, "agent_key": f"agent-{agent_id}",
                "name": snapshot["agent"]["name"]})
            # 修订：同 digest 幂等
            revision_id = await session.scalar(text(
                "SELECT id FROM admin_system.agent_config_revisions "
                "WHERE agent_id=:agent_id AND config_digest=:digest "
                "ORDER BY revision_no DESC LIMIT 1"
            ), {"agent_id": agent_id, "digest": digest})
            if revision_id is None:
                revision_id = uuid.uuid4()
                revision_no = await session.scalar(text(
                    "SELECT COALESCE(MAX(revision_no), 0) + 1 "
                    "FROM admin_system.agent_config_revisions WHERE agent_id=:agent_id"
                ), {"agent_id": agent_id})
                await session.execute(text(
                    "INSERT INTO admin_system.agent_config_revisions "
                    "(id,agent_id,revision_no,schema_version,snapshot_json,"
                    "config_digest,change_summary,published_by) "
                    "VALUES (:id,:agent_id,:revision_no,1,CAST(:snapshot AS jsonb),"
                    ":digest,'group-analysis assistant MVP',:published_by)"
                ), {
                    "id": revision_id, "agent_id": agent_id,
                    "revision_no": revision_no,
                    "snapshot": json.dumps(snapshot, ensure_ascii=False,
                                           separators=(",", ":")),
                    "digest": digest, "published_by": published_by,
                })
            await session.execute(text(
                "UPDATE admin_system.agents SET "
                "active_revision_id=:revision_id,status='enabled',updated_at=now() "
                "WHERE id=:agent_id"
            ), {"agent_id": agent_id, "revision_id": revision_id})
            if set_default:
                await session.execute(text(
                    "INSERT INTO admin_system.agent_platform_settings "
                    "(id,default_agent_id,updated_by) "
                    "VALUES (1,:agent_id,:published_by) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "default_agent_id=EXCLUDED.default_agent_id,"
                    "updated_by=EXCLUDED.updated_by,updated_at=now()"
                ), {"agent_id": agent_id, "published_by": published_by})
        return agent_id, revision_id, set_default
    finally:
        await engine.dispose()


def main() -> int:
    # --dry-run：只构建快照并校验（受管门禁 + schema 同源 + digest），
    # 不触碰数据库、不需要 sqlalchemy —— 供 CI 与操作者预演
    if "--dry-run" in sys.argv:
        try:
            agent_id = _uuid("GROUP_ANALYSIS_AGENT_ID")
            snapshot = _snapshot(
                agent_id,
                _uuid("GROUP_ANALYSIS_MODEL_CREDENTIAL_ID"),
                _uuid("GROUP_ANALYSIS_IM_CREDENTIAL_ID"),
            )
            ManagedSnapshotV1.model_validate(snapshot)
            digest = _digest(snapshot)
        except PublishConfigurationError as exc:
            print(f"[group-analysis-publish] configuration error: {exc}",
                  file=sys.stderr)
            return 2
        print(f"[group-analysis-publish] dry-run ok agent_id={agent_id} "
              f"digest={digest[:16]}... mcp_tools="
              f"{[t['exposed_name'] for t in snapshot['mcp_tools']]}")
        return 0
    try:
        agent_id, revision_id, set_default = asyncio.run(publish())
    except PublishConfigurationError as exc:
        print(f"[group-analysis-publish] configuration error: {exc}",
              file=sys.stderr)
        return 2
    except Exception as exc:  # 输出不泄露凭据/连接串
        print(f"[group-analysis-publish] database publish failed: "
              f"{type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"[group-analysis-publish] agent_id={agent_id} "
          f"revision_id={revision_id} default={set_default}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
