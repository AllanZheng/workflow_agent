"""execution-v1 信封与 V3 快照协议（严格校验，旧版本兼容不进入内核）。

规则（技术方案 §7.1，与旧实现契约逐条一致）：
- StrictModel 全部 extra=forbid：未知键 422/ValidationError。
- V3 摘要：sha256(snapshot 去掉 config_digest，sort_keys、紧凑分隔、
  ensure_ascii=False、default=str、allow_nan=False)。
- 冻结交叉引用：runtime/model/memory/system_prompt/tools/skills 全一致、无重复。
- V3 信封：execution_id 与 trace.trace_id 均为 UUID 且相等。
- V3 必需凭据：model_api_key+generation+authorization_url+execution_token；
  有工具时另需 mcp_endpoint+tool_access_token。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- V3 快照


class ModelParametersV3(StrictModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    stop: list[str] | None = None


class ModelSnapshotV3(StrictModel):
    model_id: uuid.UUID
    version_id: uuid.UUID
    provider: Literal["openai_compatible"]
    model_name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    credential_id: uuid.UUID
    parameters: ModelParametersV3 = Field(default_factory=ModelParametersV3)


class ToolSnapshotV3(StrictModel):
    tool_id: uuid.UUID
    tool_version_id: uuid.UUID
    server_version_id: uuid.UUID
    remote_name: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    description: str = ""
    schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class SkillSnapshotV3(StrictModel):
    skill_id: uuid.UUID
    metadata_version_id: uuid.UUID
    body_version_id: uuid.UUID
    name: str = Field(min_length=1)
    summary: str


class MemorySnapshotV3(StrictModel):
    profile_id: uuid.UUID
    profile_version_id: uuid.UUID
    provider: str
    credential_id: uuid.UUID | None = None
    namespace: str = ""
    team_id: str = ""
    logical_agent_id: str = ""
    service_id: str = ""
    read: bool = False
    write: bool = False


class ToolReferenceV3(StrictModel):
    tool_id: uuid.UUID
    tool_version_id: uuid.UUID


class SkillReferenceV3(StrictModel):
    skill_id: uuid.UUID
    metadata_version_id: uuid.UUID
    body_version_id: uuid.UUID


class FrozenConfigV3(StrictModel):
    runtime_id: uuid.UUID
    enabled: bool
    model_version_id: uuid.UUID
    memory_profile_version_id: uuid.UUID
    memory_read: bool
    memory_write: bool
    system_prompt: str
    tools: list[ToolReferenceV3] = Field(default_factory=list)
    skills: list[SkillReferenceV3] = Field(default_factory=list)


class ManagedSnapshotV3(StrictModel):
    schema_version: Literal[3]
    agent_id: uuid.UUID
    revision_id: uuid.UUID
    runtime_id: uuid.UUID
    runtime_capability_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: FrozenConfigV3
    system_prompt: str
    model: ModelSnapshotV3
    tools: list[ToolSnapshotV3] = Field(default_factory=list)
    skills: list[SkillSnapshotV3] = Field(default_factory=list)
    memory: MemorySnapshotV3
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _references_are_frozen_consistently(self):
        if self.config.runtime_id != self.runtime_id:
            raise ValueError("runtime_id reference mismatch")
        if self.config.model_version_id != self.model.version_id:
            raise ValueError("model version reference mismatch")
        if self.config.memory_profile_version_id != self.memory.profile_version_id:
            raise ValueError("memory profile reference mismatch")
        if self.config.system_prompt != self.system_prompt:
            raise ValueError("system prompt reference mismatch")
        config_tools = {(i.tool_id, i.tool_version_id) for i in self.config.tools}
        snapshot_tools = {(i.tool_id, i.tool_version_id) for i in self.tools}
        if config_tools != snapshot_tools or len(config_tools) != len(self.config.tools) \
                or len(snapshot_tools) != len(self.tools):
            raise ValueError("tool version references mismatch")
        config_skills = {
            (i.skill_id, i.metadata_version_id, i.body_version_id) for i in self.config.skills
        }
        snapshot_skills = {
            (i.skill_id, i.metadata_version_id, i.body_version_id) for i in self.skills
        }
        if config_skills != snapshot_skills or len(config_skills) != len(self.config.skills) \
                or len(snapshot_skills) != len(self.skills):
            raise ValueError("skill version references mismatch")
        if len({i.tool_id for i in self.tools}) != len(self.tools):
            raise ValueError("duplicate tool reference")
        if len({i.skill_id for i in self.skills}) != len(self.skills):
            raise ValueError("duplicate skill reference")
        return self


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    """V3 规范摘要：排除自身 config_digest 的规范序列化 sha256。"""
    content = {k: v for k, v in snapshot.items() if k != "config_digest"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_managed_snapshot_v3(
    snapshot: dict[str, Any], *, agent_id: uuid.UUID,
    revision_id: uuid.UUID, config_digest: str,
) -> ManagedSnapshotV3:
    """原文摘要核验 + 严格解析 + 信封身份一致。"""
    if not isinstance(snapshot, dict) or snapshot_digest(snapshot) != snapshot.get("config_digest"):
        raise ValueError("snapshot digest mismatch")
    parsed = ManagedSnapshotV3.model_validate(snapshot)
    if parsed.agent_id != agent_id:
        raise ValueError("snapshot agent reference mismatch")
    if parsed.revision_id != revision_id:
        raise ValueError("snapshot revision reference mismatch")
    if parsed.config_digest != config_digest:
        raise ValueError("envelope digest mismatch")
    return parsed


# ---------------------------------------------------------------- 记忆上下文


class MemoryContextItem(StrictModel):
    id: str
    content: str
    source_refs: list[str] = Field(default_factory=list)
    provenance_status: Literal["verified", "unknown"] = "unknown"


class MemoryContextV1(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["ok", "empty", "disabled", "timeout", "unavailable", "unsupported"]
    rendered_text: str
    included_ids: list[str] = Field(default_factory=list)
    items: list[MemoryContextItem] = Field(default_factory=list)


# ---------------------------------------------------------------- 功能阶段任务


class FunctionTask(StrictModel):
    function_key: Literal["risk_consultation"]
    stage: Literal["draft", "final", "tasks"]
    output_schema: Literal["consultation_document_v1", "consultation_tasks_v1"]
    inputs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _stage_schema_match(self):
        expected = {"draft": "consultation_document_v1", "final": "consultation_document_v1",
                    "tasks": "consultation_tasks_v1"}.get(self.stage)
        if expected != self.output_schema:
            raise ValueError(f"stage/schema mismatch: {self.stage}/{self.output_schema}")
        return self

    @property
    def knowledge_mode(self) -> str:
        mode = self.inputs.get("knowledge_mode", "off")
        return mode if mode in ("off", "optional", "required") else "off"


# ---------------------------------------------------------------- 信封


class Trace(StrictModel):
    trace_id: str


class GroupAnalysisTarget(StrictModel):
    conversation_id: int = Field(strict=True, gt=0)


class ManagedAgentConfig(StrictModel):
    agent_id: uuid.UUID
    agent_type: Literal["workflow"]
    revision_id: uuid.UUID
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: dict[str, Any]  # V3 原文；经 parse_managed_snapshot_v3 严格核验
    model_api_key: SecretStr | None = Field(default=None, repr=False)
    model_credential_generation: int | None = Field(default=None, ge=1)
    mcp_endpoint: str | None = None
    tool_access_token: SecretStr | None = Field(default=None, repr=False)
    model_authorization_url: str | None = None
    execution_token: SecretStr | None = Field(default=None, repr=False)
    _snapshot_v3: ManagedSnapshotV3 | None = PrivateAttr(default=None)

    @field_validator("snapshot")
    @classmethod
    def _validate_v3_only(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("schema_version") != 3:
            raise ValueError("snapshot_version_not_supported")
        return value

    @field_validator("model_authorization_url")
    @classmethod
    def _validate_authorization_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("model authorization URL is invalid")
        return value

    @model_validator(mode="after")
    def _validate_snapshot_contract(self):
        try:
            self._snapshot_v3 = parse_managed_snapshot_v3(
                self.snapshot, agent_id=self.agent_id,
                revision_id=self.revision_id, config_digest=self.config_digest,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(str(exc)) from exc
        if self.model_api_key is None or self.model_credential_generation is None:
            raise ValueError("V3 model credential is required")
        if not self.model_authorization_url or self.execution_token is None:
            raise ValueError("V3 model authorization configuration is required")
        if self._snapshot_v3.tools and (not self.mcp_endpoint or self.tool_access_token is None):
            raise ValueError("V3 execution-scoped MCP configuration is required")
        return self

    @property
    def snapshot_v3(self) -> ManagedSnapshotV3 | None:
        return self._snapshot_v3


class ExecutionRequest(StrictModel):
    protocol_version: Literal["execution-v1"]
    execution_id: str
    idempotency_key: str | None = None
    deadline_ms: int | None = None
    trace: Trace
    binding_generation: int = Field(ge=0, strict=True)
    continuity_key: str
    message: str
    analysis_target: GroupAnalysisTarget | None = None
    identity: dict[str, Any]
    scene: dict[str, Any]
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    memory_context: MemoryContextV1 | None = None
    function_task: FunctionTask | None = None
    model: str | None = None
    model_revision: int | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    runtime_injection: None = None
    unified_runtime_injection: None = None
    agent_config: ManagedAgentConfig

    @model_validator(mode="after")
    def _validate_v3_execution_scope(self):
        try:
            execution_id = uuid.UUID(self.execution_id)
            trace_id = uuid.UUID(self.trace.trace_id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("V3 execution identifiers must be UUIDs") from exc
        if execution_id != trace_id:
            raise ValueError("V3 trace and execution identifiers must match")
        if self.memory_context is not None:
            snapshot = self.agent_config.snapshot_v3
            if snapshot is not None and not snapshot.memory.read:
                # 记忆注入需快照显式开启（旧服务 422 语义保持）。
                raise ValueError("memory_context_not_enabled_in_snapshot")
        return self
