"""Strict V3 snapshot models shared by the execution adapter and runtime."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelParametersV3(_StrictModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    stop: list[str] | None = None


class ModelSnapshotV3(_StrictModel):
    model_id: uuid.UUID
    version_id: uuid.UUID
    provider: Literal["openai_compatible"]
    model_name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    credential_id: uuid.UUID
    parameters: ModelParametersV3 = Field(default_factory=ModelParametersV3)


class ToolSnapshotV3(_StrictModel):
    tool_id: uuid.UUID
    tool_version_id: uuid.UUID
    server_version_id: uuid.UUID
    remote_name: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    description: str = ""
    schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class SkillSnapshotV3(_StrictModel):
    skill_id: uuid.UUID
    metadata_version_id: uuid.UUID
    body_version_id: uuid.UUID
    name: str = Field(min_length=1)
    summary: str


class MemorySnapshotV3(_StrictModel):
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


class ToolReferenceV3(_StrictModel):
    tool_id: uuid.UUID
    tool_version_id: uuid.UUID


class SkillReferenceV3(_StrictModel):
    skill_id: uuid.UUID
    metadata_version_id: uuid.UUID
    body_version_id: uuid.UUID


class FrozenConfigV3(_StrictModel):
    runtime_id: uuid.UUID
    enabled: bool
    model_version_id: uuid.UUID
    memory_profile_version_id: uuid.UUID
    memory_read: bool
    memory_write: bool
    system_prompt: str
    tools: list[ToolReferenceV3] = Field(default_factory=list)
    skills: list[SkillReferenceV3] = Field(default_factory=list)


class ManagedSnapshotV3(_StrictModel):
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
            raise ValueError("memory profile version reference mismatch")
        if self.config.system_prompt != self.system_prompt:
            raise ValueError("system prompt reference mismatch")

        config_tools = {(item.tool_id, item.tool_version_id) for item in self.config.tools}
        snapshot_tools = {(item.tool_id, item.tool_version_id) for item in self.tools}
        if config_tools != snapshot_tools or len(config_tools) != len(self.config.tools) or len(snapshot_tools) != len(self.tools):
            raise ValueError("tool version references mismatch")
        config_skills = {
            (item.skill_id, item.metadata_version_id, item.body_version_id)
            for item in self.config.skills
        }
        snapshot_skills = {
            (item.skill_id, item.metadata_version_id, item.body_version_id)
            for item in self.skills
        }
        if config_skills != snapshot_skills or len(config_skills) != len(self.config.skills) or len(snapshot_skills) != len(self.skills):
            raise ValueError("skill version references mismatch")
        if len({item.tool_id for item in self.tools}) != len(self.tools):
            raise ValueError("duplicate tool reference")
        if len({item.skill_id for item in self.skills}) != len(self.skills):
            raise ValueError("duplicate skill reference")
        return self


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    """Return the Demo V3 canonical digest, excluding its own digest field."""
    content = {key: value for key, value in snapshot.items() if key != "config_digest"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_managed_snapshot_v3(
    snapshot: dict[str, Any], *, agent_id: uuid.UUID,
    revision_id: uuid.UUID, config_digest: str,
) -> ManagedSnapshotV3:
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
