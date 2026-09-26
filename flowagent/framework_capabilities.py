"""FlowAgent's deployed execution capability manifest."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from . import __version__


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SnapshotCapabilities(_StrictModel):
    versions: list[int] = Field(default_factory=lambda: [3])


class BasicCapabilities(_StrictModel):
    respond: bool = True
    stream: bool = True
    config_receipt: bool = True


class ContextCapabilities(_StrictModel):
    history_import: Literal["messages"] = "messages"
    memory_context_versions: list[int] = Field(default_factory=lambda: [1])


class ControlCapabilities(_StrictModel):
    versions: list[str] = Field(default_factory=list)
    hitl_modes: list[str] = Field(default_factory=list)
    steering_modes: list[str] = Field(default_factory=list)
    cancel: Literal["implemented", "unsupported"] = "unsupported"
    reset: Literal["implemented", "unsupported"] = "implemented"


class ToolCapabilities(_StrictModel):
    managed_mcp: bool = True
    execution_scoped_mcp: bool = True


class SkillCapabilities(_StrictModel):
    catalog_versions: list[int] = Field(default_factory=lambda: [1])
    load_versions: list[int] = Field(default_factory=lambda: [1])
    session_pinning: bool = True


class ReferenceCapabilities(_StrictModel):
    uuid: bool = True


class ModelCapabilities(_StrictModel):
    pre_call_authorization: bool = True


class FrameworkCapabilitiesV2(_StrictModel):
    schema_version: Literal[2] = 2
    framework_id: Literal["workflow"] = "workflow"
    adapter_version: str = __version__
    runtime_mode: Literal["workflow_managed"] = "workflow_managed"
    snapshot: SnapshotCapabilities = Field(default_factory=SnapshotCapabilities)
    basic: BasicCapabilities = Field(default_factory=BasicCapabilities)
    context: ContextCapabilities = Field(default_factory=ContextCapabilities)
    control: ControlCapabilities = Field(default_factory=ControlCapabilities)
    tools: ToolCapabilities = Field(default_factory=ToolCapabilities)
    skills: SkillCapabilities = Field(default_factory=SkillCapabilities)
    references: ReferenceCapabilities = Field(default_factory=ReferenceCapabilities)
    models: ModelCapabilities = Field(default_factory=ModelCapabilities)
    applied_config_fields: list[str] = Field(default_factory=lambda: [
        "schema_version", "agent_id", "revision_id", "runtime_id", "config_digest",
        "model", "tools", "skills", "memory", "system_prompt",
    ])
