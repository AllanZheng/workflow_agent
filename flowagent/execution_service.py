"""面向 Demo 的受管 execution-v1 同步服务。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)

from .agent_runtime import ManagedAgentRuntime
from .engine import build_graph, open_postgres_checkpointer
from .errors import ConfigError
from .execution.deps import model_credential_bindings
from .execution.errors import classify_exception, error_response_body
from .execution.planner import select_flow
from .execution_authorization import (
    AuthorizationError,
    ModelAuthorizer,
)
from .framework_capabilities import FrameworkCapabilitiesV2
from .function_stage import FunctionTask, run_function_stage
from .group_analysis import analyze_group
from .llm import MockLLM
from .managed_config import load_managed_spec
from .managed_config_v3 import ManagedSnapshotV3, parse_managed_snapshot_v3
from .mcp_edge_client import McpEdgeClient
from .security import bearer_ok, warn_if_disabled
from .obs_log import RequestIdMiddleware, setup_logging
from .execution.v3_runtime import chat_runtime, execution_parts


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Trace(StrictModel):
    trace_id: str


class SnapshotAgentV1(StrictModel):
    id: uuid.UUID
    type: Literal["workflow", "hermes", "nanobot"]
    name: str | None = Field(default=None, max_length=128)


class SnapshotAgentV2(StrictModel):
    id: uuid.UUID
    type: Literal["workflow", "hermes", "nanobot"]
    name: str = Field(min_length=1, max_length=128)


class SnapshotAdapter(StrictModel):
    endpoint: str = Field(min_length=1)
    timeout_ms: int = Field(default=90_000, ge=1_000, le=300_000)
    credential_id: uuid.UUID | None = None


class SnapshotModel(StrictModel):
    provider: Literal["openai_compatible"]
    name: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1)
    credential_id: uuid.UUID
    parameters: dict[str, Any] = Field(default_factory=dict)


class SnapshotModelV1(StrictModel):
    provider: Literal["openai_compatible"]
    name: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1)
    credential_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class SnapshotWorkflow(StrictModel):
    yaml: str = Field(min_length=1, max_length=524_288)
    max_tool_iterations: int = Field(default=8, ge=1, le=8)
    timeout_ms: int = Field(default=90_000, ge=1_000, le=300_000)


class SnapshotWorkflowV1(StrictModel):
    yaml: str = Field(min_length=1, max_length=524_288)
    max_tool_iterations: int = Field(default=8, ge=1, le=8)
    timeout_ms: int = Field(default=90_000, ge=1_000, le=300_000)


class SnapshotMcpToolV1(StrictModel):
    tool_id: int
    server_id: int
    exposed_name: str = Field(min_length=1, max_length=128)
    remote_name: str = Field(min_length=1, max_length=200)
    transport: Literal["streamable_http"]
    url: str = Field(min_length=1)
    credential_id: uuid.UUID | None = None
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class SnapshotSkillV1(StrictModel):
    skill_id: int | None = None
    key: str | None = Field(default=None, max_length=100)
    version: str | None = Field(default=None, max_length=50)
    content: str
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    package_hash: str = ""


class SnapshotRuntimeV1(StrictModel):
    history_limit: int = Field(default=20, ge=0, le=100)


class ManagedSnapshotV1(StrictModel):
    schema_version: Literal[1] = 1
    agent: SnapshotAgentV1
    adapter: SnapshotAdapter | None = None
    model: SnapshotModelV1 | None = None
    system_prompt: str = ""
    workflow: SnapshotWorkflowV1 | None = None
    mcp_tools: tuple[SnapshotMcpToolV1, ...] = ()
    skills: tuple[SnapshotSkillV1, ...] = ()
    runtime: SnapshotRuntimeV1 = Field(default_factory=SnapshotRuntimeV1)
    compatibility: dict[str, Any] | None = None


class SnapshotMcpToolV2(SnapshotMcpToolV1):
    schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    grant_id: uuid.UUID


class SnapshotSkillV2(StrictModel):
    skill_id: int
    key: str = Field(min_length=1, max_length=100)
    metadata_version_id: uuid.UUID
    body_version_id: uuid.UUID
    name: str = Field(min_length=1, max_length=200)
    summary: str = Field(max_length=1000)
    metadata_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_tool_ids: tuple[int, ...] = ()


class SnapshotMemoryV2(StrictModel):
    profile_id: Literal["disabled", "default"]
    read_enabled: bool = False
    write_enabled: bool = False


class SnapshotRuntimeV2(StrictModel):
    history_limit: int = Field(default=20, ge=0, le=100)
    skill_load_mode: Literal["on_demand"] = "on_demand"
    visible_text_policy: Literal["incremental"] = "incremental"


class ManagedSnapshotV2(StrictModel):
    schema_version: Literal[2] = 2
    agent: SnapshotAgentV2
    adapter: SnapshotAdapter
    model: SnapshotModel | None = None
    system_prompt: str = ""
    workflow: SnapshotWorkflow | None = None
    mcp_tools: tuple[SnapshotMcpToolV2, ...] = ()
    skills: tuple[SnapshotSkillV2, ...] = ()
    memory: SnapshotMemoryV2
    runtime: SnapshotRuntimeV2 = Field(default_factory=SnapshotRuntimeV2)


ManagedSnapshot = Annotated[
    ManagedSnapshotV1 | ManagedSnapshotV2 | ManagedSnapshotV3,
    Field(discriminator="schema_version"),
]


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


class ManagedAgentConfig(StrictModel):
    agent_id: uuid.UUID
    agent_type: Literal["workflow"]
    revision_id: uuid.UUID
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: ManagedSnapshot
    model_api_key: SecretStr | None = Field(default=None, repr=False)
    model_credential_generation: int | None = Field(default=None, ge=1)
    mcp_endpoint: str | None = None
    tool_access_token: SecretStr | None = Field(default=None, repr=False)
    model_authorization_url: str | None = None
    execution_token: SecretStr | None = Field(default=None, repr=False)
    _snapshot_v3: ManagedSnapshotV3 | None = PrivateAttr(default=None)

    # 校验`raw``v3`生成快照。
    # English: Validate raw v3 snapshot.
    @field_validator("snapshot", mode="before")
    @classmethod
    def _validate_raw_v3_snapshot(cls, value, info):
        if isinstance(value, dict) and value.get("schema_version") == 3:
            try:
                parse_managed_snapshot_v3(
                    value, agent_id=info.data["agent_id"],
                    revision_id=info.data["revision_id"],
                    config_digest=info.data["config_digest"],
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(str(exc)) from exc
        return value

    # 校验授权URL 地址。
    # English: Validate authorization URL.
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

    # 校验生成快照契约。
    # English: Validate snapshot contract.
    @model_validator(mode="after")
    def _validate_snapshot_contract(self):
        if isinstance(self.snapshot, ManagedSnapshotV3):
            self._snapshot_v3 = self.snapshot
            if self.model_api_key is None or self.model_credential_generation is None:
                raise ValueError("V3 model credential is required")
            if not self.model_authorization_url or self.execution_token is None:
                raise ValueError("V3 model authorization configuration is required")
            if self._snapshot_v3.tools and (
                not self.mcp_endpoint or self.tool_access_token is None
            ):
                raise ValueError("V3 execution-scoped MCP configuration is required")
        return self

    # 生成快照数据。
    # English: Handle snapshot data operations.
    @property
    def snapshot_data(self) -> dict[str, Any]:
        return self.snapshot.model_dump(mode="json", exclude_unset=True)

    # 生成快照`v3`。
    # English: Handle snapshot v3 operations.
    @property
    def snapshot_v3(self) -> ManagedSnapshotV3 | None:
        return self._snapshot_v3


class GroupAnalysisTarget(StrictModel):
    conversation_id: int = Field(strict=True, gt=0)


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
    # 平台下发的功能阶段任务（风险会商的初稿/终稿/任务）；走 function_stage flow。
    function_task: FunctionTask | None = None
    model: str | None = None
    model_revision: int | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    runtime_injection: None = None
    unified_runtime_injection: None = None
    agent_config: ManagedAgentConfig

    # 校验`v3`执行权限范围。
    # English: Validate v3 execution scope.
    @model_validator(mode="after")
    def _validate_v3_execution_scope(self):
        if self.agent_config.snapshot_v3 is not None:
            try:
                execution_id = uuid.UUID(self.execution_id)
                trace_id = uuid.UUID(self.trace.trace_id)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError("V3 execution identifiers must be UUIDs") from exc
            if execution_id != trace_id:
                raise ValueError("V3 trace and execution identifiers must match")
        return self


# 处理`digest`相关逻辑。
# English: Handle digest operations.
def _digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


#: 模型凭据映射表已移到 flowagent/execution/deps.py（Stage 2 架构拆分）；
#: _request_llm 通过 model_credential_bindings() 调用，行为不变。


# English: Handle truncate history operations.
def _truncate_history(history: list[dict], history_limit: int) -> list[dict]:
    """评审 7f：runtime.history_limit 真正生效。history_limit<=0 表示
    不带历史（风险会商等无状态 flow 的明确选择）；注意 Python 切片
    history[-0:] 等价于整个列表，必须单独处理 0 的情况。
    """
    if history_limit <= 0:
        return []
    return list(history[-history_limit:])


# English: Handle execution deadline seconds operations.
def _execution_deadline_seconds(snapshot: dict) -> float:
    """评审 7f：workflow.timeout_ms（优先）/ adapter.timeout_ms 真正生效，
    作为整次 V1/V2 图执行的截止时间；两者都缺省时退回 Pydantic 模型声明的
    默认值 90 秒。
    """
    workflow_cfg = snapshot.get("workflow") or {}
    adapter_cfg = snapshot.get("adapter") or {}
    timeout_ms = workflow_cfg.get("timeout_ms") or adapter_cfg.get("timeout_ms") or 90_000
    return timeout_ms / 1000


# 处理请求大语言模型相关逻辑。
# English: Handle request LLM operations.
def _request_llm(spec, config: ManagedAgentConfig):
    if config.snapshot_v3 is not None:
        model = config.snapshot_v3.model
        if config.model_api_key is None:
            raise ConfigError("managed_model_config_missing")
        from langchain_openai import ChatOpenAI

        parameters = {**model.parameters.model_dump(exclude_none=True), "max_retries": 0}  # 重试由单轮恢复统一控制。
        return ChatOpenAI(
            model=model.model_name, base_url=model.base_url,
            api_key=config.model_api_key.get_secret_value(), **parameters,
        )
    model = config.snapshot_data.get("model")
    # 只看可达节点：YAML 里存在不可达的 llm/agent 节点不应强制要求模型凭据
    # （评审 7f）。
    reachable = _reachable_nodes(spec)
    needs_model = any(spec.nodes[name].type in {"llm", "agent"} for name in reachable)
    if not needs_model:
        return MockLLM()
    model_key = config.model_api_key.get_secret_value() if config.model_api_key else None
    if model_key is None and isinstance(model, dict):
        for binding in model_credential_bindings():
            if (model.get("credential_id") == binding["credential_id"]
                    and spec.workflow.name == binding["workflow_name"]):
                model_key = os.environ.get(binding["env_var"])
                break
    if not isinstance(model, dict) or not model_key:
        raise ConfigError("managed_model_config_missing")
    from langchain_openai import ChatOpenAI

    parameters = {**(model.get("parameters") or {}), "max_retries": 0}  # 避免 SDK 和运行时叠加重试。
    return ChatOpenAI(
        model=model["name"], base_url=str(model["base_url"]),
        api_key=model_key, **parameters,
    )


# 处理`managed``system`提示词相关逻辑。
# English: Handle managed system prompt operations.
def _managed_system_prompt(
    snapshot: dict[str, Any], memory_context: MemoryContextV1 | None = None,
) -> str:
    sections = [str(snapshot.get("system_prompt") or "").strip()]
    if snapshot.get("schema_version") == 2:
        catalog = []
        for skill in snapshot.get("skills") or []:
            if isinstance(skill, dict):
                catalog.append(
                    f"- {skill.get('key')}: {skill.get('name')}. "
                    f"{skill.get('summary')} Load the full Skill with platform_load_skill when needed."
                )
        if catalog:
            sections.append("Available Skills (summaries only):\n" + "\n".join(catalog))
        if memory_context is not None and memory_context.status == "ok" and memory_context.rendered_text:
            sections.append("[Platform memory context]\n" + memory_context.rendered_text)
    else:
        for skill in snapshot.get("skills") or []:
            content = skill.get("content") if isinstance(skill, dict) else None
            if content:
                sections.append(str(content).strip())
    return "\n\n".join(section for section in sections if section)


# 处理包含`analysis`目标相关逻辑。
# English: Handle with analysis target operations.
def _with_analysis_target(system_prompt: str, target: GroupAnalysisTarget | None) -> str:
    if target is None:
        return system_prompt
    context = (
        "Authorized group analysis target:\n"
        f"- conversation_id: {target.conversation_id}\n"
        "When the user asks about this group, call the available group analysis input tool "
        "with this exact conversation_id. Do not substitute or infer another conversation ID."
    )
    return f"{system_prompt}\n\n{context}" if system_prompt else context


# 处理MCP客户端针对执行相关逻辑。
# English: Handle MCP client for execution operations.
def _mcp_client_for_execution(config: ManagedAgentConfig, legacy_workload_token: str):
    if not config.mcp_endpoint:
        return None
    if config.snapshot.schema_version == 2:
        if config.tool_access_token is None:
            raise ConfigError("execution_tool_token_missing")
        token = config.tool_access_token.get_secret_value()
    else:
        token = legacy_workload_token
    if not token:
        raise ConfigError("managed_mcp_configuration_incomplete")
    return McpEdgeClient(str(config.mcp_endpoint), token)


# 执行`managed`。
# English: Execute managed.
async def execute_managed(
    request: ExecutionRequest,
    mcp_workload_token: str = "",
    checkpointer=None,
    *, model_authorizer: ModelAuthorizer | None = None,
    skill_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    config = request.agent_config
    flow = select_flow(config, function_task=request.function_task)
    if flow == "function_stage":
        return await _execute_function_stage(
            request, model_authorizer=model_authorizer, skill_loader=skill_loader,
        )
    if flow == "qa":
        return await _execute_managed_v3(
            request, model_authorizer=model_authorizer, skill_loader=skill_loader,
        )
    snapshot = config.snapshot_data
    if _digest(snapshot) != config.config_digest:
        raise ValueError("config_digest_mismatch")
    if snapshot["schema_version"] not in {1, 2}:
        raise ConfigError("snapshot_version_not_supported")
    agent = snapshot.get("agent") or {}
    if str(agent.get("id")) != str(config.agent_id) or agent.get("type") != "workflow":
        raise ValueError("config_identity_mismatch")
    workflow = snapshot.get("workflow") or {}
    yaml_text = workflow.get("yaml")
    if not isinstance(yaml_text, str) or not yaml_text:
        raise ConfigError("managed_yaml_missing")
    spec = load_managed_spec(yaml_text)
    reachable = _reachable_nodes(spec)
    if snapshot["schema_version"] == 2 and snapshot.get("skills") and not any(
        spec.nodes[name].type == "agent" for name in reachable
    ):
        raise ConfigError("skill_loader_requires_agent_node")
    llm = _request_llm(spec, config)
    if select_flow(config, spec) == "group_analysis":
        reply, result_meta = await analyze_group(
            request, spec, llm, _mcp_client_for_execution(config, mcp_workload_token),
        )
        return {"execution_id": request.execution_id, "reply": reply,
                "used_context_refs": [],
                "metadata": {"agent_config_applied": _receipt(config), **result_meta}}
    agent_runtime = None
    if any(spec.nodes[name].type == "agent" for name in reachable):
        has_endpoint = bool(config.mcp_endpoint)
        has_token = bool(config.tool_access_token if snapshot["schema_version"] == 2 else mcp_workload_token)
        if has_endpoint != has_token:
            raise ConfigError("managed_mcp_configuration_incomplete")
        mcp_client = _mcp_client_for_execution(config, mcp_workload_token) if has_endpoint else None
        agent_runtime = ManagedAgentRuntime(llm, mcp_client)
    # 每次请求创建独立模型客户端，不进入缓存；需要模型时禁止 auto->Mock 回退。
    graph = build_graph(
        spec, llm=llm,
        checkpointer=checkpointer if checkpointer is not None else False,
        agent_runtime=agent_runtime,
    )
    initial = {
        "vars": {
            "message": request.message,
            "history": _truncate_history(
                request.history, (snapshot.get("runtime") or {}).get("history_limit", 20),
            ),
            "analysis_target": request.analysis_target.model_dump() if request.analysis_target else None,
            "system_prompt": _managed_system_prompt(
                snapshot, request.memory_context if snapshot["schema_version"] == 2 else None,
            ),
            "skills": snapshot.get("skills", []),
        },
        "trace": [], "loop_counts": {}, "error": None,
    }
    graph_config = (
        {"configurable": {"thread_id": request.execution_id}}
        if checkpointer is not None else None
    )
    final = await asyncio.wait_for(
        graph.ainvoke(initial, config=graph_config), _execution_deadline_seconds(snapshot),
    )
    reply = str((final.get("vars") or {}).get("reply", ""))
    return {
        "execution_id": request.execution_id,
        "reply": reply,
        "used_context_refs": [],
        "metadata": {"agent_config_applied": _receipt(
            config, getattr(agent_runtime, "loaded_skills", []),
        )},
    }


# 同步、流式与功能阶段使用一致的 V3 快照和授权构造。
def _v3_execution_parts(request: ExecutionRequest, model_authorizer, skill_loader):
    return execution_parts(request, model_authorizer, skill_loader, _request_llm)


# 执行功能阶段（function_stage）：结构化输出 + 知识依据统计。
# English: Execute a function stage.
async def _execute_function_stage(
    request: ExecutionRequest, *, model_authorizer: ModelAuthorizer | None,
    skill_loader: Callable[..., Any] | None,
) -> dict[str, Any]:
    config = request.agent_config
    if config.snapshot_v3 is None:
        # 功能阶段只支持 V3 快照。
        raise ConfigError("snapshot_version_not_supported")
    snapshot, llm, mcp_client, resolved_skill_loader = _v3_execution_parts(
        request, model_authorizer, skill_loader)
    reply, function_meta, runtime = await run_function_stage(
        request, llm=llm, mcp_client=mcp_client,
        system_prompt=_with_analysis_target(snapshot.system_prompt, request.analysis_target),
        skills=[skill.model_dump(mode="json") for skill in snapshot.skills],
        skill_loader=resolved_skill_loader, execution_id=request.execution_id,
        generation=request.binding_generation,
    )
    return {
        "execution_id": request.execution_id, "reply": reply, "used_context_refs": [],
        "metadata": {"agent_config_applied": _receipt(
            config, execution_id=request.execution_id, usage=runtime.usage,
            memory_context_status="unavailable",
        ), **function_meta},
    }


# 执行`managed``v3`。
# English: Execute managed v3.
async def _execute_managed_v3(
    request: ExecutionRequest, *, model_authorizer: ModelAuthorizer | None,
    skill_loader: Callable[..., Any] | None,
) -> dict[str, Any]:
    config = request.agent_config
    if config.snapshot_v3 is None:  # Internal guard for direct callers.
        raise ValueError("managed_snapshot_v3_required")
    snapshot, llm, mcp_client, resolved_skill_loader = _v3_execution_parts(
        request, model_authorizer, skill_loader)
    runtime = chat_runtime(request, snapshot, llm, mcp_client, resolved_skill_loader)
    system_prompt = _with_analysis_target(snapshot.system_prompt, request.analysis_target)
    if request.memory_context is not None:
        if not snapshot.memory.read:
            raise ValueError("memory_context_not_enabled_in_snapshot")
        rendered_memory = request.memory_context.rendered_text.strip()
        if rendered_memory:
            system_prompt = (
                f"{system_prompt}\n\nRelevant user memory context:\n{rendered_memory}"
                if system_prompt else f"Relevant user memory context:\n{rendered_memory}"
            )
    reply = await runtime.run(
        request.message, system_prompt, max_tool_iterations=8,
        history=request.history,
    )
    return {
        "execution_id": request.execution_id,
        "reply": reply,
        "used_context_refs": (
            list(request.memory_context.included_ids) if request.memory_context is not None else []
        ),
        "metadata": {"agent_config_applied": _receipt(
            config, execution_id=request.execution_id, usage=runtime.usage,
            memory_context_status=(request.memory_context.status if request.memory_context else "unavailable"),
        ), **runtime.terminal_metadata},
    }


# 处理回执相关逻辑。
# English: Handle receipt operations.
def _receipt(config: ManagedAgentConfig, loaded_skills: list[dict] | None = None, *,
             execution_id: str | None = None, usage=None,
             memory_context_status: str = "unavailable") -> dict[str, Any]:
    if config.snapshot_v3 is not None:
        snapshot = config.snapshot_v3
        return {
            "execution_id": execution_id,
            "runtime_id": str(snapshot.runtime_id),
            "schema_version": 3,
            "revision_id": str(snapshot.revision_id),
            "config_digest": snapshot.config_digest,
            "model_version_id": str(snapshot.model.version_id),
            "memory_profile_version_id": str(snapshot.memory.profile_version_id),
            "memory_context_status": memory_context_status,
            "loaded_skill_versions": list(getattr(usage, "loaded_skills", []) or []),
            "skill_load_failures": list(getattr(usage, "skill_load_failures", []) or []),
            "called_tool_versions": list(getattr(usage, "called_tools", []) or []),
            "applied_fields": ["system_prompt", "model", "tools", "skills", "memory"],
        }
    snapshot = config.snapshot_data
    details = {
        "snapshot_version": snapshot["schema_version"],
        "skill_catalog": [
            {"key": skill.get("key"), "metadata_version_id": skill.get("metadata_version_id")}
            for skill in snapshot.get("skills", []) if isinstance(skill, dict)
        ] if snapshot["schema_version"] == 2 else [],
        "skill_bodies_loaded": list(loaded_skills or []),
    }
    return {
        "revision_id": str(config.revision_id),
        "config_digest": config.config_digest,
        "applied_fields": ["workflow.yaml", "system_prompt", "skills"],
        "details": details,
    }


# 处理`reachable`节点相关逻辑。
# English: Handle reachable nodes operations.
def _reachable_nodes(spec) -> set[str]:
    reachable: set[str] = set()
    pending = [spec.workflow.start]
    while pending:
        name = pending.pop()
        if name == "END" or name in reachable or name not in spec.nodes:
            continue
        reachable.add(name)
        pending.extend(spec.nodes[name].next_targets)
    return reachable


# English: Handle invalid request response operations.
def _invalid_request_response() -> JSONResponse:
    """请求模型校验失败不回显 Pydantic 的 input/error 文本。"""
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_request", "message": "invalid request"}},
    )


# English: Handle stream managed operations.
async def stream_managed(
    request: ExecutionRequest,
    mcp_workload_token: str = "",
    checkpointer=None,
    *, model_authorizer: ModelAuthorizer | None = None,
    skill_loader: Callable[..., Any] | None = None,
):
    """执行受管图，并把终态 reply 节点的真实模型增量转为事件。"""
    seq = 0
    if request.function_task is not None:
        # 功能阶段需要完整的结构化结果，不支持流式。
        raise ConfigError("function_task_requires_respond")
    yield {"seq": seq, "kind": "meta", "trace_id": request.execution_id, "framework": "workflow"}
    seq += 1
    config = request.agent_config
    if select_flow(config) == "qa":
        snapshot, llm, mcp_client, resolved_skill_loader = _v3_execution_parts(
            request, model_authorizer, skill_loader)
        runtime = chat_runtime(request, snapshot, llm, mcp_client, resolved_skill_loader)
        reply = ""
        system_prompt = _with_analysis_target(snapshot.system_prompt, request.analysis_target)
        if request.memory_context is not None:
            if not snapshot.memory.read:
                raise ValueError("memory_context_not_enabled_in_snapshot")
            rendered_memory = request.memory_context.rendered_text.strip()
            if rendered_memory:
                system_prompt = (
                    f"{system_prompt}\n\nRelevant user memory context:\n{rendered_memory}"
                    if system_prompt else f"Relevant user memory context:\n{rendered_memory}"
                )
        async for content in runtime.stream(
            request.message, system_prompt, max_tool_iterations=8,
            history=request.history,
        ):
            reply += content
            yield {"seq": seq, "kind": "delta", "trace_id": request.execution_id, "content": content}
            seq += 1
        yield {
            "seq": seq, "kind": "done", "trace_id": request.execution_id,
            "content": reply, "framework": "workflow",
            "metadata": {"agent_config_applied": _receipt(
                config, execution_id=request.execution_id, usage=runtime.usage,
                memory_context_status=(request.memory_context.status if request.memory_context else "unavailable"),
            ), **runtime.terminal_metadata},
        }
        return
    snapshot = config.snapshot_data
    if _digest(snapshot) != config.config_digest:
        raise ValueError("config_digest_mismatch")
    if snapshot["schema_version"] not in {1, 2}:
        raise ConfigError("snapshot_version_not_supported")
    agent = snapshot.get("agent") or {}
    if str(agent.get("id")) != str(config.agent_id) or agent.get("type") != "workflow":
        raise ValueError("config_identity_mismatch")
    workflow = snapshot.get("workflow") or {}
    yaml_text = workflow.get("yaml")
    if not isinstance(yaml_text, str) or not yaml_text:
        raise ConfigError("managed_yaml_missing")
    spec = load_managed_spec(yaml_text)
    reachable = _reachable_nodes(spec)
    if snapshot["schema_version"] == 2 and snapshot.get("skills") and not any(
        spec.nodes[name].type == "agent" for name in reachable
    ):
        raise ConfigError("skill_loader_requires_agent_node")
    llm = _request_llm(spec, config)
    if select_flow(config, spec) == "group_analysis":
        reply, result_meta = await analyze_group(
            request, spec, llm, _mcp_client_for_execution(config, mcp_workload_token),
        )
        yield {"seq": seq, "kind": "delta", "trace_id": request.execution_id,
               "content": reply}
        seq += 1
        yield {"seq": seq, "kind": "done", "trace_id": request.execution_id,
               "content": reply, "framework": "workflow",
               "metadata": {"agent_config_applied": _receipt(config), **result_meta}}
        return
    agent_runtime = None
    if any(spec.nodes[name].type == "agent" for name in reachable):
        has_endpoint = bool(config.mcp_endpoint)
        has_token = bool(config.tool_access_token if snapshot["schema_version"] == 2 else mcp_workload_token)
        if has_endpoint != has_token:
            raise ConfigError("managed_mcp_configuration_incomplete")
        mcp_client = _mcp_client_for_execution(config, mcp_workload_token) if has_endpoint else None
        agent_runtime = ManagedAgentRuntime(llm, mcp_client)
    graph = build_graph(
        spec, llm=llm,
        checkpointer=checkpointer if checkpointer is not None else False,
        agent_runtime=agent_runtime,
        stream_final_reply=True,
    )
    initial = {
        "vars": {
            "message": request.message,
            "history": _truncate_history(
                request.history, (snapshot.get("runtime") or {}).get("history_limit", 20),
            ),
            "analysis_target": request.analysis_target.model_dump() if request.analysis_target else None,
            "system_prompt": _managed_system_prompt(
                snapshot, request.memory_context if snapshot["schema_version"] == 2 else None,
            ),
            "skills": snapshot.get("skills", []),
        },
        "trace": [], "loop_counts": {}, "error": None,
    }
    final: dict[str, Any] = {}
    graph_config = (
        {"configurable": {"thread_id": request.execution_id}}
        if checkpointer is not None else None
    )
    # 评审 7f：workflow.timeout_ms 覆盖整条流的耗时，而不仅是单次事件的等待；
    # asyncio.timeout() 上下文管理器要求 3.11+，这里用剩余预算逐步 wait_for
    # 兼容 pyproject 声明的 3.10 最低版本。
    deadline_s = _execution_deadline_seconds(snapshot)
    started = time.monotonic()
    stream_iter = graph.astream(
        initial, config=graph_config, stream_mode=["custom", "values"]
    ).__aiter__()
    while True:
        remaining = deadline_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"execution exceeded {deadline_s}s deadline")
        try:
            mode, event = await asyncio.wait_for(stream_iter.__anext__(), remaining)
        except StopAsyncIteration:
            break
        if mode == "custom" and isinstance(event, dict) and event.get("kind") == "delta":
            content = event.get("content")
            if isinstance(content, str) and content:
                yield {"seq": seq, "kind": "delta", "trace_id": request.execution_id, "content": content}
                seq += 1
        elif mode == "values" and isinstance(event, dict):
            final = event
    reply = str((final.get("vars") or {}).get("reply", ""))
    yield {
        "seq": seq, "kind": "done", "trace_id": request.execution_id,
        "content": reply, "framework": "workflow",
        "metadata": {"agent_config_applied": _receipt(
            config, getattr(agent_runtime, "loaded_skills", []),
        )},
    }


# 构建执行`app`。
# English: Build execution app.
def build_execution_app(
    internal_token: str, mcp_workload_token: str = "", *,
    model_authorizer: ModelAuthorizer | None = None,
    skill_loader: Callable[..., Any] | None = None,
) -> FastAPI:
    if not internal_token:
        # D-11：入站鉴权按评审决策放开，仅告警，不再拒绝启动。
        warn_if_disabled("execution-v1", internal_token)
    # D-2：三条现有 flow（V3 问答、群分析、YAML/风险会商）都不需要 HITL，
    # 默认不再依赖 Postgres；只写不读、从不清理的 checkpoint 因此消失。
    # 显式设置 FLOWAGENT_CHECKPOINT=postgres 仍可按需开启。
    checkpoint_mode = (os.getenv("FLOWAGENT_CHECKPOINT") or "disabled").strip().lower()

    # 处理`lifespan`相关逻辑。
    # English: Handle lifespan operations.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if checkpoint_mode == "postgres":
            async with open_postgres_checkpointer() as saver:
                app.state.checkpointer = saver
                yield
        elif checkpoint_mode in {"memory", "disabled"}:
            # Unit tests and local stateless runs may opt out. The execution image
            # sets FLOWAGENT_CHECKPOINT=postgres and fails startup without a DSN.
            app.state.checkpointer = None
            yield
        else:
            raise ConfigError("FLOWAGENT_CHECKPOINT must be postgres, memory, or disabled")

    setup_logging("workflow_agent")
    app = FastAPI(title="FlowAgent execution-v1", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)

    # 处理能力相关逻辑。
    # English: Handle capabilities operations.
    @app.get("/capabilities")
    async def capabilities(request: Request):
        authorization = request.headers.get("authorization", "")
        if not bearer_ok(authorization, internal_token):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        return {
            "schema_version": 1,
            "framework_id": "workflow",
            "adapter_version": os.environ.get("AGENT_PLATFORM_BUILD_ID") or "unknown",
            "runtime_mode": "workflow_managed",
            "basic": {"respond": True, "stream": True, "config_receipt": True},
            "context": {"history_import": "messages", "memory_context_versions": []},
            "control": {
                "versions": [], "hitl_modes": [], "steering_modes": [],
                "cancel": "unsupported", "reset": "unsupported",
            },
            "tools": {"managed_mcp": bool(mcp_workload_token)},
            "applied_config_fields": ["workflow.yaml", "system_prompt", "skills"],
        }

    # 处理能力`v2`相关逻辑。
    # English: Handle capabilities v2 operations.
    @app.get("/capabilities/v2")
    async def capabilities_v2(request: Request):
        authorization = request.headers.get("authorization", "")
        if not bearer_ok(authorization, internal_token):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        capabilities = FrameworkCapabilitiesV2(
            adapter_version=os.environ.get("AGENT_PLATFORM_BUILD_ID") or "unknown",
            snapshot={"versions": [1, 2, 3]},
            basic={"respond": True, "stream": True, "config_receipt": True},
            context={"history_import": "messages", "memory_context_versions": [1]},
            control={
                "versions": [], "hitl_modes": [], "steering_modes": [],
                "cancel": "unsupported", "reset": "unsupported",
            },
            tools={"managed_mcp": bool(mcp_workload_token), "execution_scoped_mcp": True},
            skills={"catalog_versions": [1], "load_versions": [1], "session_pinning": True},
            applied_config_fields=[
                "workflow.yaml", "system_prompt", "skills", "schema_version",
                "agent_id", "revision_id", "runtime_id", "config_digest",
                "model", "tools", "memory",
            ],
            editable_fields=[
                "adapter.endpoint", "adapter.timeout_ms", "model", "system_prompt",
                "workflow", "mcp_tools", "skills", "memory", "runtime.history_limit",
            ],
        )
        return capabilities.model_dump(mode="json")

    # 处理健康状态相关逻辑。
    # English: Handle health operations.
    @app.get("/health")
    async def health():
        return {"ok": True}

    # 执行响应对应的流程。
    # English: Handle the current operation.
    @app.post("/execution-v1/respond")
    async def respond(request: Request):
        authorization = request.headers.get("authorization", "")
        if not bearer_ok(authorization, internal_token):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        try:
            payload = ExecutionRequest.model_validate(await request.json())
        except ValueError:
            return _invalid_request_response()
        try:
            return await execute_managed(
                payload, mcp_workload_token,
                checkpointer=getattr(request.app.state, "checkpointer", None),
                model_authorizer=model_authorizer, skill_loader=skill_loader,
            )
        except (AuthorizationError, TimeoutError, asyncio.TimeoutError, ValueError, ConfigError) as exc:
            # D-12：respond 与 stream 共用 classify_exception，错误码/HTTP 状态
            # 与改造前逐条一致，只新增 stage/retryable/fallback_eligible/message。
            error = classify_exception(exc)
            return JSONResponse(status_code=error.http_status, content=error_response_body(error))
        except Exception as exc:
            # 现状：respond() 此前没有兜底 except，未分类异常会原样向上抛出；
            # 这里补上与 respond_stream 一致的兜底，避免内部细节泄露。
            error = classify_exception(exc)
            return JSONResponse(status_code=error.http_status, content=error_response_body(error))

    # 响应流式处理。
    # English: Handle stream.
    @app.post("/execution-v1/respond/stream")
    async def respond_stream(request: Request):
        authorization = request.headers.get("authorization", "")
        if not bearer_ok(authorization, internal_token):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        try:
            payload = ExecutionRequest.model_validate(await request.json())
        except ValueError:
            return _invalid_request_response()

        # 处理事件相关逻辑。
        # English: Handle events operations.
        async def events():
            emitted = 0
            try:
                async for event in stream_managed(
                    payload, mcp_workload_token,
                    checkpointer=getattr(request.app.state, "checkpointer", None),
                    model_authorizer=model_authorizer, skill_loader=skill_loader,
                ):
                    emitted = event["seq"] + 1
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 对外只给稳定错误码，不能泄露配置、工具或密钥。
                # D-12：与 respond() 共用 classify_exception，修复此前流式路径
                # 兜底粗糙（一律 execution_failed）导致的错误码不一致（评审 §7.2）。
                error = classify_exception(exc)
                event = {
                    "seq": emitted, "kind": "error", "trace_id": payload.execution_id,
                    "error_code": error.code, "error_message": error.message,
                    "stage": error.stage, "retryable": error.retryable,
                    "fallback_eligible": error.fallback_eligible,
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    # 执行重置对应的流程。
    # English: Reset the current operation.
    @app.post("/execution-v1/reset")
    async def reset(request: Request):
        authorization = request.headers.get("authorization", "")
        if not bearer_ok(authorization, internal_token):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(status_code=422, content={"error": {"code": "invalid_reset"}})
        if not isinstance(body, dict):
            return JSONResponse(status_code=422, content={"error": {"code": "invalid_reset"}})
        continuity_key = body.get("continuity_key")
        generation = body.get("generation")
        if not isinstance(continuity_key, str) or not continuity_key or not isinstance(generation, int):
            return JSONResponse(status_code=422, content={"error": {"code": "invalid_reset"}})
        # 受管 Workflow 首期显式关闭 checkpoint，无服务端会话状态需要清理。
        return {"ok": True, "continuity_key": continuity_key, "generation": generation}

    return app
