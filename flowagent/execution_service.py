"""面向 Demo 的受管 execution-v1 同步服务。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import os
import uuid
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .engine import build_graph, open_postgres_checkpointer
from .errors import ConfigError
from .llm import MockLLM
from .managed_config import load_managed_spec
from .agent_runtime import ManagedAgentRuntime
from .mcp_edge_client import McpEdgeClient
from .group_analysis import analyze_group


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
    ManagedSnapshotV1 | ManagedSnapshotV2,
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


class FrameworkCapabilitiesV2(StrictModel):
    schema_version: Literal[2] = 2
    framework_id: Literal["workflow"] = "workflow"
    adapter_version: str
    runtime_mode: Literal["workflow_managed"] = "workflow_managed"
    snapshot: dict[str, list[int]]
    basic: dict[str, bool]
    context: dict[str, Any]
    control: dict[str, Any]
    tools: dict[str, bool]
    skills: dict[str, Any]
    applied_config_fields: list[str]
    editable_fields: list[str] = Field(default_factory=list)


class ManagedAgentConfig(StrictModel):
    agent_id: uuid.UUID
    agent_type: Literal["workflow"]
    revision_id: uuid.UUID
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: ManagedSnapshot
    model_api_key: SecretStr | None = Field(default=None, repr=False)
    mcp_endpoint: str | None = None
    tool_access_token: SecretStr | None = Field(default=None, repr=False)

    @property
    def snapshot_data(self) -> dict[str, Any]:
        return self.snapshot.model_dump(mode="json", exclude_unset=True)


class GroupAnalysisTarget(StrictModel):
    conversation_id: int = Field(strict=True, gt=0)


class ExecutionRequest(StrictModel):
    protocol_version: Literal["execution-v1"]
    execution_id: str
    idempotency_key: str | None = None
    deadline_ms: int | None = None
    trace: Trace
    binding_generation: int
    continuity_key: str
    message: str
    analysis_target: "GroupAnalysisTarget | None" = None
    identity: dict[str, Any]
    scene: dict[str, Any]
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    model_revision: int | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    runtime_injection: None = None
    unified_runtime_injection: None = None
    memory_context: MemoryContextV1 | None = None
    agent_config: ManagedAgentConfig


def _digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_llm(spec, config: ManagedAgentConfig):
    model = config.snapshot_data.get("model")
    needs_model = any(node.type in {"llm", "agent"} for node in spec.nodes.values())
    if not needs_model:
        return MockLLM()
    if not isinstance(model, dict) or config.model_api_key is None:
        raise ConfigError("managed_model_config_missing")
    from langchain_openai import ChatOpenAI

    parameters = dict(model.get("parameters") or {})
    return ChatOpenAI(
        model=model["name"], base_url=str(model["base_url"]),
        api_key=config.model_api_key.get_secret_value(), **parameters,
    )


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


async def execute_managed(
    request: ExecutionRequest,
    mcp_workload_token: str = "",
    checkpointer=None,
) -> dict[str, Any]:
    config = request.agent_config
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
    if spec.workflow.group_analysis_mode == "deterministic":
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
            "message": request.message, "history": request.history,
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
    final = await graph.ainvoke(initial, config=graph_config)
    reply = str((final.get("vars") or {}).get("reply", ""))
    return {
        "execution_id": request.execution_id,
        "reply": reply,
        "used_context_refs": [],
        "metadata": {"agent_config_applied": _receipt(
            config, getattr(agent_runtime, "loaded_skills", []),
        )},
    }


def _receipt(config: ManagedAgentConfig, loaded_skills: list[dict] | None = None) -> dict[str, Any]:
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


def _invalid_request_response() -> JSONResponse:
    """请求模型校验失败不回显 Pydantic 的 input/error 文本。"""
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_request", "message": "invalid request"}},
    )


async def stream_managed(
    request: ExecutionRequest,
    mcp_workload_token: str = "",
    checkpointer=None,
):
    """执行受管图，并把终态 reply 节点的真实模型增量转为事件。"""
    seq = 0
    yield {"seq": seq, "kind": "meta", "trace_id": request.execution_id, "framework": "workflow"}
    seq += 1
    config = request.agent_config
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
    if spec.workflow.group_analysis_mode == "deterministic":
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
            "message": request.message, "history": request.history,
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
    async for mode, event in graph.astream(
        initial, config=graph_config, stream_mode=["custom", "values"]
    ):
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


def build_execution_app(internal_token: str, mcp_workload_token: str = "") -> FastAPI:
    if not internal_token:
        raise ValueError("FRAMEWORK_INTERNAL_TOKEN is required")
    checkpoint_mode = (os.getenv("FLOWAGENT_CHECKPOINT") or "postgres").strip().lower()

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

    app = FastAPI(title="FlowAgent execution-v1", lifespan=lifespan)

    @app.get("/capabilities")
    async def capabilities(request: Request):
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {internal_token}"
        if not hmac.compare_digest(authorization, expected):
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

    @app.get("/capabilities/v2")
    async def capabilities_v2(request: Request):
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {internal_token}"
        if not hmac.compare_digest(authorization, expected):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        capabilities = FrameworkCapabilitiesV2(
            adapter_version=os.environ.get("AGENT_PLATFORM_BUILD_ID") or "unknown",
            snapshot={"versions": [1, 2]},
            basic={"respond": True, "stream": True, "config_receipt": True},
            context={"history_import": "messages", "memory_context_versions": [1]},
            control={
                "versions": [], "hitl_modes": [], "steering_modes": [],
                "cancel": "unsupported", "reset": "unsupported",
            },
            tools={"managed_mcp": bool(mcp_workload_token), "execution_scoped_mcp": bool(mcp_workload_token)},
            skills={"catalog_versions": [1], "load_versions": [1], "session_pinning": True},
            applied_config_fields=["workflow.yaml", "system_prompt", "skills"],
            editable_fields=[
                "adapter.endpoint", "adapter.timeout_ms", "model", "system_prompt",
                "workflow", "mcp_tools", "skills", "memory", "runtime.history_limit",
            ],
        )
        return capabilities.model_dump(mode="json")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/execution-v1/respond")
    async def respond(request: Request):
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {internal_token}"
        if not hmac.compare_digest(authorization, expected):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        try:
            payload = ExecutionRequest.model_validate(await request.json())
        except ValueError:
            return _invalid_request_response()
        try:
            return await execute_managed(
                payload, mcp_workload_token,
                checkpointer=getattr(request.app.state, "checkpointer", None),
            )
        except ValueError as exc:
            code = str(exc)
            status = 409 if code in {"config_digest_mismatch", "config_identity_mismatch"} else 422
            return JSONResponse(status_code=status, content={"error": {"code": code}})
        except ConfigError as exc:
            return JSONResponse(status_code=422, content={"error": {"code": "managed_config_invalid", "message": str(exc)}})

    @app.post("/execution-v1/respond/stream")
    async def respond_stream(request: Request):
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {internal_token}"
        if not hmac.compare_digest(authorization, expected):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        try:
            payload = ExecutionRequest.model_validate(await request.json())
        except ValueError:
            return _invalid_request_response()

        async def events():
            emitted = 0
            try:
                async for event in stream_managed(
                    payload, mcp_workload_token,
                    checkpointer=getattr(request.app.state, "checkpointer", None),
                ):
                    emitted = event["seq"] + 1
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                raise
            except Exception:  # 对外只给稳定错误码，不能泄露配置、工具或密钥。
                event = {
                    "seq": emitted, "kind": "error", "trace_id": payload.execution_id,
                    "error_code": "execution_failed", "error_message": "managed execution failed",
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/execution-v1/reset")
    async def reset(request: Request):
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {internal_token}"
        if not hmac.compare_digest(authorization, expected):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
        body = await request.json()
        continuity_key = body.get("continuity_key")
        generation = body.get("generation")
        if not isinstance(continuity_key, str) or not continuity_key or not isinstance(generation, int):
            return JSONResponse(status_code=422, content={"error": {"code": "invalid_reset"}})
        # 受管 Workflow 首期显式关闭 checkpoint，无服务端会话状态需要清理。
        return {"ok": True, "continuity_key": continuity_key, "generation": generation}

    return app
