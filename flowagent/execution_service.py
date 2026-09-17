"""面向 Demo 的受管 execution-v1 同步服务。"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .engine import build_graph
from .errors import ConfigError
from .llm import MockLLM
from .managed_config import load_managed_spec


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Trace(StrictModel):
    trace_id: str


class ManagedAgentConfig(StrictModel):
    agent_id: uuid.UUID
    agent_type: Literal["workflow"]
    revision_id: uuid.UUID
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: dict[str, Any]
    model_api_key: SecretStr | None = Field(default=None, repr=False)
    mcp_endpoint: str | None = None


class ExecutionRequest(StrictModel):
    protocol_version: Literal["execution-v1"]
    execution_id: str
    idempotency_key: str | None = None
    deadline_ms: int | None = None
    trace: Trace
    binding_generation: int
    continuity_key: str
    message: str
    identity: dict[str, Any]
    scene: dict[str, Any]
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    model_revision: int | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    runtime_injection: None = None
    unified_runtime_injection: None = None
    agent_config: ManagedAgentConfig


def _digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_llm(spec, config: ManagedAgentConfig):
    model = config.snapshot.get("model")
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


async def execute_managed(request: ExecutionRequest) -> dict[str, Any]:
    config = request.agent_config
    if _digest(config.snapshot) != config.config_digest:
        raise ValueError("config_digest_mismatch")
    agent = config.snapshot.get("agent") or {}
    if str(agent.get("id")) != str(config.agent_id) or agent.get("type") != "workflow":
        raise ValueError("config_identity_mismatch")
    workflow = config.snapshot.get("workflow") or {}
    yaml_text = workflow.get("yaml")
    if not isinstance(yaml_text, str) or not yaml_text:
        raise ConfigError("managed_yaml_missing")
    spec = load_managed_spec(yaml_text)
    # 每次请求创建独立模型客户端，不进入缓存；需要模型时禁止 auto->Mock 回退。
    graph = build_graph(spec, llm=_request_llm(spec, config), checkpointer=False)
    final = await graph.ainvoke({
        "vars": {
            "message": request.message, "history": request.history,
            "system_prompt": config.snapshot.get("system_prompt", ""),
            "skills": config.snapshot.get("skills", []),
        },
        "trace": [], "loop_counts": {}, "error": None,
    })
    reply = str((final.get("vars") or {}).get("reply", ""))
    return {
        "execution_id": request.execution_id,
        "reply": reply,
        "used_context_refs": [],
        "metadata": {"agent_config_applied": {
            "revision_id": str(config.revision_id),
            "config_digest": config.config_digest,
            "applied_fields": ["workflow.yaml", "system_prompt", "skills"],
        }},
    }


def build_execution_app(internal_token: str) -> FastAPI:
    if not internal_token:
        raise ValueError("FRAMEWORK_INTERNAL_TOKEN is required")
    app = FastAPI(title="FlowAgent execution-v1")

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
            return await execute_managed(payload)
        except ValueError as exc:
            code = str(exc)
            status = 409 if code in {"config_digest_mismatch", "config_identity_mismatch"} else 422
            return JSONResponse(status_code=status, content={"error": {"code": code}})
        except ConfigError as exc:
            return JSONResponse(status_code=422, content={"error": {"code": "managed_config_invalid", "message": str(exc)}})

    return app
