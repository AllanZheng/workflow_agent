"""平台能力运行时：执行期 MCP（固定工具别名）、Skill 装载与模型预授权。

移植自旧 mcp_edge_client.py / skill_runtime.py / execution_authorization.py，
适配原生内核能力协议（Tools/Skills/Model）。契约逐条保持（V01–V04 依据）：
- 固定工具模式：别名 tool_<uuidhex>；服务端目录可见才暴露；PLATFORM_TOOLS 透传。
- 调用发送别名 + X-Agent-Tool-ID/X-Agent-Tool-Version-ID 头；isError → error_code。
- Skill：POST /skills/{id}/load，回显五字段 + body_hash(sha256, CRLF 归一)。
- 模型预授权：每次模型请求前 authorize()；失败 fail-closed（AUTHORIZATION_DENIED）。
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from typing import Any, AsyncIterator, Mapping

from flowagent.adapters.platform.protocol import SkillSnapshotV3
from flowagent.core.contracts import (
    ErrorCode,
    FlowAgentCoreError,
    Model,
    ModelEvent,
    ToolCallSpec,
    ToolResultData,
)

logger = logging.getLogger("flowagent.platform.capabilities")

PLATFORM_TOOLS = frozenset({
    "platform_resolve_group", "get_risk_consultation_context",
    "analyze_risk_consultation", "generate_risk_analysis_report",
})


class McpEdgeError(RuntimeError):
    """MCP Edge 调用失败；对外信息不含密钥与远端响应正文。"""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code
        logger.warning("mcp_edge_failure %s", message)


def tool_alias(tool_id: str) -> str:
    return "tool_" + uuid.UUID(str(tool_id)).hex


# ---------------------------------------------------------------- 执行期 MCP


class ExecutionMcpTools:
    """V3 执行期 MCP（固定工具）：原生 Tools 协议 + 版本绑定记录。"""

    def __init__(
        self,
        *,
        endpoint: str,
        workload_token: str,
        fixed_tools: list[dict[str, Any]],
        execution_id: str,
        agent_id: str,
        revision_id: str,
        runtime_id: str,
        generation: int,
        timeout_seconds: float = 15.0,
        session_factory: Any | None = None,
    ) -> None:
        endpoint, workload_token = endpoint.strip(), workload_token.strip()
        if not endpoint or not workload_token:
            raise ValueError("MCP endpoint and workload token are required")
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._session_factory = session_factory or _sdk_session
        self._identity_headers = {
            "Authorization": f"Bearer {workload_token}",
            "X-Execution-ID": execution_id,
            "X-Agent-ID": str(agent_id),
            "X-Agent-Revision-ID": str(revision_id),
            "X-Runtime-ID": str(runtime_id),
            "X-Binding-Generation": str(generation),
        }
        # 固定工具 UUID 规范化 + 唯一性（与旧实现一致）。
        self._pinned: dict[str, dict[str, Any]] = {}
        for raw in fixed_tools:
            tool = dict(raw)
            for key in ("tool_id", "tool_version_id", "server_version_id"):
                tool[key] = str(uuid.UUID(str(tool[key])))
            if not tool.get("remote_name"):
                raise ValueError("invalid pinned MCP remote_name")
            alias = tool_alias(tool["tool_id"])
            if alias in self._pinned:
                raise ValueError("duplicate pinned MCP tool id")
            self._pinned[alias] = tool
        self._visible_aliases: set[str] = set()
        self._visible_platform_tools: set[str] = set()

    @property
    def tool_bindings(self) -> dict[str, dict[str, Any]]:
        return dict(self._pinned)

    def remote_name(self, name: str) -> str | None:
        pinned = self._pinned.get(name)
        return str(pinned["remote_name"]) if pinned else name

    async def _call_session(self, headers: dict[str, str], op: Any) -> Any:
        async with self._session_factory(
            self._endpoint, headers, self._timeout_seconds
        ) as session:
            return await op(session)

    async def list(self) -> list[ToolCallSpec]:
        try:
            result = await self._call_session(
                dict(self._identity_headers), lambda s: s.list_tools())
        except Exception as exc:  # noqa: BLE001 - 发现失败不阻止进程（惰性）
            raise FlowAgentCoreError(
                ErrorCode.CAPABILITY_UNAVAILABLE, f"MCP tool discovery failed: {exc}") from exc
        visible = {tool.name for tool in result.tools}
        self._visible_aliases = {a for a in self._pinned if a in visible}
        self._visible_platform_tools = {t.name for t in result.tools if t.name in PLATFORM_TOOLS}
        specs = [
            ToolCallSpec(
                name=alias,
                description=str(pinned.get("description") or ""),
                input_schema=dict(pinned.get("input_schema") or {"type": "object"}),
            )
            for alias, pinned in self._pinned.items() if alias in self._visible_aliases
        ]
        for tool in result.tools:
            if tool.name in PLATFORM_TOOLS:
                schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
                if hasattr(schema, "model_dump"):
                    schema = schema.model_dump(mode="json")
                specs.append(ToolCallSpec(
                    name=tool.name, description=tool.description or "",
                    input_schema=dict(schema or {"type": "object"})))
        return specs

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        remote_name = name
        call_headers = dict(self._identity_headers)
        if name in PLATFORM_TOOLS and name in self._visible_platform_tools:
            pass  # 平台原生工具由服务端目录授权
        elif name in self._pinned and name in self._visible_aliases:
            pinned = self._pinned[name]
            remote_name = str(pinned["remote_name"])
            call_headers.update({
                "X-Agent-Tool-ID": str(pinned["tool_id"]),
                "X-Agent-Tool-Version-ID": str(pinned["tool_version_id"]),
            })
        else:
            return ToolResultData.error(
                "MCP tool is not pinned or currently authorized",
                ErrorCode.CAPABILITY_UNAVAILABLE.value)
        try:
            result = await self._call_session(
                call_headers, lambda s: s.call_tool(name, dict(arguments or {})))
        except Exception as exc:  # noqa: BLE001
            return ToolResultData.error(f"MCP tool call failed: {remote_name}: {exc}")
        if getattr(result, "is_error", False):
            structured = getattr(result, "structured_content", None)
            code = structured.get("error_code") if isinstance(structured, dict) else None
            return ToolResultData.error(
                f"MCP tool returned an error: {remote_name}"
                + (f" ({code})" if isinstance(code, str) else ""),
                code if isinstance(code, str) else ErrorCode.MCP_TOOL_ERROR.value)
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return ToolResultData.success(structured)
        return ToolResultData.success(_join_text(getattr(result, "content", None)))


def _join_text(content: Any) -> str:
    parts = [getattr(b, "text", "") for b in content or [] if getattr(b, "text", None)]
    return "\n".join(parts) if parts else "(no content)"


async def _sdk_session(endpoint: str, headers: Mapping[str, str], timeout_seconds: float):
    """真实 SDK 会话（惰性导入）；未启用执行服务的基础安装不加载 MCP 依赖。"""
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _cm():
        async with httpx.AsyncClient(headers=dict(headers), timeout=timeout_seconds) as http_client:
            async with streamablehttp_client(endpoint, http_client=http_client) as streams:
                read_stream, write_stream = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

    return _cm()


# ---------------------------------------------------------------- Skill 装载

_TOOL_REF = re.compile(
    r"\{\{tool:([a-z0-9][a-z0-9-]{0,98})/([A-Za-z0-9][A-Za-z0-9_.-]{0,159})\}\}"
)


class SkillLoadError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class PlatformSkills:
    """V3 绑定 Skill：目录来自快照；正文经装载端点按固定版本获取。"""

    def __init__(
        self,
        *,
        skills: list[SkillSnapshotV3],
        loader: Any,  # async (execution_id, generation, skill_dict) -> str | dict
    ) -> None:
        self._skills = list(skills)
        self._loader = loader
        self._loaded: dict[str, str] = {}

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {"skill_id": str(s.skill_id), "name": s.name, "summary": s.summary,
             "metadata_version_id": str(s.metadata_version_id),
             "body_version_id": str(s.body_version_id)}
            for s in self._skills
        ]

    async def load(self, skill_id: str) -> str:
        if skill_id in self._loaded:
            return self._loaded[skill_id]
        match = next((s for s in self._skills if str(s.skill_id) == skill_id), None)
        if match is None:
            raise SkillLoadError("skill_binding_invalid")
        try:
            body = await self._loader(
                execution_id=self._execution_id, generation=self._generation,
                skill={"skill_id": str(match.skill_id),
                       "metadata_version_id": str(match.metadata_version_id),
                       "body_version_id": str(match.body_version_id),
                       "name": match.name, "summary": match.summary},
            )
        except SkillLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SkillLoadError("skill_load_unavailable") from exc
        if isinstance(body, dict):
            self._validate_loaded_body(match, body)
            body_text = str(body.get("body_text") or "")
        else:
            body_text = str(body)
        self._loaded[skill_id] = body_text
        return body_text

    def bind_execution(self, execution_id: str, generation: int) -> None:
        self._execution_id = execution_id
        self._generation = generation

    def _validate_loaded_body(self, skill: SkillSnapshotV3, body: dict[str, Any]) -> None:
        for key in ("skill_id", "metadata_version_id", "body_version_id"):
            if str(body.get(key)) != str(getattr(skill, key)):
                raise SkillLoadError("skill_load_binding_mismatch")
        text = str(body.get("body_text") or "")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if body.get("body_hash") != digest:
            raise SkillLoadError("skill_load_binding_mismatch")


async def http_skill_loader(
    *, model_authorization_url: str, execution_token: str, runtime_id: str,
    transport: Any | None = None, timeout_seconds: float = 3.0,
    execution_id: str = "", generation: int = 0, skill: dict[str, Any],
) -> str | dict[str, Any]:
    """装载端点：URL 契约与旧实现一致（.../executions/{id}/model-authorization 派生）。"""
    import httpx

    base = model_authorization_url.rstrip("/")
    marker = "/model-authorization"
    if not base.endswith(marker):
        raise SkillLoadError("skill_loader_endpoint_invalid")
    load_url = f"{base.rsplit('/', 1)[0]}/skills/{skill['skill_id']}/load"
    payload = {
        "generation": generation,
        "metadata_version_id": skill["metadata_version_id"],
        "body_version_id": skill["body_version_id"],
    }
    client = httpx.AsyncClient(
        timeout=timeout_seconds, transport=transport,
        headers={"Authorization": f"Bearer {execution_token}", "X-Runtime-ID": runtime_id},
    )
    try:
        response = await client.post(load_url, json=payload)
    except httpx.HTTPError as exc:
        raise SkillLoadError("skill_load_unavailable") from exc
    finally:
        await client.aclose()
    if response.status_code < 200 or response.status_code >= 300:
        raise SkillLoadError("skill_load_denied")
    try:
        return response.json()
    except ValueError as exc:
        raise SkillLoadError("skill_load_unavailable") from exc


# ---------------------------------------------------------------- 模型预授权


class AuthorizationError(FlowAgentCoreError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.AUTHORIZATION_DENIED, message)


class ModelAuthorizer:
    """预授权协议：每次实际模型请求（含格式修复）前调用。"""

    async def authorize(
        self, *, execution_id: str, model_version_id: str, generation: int
    ) -> dict[str, Any]:
        return {"credential_generation": 0, "request_id": None}


class DemoModelAuthorizer(ModelAuthorizer):
    """Demo 授权端点客户端：非 2xx / 不可达 / 形状错误均 fail-closed。"""

    def __init__(
        self, url: str, execution_token: str, runtime_id: str, *,
        credential_generation: int | None = None,
        timeout_seconds: float = 3.0, transport: Any | None = None,
    ) -> None:
        self._url = url
        self._token = execution_token
        self._runtime_id = runtime_id
        self._credential_generation = credential_generation
        self._timeout = timeout_seconds
        self._transport = transport

    async def authorize(
        self, *, execution_id: str, model_version_id: str, generation: int
    ) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {
            "execution_id": execution_id, "model_version_id": model_version_id,
            "generation": generation,
        }
        if self._credential_generation is not None:
            payload["credential_generation"] = self._credential_generation
        client = httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport,
            headers={"Authorization": f"Bearer {self._token}", "X-Runtime-ID": self._runtime_id},
        )
        try:
            response = await client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise AuthorizationError(f"model authorization unavailable: {exc}") from exc
        finally:
            await client.aclose()
        if response.status_code == 409:
            raise AuthorizationError("model_credential_rotated")
        if response.status_code < 200 or response.status_code >= 300:
            raise AuthorizationError(f"model authorization denied: {response.status_code}")
        try:
            decision = response.json()
        except ValueError as exc:
            raise AuthorizationError("model authorization invalid body") from exc
        if not isinstance(decision, dict) or "credential_generation" not in decision:
            raise AuthorizationError("model authorization invalid body")
        if (self._credential_generation is not None
                and decision.get("credential_generation") != self._credential_generation):
            raise AuthorizationError("model_credential_rotated")
        return decision


class AuthorizedModel(Model):
    """包装原生模型：每次 stream 前预授权；拒绝即终局（不重试）。"""

    def __init__(self, inner: Model, authorizer: ModelAuthorizer, *,
                 execution_id: str, model_version_id: str, generation: int) -> None:
        self._inner = inner
        self._authorizer = authorizer
        self._execution_id = execution_id
        self._model_version_id = model_version_id
        self._generation = generation

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        limits: Mapping[str, Any],
    ) -> AsyncIterator[ModelEvent]:
        await self._authorizer.authorize(
            execution_id=self._execution_id,
            model_version_id=self._model_version_id,
            generation=self._generation,
        )
        async for event in self._inner.stream(messages, tools, limits):
            yield event
