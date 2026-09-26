"""Demo MCP Edge 的轻量客户端。

MVP 阶段每次操作建立并关闭独立 MCP 会话，避免在 Workflow 进程内维护
跨请求连接状态；认证使用 Admin 发布配置中下发的 workload bearer。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol
import uuid


class McpEdgeError(RuntimeError):
    """MCP Edge 调用失败，对外信息刻意不包含密钥和远端响应正文。"""


class _Session(Protocol):
    async def list_tools(self) -> Any: ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


SessionFactory = Callable[
    [str, Mapping[str, str], float],
    Any,
]


@asynccontextmanager
async def _sdk_session(
    endpoint: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> AsyncIterator[_Session]:
    """延迟导入 MCP SDK，让未启用执行服务的基础安装仍可加载包。"""
    try:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover - 由部署依赖检查覆盖
        raise McpEdgeError("MCP client dependency is not installed") from exc

    async with httpx.AsyncClient(headers=dict(headers), timeout=timeout_seconds) as http_client:
        async with streamable_http_client(endpoint, http_client=http_client) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


class McpEdgeClient:
    """通过 Demo 暴露的 Streamable HTTP MCP 端点发现和调用工具。"""

    def __init__(
        self,
        endpoint: str,
        workload_token: str,
        *,
        timeout_seconds: float = 15,
        session_factory: SessionFactory | None = None,
        fixed_tools: list[Mapping[str, Any]] | None = None,
        execution_id: str | None = None,
        agent_id: str | None = None,
        revision_id: str | None = None,
        runtime_id: str | None = None,
        generation: int | None = None,
    ) -> None:
        endpoint = endpoint.strip()
        workload_token = workload_token.strip()
        if not endpoint:
            raise ValueError("MCP endpoint is required")
        if not workload_token:
            raise ValueError("MCP workload token is required")
        if timeout_seconds <= 0:
            raise ValueError("MCP timeout must be positive")

        self._endpoint = endpoint
        self._workload_token = workload_token
        self._timeout_seconds = timeout_seconds
        self._session_factory = session_factory or _sdk_session
        self._fixed_tools = None if fixed_tools is None else [dict(tool) for tool in fixed_tools]
        self._execution_id = execution_id
        self._agent_id = agent_id
        self._revision_id = revision_id
        self._runtime_id = runtime_id
        self._generation = generation
        self._tool_bindings: dict[str, dict[str, Any]] = {}
        self._visible_tool_aliases: set[str] = set()
        if self._fixed_tools is not None:
            for tool in self._fixed_tools:
                for reference in ("tool_id", "tool_version_id", "server_version_id"):
                    try:
                        tool[reference] = str(uuid.UUID(str(tool[reference])))
                    except (KeyError, ValueError, TypeError) as exc:
                        raise ValueError(f"invalid pinned MCP {reference}") from exc
                if not isinstance(tool.get("remote_name"), str) or not tool["remote_name"]:
                    raise ValueError("invalid pinned MCP remote_name")
            tool_ids = [tool["tool_id"] for tool in self._fixed_tools]
            if len(tool_ids) != len(set(tool_ids)):
                raise ValueError("duplicate pinned MCP tool id")
            for value in (execution_id, agent_id, revision_id, runtime_id):
                if value is None:
                    raise ValueError("execution-scoped MCP requires UUID identity context")
                try:
                    uuid.UUID(str(value))
                except (ValueError, TypeError) as exc:
                    raise ValueError("execution-scoped MCP identity must be UUID") from exc
            if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
                raise ValueError("execution-scoped MCP requires a non-negative generation")
            self._tool_bindings = {
                self._tool_alias(tool["tool_id"]): tool for tool in self._fixed_tools
            }

    def __repr__(self) -> str:
        return f"McpEdgeClient(endpoint={self._endpoint!r}, timeout_seconds={self._timeout_seconds!r})"

    @property
    def tool_bindings(self) -> dict[str, dict[str, Any]]:
        return dict(self._tool_bindings)

    @staticmethod
    def _tool_alias(tool_id: str) -> str:
        return "tool_" + uuid.UUID(str(tool_id)).hex

    def _session(self, extra_headers: Mapping[str, str] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._workload_token}"}
        if self._execution_id is not None:
            headers.update({
                "X-Execution-ID": self._execution_id,
                "X-Agent-ID": str(self._agent_id),
                "X-Agent-Revision-ID": str(self._revision_id),
                "X-Runtime-ID": str(self._runtime_id),
                "X-Binding-Generation": str(self._generation),
            })
        if extra_headers:
            headers.update(extra_headers)
        return self._session_factory(self._endpoint, headers, self._timeout_seconds)

    async def list_tools(self) -> list[dict[str, Any]]:
        try:
            async with self._session() as session:
                result = await session.list_tools()
        except McpEdgeError:
            raise
        except Exception:
            raise McpEdgeError("MCP tool discovery failed") from None

        if self._fixed_tools is not None:
            visible_names = {tool.name for tool in result.tools}
            tools = []
            self._visible_tool_aliases = set()
            for pinned in self._fixed_tools:
                alias = self._tool_alias(pinned["tool_id"])
                if alias not in visible_names:
                    continue
                self._visible_tool_aliases.add(alias)
                tools.append({
                    "name": alias,
                    "description": pinned.get("description") or "",
                    "input_schema": pinned.get("input_schema") or {"type": "object"},
                    "tool_id": pinned["tool_id"],
                    "tool_version_id": pinned["tool_version_id"],
                    "server_version_id": pinned["server_version_id"],
                    "remote_name": pinned["remote_name"],
                })
            return tools

        tools = []
        for tool in result.tools:
            schema = getattr(tool, "input_schema", None) or {"type": "object"}
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump(mode="json")
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": schema,
            })
        return tools

    async def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        remote_name = name
        call_headers = None
        if self._fixed_tools is not None:
            pinned = self._tool_bindings.get(name)
            if pinned is None or name not in self._visible_tool_aliases:
                raise McpEdgeError("MCP tool is not pinned or currently authorized")
            remote_name = str(pinned["remote_name"])
            call_headers = {
                "X-Agent-Tool-ID": str(pinned["tool_id"]),
                "X-Agent-Tool-Version-ID": str(pinned["tool_version_id"]),
            }
        try:
            async with self._session(call_headers) as session:
                # The edge receives the stable alias advertised in list_tools and
                # resolves the pinned remote name after checking the UUID headers.
                result = await session.call_tool(name, dict(arguments or {}))
        except McpEdgeError:
            raise
        except Exception:
            raise McpEdgeError(f"MCP tool call failed: {remote_name}") from None

        if result.is_error:
            raise McpEdgeError(f"MCP tool returned an error: {remote_name}")
        if result.structured_content is not None:
            return result.structured_content
        return self._normalize_content(result.content)

    @staticmethod
    def _normalize_content(content: list[Any]) -> Any:
        normalized: list[Any] = []
        for block in content:
            text = getattr(block, "text", None)
            if text is not None:
                normalized.append(text)
            elif hasattr(block, "model_dump"):
                normalized.append(block.model_dump(mode="json"))
            else:
                normalized.append(str(block))

        if all(isinstance(item, str) for item in normalized):
            return "\n".join(normalized)
        return normalized
