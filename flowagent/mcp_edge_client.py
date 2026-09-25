"""Demo MCP Edge 的轻量客户端。

MVP 阶段每次操作建立并关闭独立 MCP 会话，避免在 Workflow 进程内维护
跨请求连接状态；认证使用 Admin 发布配置中下发的 workload bearer。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol


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
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover - 由部署依赖检查覆盖
        raise McpEdgeError("MCP client dependency is not installed") from exc

    async with httpx2.AsyncClient(headers=dict(headers), timeout=timeout_seconds) as http_client:
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

    def __repr__(self) -> str:
        return f"McpEdgeClient(endpoint={self._endpoint!r}, timeout_seconds={self._timeout_seconds!r})"

    def _session(self) -> Any:
        headers = {"Authorization": f"Bearer {self._workload_token}"}
        return self._session_factory(self._endpoint, headers, self._timeout_seconds)

    async def list_tools(self) -> list[dict[str, Any]]:
        try:
            async with self._session() as session:
                result = await session.list_tools()
        except McpEdgeError:
            raise
        except Exception as exc:
            raise McpEdgeError("MCP tool discovery failed") from exc

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
        try:
            async with self._session() as session:
                result = await session.call_tool(name, dict(arguments or {}))
        except McpEdgeError:
            raise
        except Exception as exc:
            raise McpEdgeError(f"MCP tool call failed: {name}") from exc

        if result.is_error:
            error_code = (result.structured_content or {}).get("error_code") if isinstance(result.structured_content, dict) else None
            suffix = f" ({error_code})" if isinstance(error_code, str) else ""
            raise McpEdgeError(f"MCP tool returned an error: {name}{suffix}")
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
