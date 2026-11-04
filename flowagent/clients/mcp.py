"""MCP 客户端：streamable HTTP 会话、工具目录与调用、错误四态映射。

约束（技术方案 §3.2/§6.2）：
- 原生工具名 `server__tool`；规范化碰撞报错。
- allowed_tools 为白名单；惰性发现（非必需服务离线不阻止进程启动）。
- 会话生命周期归属单次执行：进入/退出由同一所有者任务负责。
- 错误映射：isError → error；发送后超时（写结果不明）→ unknown；传输失败 → error。
"""
from __future__ import annotations

import asyncio
from typing import Any

from flowagent.core.contracts import ErrorCode, FlowAgentCoreError, ToolCallSpec, ToolResultData, Tools


class McpServerConfig:
    """profile 中的 MCP 服务配置（§6.2）：名称、transport、URL 环境变量、白名单。"""

    def __init__(
        self, *, name: str, transport: str = "streamable_http",
        url: str | None = None, url_env: str | None = None,
        headers: dict[str, str] | None = None, headers_env: str | None = None,
        allowed_tools: list[str] | None = None, timeout_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.transport = transport
        self.url = url
        self.url_env = url_env
        self.headers = dict(headers or {})
        self.headers_env = headers_env
        self.allowed_tools = list(allowed_tools or [])
        self.timeout_seconds = timeout_seconds

    def resolve_url(self) -> str:
        import os

        url = self.url or (os.environ.get(self.url_env or "") if self.url_env else None)
        if not url:
            raise FlowAgentCoreError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"mcp server {self.name} url unresolved",
            )
        return url

    def resolve_headers(self) -> dict[str, str]:
        import json as _json
        import os

        headers = dict(self.headers)
        if self.headers_env and os.environ.get(self.headers_env):
            try:
                extra = _json.loads(os.environ[self.headers_env])
            except _json.JSONDecodeError as exc:
                raise FlowAgentCoreError(
                    ErrorCode.CAPABILITY_UNAVAILABLE, f"mcp headers env not json: {self.headers_env}"
                ) from exc
            if isinstance(extra, dict):
                headers.update({str(k): str(v) for k, v in extra.items()})
        return headers


def normalize_tool_name(server: str, tool: str) -> str:
    """`server__tool` 规范名；工具名内再含 `__` 视为规范化碰撞。"""
    if "__" in tool:
        raise FlowAgentCoreError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"tool name collision: {server}__{tool}",
        )
    return f"{server}__{tool}"


class McpSession(Tools):
    """单次执行持有的 MCP 会话：list 白名单过滤 + call 四态映射。"""

    def __init__(self, config: McpServerConfig, *, session_factory: Any | None = None) -> None:
        self._config = config
        self._session: Any | None = None
        self._factory = session_factory  # 测试注入 (url, headers, timeout) -> async ctx
        self._catalog: dict[str, dict[str, Any]] = {}  # 规范名 -> 原始描述

    # ---- 生命周期 ----
    async def __aenter__(self) -> "McpSession":
        await self.open()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def open(self) -> None:
        if self._session is not None:
            return
        url = self._config.resolve_url()
        headers = self._config.resolve_headers()
        try:
            if self._factory is not None:
                self._session_cm = self._factory(url, headers, self._config.timeout_seconds)
            else:
                self._session_cm = _sdk_transport(
                    url, headers, self._config.timeout_seconds)
            entered = await self._session_cm.__aenter__()
            if isinstance(entered, (tuple, list)) and len(entered) == 3:
                # 真实 SDK 形态：(read, write, _) → 包 ClientSession。
                read, write, _ = entered
                from mcp import ClientSession

                self._session = ClientSession(read, write, timeout_seconds=self._config.timeout_seconds)
                await self._session.__aenter__()
            else:
                # 测试替身形态：工厂直接产出带 initialize/list_tools/call_tool 的会话。
                self._session = entered
            initialize = getattr(self._session, "initialize", None)
            if initialize is not None:
                await initialize()
        except FlowAgentCoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - 发现失败：惰性服务不阻止启动，调用时报错
            self._session = None
            raise FlowAgentCoreError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"mcp session open failed: {self._config.name}: {exc}",
            ) from exc

    async def close(self) -> None:
        if self._session is not None:
            try:
                exit_hook = getattr(self._session, "__aexit__", None)
                if exit_hook is not None:
                    await exit_hook(None, None, None)
            except Exception:  # noqa: BLE001 - 关闭不抛出
                pass
            finally:
                self._session = None
        if getattr(self, "_session_cm", None) is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 - 关闭不抛出
                pass
            self._session_cm = None

    # ---- 目录 ----
    async def list(self) -> list[ToolCallSpec]:
        if self._session is None:
            raise FlowAgentCoreError(ErrorCode.CAPABILITY_UNAVAILABLE, "mcp session not open")
        try:
            result = await self._session.list_tools()
        except Exception as exc:  # noqa: BLE001
            raise FlowAgentCoreError(
                ErrorCode.MCP_TOOL_ERROR, f"mcp list_tools failed: {exc}") from exc
        allowed = set(self._config.allowed_tools)
        specs: list[ToolCallSpec] = []
        for tool in result.tools:
            if allowed and tool.name not in allowed:
                continue
            canonical = normalize_tool_name(self._config.name, tool.name)
            self._catalog[canonical] = {"remote_name": tool.name}
            schema = (getattr(tool, "input_schema", None)
                      or getattr(tool, "inputSchema", None) or {})
            specs.append(ToolCallSpec(
                name=canonical,
                description=(getattr(tool, "description", None) or ""),
                input_schema=schema if isinstance(schema, dict) else {},
            ))
        return specs

    # ---- 调用 ----
    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        if self._session is None:
            return ToolResultData.error("mcp session not open", ErrorCode.CAPABILITY_UNAVAILABLE.value)
        info = self._catalog.get(name)
        remote = info["remote_name"] if info else name.rsplit("__", 1)[-1]
        if info is None:
            # 未列出的工具直接拒绝；不隐式放行白名单外能力。
            return ToolResultData.error(
                f"tool not in catalog: {name}", ErrorCode.TOOL_ARGUMENT_INVALID.value)
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(remote, arguments), self._config.timeout_seconds)
        except asyncio.TimeoutError:
            # 发送后超时：结果不明（写可能已生效）→ unknown，不得当作失败重放。
            return ToolResultData.unknown(f"mcp call timeout: {remote}")
        except Exception as exc:  # noqa: BLE001
            return ToolResultData.error(f"mcp call failed: {remote}: {exc}")
        if getattr(result, "is_error", None) or getattr(result, "isError", False):
            code = getattr(result, "error_code", None)
            return ToolResultData.error(
                f"mcp tool error: {remote}", code or ErrorCode.MCP_TOOL_ERROR.value)
        structured = (getattr(result, "structured_content", None)
                      or getattr(result, "structuredContent", None))
        if structured is not None:
            return ToolResultData.success(structured)
        return ToolResultData.success(_join_text(result))


def _join_text(result: Any) -> str:
    """content 文本块拼接；无文本块时返回可读占位（不虚构结构化数据）。"""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts) if parts else "(no content)"


def _sdk_transport(url: str, headers: dict[str, str], timeout_seconds: float) -> Any:
    """真实 SDK transport：mcp 2.x（Client + httpx2）优先，回退 1.x。

    2.x 经 __aenter__ 得到带 list_tools/call_tool 的 Client；
    1.x 得到 (read, write, _) 三元组（调用方包 ClientSession）。
    """
    from contextlib import asynccontextmanager

    try:
        import httpx2
        from mcp.client.client import Client as _MCPClient
        from mcp.client.streamable_http import streamable_http_client as _new_transport

        @asynccontextmanager
        async def _sdk_new():
            async with httpx2.AsyncClient(
                    headers=dict(headers), timeout=timeout_seconds) as http_client:
                transport = _new_transport(url, http_client=http_client)
                async with _MCPClient(
                        transport, read_timeout_seconds=timeout_seconds) as client:
                    yield client

        return _sdk_new()
    except ImportError:
        pass
    from mcp.client.streamable_http import streamablehttp_client as _old_transport

    return _old_transport(url, headers=headers, timeout=timeout_seconds)
