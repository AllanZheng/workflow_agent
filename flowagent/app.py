"""应用装配：build_runner(profile) / build_app(settings)。

职责（技术方案 §5.1）：能力与工作流组装；禁止业务统计与平台协议解析。
- 凭据、Skill 缓存、证据、usage 按执行隔离：每次执行新建 Model/MCP 会话。
- 平台模式经 platform/ 适配层另行组装（M3），不继承本文件 profile 凭据。
"""
from __future__ import annotations

from typing import Any

from flowagent.clients.mcp import McpServerConfig, McpSession
from flowagent.clients.model import OpenAICompatibleModel
from flowagent.config import (
    build_budget,
    build_mcp_configs,
    resolve_model_spec,
    select_profile,
    workflow_enabled_map,
)
from flowagent.core.context import RunContext
from flowagent.core.runner import Runner, RuntimePolicy
from flowagent.core.contracts import ToolCallSpec, ToolResultData, Tools
from flowagent.workflows.chat import build_chat_workflow


class CompositeTools(Tools):
    """多 MCP 服务器聚合：目录合并（server__tool 天然无碰撞），按前缀路由调用。"""

    def __init__(self, servers: list[McpServerConfig]) -> None:
        self._servers = servers
        self._sessions: list[McpSession] = []
        self._routes: dict[str, McpSession] = {}

    async def __aenter__(self) -> "CompositeTools":
        # 惰性连接失败的服务跳过（非必需服务离线不阻止执行）。
        for config in self._servers:
            session = McpSession(config)
            try:
                await session.open()
                await session.list()
            except Exception:  # noqa: BLE001
                continue
            self._sessions.append(session)
            self._routes[config.name] = session
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        for session in self._sessions:
            await session.close()
        self._sessions.clear()
        self._routes.clear()

    async def list(self) -> list[ToolCallSpec]:
        specs: list[ToolCallSpec] = []
        for session in self._sessions:
            specs.extend(await session.list())
        return specs

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        server = name.split("__", 1)[0] if "__" in name else ""
        session = self._routes.get(server)
        if session is None:
            return ToolResultData.error(
                f"tool server unavailable: {server}", "CAPABILITY_UNAVAILABLE")
        return await session.call(name, arguments)


def build_runner(
    profile: dict, *, fallback_owner: str = "local", extra_workflows: dict | None = None
) -> Runner:
    """从 profile 组装 Runner：模型/预算/策略/工作流注册表。"""
    model_spec = resolve_model_spec(profile)
    mcp_configs = build_mcp_configs(profile)
    policy = RuntimePolicy(
        workflow_enabled=workflow_enabled_map(profile),
        max_active_runs=int(profile.get("max_active_runs", 8)),
    )
    workflows = {"chat": build_chat_workflow()}
    workflows.update(extra_workflows or {})

    def context_factory() -> RunContext:
        # 每次执行独立 Model 实例与 MCP 会话（执行隔离）；tools 于执行内打开。
        model = OpenAICompatibleModel(
            base_url=model_spec["base_url"], api_key=model_spec["api_key"],
            model_name=model_spec["model_name"], temperature=model_spec["temperature"],
            extra_body=model_spec.get("extra_body"),
        )
        ctx = RunContext(model=model, budget=build_budget(profile))
        if mcp_configs:
            # CompositeTools 的开闭由 Runner finally 经 reclaim 钩子处理（见 _attach）。
            _attach_lazy_tools(ctx, mcp_configs)
        return ctx

    return Runner(
        workflows=workflows, context_factory=context_factory,
        fallback_owner=fallback_owner, policy=policy,
    )


def _attach_lazy_tools(ctx: RunContext, mcp_configs: list[McpServerConfig]) -> None:
    """把 MCP 会话挂到 ctx.tools（惰性打开），会话生命周期=单次执行。"""
    composite = CompositeTools(mcp_configs)
    state = {"opened": False}

    async def _open() -> CompositeTools:
        if not state["opened"]:
            await composite.__aenter__()
            state["opened"] = True
        return composite

    class _LazyTools(Tools):
        async def list(self) -> list[ToolCallSpec]:
            return await (await _open()).list()

        async def call(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
            return await (await _open()).call(name, arguments)

    async def _close() -> None:
        if state["opened"]:
            await composite.__aexit__(None, None, None)
            state["opened"] = False

    ctx.tools = _LazyTools()
    ctx.closers.append(_close)


def build_app(settings: dict) -> Any:
    """原生 HTTP 应用（fastapi 在 api extra 中；装配在 adapters/http.py）。"""
    from flowagent.adapters.http import build_http_app

    return build_http_app(settings)
