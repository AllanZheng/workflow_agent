"""引擎选择与基准共用替身。

- build_engine(name, manifest)：baseline-v3 可用；native 在 M1 内核落地后接入。
- BenchMcp：脚本化 MCP 客户端（list_tools/call_tool/tool_bindings/calls 记录），
  协议与 tests/af_helpers.PinnedMcp 一致，但基准目录不依赖 tests 包。
"""
from __future__ import annotations

import uuid
from typing import Any

from benchmarks._compat import bootstrap_paths

bootstrap_paths()


class BenchMcp:
    """脚本化 MCP：远端名 -> 结果；异常注入；记录实际调用。"""

    def __init__(
        self,
        tools: list[dict[str, Any]],
        results: dict[str, Any] | None = None,
        visible: list[str] | None = None,
    ) -> None:
        # tools: [{name(远端名), description, input_schema}]；别名按 V3 规则生成。
        self.aliases: dict[str, str] = {f"tool_{uuid.uuid4().hex}": t["name"] for t in tools}
        self.tool_defs = {t["name"]: t for t in tools}
        self.results: dict[str, Any] = dict(results or {})
        self.visible = set(self.tool_defs) if visible is None else set(visible)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_bindings = {
            alias: {"remote_name": remote, "tool_id": str(uuid.uuid4()),
                    "tool_version_id": str(uuid.uuid4()), "server_version_id": str(uuid.uuid4())}
            for alias, remote in self.aliases.items()
        }

    def alias_of(self, remote: str) -> str:
        return next(a for a, name in self.aliases.items() if name == remote)

    async def list_tools(self) -> list[dict[str, Any]]:
        out = []
        for alias, remote in self.aliases.items():
            if remote not in self.visible:
                continue
            definition = self.tool_defs[remote]
            out.append({
                "name": alias, "description": definition.get("description", remote),
                "input_schema": definition.get("input_schema", {"type": "object"}),
                "remote_name": remote,
            })
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        remote = self.aliases.get(name, name)
        self.calls.append((remote, dict(arguments or {})))
        result = self.results.get(remote)
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result(arguments or {})
        return result


def build_engine(name: str, manifest: dict[str, Any]) -> Any:
    """引擎工厂：新引擎接入时在此登记，run.py 的 --engine 与 manifest.engine 必须一致。"""
    if name == "baseline-v3":
        from benchmarks.baseline_v3 import BaselineV3Engine
        return BaselineV3Engine(manifest)
    if name == "native":
        raise SystemExit(
            "native engine 尚未实现（M1 内核落地后接入）；当前仅支持 baseline-v3"
        )
    raise SystemExit(f"unknown engine: {name}")


def build_group_terminal_tool(llm: Any, mcp_client: Any, target: Any, requested_message: str = "") -> Any:
    """V3 群分析终端工具（analyze_group_messages），用于入口选择类用例。"""
    from flowagent.local_tools import build_group_analysis_tool
    return build_group_analysis_tool(llm, mcp_client, target, requested_message=requested_message)
