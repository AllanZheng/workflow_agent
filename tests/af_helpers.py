"""智能体功能测试的共用替身：脚本化模型与带固定工具别名的 MCP 客户端。"""
from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage

from flowagent.mcp_edge_client import McpEdgeError


class ScriptedLlm:
    """bind_tools 返回的模型按脚本逐条回复；未绑定的 ainvoke 用于确定性分类与报告。"""

    def __init__(self, replies: list[AIMessage], *, report_text: str | None = None,
                 classify=None):
        self.replies = list(replies)
        self.bound_calls = 0
        self.raw_calls: list[str] = []
        self.report_text = report_text or "## 总结\n本群讨论集中在进展与风险。\n\n## 汇报要点\n- 无"
        self.classify = classify or (lambda index, ref: "问题" if index % 2 == 0 else "通知")
        self.schemas: list[dict] = []

    def bind_tools(self, tools):
        self.schemas = tools
        outer = self

        class Bound:
            async def ainvoke(self, messages):
                outer.bound_calls += 1
                return outer.replies.pop(0)

            # 流式：单个块即完整回复（工具调用轮次 content 为空）。
            async def astream(self, messages):
                outer.bound_calls += 1
                yield outer.replies.pop(0)

        return Bound()

    async def ainvoke(self, messages):
        system, prompt = messages[0].content, str(messages[-1].content)
        self.raw_calls.append(prompt)
        if "分类器" in system:
            refs = re.findall(r'"message_ref": "([^"]+)"', prompt)
            assignments = [{"message_ref": ref, "category": self.classify(i, ref)}
                           for i, ref in enumerate(refs)]
            import json
            return AIMessage(content=json.dumps(
                {"assignments": assignments, "summary": "批次纪要"}, ensure_ascii=False))
        return AIMessage(content=self.report_text)


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])


class PinnedMcp:
    """模拟 V3 固定工具 MCP 客户端：别名 -> 远端名，按远端名脚本化结果。"""

    def __init__(self, remotes: list[str], results: dict[str, Any] | None = None,
                 visible: list[str] | None = None):
        self.aliases = {f"tool_{uuid.uuid4().hex}": remote for remote in remotes}
        self.results = dict(results or {})
        self.visible = set(remotes if visible is None else visible)
        self.calls: list[tuple[str, dict]] = []
        self.tool_bindings = {
            alias: {"remote_name": remote, "tool_id": str(uuid.uuid4()),
                    "tool_version_id": str(uuid.uuid4()), "server_version_id": str(uuid.uuid4())}
            for alias, remote in self.aliases.items()
        }

    def alias_of(self, remote: str) -> str:
        return next(alias for alias, name in self.aliases.items() if name == remote)

    async def list_tools(self):
        return [{"name": alias, "description": remote, "input_schema": {"type": "object"},
                 "remote_name": remote}
                for alias, remote in self.aliases.items() if remote in self.visible]

    async def call_tool(self, name, arguments):
        remote = self.aliases.get(name, name)
        self.calls.append((remote, dict(arguments)))
        result = self.results.get(remote)
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result(arguments)
        return result


def group_input(conversation_id: int = 5, count: int = 4) -> dict[str, Any]:
    messages = [{"message_ref": f"m{i}", "timestamp": "2026-10-01T10:00:00+08:00",
                 "text": f"消息{i}"} for i in range(1, count + 1)]
    return {"conversation_id": conversation_id, "messages": messages, "status": "complete",
            "partial_reasons": [], "counts": {"fetched": count},
            "window": {"start": "2026-09-30T00:00:00+08:00", "end_exclusive": "2026-10-01T00:00:00+08:00"}}


__all__ = ["ScriptedLlm", "PinnedMcp", "tool_call", "group_input", "McpEdgeError"]
