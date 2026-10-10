"""V3 对话运行时的本地工具：确定性群分析终端工具。

模型调用 analyze_group_messages 后，运行时直接把确定性流水线生成的报告作为最终回复，
计数、比例与覆盖率由程序计算，不再交回模型改写。
"""

from __future__ import annotations

from typing import Any

from .agent_runtime import LocalTool
from .group_analysis import run_group_analysis
from .mcp_edge_client import McpEdgeError

GROUP_ANALYSIS_TOOL_NAME = "analyze_group_messages"
_FETCH_REMOTE = "fetch_group_analysis_input"
_ALLOWED_ARGS = frozenset({"group_query", "conversation_id", "start", "end", "days", "taxonomy"})
_REPORT_SYSTEM = "你是群消息分析助手，仅依据给定的统计与分批纪要，输出客观、简洁的中文群分析报告。"

_PARAMETERS = {
    "type": "object",
    "properties": {
        "group_query": {"type": "string", "maxLength": 100,
                        "description": "目标群名称；已锁定目标群时可省略"},
        "conversation_id": {"type": "integer", "minimum": 1, "description": "明确给出的群 ID"},
        "start": {"type": "string", "description": "起始时间，ISO 8601 带时区；北京时间示例 2026-10-10T13:00:00+08:00"},
        "end": {"type": "string", "description": "结束时间，不含，ISO 8601 带时区；北京时间示例 2026-10-10T13:10:00+08:00"},
        "days": {"type": "integer", "minimum": 1, "maximum": 31, "description": "最近 N 天"},
        "taxonomy": {"type": "string", "description": "用户指定的分类，例如“按技术类、管理类分析”"},
    },
    "additionalProperties": False,
}


# 构造群分析终端工具；调用时走与 V2 相同的确定性流水线。
# English: Build the deterministic group-analysis terminal tool.
def build_group_analysis_tool(llm, mcp_client, target) -> LocalTool:
    # 固定工具在 V3 下使用别名，这里按远端名找到别名再调用。
    # English: Adapt remote tool names to V3 pinned aliases.
    async def call_tool(name: str, arguments: dict[str, Any]):
        if name == _FETCH_REMOTE:
            alias = next((key for key, binding in mcp_client.tool_bindings.items()
                          if binding.get("remote_name") == _FETCH_REMOTE), None)
            if alias is None:
                raise McpEdgeError("MCP tool is not pinned or currently authorized")
            return await mcp_client.call_tool(alias, arguments)
        return await mcp_client.call_tool(name, arguments)

    # 执行一次群分析并返回（回复文本, 元数据）。
    # English: Run one analysis and return (reply, metadata).
    async def handler(arguments: dict[str, Any]) -> tuple[str, dict]:
        unknown = set(arguments) - _ALLOWED_ARGS
        if unknown:
            raise ValueError("invalid_group_analysis_arguments")
        parsed = {
            "group_query": arguments.get("group_query") or "",
            "group_id": arguments.get("conversation_id"),
            "start": arguments.get("start") or "",
            "end": arguments.get("end") or "",
            "days": arguments.get("days"),
            "taxonomy_override": arguments.get("taxonomy") or "",
        }
        return await run_group_analysis(
            parsed, target, llm, call_tool, lambda: _REPORT_SYSTEM, use_async=True)

    return LocalTool(
        name=GROUP_ANALYSIS_TOOL_NAME,
        description=("对指定群的消息做确定性分类统计并生成报告；"
                     "用户要求分析群消息、统计群讨论情况时调用。调用后报告即为最终回复。"),
        parameters=_PARAMETERS, handler=handler, requires_remote=_FETCH_REMOTE,
    )
