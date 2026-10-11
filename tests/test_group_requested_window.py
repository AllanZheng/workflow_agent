"""模型漏掉或改错时间参数时，两种群分析入口仍须遵守用户明确的时间范围。"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from flowagent.group_analysis import analyze_group
from flowagent.local_tools import build_group_analysis_tool


def execute(version, message, model_arguments):
    calls = []

    class Model:
        def invoke(self, messages):
            if messages[0].content == "parse":
                return AIMessage(content=json.dumps(model_arguments))
            return AIMessage(content="## 总结\n指定范围内无消息。")

        async def ainvoke(self, messages):
            return self.invoke(messages)

    class Mcp:
        tool_bindings = {"fetch_group_analysis_input": {"remote_name": "fetch_group_analysis_input"}}

        async def call_tool(self, name, arguments):
            calls.append(arguments)
            return {"conversation_id": 21, "messages": [], "status": "complete",
                    "counts": {}, "window": {"start": arguments.get("start", "default-start"),
                    "end_exclusive": arguments.get("end", "default-end")}}

    target = SimpleNamespace(conversation_id=21)
    if version == "v3":
        tool = build_group_analysis_tool(Model(), Mcp(), target, requested_message=message)
        result = asyncio.run(tool.handler(model_arguments))
    else:
        request = SimpleNamespace(message=message, history=[], analysis_target=target)
        node = SimpleNamespace(system="parse", prompt="{message}")
        result = asyncio.run(analyze_group(request, SimpleNamespace(nodes={
            "parse_args": node, "report": SimpleNamespace(system="report")}), Model(), Mcp()))
    return result, calls


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("message,start,end", [
    ("分析本群2026年10月10日北京时间13:00至13:10的消息", "2026-10-10T13:00:00+08:00", "2026-10-10T13:10:00+08:00"),
    ("分析本群2026-10-10 13:00到13:10的消息", "2026-10-10T13:00:00+08:00", "2026-10-10T13:10:00+08:00"),
    ("分析2026-10-10T05:00:00Z至2026-10-10T05:10:00Z的消息", "2026-10-10T05:00:00+00:00", "2026-10-10T05:10:00+00:00"),
    ("分析2026年10月10日23:00到2026年10月11日01:00的消息", "2026-10-10T23:00:00+08:00", "2026-10-11T01:00:00+08:00"),
    ("当前北京时间2026-10-11 16:00。用户问题：分析2026-10-10T13:00:00+08:00至13:10:00+08:00的消息", "2026-10-10T13:00:00+08:00", "2026-10-10T13:10:00+08:00"),
])
@pytest.mark.parametrize("arguments", [{"days": 1}, {"start": "2026-10-11T00:00:00Z", "end": "now", "days": 7}])
def test_requested_window_overrides_missing_or_wrong_model_window(version, message, start, end, arguments):
    (_, metadata), calls = execute(version, message, arguments)
    assert calls == [{"conversation_id": 21, "start": start, "end": end}]
    assert metadata["analysis_result"]["window_start"] == start
    assert metadata["analysis_result"]["window_end_exclusive"] == end


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("message", [
    "分析2026年2月30日13:00至13:10的消息", "分析2026年10月10日14:00至13:00的消息",
    "分析2026年10月10日13:00至2026年12月10日13:10的消息",
])
def test_invalid_explicit_window_requests_clarification_without_fetch(version, message):
    (reply, metadata), calls = execute(version, message, {"days": 1})
    assert "请确认" in reply and "时间" in reply
    assert metadata["analysis_result"] is None and calls == []


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_relative_window_and_unspecified_window_keep_supported_behavior(version):
    _, calls = execute(version, "分析本群最近3天的消息", {"days": 1})
    assert calls == [{"conversation_id": 21, "days": 3}]
    _, calls = execute(version, "分析本群", {"days": 7})
    assert calls == [{"conversation_id": 21, "days": 7}]
