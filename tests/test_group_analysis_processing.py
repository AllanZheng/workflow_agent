"""Tests for the program-owned analysis contract."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from flowagent.group_analysis import _batches, _taxonomy, analyze_group
from flowagent.managed_config import load_managed_spec
from pathlib import Path


YAML = Path(__file__).parents[1] / "configs" / "managed" / "group_analysis.yaml"


def test_custom_taxonomy_keeps_only_requested_categories():
    assert _taxonomy("按消息性质分技术类/管理类") == ("技术类", "管理类", "未归类")


def test_batches_never_split_a_message():
    messages = [{"message_ref": f"m{i}", "text": "消息" * 5} for i in range(3)]
    assert [len(batch) for batch in _batches(messages, max_chars=11)] == [1, 1, 1]


@pytest.mark.anyio
async def test_unclear_taxonomy_asks_for_categories_without_reading_im():
    class Client:
        async def call_tool(self, name, args):
            raise AssertionError("IM must not be called")

    class LLM:
        def invoke(self, messages):
            return AIMessage(content='{"start":"","end":"","days":7,"taxonomy_override":"换个口径"}')

    request = SimpleNamespace(analysis_target=SimpleNamespace(conversation_id=5),
                              message="换个口径", history=[])
    spec = load_managed_spec(YAML.read_text(encoding="utf-8"))
    reply, metadata = await analyze_group(request, spec, LLM(), Client())
    assert "分类名称" in reply
    assert metadata["analysis_result"] is None


@pytest.mark.anyio
async def test_invalid_model_classification_cannot_be_counted():
    class Client:
        async def call_tool(self, name, args):
            assert name == "fetch_group_analysis_input"
            assert args["conversation_id"] == 5
            return {"conversation_id": 5,
                    "window": {"start": "2026-09-01T00:00:00+00:00",
                               "end_exclusive": "2026-09-02T00:00:00+00:00"},
                    "status": "complete", "partial_reasons": [],
                    "counts": {"fetched_unique": 1, "included": 1},
                    "messages": [{"message_ref": "m1", "text": "[09-01 10:00] 阿明: 配置出错"}]}

    class LLM:
        def invoke(self, messages):
            prompt = messages[-1].content
            if "请输出解析结果 JSON" in prompt:
                return AIMessage(content='{"start":"","end":"","days":1,"taxonomy_override":""}')
            if "只输出 JSON" in prompt:
                return AIMessage(content='{"assignments":[{"message_ref":"forged","category":"问题"}]}')
            return AIMessage(content="## 总结\n有一条消息。\n## 汇报要点\n- 需跟进")

    request = SimpleNamespace(analysis_target=SimpleNamespace(conversation_id=5),
                              message="分析最近一天", history=[])
    spec = load_managed_spec(YAML.read_text(encoding="utf-8"))
    reply, metadata = await analyze_group(request, spec, LLM(), Client())
    result = metadata["analysis_result"]
    assert result["statistics"]["total_messages"] == 0
    assert result["coverage"]["classification_failed"] == 1
    assert result["status"] == "partial"
    assert "forged" not in reply


@pytest.mark.anyio
async def test_group_name_is_resolved_before_messages_are_read():
    calls = []

    class Client:
        async def call_tool(self, name, args):
            calls.append((name, args))
            if name == "platform_resolve_group":
                return {"status": "matched", "conversation_id": 5, "name": "验收测试群"}
            assert name == "fetch_group_analysis_input"
            return {"conversation_id": 5,
                    "window": {"start": "2026-09-01T00:00:00+00:00",
                               "end_exclusive": "2026-09-02T00:00:00+00:00"},
                    "status": "complete", "partial_reasons": [],
                    "counts": {"fetched_unique": 0, "included": 0}, "messages": []}

    class LLM:
        def invoke(self, messages):
            if "请输出解析结果 JSON" in messages[-1].content:
                return AIMessage(content='{"group_query":"验收测试群","group_id":null,"start":"","end":"","days":1,"taxonomy_override":""}')
            return AIMessage(content="## 总结\n本窗口无消息。\n## 汇报要点\n- 无")

    request = SimpleNamespace(analysis_target=None, message="分析验收测试群最近一天", history=[])
    spec = load_managed_spec(YAML.read_text(encoding="utf-8"))
    _, metadata = await analyze_group(request, spec, LLM(), Client())
    assert calls[0] == ("platform_resolve_group", {"name_query": "验收测试群"})
    assert calls[1][0] == "fetch_group_analysis_input"
    assert calls[1][1]["conversation_id"] == 5
    assert metadata["analysis_result"]["conversation_id"] == 5


@pytest.mark.anyio
async def test_ambiguous_group_name_does_not_read_messages():
    class Client:
        async def call_tool(self, name, args):
            assert name == "platform_resolve_group"
            return {"status": "ambiguous", "candidates": [
                {"conversation_id": 5, "name": "项目群"},
                {"conversation_id": 6, "name": "项目群"},
            ]}

    class LLM:
        def invoke(self, messages):
            return AIMessage(content='{"group_query":"项目群","group_id":null,"start":"","end":"","days":null,"taxonomy_override":""}')

    request = SimpleNamespace(analysis_target=None, message="分析项目群", history=[])
    spec = load_managed_spec(YAML.read_text(encoding="utf-8"))
    reply, metadata = await analyze_group(request, spec, LLM(), Client())
    assert "ID 5" in reply and "ID 6" in reply
    assert metadata["analysis_result"] is None
