"""模型分类异常不能被伪装为没有群消息，重试须收到校验反馈。"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from flowagent.group_analysis import run_group_analysis


@pytest.mark.parametrize("recover", [False, True])
def test_classification_retry_feedback_and_honest_failure(recover):
    calls = []

    class Model:
        async def ainvoke(self, messages):
            if "分类器" in messages[0].content:
                calls.append(messages[-1].content)
                ref = "m1" if recover and len(calls) == 2 else "forged"
                return AIMessage(content=json.dumps({"assignments": [
                    {"message_ref": ref, "category": "问题"}], "summary": "备用泵异常"}))
            assert recover, "分类全部失败后不得要求模型凭空生成报告"
            return AIMessage(content="## 总结\n备用泵异常。")

    async def fetch(name, args):
        return {"conversation_id": 21, "messages": [{"message_ref": "m1", "text": "备用泵异常"}],
                "status": "complete", "counts": {"included": 1},
                "window": {"start": "2026-10-10T05:00:00Z", "end_exclusive": "2026-10-10T05:10:00Z"}}

    reply, metadata = asyncio.run(run_group_analysis({}, SimpleNamespace(conversation_id=21),
        Model(), fetch, lambda: "report", use_async=True))
    assert len(calls) == 2 and "校验失败" in calls[1] and "m1" in calls[1]
    if recover:
        assert metadata["analysis_result"]["status"] == "complete"
        assert metadata["analysis_result"]["statistics"]["total_messages"] == 1
    else:
        assert "已读取 1 条" in reply and "失败" in reply and "请重试" in reply
        assert "无消息" not in reply and metadata["analysis_result"]["status"] == "partial"


def test_unrequested_model_taxonomy_uses_defaults():
    calls = []

    async def fetch(name, args):
        calls.append(name)
        return {"conversation_id": 21, "messages": [], "status": "complete", "counts": {},
                "window": {"start": "start", "end_exclusive": "end"}}

    class Model:
        async def ainvoke(self, messages):
            return AIMessage(content="## 总结\n当前范围无消息。")

    _, meta = asyncio.run(run_group_analysis({"taxonomy_override": "泵站状态、责任人、期限"},
        SimpleNamespace(conversation_id=21), Model(), fetch, lambda: "report", use_async=True,
        requested_message="分析本群2026年10月10日13:00至13:10的消息，汇总泵站状态、责任人及复核期限。"))
    assert calls == ["fetch_group_analysis_input"]
    assert meta["analysis_result"]["effective_taxonomy"]["categories"] == ["问题", "通知", "讨论", "闲聊", "其他"]
