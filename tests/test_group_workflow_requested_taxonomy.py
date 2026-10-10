"""旧工作流参数解析遗漏用户分类时，也必须保留原始请求指定的类别。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from flowagent.group_analysis import analyze_group


@pytest.mark.parametrize("model_taxonomy", ["", "按技术类、管理类分类"])
def test_workflow_preserves_requested_taxonomy(model_taxonomy):
    class Model:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                value = {"taxonomy_override": model_taxonomy}
            elif self.calls == 2:
                value = {"assignments": [{"message_ref": "m10", "category": "风险类"}], "summary": "备用泵异常"}
            else:
                return SimpleNamespace(content="演练风险汇总")
            return SimpleNamespace(content=json.dumps(value, ensure_ascii=False))

    class Mcp:
        async def call_tool(self, name, arguments):
            assert name == "fetch_group_analysis_input"
            return {"conversation_id": 21, "messages": [{"id": 10, "message_ref": "m10", "text": "备用泵异常",
                    "created_at": "2026-10-10T05:05:00Z", "sender_name": "演练员"}],
                    "window": {"start": "2026-10-10T05:00:00Z", "end_exclusive": "2026-10-10T05:10:00Z"}}

    node = SimpleNamespace(system="", prompt="{message}")
    request = SimpleNamespace(message="按通知类、风险类、待办类、其他类分类，引用消息原文。",
                              history=[], analysis_target=SimpleNamespace(conversation_id=21))
    reply, metadata = asyncio.run(analyze_group(request, SimpleNamespace(nodes={"parse_args": node, "report": node}), Model(), Mcp()))
    categories = metadata["analysis_result"]["statistics"]["categories"]
    # 断言最终分类统计，而不是只检查解析函数，覆盖真实旧工作流路径。
    assert [item["name"] for item in categories] == ["通知类", "风险类", "待办类", "其他类", "未归类"]
    assert categories[1]["count"] == 1
    assert "风险类" in reply
