import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from flowagent.execution.v3_runtime import chat_local_tools


@pytest.mark.parametrize('model_taxonomy', ['', '按技术类、管理类分析'])
@pytest.mark.parametrize('prefix', ['', '你是本群的统一AI助手。按问题自动选择普通问答、知识检索、群分析、风险查询和独立报告工具。\n用户问题：'])
def test_original_requested_categories_take_precedence_over_model_arguments(model_taxonomy, prefix):
    # 原始请求已经明确列出分类时，模型遗漏或改写参数不能改变分类口径。
    request = SimpleNamespace(
        message=prefix + '分析本群13:00到13:10的消息，按通知类、风险类、待办类、其他类分类，汇总泵站状态。',
        analysis_target=SimpleNamespace(conversation_id=21))
    snapshot = SimpleNamespace(tools=[SimpleNamespace(remote_name='fetch_group_analysis_input')])

    class Mcp:
        tool_bindings = {'fetch_group_analysis_input': {'remote_name': 'fetch_group_analysis_input'}}

        async def call_tool(self, name, arguments):
            return {'conversation_id': 21, 'messages': [], 'status': 'complete',
                    'partial_reasons': [], 'counts': {'fetched': 0},
                    'window': {'start': '2026-10-10T13:00:00+08:00',
                               'end_exclusive': '2026-10-10T13:10:00+08:00'}}

    class Model:
        async def ainvoke(self, messages):
            return AIMessage(content='## 总结\n无消息。')

    tool = chat_local_tools(request, snapshot, Model(), Mcp())[0]
    _, metadata = asyncio.run(tool.handler({'taxonomy': model_taxonomy}))
    names = [item['name'] for item in metadata['analysis_result']['statistics']['categories']]
    assert names == ['通知类', '风险类', '待办类', '其他类', '未归类']
