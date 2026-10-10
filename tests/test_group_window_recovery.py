import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from flowagent.agent_runtime import ManagedAgentRuntime
from flowagent.local_tools import build_group_analysis_tool
from flowagent.mcp_edge_client import McpEdgeError


@pytest.mark.parametrize('streaming', [False, True])
def test_group_window_validation_returns_to_model_before_terminal_report(streaming):
    # 首次时间格式非法时必须反馈错误；修正后才生成终端报告与真实统计。
    calls, histories = [], []

    class Mcp:
        tool_bindings = {'fetch_group_analysis_input': {'remote_name': 'fetch_group_analysis_input'}}

        async def list_tools(self):
            return [{'name': 'fetch_group_analysis_input', 'remote_name': 'fetch_group_analysis_input',
                     'input_schema': {'type': 'object'}}]

        async def call_tool(self, name, arguments):
            calls.append(arguments)
            if len(calls) == 1:
                error = McpEdgeError('MCP tool returned an error: fetch_group_analysis_input (invalid_arguments)')
                error.code = 'invalid_arguments'
                raise error
            return {'conversation_id': 21, 'messages': [], 'status': 'complete',
                    'partial_reasons': [], 'counts': {'fetched': 0},
                    'window': {'start': arguments['start'], 'end_exclusive': arguments['end']}}

    class Model:
        def bind_tools(self, schemas):
            return self

        async def ainvoke(self, messages):
            if '分类器' in messages[0].content:
                return AIMessage(content='{"assignments":[],"summary":"无消息"}')
            if messages[0].content.startswith('你是群消息分析助手'):
                return AIMessage(content='## 总结\n时间窗内没有消息。')
            histories.append(list(messages))
            args = ({'start': '2026年10月10日13:00', 'end': '2026年10月10日13:10'}
                    if len(histories) == 1 else
                    {'start': '2026-10-10T13:00:00+08:00', 'end': '2026-10-10T13:10:00+08:00'})
            return AIMessage(content='', tool_calls=[{'name': 'analyze_group_messages',
                                                     'id': str(len(histories)), 'args': args}])

        async def astream(self, messages):
            yield await self.ainvoke(messages)

    model, mcp = Model(), Mcp()
    runtime = ManagedAgentRuntime(model, mcp, tolerate_tool_errors=True,
                                 local_tools=[build_group_analysis_tool(
                                     model, mcp, SimpleNamespace(conversation_id=21))])

    async def execute():
        if streaming:
            return ''.join([piece async for piece in runtime.stream('分析本群', 'system', 8)])
        return await runtime.run('分析本群', 'system', 8)

    reply = asyncio.run(execute())
    assert '时间窗内没有消息' in reply
    assert len(calls) == 2 and len(histories) == 2
    assert any(isinstance(message, ToolMessage) and 'ISO' in message.content
               for message in histories[1])
    assert runtime.terminal_metadata['analysis_result']['status'] == 'complete'
