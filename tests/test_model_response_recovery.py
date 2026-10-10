"""验证模型恢复不会把空内容当成功，也不会重复执行工具。"""
import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from flowagent.agent_runtime import ManagedAgentRuntime
from flowagent.execution.errors import classify_exception


class Model:
    def __init__(self, attempts):
        self.attempts = iter(attempts)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        result = next(self.attempts)
        if isinstance(result, Exception):
            raise result
        return result

    async def astream(self, messages):
        result = await self.ainvoke(messages)
        for item in result:
            if isinstance(item, Exception):
                raise item
            yield item


class Tools:
    def __init__(self):
        self.calls = 0

    async def list_tools(self):
        return [{"name": "report"}]

    async def call_tool(self, name, arguments):
        self.calls += 1
        return "已生成报告，文件 ID 32"


def execute(model, streaming, tools=None):
    # 两条入口共用的同步和流式协议都必须满足同一验收。
    async def run():
        runtime = ManagedAgentRuntime(model, tools)
        if streaming:
            return "".join([text async for text in runtime.stream("请回答", "", 2)])
        return await runtime.run("请回答", "", 2)
    return asyncio.run(run())


def response(text, streaming, **kwargs):
    return [AIMessageChunk(content=text, **kwargs)] if streaming else AIMessage(content=text, **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("empty", ["", " \n\t"])
def test_empty_response_retries_once_and_returns_real_answer(streaming, empty):
    model = Model([response(empty, streaming), response("真实答案", streaming)])
    assert execute(model, streaming) == "真实答案"
    assert model.calls == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_repeated_empty_response_fails_with_stable_code(streaming):
    model = Model([response("", streaming), response(" ", streaming)])
    with pytest.raises(Exception) as caught:
        execute(model, streaming)
    assert classify_exception(caught.value).code == "MODEL_EMPTY_RESPONSE"
    assert model.calls == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_retry_after_tool_does_not_execute_tool_twice(streaming):
    call = {"id": "report-1", "name": "report", "args": {}}
    model = Model([response("", streaming, tool_calls=[call]), response("", streaming),
                   response("报告已生成", streaming)])
    tools = Tools()
    assert execute(model, streaming, tools) == "报告已生成"
    assert tools.calls == 1
    assert model.calls == 3


def test_stream_without_chunks_retries():
    model = Model([[], response("恢复正常", True)])
    assert execute(model, True) == "恢复正常"
    assert model.calls == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_timeout_before_output_retries(streaming):
    model = Model([TimeoutError("private diagnostics"), response("恢复正常", streaming)])
    assert execute(model, streaming) == "恢复正常"
    assert model.calls == 2


def test_partial_stream_is_not_replayed():
    model = Model([[AIMessageChunk(content="已发送"), TimeoutError("private")]])
    with pytest.raises(Exception) as caught:
        execute(model, True)
    assert classify_exception(caught.value).code == "MODEL_TIMEOUT"
    assert model.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_nontransient_errors_are_not_retried(streaming):
    model = Model([ValueError("invalid_request")])
    with pytest.raises(ValueError):
        execute(model, streaming)
    assert model.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_text_blocks_are_returned_as_text(streaming):
    model = Model([response([{"type": "text", "text": "实际内容"}], streaming)])
    assert execute(model, streaming) == "实际内容"


@pytest.mark.parametrize("streaming", [False, True])
def test_unknown_tool_retries_before_any_tool_is_executed(streaming):
    # 整轮先校验，避免合法调用已执行后才发现同批次包含未知工具。
    calls = [{"id": "valid-1", "name": "report", "args": {}},
             {"id": "invalid-1", "name": "invented_tool", "args": {}}]
    model = Model([response("", streaming, tool_calls=calls),
                   response("恢复后的真实回答", streaming)])
    tools = Tools()
    assert execute(model, streaming, tools) == "恢复后的真实回答"
    assert model.calls == 2
    assert tools.calls == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_repeated_unknown_tool_fails_without_execution(streaming):
    # 连续两次无效即明确失败，不能把未知工具当作授权调用。
    call = {"id": "invalid-1", "name": "invented_tool", "args": {}}
    model = Model([response("", streaming, tool_calls=[call]) for _ in range(2)])
    tools = Tools()
    with pytest.raises(Exception) as caught:
        execute(model, streaming, tools)
    assert classify_exception(caught.value).code == "MODEL_INVALID_RESPONSE"
    assert model.calls == 2
    assert tools.calls == 0
