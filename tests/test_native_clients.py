"""原生 V3 内核客户端测试：OpenAI 兼容模型流式聚合 + MCP 会话四态。

覆盖行为（本文件局部编号）：
- MC-01 chunk 聚合：正文增量、工具调用增量拼接、completed 汇总与 usage 提取
- MC-02 空 tools 的请求不下发 tools 字段
- MC-03 parse_tool_call_arguments：dict 透传、空串归空、非法/非对象 JSON 报错
- MC-04 SDK 异常映射稳定错误码；内核错误原样传播不被二次包装
- MP-01 规范工具名 server__tool；工具名内含 __ 视为碰撞
- MP-02 会话生命周期：open 初始化、close 释放（工厂/会话退出均被记录）
- MP-03 目录：allowed_tools 过滤 + 规范名 + inputSchema 透传
- MP-04 调用映射：结构化成功 / 文本块拼接 / isError / 超时 unknown / 传输异常 error
- MP-05 目录守卫：未列出的工具直接拒绝（TOOL_ARGUMENT_INVALID）
- MP-06 KERNEL-BUG：会话工厂契约与真实 mcp.ClientSession 参数不兼容（xfail）

离线约束：模型侧注入假 AsyncOpenAI（无网络）；MCP 侧用假会话工厂 +
本文件内的 _FakeClientSession 顶替 mcp.ClientSession，全部在进程内完成。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from flowagent.clients.mcp import McpServerConfig, McpSession, normalize_tool_name
from flowagent.clients.model import OpenAICompatibleModel, parse_tool_call_arguments
from flowagent.core.contracts import ErrorCode, FlowAgentCoreError

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------ 模型侧假件
class _FakeCompletions:
    """假 chat.completions：记录请求；按脚本产出 chunk 或抛异常。

    chunks 中的 BaseException 元素表示“流中途抛错”，用于验证内核错误原样传播。
    """

    def __init__(self, chunks: list[Any] = (), *, error: BaseException | None = None) -> None:
        self.chunks = list(chunks)
        self.error = error
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        chunks = list(self.chunks)

        async def _iter():
            for item in chunks:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return _iter()


def _model(completions: _FakeCompletions) -> OpenAICompatibleModel:
    """注入假 client：AsyncOpenAI 替身只需暴露 chat.completions.create。"""
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAICompatibleModel(
        base_url="http://fake.local/v1", api_key="test-key",
        model_name="fake-model", client=client,
    )


def _chunk(
    *, content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
    empty_choices: bool = False,
) -> SimpleNamespace:
    """构造一个流式 chunk（形状对齐 OpenAI SDK 的 ChatCompletionChunk）。"""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choices = [] if empty_choices else [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    return SimpleNamespace(choices=choices, usage=usage)


def _tc(index: int, *, id: str | None = None, name: str | None = None,
        arguments: str | None = None) -> SimpleNamespace:
    """工具调用增量片段；arguments 可跨 chunk 分片到达。"""
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


def _usage_chunk(prompt: int, completion: int) -> SimpleNamespace:
    """仅带 usage 的尾 chunk（choices 为空，这与 OpenAI 流式实现一致）。"""
    return _chunk(empty_choices=True, usage=SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion))


# ------------------------------------------------------------------ MC-01/MC-02
async def test_stream_aggregates_text_tool_calls_and_usage() -> None:
    """MC-01：两个正文 chunk + 两段工具参数 + finish_reason + usage 尾 chunk。"""
    completions = _FakeCompletions([
        _chunk(content="你"),
        _chunk(content="好"),
        _chunk(tool_calls=[_tc(0, id="c1", name="echo", arguments='{"a"')]),
        _chunk(tool_calls=[_tc(0, arguments=":1}")]),
        _chunk(finish_reason="tool_calls"),
        _usage_chunk(10, 5),
    ])
    model = _model(completions)

    events = [event async for event in model.stream(
        [{"role": "user", "content": "hi"}],
        [{"type": "function", "function": {"name": "echo"}}],
        {},
    )]

    text_deltas = [e.text for e in events if e.kind == "text_delta"]
    tool_deltas = [e for e in events if e.kind == "tool_delta"]
    completed = [e for e in events if e.kind == "completed"]

    assert text_deltas == ["你", "好"]
    # 工具参数分片在增量事件中应体现累积值（先 '{"a"'，后补齐为 '{"a":1}'）。
    assert [e.tool_calls[0]["arguments"] for e in tool_deltas] == ['{"a"', '{"a":1}']
    assert len(completed) == 1
    final = completed[0]
    assert final.text == "你好"
    assert final.tool_calls == [{"id": "c1", "name": "echo", "arguments": '{"a":1}'}]
    assert final.finish_reason == "tool_calls"
    assert final.usage == {"input_tokens": 10, "output_tokens": 5}


async def test_stream_omits_tools_key_when_empty() -> None:
    """MC-02：无工具时请求体不得携带 tools 字段（部分端点对空数组不兼容）。"""
    completions = _FakeCompletions([_chunk(content="ok", finish_reason="stop")])
    model = _model(completions)

    _ = [event async for event in model.stream([{"role": "user", "content": "x"}], [], {})]

    request = completions.requests[0]
    assert "tools" not in request
    assert request["model"] == "fake-model"
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["temperature"] == 0.0  # 默认温度来自构造函数


# ------------------------------------------------------------------ MC-03
def test_parse_arguments_dict_passthrough() -> None:
    """MC-03：已是 dict 的参数原样返回。"""
    args = {"a": 1, "b": "文本"}
    assert parse_tool_call_arguments({"name": "echo", "arguments": args}) == args


@pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
def test_parse_arguments_empty_string_yields_empty_dict(raw: str) -> None:
    """MC-03：空/纯空白参数字符串视为无参数。"""
    assert parse_tool_call_arguments({"name": "echo", "arguments": raw}) == {}


@pytest.mark.parametrize("raw", ['{"a"', "[1]", '"text"', "123"])
def test_parse_arguments_invalid_or_non_object_json_raises(raw: str) -> None:
    """MC-03：非法 JSON 与非对象 JSON 都归为 MODEL_INVALID_RESPONSE。"""
    with pytest.raises(FlowAgentCoreError) as info:
        parse_tool_call_arguments({"name": "echo", "arguments": raw})
    assert info.value.code is ErrorCode.MODEL_INVALID_RESPONSE


def test_parse_arguments_missing_key_yields_empty_dict() -> None:
    """MC-03：缺 arguments 键按无参数处理（不虚构）。"""
    assert parse_tool_call_arguments({"name": "echo"}) == {}


# ------------------------------------------------------------------ MC-04
def _openai_request():
    import httpx

    return httpx.Request("POST", "http://fake.local/v1/chat/completions")


async def test_sdk_timeout_maps_to_model_timeout() -> None:
    """MC-04：openai.APITimeoutError → MODEL_TIMEOUT。"""
    import openai

    completions = _FakeCompletions(error=openai.APITimeoutError(request=_openai_request()))
    model = _model(completions)

    with pytest.raises(FlowAgentCoreError) as info:
        _ = [event async for event in model.stream([], [], {})]
    assert info.value.code is ErrorCode.MODEL_TIMEOUT


async def test_sdk_connection_error_maps_to_model_upstream_error() -> None:
    """MC-04：openai.APIConnectionError → MODEL_UPSTREAM_ERROR。"""
    import openai

    completions = _FakeCompletions(error=openai.APIConnectionError(request=_openai_request()))
    model = _model(completions)

    with pytest.raises(FlowAgentCoreError) as info:
        _ = [event async for event in model.stream([], [], {})]
    assert info.value.code is ErrorCode.MODEL_UPSTREAM_ERROR


async def test_unknown_exception_maps_to_model_upstream_error() -> None:
    """MC-04：未识别的异常同样落到 MODEL_UPSTREAM_ERROR（保留类型名便于定位）。"""
    completions = _FakeCompletions(error=ValueError("boom"))
    model = _model(completions)

    with pytest.raises(FlowAgentCoreError) as info:
        _ = [event async for event in model.stream([], [], {})]
    assert info.value.code is ErrorCode.MODEL_UPSTREAM_ERROR
    assert "ValueError" in str(info.value)


async def test_core_error_propagates_unchanged_from_stream() -> None:
    """MC-04：流中途的内核错误必须原样抛出，不得被包装成上游错误。"""
    original = FlowAgentCoreError(ErrorCode.MODEL_INVALID_RESPONSE, "坏分片")
    completions = _FakeCompletions([_chunk(content="半"), original])
    model = _model(completions)

    with pytest.raises(FlowAgentCoreError) as info:
        _ = [event async for event in model.stream([], [], {})]
    assert info.value is original


# ------------------------------------------------------------------ MCP 假件
class _RemoteTool:
    """假 MCP 工具描述（camelCase 的 inputSchema 与 SDK 对齐）。"""

    def __init__(self, name: str, description: str | None = "",
                 input_schema: Any = ...) -> None:
        self.name = name
        self.description = description
        # 省略时给标准对象 schema；显式传 None/非 dict 用于验证内核的归一化。
        self.inputSchema = {"type": "object"} if input_schema is ... else input_schema


class _CallResult:
    """假 call_tool 返回值：isError/structuredContent/content 三源。"""

    def __init__(self, *, isError: bool = False, structuredContent: Any = None,
                 content: list[Any] | None = None, error_code: str | None = None) -> None:
        self.isError = isError
        self.structuredContent = structuredContent
        self.content = list(content or [])
        self.error_code = error_code


class _FakeMcpServer:
    """假 MCP 服务端：目录 + 调用脚本 + 生命周期事件流水。"""

    def __init__(self, tools: list[_RemoteTool] | None = None,
                 results: dict[str, Any] | None = None) -> None:
        self.tools = list(tools or [])
        self.results = dict(results or {})
        self.events: list[str] = []
        self.initialize_calls = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.events.append("initialize")

    async def list_tools(self) -> SimpleNamespace:
        self.events.append("list_tools")
        return SimpleNamespace(tools=list(self.tools))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        script = self.results.get(name)
        if isinstance(script, BaseException):
            # 支持注入 asyncio.TimeoutError / 传输异常等脚本。
            raise script
        return script if script is not None else _CallResult(structuredContent={"ok": True})


class _FakeTransportCM:
    """工厂返回的传输上下文：内核约定解包 (read, write, transport)。"""

    def __init__(self, server: _FakeMcpServer) -> None:
        self._server = server

    async def __aenter__(self):
        self._server.events.append("transport-enter")
        return self._server, "fake-write", None

    async def __aexit__(self, *exc_info: Any) -> None:
        self._server.events.append("transport-exit")


class _FakeClientSession:
    """顶替 mcp.ClientSession：内核把 read_stream 当会话对象，这里直接转调假服务端。"""

    def __init__(self, read_stream: Any, write_stream: Any,
                 timeout_seconds: float | None = None) -> None:
        self.server: _FakeMcpServer = read_stream
        self.timeout_seconds = timeout_seconds
        self.server.events.append("session-init")

    async def __aenter__(self) -> "_FakeClientSession":
        self.server.events.append("session-enter")
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self.server.events.append("session-exit")

    async def initialize(self) -> None:
        await self.server.initialize()

    async def list_tools(self) -> SimpleNamespace:
        return await self.server.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self.server.call_tool(name, arguments)


class _FakeFactory:
    """假 session_factory：记录 (url, headers, timeout) 供断言。"""

    def __init__(self, server: _FakeMcpServer) -> None:
        self.server = server
        self.params: list[dict[str, Any]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> _FakeTransportCM:
        self.params.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeTransportCM(self.server)


@pytest.fixture
def fake_mcp(monkeypatch):
    """离线顶替 mcp.ClientSession。

    内核在 open() 内 `from mcp import ClientSession`（运行时取属性），
    故 monkeypatch 模块属性即可生效；工厂按内核实际契约产出三元组。
    """
    import mcp

    monkeypatch.setattr(mcp, "ClientSession", _FakeClientSession)


def _config(*, allowed_tools: list[str] | None = None,
            timeout_seconds: float = 5.0) -> McpServerConfig:
    return McpServerConfig(
        name="wiki", url="http://fake.local/mcp",
        allowed_tools=allowed_tools, timeout_seconds=timeout_seconds,
    )


async def _opened(session: McpSession) -> McpSession:
    """便捷开启会话（不隐藏 open 失败，便于用例直接断言异常）。"""
    await session.open()
    return session


# ------------------------------------------------------------------ MP-01
def test_normalize_tool_name_canonical_form() -> None:
    """MP-01：规范名为 server__tool。"""
    assert normalize_tool_name("wiki", "search") == "wiki__search"


def test_normalize_tool_name_rejects_inner_double_underscore() -> None:
    """MP-01：工具名内再含 __ 会破坏规范名解析，必须拒绝。"""
    with pytest.raises(FlowAgentCoreError) as info:
        normalize_tool_name("wiki", "a__b")
    assert info.value.code is ErrorCode.CAPABILITY_UNAVAILABLE


# ------------------------------------------------------------------ MP-02
async def test_open_initializes_and_close_releases_session(fake_mcp) -> None:
    """MP-02：open 完成 initialize；close 依次释放会话与传输。"""
    server = _FakeMcpServer(tools=[_RemoteTool("search")])
    factory = _FakeFactory(server)
    session = McpSession(_config(), session_factory=factory)

    await session.open()

    assert server.initialize_calls == 1
    # 工厂拿到解析后的 URL、头（本次为空）与超时。
    assert factory.params == [{"url": "http://fake.local/mcp", "headers": {}, "timeout": 5.0}]
    assert server.events == ["transport-enter", "session-init", "session-enter", "initialize"]

    await session.close()

    # 会话先退、传输后退；顺序即所有权顺序。
    assert server.events[-2:] == ["session-exit", "transport-exit"]


# ------------------------------------------------------------------ MP-03
async def test_list_filters_allowed_tools_and_canonicalizes(fake_mcp) -> None:
    """MP-03：白名单过滤 + 规范名 + inputSchema 透传。"""
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    server = _FakeMcpServer(tools=[
        _RemoteTool("search", description="检索", input_schema=schema),
        _RemoteTool("fetch", description="抓取"),
    ])
    session = McpSession(_config(allowed_tools=["search"]), session_factory=_FakeFactory(server))
    await session.open()

    specs = await session.list()

    assert [spec.name for spec in specs] == ["wiki__search"]
    assert specs[0].description == "检索"
    assert specs[0].input_schema == schema


async def test_list_without_allowlist_returns_all_tools(fake_mcp) -> None:
    """MP-03：未配置白名单时目录应包含全部远端工具。"""
    server = _FakeMcpServer(tools=[
        _RemoteTool("search"), _RemoteTool("fetch", description=None, input_schema=None),
    ])
    session = McpSession(_config(), session_factory=_FakeFactory(server))
    await session.open()

    specs = await session.list()

    assert sorted(spec.name for spec in specs) == ["wiki__fetch", "wiki__search"]
    # description 缺失归一为空串；非 dict schema 归一为空 dict。
    fetch = next(spec for spec in specs if spec.name == "wiki__fetch")
    assert fetch.description == ""
    assert fetch.input_schema == {}


# ------------------------------------------------------------------ MP-04
async def test_call_structured_content_success(fake_mcp) -> None:
    """MP-04：structuredContent 优先作为成功值，且远端工具名去掉 server 前缀。"""
    payload = {"hits": [{"id": 1}]}
    server = _FakeMcpServer(
        tools=[_RemoteTool("search")],
        results={"search": _CallResult(structuredContent=payload)},
    )
    session = McpSession(_config(), session_factory=_FakeFactory(server))
    await _opened(session)
    await session.list()

    result = await session.call("wiki__search", {"q": "巡检"})

    assert result.status == "success"
    assert result.value == payload
    assert server.calls == [("search", {"q": "巡检"})]


async def test_call_text_blocks_joined_when_no_structured(fake_mcp) -> None:
    """MP-04：无 structuredContent 时拼接 content 文本块（换行分隔）。"""
    server = _FakeMcpServer(
        tools=[_RemoteTool("search")],
        results={"search": _CallResult(content=[
            SimpleNamespace(text="第一段"), SimpleNamespace(text="第二段"),
        ])},
    )
    session = McpSession(_config(), session_factory=_FakeFactory(server))
    await _opened(session)
    await session.list()

    result = await session.call("wiki__search", {})

    assert result.status == "success"
    assert result.value == "第一段\n第二段"


async def test_call_is_error_maps_to_mcp_tool_error(fake_mcp) -> None:
    """MP-04：isError=True → status error，错误码 MCP_TOOL_ERROR。"""
    server = _FakeMcpServer(
        tools=[_RemoteTool("search")],
        results={"search": _CallResult(isError=True)},
    )
    session = McpSession(_config(), session_factory=_FakeFactory(server))
    await _opened(session)
    await session.list()

    result = await session.call("wiki__search", {})

    assert result.status == "error"
    assert result.error_code == ErrorCode.MCP_TOOL_ERROR.value
    assert "mcp tool error" in (result.message or "")


async def test_call_timeout_maps_to_unknown(fake_mcp) -> None:
    """MP-04：发送后超时的结果不明 → unknown（不得当作可重放的失败）。

    用即时抛出的 asyncio.TimeoutError 模拟 wait_for 超时，避免真实 sleep 等待。
    """
    server = _FakeMcpServer(
        tools=[_RemoteTool("search")],
        results={"search": asyncio_timeout_error()},
    )
    session = McpSession(_config(timeout_seconds=0.1), session_factory=_FakeFactory(server))
    await _opened(session)
    await session.list()

    result = await session.call("wiki__search", {})

    assert result.status == "unknown"
    assert "timeout" in (result.message or "")


async def test_call_transport_exception_maps_to_error(fake_mcp) -> None:
    """MP-04：传输层异常（未完成发送）→ status error。"""
    server = _FakeMcpServer(
        tools=[_RemoteTool("search")],
        results={"search": RuntimeError("connection reset")},
    )
    session = McpSession(_config(), session_factory=_FakeFactory(server))
    await _opened(session)
    await session.list()

    result = await session.call("wiki__search", {})

    assert result.status == "error"
    assert "connection reset" in (result.message or "")


# ------------------------------------------------------------------ MP-05
async def test_call_unlisted_tool_rejected_by_catalog(fake_mcp) -> None:
    """MP-05：目录外的工具一律拒绝，绝不放行白名单外能力。"""
    server = _FakeMcpServer(tools=[_RemoteTool("search"), _RemoteTool("fetch")])
    session = McpSession(_config(allowed_tools=["search"]), session_factory=_FakeFactory(server))
    await _opened(session)
    await session.list()

    result = await session.call("wiki__fetch", {"url": "x"})

    assert result.status == "error"
    assert result.error_code == ErrorCode.TOOL_ARGUMENT_INVALID.value
    assert server.calls == []  # 未触达远端


# ------------------------------------------------------------------ MP-06
async def test_open_accepts_documented_session_factory_contract() -> None:
    """MP-06：按约定契约，工厂产出会话对象即可 open 成功并 list。

    当前实现会因解包失败（TypeError）被包装成 CAPABILITY_UNAVAILABLE。
    """
    server = _FakeMcpServer(tools=[_RemoteTool("search")])

    class _DocumentedCM:
        async def __aenter__(self):
            return server  # 约定：直接产出会话对象，而不是 (read, write, transport)

        async def __aexit__(self, *exc_info: Any) -> None:
            return None

    session = McpSession(_config(), session_factory=lambda *args: _DocumentedCM())

    await session.open()
    specs = await session.list()
    assert [spec.name for spec in specs] == ["wiki__search"]


def asyncio_timeout_error() -> TimeoutError:
    """asyncio.TimeoutError 即内置 TimeoutError（3.11+），单独封装便于阅读。"""
    import asyncio

    return asyncio.TimeoutError("simulated wait_for timeout")
