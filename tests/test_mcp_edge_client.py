import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from flowagent.mcp_edge_client import McpEdgeClient, McpEdgeError


class FakeSession:
    # 初始化对象状态并建立所需依赖。
    # English: Initialize object state and required dependencies.
    def __init__(self, tool_name="search"):
        self.calls = []
        self.tool_name = tool_name

    # 列出工具。
    # English: List tools.
    async def list_tools(self):
        tool = SimpleNamespace(
            name=self.tool_name,
            description=None,
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        return SimpleNamespace(tools=[tool])

    # 处理`call`工具相关逻辑。
    # English: Handle call tool operations.
    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            is_error=False,
            structured_content={"items": ["a"]},
            content=[SimpleNamespace(text="fallback")],
        )


# 构建工厂方法。
# English: Create factory.
def make_factory(session, captured):
    # 处理工厂方法相关逻辑。
    # English: Handle factory operations.
    @asynccontextmanager
    async def factory(endpoint, headers, timeout_seconds):
        captured.update(endpoint=endpoint, headers=headers, timeout_seconds=timeout_seconds)
        yield session

    return factory


# 验证“列出工具与`passes`工作负载Bearer 令牌`transport`”场景下的预期行为。
# English: Verify lists tools and passes workload bearer to transport.
def test_lists_tools_and_passes_workload_bearer_to_transport():
    captured = {}
    client = McpEdgeClient(
        "http://demo.test/mcp",
        "workload-secret",
        timeout_seconds=7,
        session_factory=make_factory(FakeSession(), captured),
    )

    tools = asyncio.run(client.list_tools())

    assert tools == [{
        "name": "search",
        "description": "",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }]
    assert captured == {
        "endpoint": "http://demo.test/mcp",
        "headers": {"Authorization": "Bearer workload-secret"},
        "timeout_seconds": 7,
    }
    assert "workload-secret" not in repr(client)


# 验证“`calls`工具与`prefers``structured`结果”场景下的预期行为。
# English: Verify calls tool and prefers structured result.
def test_calls_tool_and_prefers_structured_result():
    session = FakeSession()
    client = McpEdgeClient("http://demo.test/mcp", "token", session_factory=make_factory(session, {}))

    result = asyncio.run(client.call_tool("search", {"q": "x"}))

    assert result == {"items": ["a"]}
    assert session.calls == [("search", {"q": "x"})]


# 验证“`v3`列出仅`frozen`工具版本与发送`their`UUID 标识权限范围”场景下的预期行为。
# English: Verify v3 lists only frozen tool versions and sends their UUID scope.
def test_v3_lists_only_frozen_tool_versions_and_sends_their_uuid_scope():
    session = FakeSession("tool_604b620930da4e57bf9373c6867cc101")
    captured = {}
    frozen = [{
        "tool_id": "604b6209-30da-4e57-bf93-73c6867cc101",
        "tool_version_id": "704b6209-30da-4e57-bf93-73c6867cc101",
        "server_version_id": "804b6209-30da-4e57-bf93-73c6867cc101",
        "remote_name": "search",
        "description": "Pinned description",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }]
    client = McpEdgeClient(
        "http://demo.test/mcp/execution/f04b6209-30da-4e57-bf93-73c6867cc101/",
        "execution-token",
        session_factory=make_factory(session, captured),
        fixed_tools=frozen,
        execution_id="f04b6209-30da-4e57-bf93-73c6867cc101",
        agent_id="904b6209-30da-4e57-bf93-73c6867cc101",
        revision_id="a04b6209-30da-4e57-bf93-73c6867cc101",
        runtime_id="b04b6209-30da-4e57-bf93-73c6867cc101",
        generation=4,
    )

    tools = asyncio.run(client.list_tools())
    result = asyncio.run(client.call_tool(tools[0]["name"], {"query": "x"}))

    assert tools == [{
        "name": "tool_604b620930da4e57bf9373c6867cc101",
        "description": "Pinned description",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        "tool_id": frozen[0]["tool_id"],
        "tool_version_id": frozen[0]["tool_version_id"],
        "server_version_id": frozen[0]["server_version_id"],
        "remote_name": "search",
    }]
    assert result == {"items": ["a"]}
    assert session.calls == [(tools[0]["name"], {"query": "x"})]
    assert captured["headers"]["X-Agent-Tool-ID"] == frozen[0]["tool_id"]
    assert captured["headers"]["X-Agent-Tool-Version-ID"] == frozen[0]["tool_version_id"]
    assert captured["headers"]["X-Agent-Revision-ID"] == "a04b6209-30da-4e57-bf93-73c6867cc101"
    assert captured["headers"]["X-Runtime-ID"] == "b04b6209-30da-4e57-bf93-73c6867cc101"


# 验证“`v3`允许仅`advertised``platform`群组`resolver`不包含UUID 标识请求头”场景下的预期行为。
# English: Verify v3 allows only advertised platform group resolver without UUID headers.
def test_v3_allows_only_advertised_platform_group_resolver_without_uuid_headers():
    class ResolverSession(FakeSession):
        # 列出工具。
        # English: List tools.
        async def list_tools(self):
            pinned = SimpleNamespace(
                name="tool_604b620930da4e57bf9373c6867cc101",
                description="Pinned", input_schema={"type": "object"},
            )
            resolver = SimpleNamespace(
                name="platform_resolve_group", description="Resolve a group",
                input_schema={"type": "object", "required": ["name_query"]},
            )
            return SimpleNamespace(tools=[pinned, resolver])

    session = ResolverSession()
    captured = {}
    client = McpEdgeClient(
        "http://demo.test/mcp/execution/f04b6209-30da-4e57-bf93-73c6867cc101/",
        "execution-token", session_factory=make_factory(session, captured),
        fixed_tools=[{
            "tool_id": "604b6209-30da-4e57-bf93-73c6867cc101",
            "tool_version_id": "704b6209-30da-4e57-bf93-73c6867cc101",
            "server_version_id": "804b6209-30da-4e57-bf93-73c6867cc101",
            "remote_name": "fetch_group_analysis_input",
            "description": "Pinned", "input_schema": {"type": "object"},
        }],
        execution_id="f04b6209-30da-4e57-bf93-73c6867cc101",
        agent_id="904b6209-30da-4e57-bf93-73c6867cc101",
        revision_id="a04b6209-30da-4e57-bf93-73c6867cc101",
        runtime_id="b04b6209-30da-4e57-bf93-73c6867cc101", generation=4,
    )

    tools = asyncio.run(client.list_tools())
    result = asyncio.run(client.call_tool(
        "platform_resolve_group", {"name_query": "风险会商群"},
    ))

    assert "platform_resolve_group" in {tool["name"] for tool in tools}
    assert result == {"items": ["a"]}
    assert session.calls[-1][0] == "platform_resolve_group"
    assert "X-Agent-Tool-ID" not in captured["headers"]


# 验证“`v3`拒绝工具`alias``outside``fixed`生成快照”场景下的预期行为。
# English: Verify v3 rejects tool alias outside the fixed snapshot.
def test_v3_rejects_tool_alias_outside_the_fixed_snapshot():
    client = McpEdgeClient(
        "http://demo.test/mcp/execution/f04b6209-30da-4e57-bf93-73c6867cc101/",
        "execution-token",
        session_factory=make_factory(FakeSession("tool_unknown"), {}),
        fixed_tools=[{
            "tool_id": "604b6209-30da-4e57-bf93-73c6867cc101",
            "tool_version_id": "704b6209-30da-4e57-bf93-73c6867cc101",
            "server_version_id": "804b6209-30da-4e57-bf93-73c6867cc101",
            "remote_name": "search", "description": "", "input_schema": {"type": "object"},
        }],
        execution_id="f04b6209-30da-4e57-bf93-73c6867cc101",
        agent_id="904b6209-30da-4e57-bf93-73c6867cc101",
        revision_id="a04b6209-30da-4e57-bf93-73c6867cc101",
        runtime_id="b04b6209-30da-4e57-bf93-73c6867cc101",
        generation=4,
    )

    with pytest.raises(McpEdgeError, match="not pinned"):
        asyncio.run(client.call_tool("tool_unknown", {}))


# 验证“规范化文本结果当`structured`结果判断是否`absent`”场景下的预期行为。
# English: Verify normalizes text result when structured result is absent.
def test_normalizes_text_result_when_structured_result_is_absent():
    session = FakeSession()

    # 处理`call`工具相关逻辑。
    # English: Handle call tool operations.
    async def call_tool(name, arguments):
        return SimpleNamespace(
            is_error=False,
            structured_content=None,
            content=[SimpleNamespace(text="first"), SimpleNamespace(text="second")],
        )

    session.call_tool = call_tool
    client = McpEdgeClient("http://demo.test/mcp", "token", session_factory=make_factory(session, {}))

    assert asyncio.run(client.call_tool("search", {})) == "first\nsecond"


# 验证“`remote`工具错误`does`不`leak`令牌或响应内容体”场景下的预期行为。
# English: Verify remote tool error does not leak token or response body.
def test_remote_tool_error_does_not_leak_token_or_response_body():
    session = FakeSession()

    # 处理`call`工具相关逻辑。
    # English: Handle call tool operations.
    async def call_tool(name, arguments):
        return SimpleNamespace(
            is_error=True,
            structured_content=None,
            content=[SimpleNamespace(text="remote-sensitive-detail")],
        )

    session.call_tool = call_tool
    client = McpEdgeClient(
        "http://demo.test/mcp",
        "workload-secret",
        session_factory=make_factory(session, {}),
    )

    with pytest.raises(McpEdgeError) as exc:
        asyncio.run(client.call_tool("search", {}))

    assert "search" in str(exc.value)
    assert "workload-secret" not in str(exc.value)
    assert "remote-sensitive-detail" not in str(exc.value)


# 验证“拒绝缺失连接配置”场景下的预期行为。
# English: Verify rejects missing connection configuration.
@pytest.mark.parametrize("endpoint, token", [("", "token"), ("http://demo.test/mcp", "")])
def test_rejects_missing_connection_configuration(endpoint, token):
    with pytest.raises(ValueError):
        McpEdgeClient(endpoint, token)
