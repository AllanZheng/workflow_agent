import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from flowagent.mcp_edge_client import McpEdgeClient, McpEdgeError


class FakeSession:
    def __init__(self):
        self.calls = []

    async def list_tools(self):
        tool = SimpleNamespace(
            name="search",
            description=None,
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        return SimpleNamespace(tools=[tool])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            is_error=False,
            structured_content={"items": ["a"]},
            content=[SimpleNamespace(text="fallback")],
        )


def make_factory(session, captured):
    @asynccontextmanager
    async def factory(endpoint, headers, timeout_seconds):
        captured.update(endpoint=endpoint, headers=headers, timeout_seconds=timeout_seconds)
        yield session

    return factory


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


def test_calls_tool_and_prefers_structured_result():
    session = FakeSession()
    client = McpEdgeClient("http://demo.test/mcp", "token", session_factory=make_factory(session, {}))

    result = asyncio.run(client.call_tool("search", {"q": "x"}))

    assert result == {"items": ["a"]}
    assert session.calls == [("search", {"q": "x"})]


def test_normalizes_text_result_when_structured_result_is_absent():
    session = FakeSession()

    async def call_tool(name, arguments):
        return SimpleNamespace(
            is_error=False,
            structured_content=None,
            content=[SimpleNamespace(text="first"), SimpleNamespace(text="second")],
        )

    session.call_tool = call_tool
    client = McpEdgeClient("http://demo.test/mcp", "token", session_factory=make_factory(session, {}))

    assert asyncio.run(client.call_tool("search", {})) == "first\nsecond"


def test_remote_tool_error_does_not_leak_token_or_response_body():
    session = FakeSession()

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


@pytest.mark.parametrize("endpoint, token", [("", "token"), ("http://demo.test/mcp", "")])
def test_rejects_missing_connection_configuration(endpoint, token):
    with pytest.raises(ValueError):
        McpEdgeClient(endpoint, token)
