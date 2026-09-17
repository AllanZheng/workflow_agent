import asyncio

import pytest
from langchain_core.messages import AIMessage

from flowagent.agent_runtime import ManagedAgentRuntime, ToolIterationLimit


class FakeMcp:
    def __init__(self): self.calls = []
    async def list_tools(self):
        return [{"name": "search", "description": "search", "input_schema": {"type": "object"}}]
    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return "tool-result"


class FakeBoundLlm:
    def __init__(self, replies): self.replies = iter(replies)
    async def ainvoke(self, messages): return next(self.replies)


class FakeLlm:
    def __init__(self, replies): self.bound = FakeBoundLlm(replies); self.tools = None
    def bind_tools(self, tools): self.tools = tools; return self.bound


def test_agent_calls_mcp_tool_then_returns_final_answer():
    llm = FakeLlm([
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "search", "args": {"q": "x"}}]),
        AIMessage(content="final"),
    ])
    mcp = FakeMcp()
    answer = asyncio.run(ManagedAgentRuntime(llm, mcp).run("hello", "system", 8))
    assert answer == "final"
    assert mcp.calls == [("search", {"q": "x"})]
    assert llm.tools[0]["function"]["name"] == "search"


def test_agent_stops_at_configured_tool_iteration_limit():
    calls = [AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "search", "args": {}}]) for i in range(3)]
    with pytest.raises(ToolIterationLimit):
        asyncio.run(ManagedAgentRuntime(FakeLlm(calls), FakeMcp()).run("hello", "", 2))
