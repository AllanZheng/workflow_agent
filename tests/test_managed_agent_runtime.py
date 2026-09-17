import asyncio

import pytest
from langchain_core.messages import AIMessage

from flowagent.agent_runtime import ManagedAgentRuntime, ToolIterationLimit
from flowagent.engine import build_graph
from flowagent.managed_config import load_managed_spec


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


def test_managed_graph_executes_agent_node_with_request_runtime():
    spec = load_managed_spec("""
workflow:
  name: managed
  start: main
nodes:
  main:
    type: agent
    output_key: reply
    max_tool_iterations: 3
    next: END
""")

    class FakeRuntime:
        def __init__(self): self.calls = []
        async def run(self, message, system_prompt, max_tool_iterations):
            self.calls.append((message, system_prompt, max_tool_iterations))
            return "managed-answer"

    runtime = FakeRuntime()
    graph = build_graph(spec, llm=FakeLlm([]), checkpointer=False, agent_runtime=runtime)
    final = asyncio.run(graph.ainvoke({
        "vars": {"message": "hello", "system_prompt": "rules"},
        "trace": [], "loop_counts": {}, "error": None,
    }))

    assert final["vars"]["reply"] == "managed-answer"
    assert runtime.calls == [("hello", "rules", 3)]
