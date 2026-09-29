import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from flowagent.agent_runtime import ManagedAgentRuntime, ToolIterationLimit
from flowagent.engine import build_graph
from flowagent.managed_config import load_managed_spec


class FakeMcp:
    # 初始化对象状态并建立所需依赖。
    def __init__(self): self.calls = []
    # 列出工具。
    async def list_tools(self):
        return [{"name": "search", "description": "search", "input_schema": {"type": "object"}}]
    # 处理`call`工具相关逻辑。
    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return "tool-result"


class FakeBoundLlm:
    # 初始化对象状态并建立所需依赖。
    def __init__(self, replies): self.replies = iter(replies)
    # 处理`ainvoke`相关逻辑。
    async def ainvoke(self, messages): return next(self.replies)


class FakeLlm:
    # 初始化对象状态并建立所需依赖。
    def __init__(self, replies): self.bound = FakeBoundLlm(replies); self.tools = None
    # 处理`bind`工具相关逻辑。
    def bind_tools(self, tools): self.tools = tools; return self.bound


# 验证“群组`resolution``needing`用户输入`finishes`不包含`repeated``calls`”场景下的预期行为。
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("resolution,expected", [
    ({"status": "not_found"}, "没有找到"),
    ({"status": "ambiguous", "candidates": [
        {"conversation_id": 6, "name": "风险会商一组"},
        {"conversation_id": 7, "name": "风险会商二组"},
    ]}, "风险会商一组（ID 6）"),
])
def test_group_resolution_needing_user_input_finishes_without_repeated_calls(streaming, resolution, expected):
    class ResolvingMcp:
        # 初始化对象状态并建立所需依赖。
        def __init__(self): self.calls = 0
        # 列出工具。
        async def list_tools(self):
            return [{"name": "platform_resolve_group"}]
        # 处理`call`工具相关逻辑。
        async def call_tool(self, name, arguments):
            self.calls += 1
            return resolution

    class RepeatingModel:
        # 处理`bind`工具相关逻辑。
        def bind_tools(self, tools): return self
        # 处理`ainvoke`相关逻辑。
        async def ainvoke(self, messages):
            return AIMessage(content="", tool_calls=[{
                "id": "resolve", "name": "platform_resolve_group", "args": {"name_query": "风险会商群"},
            }])
        # 处理`astream`相关逻辑。
        async def astream(self, messages):
            response = await self.ainvoke(messages)
            yield AIMessageChunk(content="", tool_calls=response.tool_calls)

    mcp = ResolvingMcp()
    runtime = ManagedAgentRuntime(RepeatingModel(), mcp)
    # 执行执行对应的流程。
    async def execute():
        if streaming:
            return "".join([part async for part in runtime.stream("总结风险会商群", "", 8)])
        return await runtime.run("总结风险会商群", "", 8)
    assert expected in asyncio.run(execute())
    assert mcp.calls == 1


# 验证“代理`calls`MCP工具`then``returns``final``answer`”场景下的预期行为。
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


# 验证“代理停止`at`已配置工具`iteration`数量上限”场景下的预期行为。
def test_agent_stops_at_configured_tool_iteration_limit():
    calls = [AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "search", "args": {}}]) for i in range(3)]
    with pytest.raises(ToolIterationLimit):
        asyncio.run(ManagedAgentRuntime(FakeLlm(calls), FakeMcp()).run("hello", "", 2))


# 验证“代理`places`历史记录之前当前项消息”场景下的预期行为。
def test_agent_places_history_before_current_message():
    captured = {}

    class CapturingModel:
        # 处理`ainvoke`相关逻辑。
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return AIMessage(content="answer")

    class CapturingLlm:
        # 处理`bind`工具相关逻辑。
        def bind_tools(self, tools): return CapturingModel()

    result = asyncio.run(ManagedAgentRuntime(CapturingLlm(), FakeMcp()).run(
        "current", "rules", 2,
        history=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    ))

    assert result == "answer"
    assert [message.type for message in captured["messages"]] == [
        "system", "human", "ai", "human",
    ]
    assert isinstance(captured["messages"][-1], HumanMessage)
    assert captured["messages"][-1].content == "current"


# 验证“`managed`图执行代理节点包含请求运行时”场景下的预期行为。
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
        # 初始化对象状态并建立所需依赖。
        def __init__(self): self.calls = []
        # 执行当前任务流程并返回运行结果。
        async def run(self, message, system_prompt, max_tool_iterations, history=None):
            self.calls.append((message, system_prompt, max_tool_iterations, history))
            return "managed-answer"

    runtime = FakeRuntime()
    graph = build_graph(spec, llm=FakeLlm([]), checkpointer=False, agent_runtime=runtime)
    final = asyncio.run(graph.ainvoke({
        "vars": {"message": "hello", "system_prompt": "rules"},
        "trace": [], "loop_counts": {}, "error": None,
    }))

    assert final["vars"]["reply"] == "managed-answer"
    assert runtime.calls == [("hello", "rules", 3, [])]


# ---------------------------------------------------------------- P3 会话连续性


def test_session_plan_starts_new_without_checkpoint():
    from flowagent.session_continuity import plan_session

    decision = plan_session(has_checkpoint=False,
                            checkpoint_config_digest=None,
                            request_config_digest="d1")
    assert decision.action == "start_new"
    assert decision.error_code is None


# 验证“会话`plan``reuses`仅启用`identical``digest`与结构”场景下的预期行为。
def test_session_plan_reuses_only_on_identical_digest_and_schema():
    from flowagent.session_continuity import plan_session

    same = plan_session(has_checkpoint=True, checkpoint_config_digest="d1",
                        request_config_digest="d1", checkpoint_schema_version=3,
                        request_schema_version=3)
    assert same.action == "reuse_checkpoint"

    changed_config = plan_session(has_checkpoint=True, checkpoint_config_digest="d1",
                                  request_config_digest="d2")
    assert changed_config.action == "restart_required"
    assert changed_config.error_code == "SESSION_RESTART_REQUIRED"

    changed_schema = plan_session(has_checkpoint=True, checkpoint_config_digest="d1",
                                  request_config_digest="d1",
                                  checkpoint_schema_version=1, request_schema_version=3)
    assert changed_schema.action == "restart_required"


# 验证“会话`plan``never``silently``continues`未知`digest`”场景下的预期行为。
def test_session_plan_never_silently_continues_unknown_digest():
    from flowagent.session_continuity import plan_session

    decision = plan_session(has_checkpoint=True, checkpoint_config_digest=None,
                            request_config_digest="d1")
    assert decision.action == "restart_required"
    assert decision.error_code == "SESSION_RESTART_REQUIRED"
