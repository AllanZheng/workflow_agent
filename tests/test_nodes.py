"""单元测试：节点执行器（模板/工具/LLM 三类节点 + 错误降级）。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from flowagent.config import NodeConfig
from flowagent.errors import FlowAgentError, OutputParseError, ToolError
from flowagent.llm import MockLLM
from flowagent.nodes import execute_node, make_node_executor


class FakeRealLLM:
    """模拟“非 Mock”的真实 LLM：忽略 mock_response，按固定脚本回复。"""

    name = "fake-real"

    # 初始化对象状态并建立所需依赖。
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list] = []

    # 处理`invoke`相关逻辑。
    def invoke(self, messages, config=None, **kwargs) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.replies.pop(0))


# 处理节点相关逻辑。
def _node(**kwargs) -> NodeConfig:
    base = {"type": "template", "template": "x", "output_key": "o", "next": "END"}
    base.update(kwargs)
    return NodeConfig.model_validate(base)


class TestExecuteNode:
    # 验证“模板节点”场景下的预期行为。
    def test_template_node(self):
        node = _node(template="你好 {name}", output_key="msg")
        assert execute_node(node, {"vars": {"name": "Bob"}}, None) == {"msg": "你好 Bob"}

    # 验证“工具节点包含`var``arg`”场景下的预期行为。
    def test_tool_node_with_var_arg(self):
        node = _node(
            type="tool",
            tool="flowagent.tools:bump_counter",
            args={"count": "$n"},
            output_key="n",
        )
        out = execute_node(node, {"vars": {"n": 4}}, None)
        assert out == {"n": 5}

    # 验证“工具节点`literal`仅”场景下的预期行为。
    def test_tool_node_literal_only(self):
        node = _node(
            type="tool",
            tool="flowagent.tools:knowledge_base",
            args={"query": "退款政策是什么"},
            output_key="answer",
        )
        out = execute_node(node, {"vars": {}}, None)
        assert "退款" in out["answer"]

    # 验证“工具`bad``ref`抛出”场景下的预期行为。
    def test_tool_bad_ref_raises(self):
        node = _node(type="tool", tool="not-a-ref", args={}, output_key="o")
        with pytest.raises(FlowAgentError):
            execute_node(node, {"vars": {}}, None)

    # 验证“大语言模型节点`uses``mock`响应”场景下的预期行为。
    def test_llm_node_uses_mock_response(self):
        node = _node(
            type="llm",
            prompt="分类：{x}",
            parser="classifier",
            labels=["退款"],
            output_key="cat",
            mock_response="我认为是退款",
        )
        out = execute_node(node, {"vars": {"x": 1}}, MockLLM())
        assert out == {"cat": "退款"}

    # 验证“大语言模型节点包含`real`大语言模型`ignores``mock`响应”场景下的预期行为。
    def test_llm_node_with_real_llm_ignores_mock_response(self):
        fake = FakeRealLLM(['{"score": 8, "reason": "good"}'])
        node = _node(
            type="llm",
            prompt="p",
            system="s",
            parser="json",
            output_key="review",
            mock_response="这个不应被使用",
        )
        out = execute_node(node, {"vars": {"x": 1}}, fake)
        assert out == {"review": {"score": 8, "reason": "good"}}
        # system + human 两条消息均已渲染
        assert len(fake.calls[0]) == 2
        assert fake.calls[0][0].content == "s"

    # 验证“大语言模型节点内容`block`列出”场景下的预期行为。
    def test_llm_node_content_block_list(self):
        class BlockLLM(FakeRealLLM):
            # 处理`invoke`相关逻辑。
            def invoke(self, messages, config=None, **kwargs):
                return SimpleNamespace(content=["部分1", "部分2"])

        node = _node(type="llm", prompt="p", parser="text", output_key="t")
        out = execute_node(node, {"vars": {}}, BlockLLM([]))
        assert out == {"t": "部分1部分2"}

    # 验证“大语言模型节点不包含大语言模型抛出”场景下的预期行为。
    def test_llm_node_without_llm_raises(self):
        node = _node(type="llm", prompt="p", output_key="t")
        with pytest.raises(FlowAgentError):
            execute_node(node, {"vars": {}}, None)

    # 验证“未知类型抛出”场景下的预期行为。
    def test_unknown_type_raises(self):
        bogus = SimpleNamespace(
            type="magic", output_key="o", args={}, parser="text", labels=[]
        )
        with pytest.raises(FlowAgentError):
            execute_node(bogus, {"vars": {}}, None)


class TestNodeExecutorWrapper:
    # 验证“`success``increments`数量与追踪”场景下的预期行为。
    def test_success_increments_counts_and_trace(self, mock_llm):
        node = _node(template="hi", output_key="msg", next="END")
        executor = make_node_executor("step1", node, mock_llm)
        update = executor({"vars": {}, "loop_counts": {}, "trace": []})
        assert update["loop_counts"] == {"step1": 1}
        assert update["trace"][0]["node"] == "step1"
        assert update["trace"][0]["status"] == "ok"
        assert update["vars"] == {"msg": "hi"}

    # 验证“`failure`不包含启用错误抛出”场景下的预期行为。
    def test_failure_without_on_error_raises(self):
        node = _node(
            type="tool",
            tool="flowagent.tools:unreliable_tool",
            args={"text": "触发故障"},
            output_key="r",
            next="END",
        )
        executor = make_node_executor("boom", node, None)
        with pytest.raises(ToolError):
            executor({"vars": {}, "loop_counts": {}, "trace": []})

    # 验证“`failure`包含启用错误`degrades`”场景下的预期行为。
    def test_failure_with_on_error_degrades(self):
        node = _node(
            type="tool",
            tool="flowagent.tools:unreliable_tool",
            args={"text": "触发故障"},
            output_key="r",
            on_error={"goto": "manual", "set": {"r": "人工接管"}},
            next="END",
        )
        executor = make_node_executor("boom", node, None)
        update = executor({"vars": {}, "loop_counts": {}, "trace": []})
        assert update["error"]["node"] == "boom"
        assert update["error"]["type"] == "ToolError"
        assert update["vars"] == {"r": "人工接管"}
        assert update["trace"][0]["status"] == "error"

    # 验证“追踪`summarize``truncates``long`值”场景下的预期行为。
    def test_trace_summarize_truncates_long_values(self, mock_llm):
        node = _node(template="x" * 300, output_key="big", next="END")
        executor = make_node_executor("n", node, mock_llm)
        update = executor({"vars": {}, "loop_counts": {}, "trace": []})
        assert update["trace"][0]["outputs"]["big"].endswith("...")
        assert len(update["trace"][0]["outputs"]["big"]) <= 120


class TestOutputSchemaNode:
    """4.2：llm 节点声明 output_schema 后，畸形输出失败关闭。"""

    SCORE_SCHEMA = {
        "score": {"type": "integer", "min": 1, "max": 10},
        "reason": {"type": "string"},
    }

    # 处理节点相关逻辑。
    def _node(self, mock: str) -> NodeConfig:
        return NodeConfig(
            type="llm", prompt="打分", parser="json", output_key="qa",
            mock_response=mock, output_schema=self.SCORE_SCHEMA, next="END",
        )

    # 验证“缺失`score`标记失败`closed`”场景下的预期行为。
    def test_missing_score_fails_closed(self):
        with pytest.raises(OutputParseError):
            execute_node(self._node('{"reason": "missing"}'), {}, MockLLM(), "qa")

    # 验证“`score``out``range`标记失败`closed`”场景下的预期行为。
    def test_score_out_of_range_fails_closed(self):
        with pytest.raises(OutputParseError):
            execute_node(self._node('{"score": 99, "reason": "x"}'), {}, MockLLM(), "qa")

    # 验证“有效`mock``passes`”场景下的预期行为。
    def test_valid_mock_passes(self):
        out = execute_node(self._node('{"score": 9, "reason": "ok"}'), {}, MockLLM(), "qa")
        assert out["qa"]["score"] == 9

    # 验证“缺失`score``degrades``via`启用错误”场景下的预期行为。
    def test_missing_score_degrades_via_on_error(self):
        from flowagent import WorkflowEngine, load_spec

        spec = load_spec({
            "workflow": {"name": "sc2", "start": "qa"},
            "nodes": {
                "qa": {
                    "type": "llm", "prompt": "打分", "parser": "json",
                    "output_key": "qa", "mock_response": '{"reason": "m"}',
                    "output_schema": {"score": {"type": "integer"}},
                    "on_error": {"goto": "manual", "set": {}},
                    "next": "END",
                },
                "manual": {
                    "type": "template", "template": "转人工复核",
                    "output_key": "final", "next": "END",
                },
            },
        })
        final = WorkflowEngine(spec, MockLLM()).run({})
        assert final["vars"]["final"] == "转人工复核"
        assert final["error"]["type"] == "OutputParseError"

    # 验证“结构包含`non`JSON参数解析器`rejected``at`加载”场景下的预期行为。
    def test_schema_with_non_json_parser_rejected_at_load(self):
        from flowagent import load_spec
        from flowagent.errors import ConfigError

        with pytest.raises(ConfigError):
            load_spec({
                "workflow": {"name": "bad", "start": "n"},
                "nodes": {
                    "n": {
                        "type": "llm", "prompt": "x", "parser": "text",
                        "output_key": "o", "output_schema": {"a": {}},
                        "next": "END",
                    }
                },
            })
