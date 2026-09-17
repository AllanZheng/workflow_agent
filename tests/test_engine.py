"""集成测试：引擎编译、端到端执行、循环护栏、错误降级、checkpoint。"""

import pytest

from flowagent.config import load_spec
from flowagent.engine import WorkflowEngine, build_graph
from flowagent.errors import ToolError
from flowagent.llm import MockLLM

LOOP_SPEC = {
    "workflow": {"name": "loop", "start": "bump", "recursion_limit": 30},
    "nodes": {
        "bump": {
            "type": "tool",
            "tool": "flowagent.tools:bump_counter",
            "args": {"count": "$count"},
            "output_key": "count",
            "next": {
                "field": "count",
                "rules": [
                    {
                        "when": {"op": "lt", "value": 99},
                        "goto": "bump",
                        "max_loops": 3,
                        "loop_exhaust": "done",
                    }
                ],
                "default": "done",
            },
        },
        "done": {
            "type": "template",
            "template": "完成，计数={count}",
            "output_key": "msg",
            "next": "END",
        },
    },
}

ERROR_SPEC = {
    "workflow": {"name": "err", "start": "break"},
    "nodes": {
        "break": {
            "type": "tool",
            "tool": "flowagent.tools:unreliable_tool",
            "args": {"text": "$ticket"},
            "output_key": "result",
            "on_error": {"goto": "manual", "set": {"result": "人工接管"}},
            "next": {"field": "result", "rules": [], "default": "manual", "on_error": "manual"},
        },
        "manual": {
            "type": "template",
            "template": "已转人工（处理结果：{result}）",
            "output_key": "msg",
            "next": "END",
        },
    },
}


class TestBuildGraph:
    def test_build_returns_compiled_graph(self):
        spec = load_spec(LOOP_SPEC)
        graph = build_graph(spec, MockLLM())
        assert graph is not None

    def test_build_requires_spec(self):
        with pytest.raises(AttributeError):
            build_graph(None, MockLLM())  # type: ignore[arg-type]

    def test_build_graph_auto_creates_llm(self, monkeypatch):
        # 不传 llm 时引擎应自动走 create_llm()（无 key 环境 → MockLLM）
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        graph = build_graph(load_spec(LOOP_SPEC))
        assert graph is not None


class TestLoopGuard:
    def test_max_loops_caps_execution(self):
        engine = WorkflowEngine(load_spec(LOOP_SPEC), MockLLM())
        final = engine.run({"count": 0})
        # 循环护栏：bump 恰好执行 3 次后强制走 loop_exhaust
        assert final["vars"]["count"] == 3
        assert final["loop_counts"]["bump"] == 3
        assert final["vars"]["msg"] == "完成，计数=3"
        nodes_run = [step["node"] for step in final["trace"]]
        assert nodes_run == ["bump", "bump", "bump", "done"]


class TestErrorDegrade:
    def test_on_error_routes_to_fallback(self):
        engine = WorkflowEngine(load_spec(ERROR_SPEC), MockLLM())
        final = engine.run({"ticket": "请触发故障"})
        assert final["error"]["node"] == "break"
        assert final["error"]["type"] == "ToolError"
        assert final["vars"]["result"] == "人工接管"
        assert final["trace"][0]["status"] == "error"

    def test_tool_success_no_error(self):
        engine = WorkflowEngine(load_spec(ERROR_SPEC), MockLLM())
        final = engine.run({"ticket": "正常输入"})
        assert final["error"] is None
        assert final["vars"]["result"] == "processed:正常输入"
        assert final["vars"]["msg"] == "已转人工（处理结果：processed:正常输入）"

    def test_failure_without_on_error_propagates(self):
        spec = {
            "workflow": {"name": "err2", "start": "break"},
            "nodes": {
                "break": {
                    "type": "tool",
                    "tool": "flowagent.tools:unreliable_tool",
                    "args": {"text": "$ticket"},
                    "output_key": "result",
                    "next": "END",
                }
            },
        }
        engine = WorkflowEngine(load_spec(spec), MockLLM())
        with pytest.raises(ToolError):
            engine.run({"ticket": "触发故障"})


class TestCheckpoint:
    def test_state_persisted_by_thread_id(self):
        spec = load_spec(LOOP_SPEC)
        engine = WorkflowEngine(spec, MockLLM())
        final = engine.run({"count": 0}, thread_id="ck-1")
        snap = engine.get_state("ck-1").values
        assert snap["vars"] == final["vars"]
        assert snap["loop_counts"]["bump"] == 3

    def test_rerun_same_thread_id_merges_state(self):
        spec = load_spec(LOOP_SPEC)
        engine = WorkflowEngine(spec, MockLLM())
        engine.run({"count": 0}, thread_id="ck-2")
        # 同一 thread_id 续跑：新输入与检查点状态合并
        final = engine.run({"tag": "second"}, thread_id="ck-2")
        assert final["vars"]["tag"] == "second"
        assert final["vars"]["count"] >= 1


class TestMermaid:
    def test_mermaid_contains_nodes(self):
        engine = WorkflowEngine(load_spec(LOOP_SPEC), MockLLM())
        text = engine.mermaid()
        assert "bump" in text and "done" in text
