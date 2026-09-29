"""检查点写入策略（durability）与并发安全测试。"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from flowagent import WorkflowEngine, load_spec
from flowagent.llm import MockLLM

SPEC = {
    "workflow": {"name": "dur", "start": "a"},
    "nodes": {
        "a": {
            "type": "template",
            "template": "值:{x}",
            "output_key": "msg",
            "next": "END",
        }
    },
}


class TestDurabilityConfig:
    # 验证“有效值`accepted`”场景下的预期行为。
    @pytest.mark.parametrize("value", ["exit", "async", "all"])
    def test_valid_values_accepted(self, value):
        data = {"workflow": {"name": "d", "start": "a", "durability": value}, "nodes": SPEC["nodes"]}
        spec = load_spec(data)
        assert spec.workflow.durability == value

    # 验证“无效值`rejected`”场景下的预期行为。
    def test_invalid_value_rejected(self):
        from flowagent.errors import ConfigError

        data = {
            "workflow": {"name": "d", "start": "a", "durability": "never"},
            "nodes": SPEC["nodes"],
        }
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“默认项判断是否`exit`”场景下的预期行为。
    def test_default_is_exit(self):
        assert load_spec(SPEC).workflow.durability == "exit"

    # 验证“`engine``passes``durability``invoke`”场景下的预期行为。
    def test_engine_passes_durability_to_invoke(self, monkeypatch):
        engine = WorkflowEngine(load_spec(SPEC), MockLLM())
        captured = {}

        # 处理`fake``invoke`相关逻辑。
        def fake_invoke(initial, config, **kwargs):
            captured.update(kwargs)
            return {"vars": {"msg": "ok"}, "trace": [], "loop_counts": {}, "error": None}

        monkeypatch.setattr(engine.graph, "invoke", fake_invoke)
        engine.run({"x": "1"})
        assert captured["durability"] == "exit"

    # 验证“`engine``uses`已配置`durability`”场景下的预期行为。
    def test_engine_uses_configured_durability(self):
        data = {
            "workflow": {"name": "d", "start": "a", "durability": "all"},
            "nodes": SPEC["nodes"],
        }
        engine = WorkflowEngine(load_spec(data), MockLLM())
        assert engine.durability == "all"
        final = engine.run({"x": "全量"})
        assert final["vars"]["msg"] == "值:全量"

    # 验证“`durability`应用流式处理”场景下的预期行为。
    def test_durability_applies_to_stream(self):
        captured = {}

        # 处理`fake`流式处理相关逻辑。
        def fake_stream(initial, config, **kwargs):
            captured.update(kwargs)
            yield {"vars": {}, "trace": [], "loop_counts": {}, "error": None}

        engine = WorkflowEngine(load_spec(SPEC), MockLLM())
        engine.graph.stream = fake_stream
        list(engine.stream({"x": "1"}))
        assert captured["durability"] == "exit"


class TestConcurrency:
    def test_parallel_runs_same_engine_distinct_threads(self):
        """同一引擎实例被多线程并发调用（不同 thread_id）时应互不串扰。"""
        engine = WorkflowEngine(load_spec(SPEC), MockLLM())

        # 运行`one`。
        def run_one(i: int):
            final = engine.run({"x": str(i)}, thread_id=f"conc-{i}")
            return i, final["vars"]["msg"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run_one, range(16)))

        for i, msg in results:
            assert msg == f"值:{i}"
        # 各 thread 的检查点互相独立
        snap = engine.get_state("conc-7").values
        assert snap["vars"]["msg"] == "值:7"
