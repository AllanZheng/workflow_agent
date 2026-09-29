"""集成测试：human 节点（LangGraph interrupt 人工介入）。"""

import pytest

from flowagent import WorkflowEngine, load_spec
from flowagent.errors import ConfigError
from flowagent.llm import MockLLM

HITL_SPEC = {
    "workflow": {"name": "hitl", "start": "ask", "recursion_limit": 20},
    "nodes": {
        "ask": {
            "type": "human",
            "message": "请审批：{ticket}",
            "output_key": "decision",
            "next": {
                "field": "decision",
                "rules": [{"when": {"op": "eq", "value": "批准"}, "goto": "END"}],
                "default": "redo",
            },
        },
        "redo": {
            "type": "template",
            "template": "已驳回:{ticket}",
            "output_key": "msg",
            "next": "END",
        },
    },
}


# 处理`engine`相关逻辑。
@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine(load_spec(HITL_SPEC), MockLLM())


class TestHumanNode:
    # 验证“`pause`生成`interrupt`载荷”场景下的预期行为。
    def test_pause_produces_interrupt_payload(self, engine):
        paused = engine.run({"ticket": "T-1"}, thread_id="h1")
        payload = WorkflowEngine.waiting_on(paused)
        assert isinstance(payload, dict)
        assert payload["node"] == "ask"
        assert payload["question"] == "请审批：T-1"

    # 验证“`resume`包含拒绝路由`redo`”场景下的预期行为。
    def test_resume_with_reject_routes_to_redo(self, engine):
        engine.run({"ticket": "T-2"}, thread_id="h2")
        final = engine.resume("拒绝", thread_id="h2")
        assert final["vars"]["decision"] == "拒绝"
        assert final["vars"]["msg"] == "已驳回:T-2"
        # interrupt 暂停不提交执行痕迹；恢复后节点完整执行一次
        assert final["loop_counts"]["ask"] == 1

    # 验证“`resume`包含`approach`路由`end`”场景下的预期行为。
    def test_resume_with_approach_routes_to_end(self, engine):
        engine.run({"ticket": "T-3"}, thread_id="h3")
        final = engine.resume("批准", thread_id="h3")
        assert final["vars"]["decision"] == "批准"
        assert "msg" not in final["vars"]

    # 验证“默认项输出密钥当`unset`”场景下的预期行为。
    def test_default_output_key_when_unset(self):
        spec = {
            "workflow": {"name": "d", "start": "ask"},
            "nodes": {
                "ask": {"type": "human", "message": "m", "next": "END"},
            },
        }
        engine = WorkflowEngine(load_spec(spec), MockLLM())
        paused = engine.run({}, thread_id="d1")
        assert WorkflowEngine.waiting_on(paused) is not None
        final = engine.resume("ok", thread_id="d1")
        assert final["vars"]["human_decision"] == "ok"

    # 验证“`waiting`启用无针对`normal`状态”场景下的预期行为。
    def test_waiting_on_none_for_normal_state(self):
        engine = WorkflowEngine(
            load_spec(
                {
                    "workflow": {"name": "n", "start": "a"},
                    "nodes": {
                        "a": {"type": "template", "template": "x", "output_key": "o", "next": "END"}
                    },
                }
            ),
            MockLLM(),
        )
        final = engine.run({})
        assert WorkflowEngine.waiting_on(final) is None
        assert WorkflowEngine.waiting_on({}) is None

    # 验证“`human`不包含消息`rejected`”场景下的预期行为。
    def test_human_without_message_rejected(self):
        with pytest.raises(ConfigError):
            load_spec(
                {
                    "workflow": {"name": "bad", "start": "a"},
                    "nodes": {"a": {"type": "human", "next": "END"}},
                }
            )

    def test_rerun_same_thread_uses_latest_inputs(self, engine):
        """同 thread 重跑即重启：新输入合并后重新渲染暂停问题。"""
        engine.run({"ticket": "first"}, thread_id="h5")
        paused = engine.run({"ticket": "second"}, thread_id="h5")
        assert WorkflowEngine.waiting_on(paused)["question"] == "请审批：second"


class TestHumanOptionEnforcement:
    """4.1：声明 options 即强校验——无效答复不进入业务分支，重新询问。"""

    OPT_SPEC = {
        "workflow": {"name": "opt", "start": "ask", "recursion_limit": 20},
        "nodes": {
            "ask": {
                "type": "human",
                "message": "请回复 批准 或 拒绝：{ticket}",
                "options": ["批准", "拒绝"],
                "output_key": "decision",
                "next": {
                    "field": "decision",
                    "rules": [{"when": {"op": "eq", "value": "批准"}, "goto": "END"}],
                    "default": "redo",
                },
            },
            "redo": {
                "type": "template",
                "template": "已驳回:{ticket}",
                "output_key": "msg",
                "next": "END",
            },
        },
    }

    # 验证“无效`answer``reinterrupts`包含错误`hint`”场景下的预期行为。
    def test_invalid_answer_reinterrupts_with_error_hint(self):
        eng = WorkflowEngine(load_spec(self.OPT_SPEC), MockLLM())
        eng.run({"ticket": "T-9"}, thread_id="ie1")
        # “不批准”不在 options 内：必须再次暂停并携带错误提示，而不是放行
        again = eng.resume("不批准", thread_id="ie1")
        payload = WorkflowEngine.waiting_on(again)
        assert payload is not None
        assert "不批准" in payload.get("error", "")
        assert "msg" not in again["vars"]

    # 验证“`second``attempt`包含有效`answer`完成”场景下的预期行为。
    def test_second_attempt_with_valid_answer_completes(self):
        eng = WorkflowEngine(load_spec(self.OPT_SPEC), MockLLM())
        eng.run({"ticket": "T-10"}, thread_id="ie2")
        eng.resume("不批准", thread_id="ie2")      # 无效 → 重新询问
        final = eng.resume("批准", thread_id="ie2")  # 有效 → 正常完成
        assert final["vars"]["decision"] == "批准"
        assert "msg" not in final["vars"]

    # 验证“无`options``keeps``free`文本”场景下的预期行为。
    def test_no_options_keeps_free_text(self):
        spec = {
            "workflow": {"name": "free", "start": "ask", "recursion_limit": 20},
            "nodes": {
                "ask": {
                    "type": "human",
                    "message": "任意回复",
                    "output_key": "d",
                    "next": "END",
                }
            },
        }
        eng = WorkflowEngine(load_spec(spec), MockLLM())
        eng.run({}, thread_id="ft1")
        final = eng.resume("不批准，理由略", thread_id="ft1")
        assert final["vars"]["d"] == "不批准，理由略"
