"""多级人工审批链：连续多个 human 节点应依次暂停/恢复。"""


from flowagent import WorkflowEngine, load_spec
from flowagent.llm import MockLLM

CHAIN_SPEC = {
    "workflow": {"name": "chain", "start": "lead_approve", "recursion_limit": 30},
    "nodes": {
        "lead_approve": {
            "type": "human",
            "message": "主管审批：{ticket}",
            "options": ["同意", "驳回"],
            "output_key": "lead_decision",
            "next": {
                "field": "lead_decision",
                "rules": [{"when": {"op": "eq", "value": "同意"}, "goto": "manager_approve"}],
                "default": "rejected",
            },
        },
        "manager_approve": {
            "type": "human",
            "message": "经理复核：{ticket}（主管已同意）",
            "options": ["批准", "否决"],
            "output_key": "manager_decision",
            "next": {
                "field": "manager_decision",
                "rules": [{"when": {"op": "eq", "value": "批准"}, "goto": "done"}],
                "default": "rejected",
            },
        },
        "done": {
            "type": "template",
            "template": "两级审批通过：{ticket}",
            "output_key": "msg",
            "next": "END",
        },
        "rejected": {
            "type": "template",
            "template": "审批未通过：{ticket}",
            "output_key": "msg",
            "next": "END",
        },
    },
}


class TestHumanChain:
    # 验证“`two``stage``approval``chain`”场景下的预期行为。
    def test_two_stage_approval_chain(self):
        engine = WorkflowEngine(load_spec(CHAIN_SPEC), MockLLM())
        # 第一级：主管
        p1 = engine.run({"ticket": "T-9"}, thread_id="chain1")
        assert WorkflowEngine.waiting_on(p1)["question"] == "主管审批：T-9"
        # 主管同意 → 第二级：经理再次暂停
        p2 = engine.resume("同意", thread_id="chain1")
        payload = WorkflowEngine.waiting_on(p2)
        assert payload["question"] == "经理复核：T-9（主管已同意）"
        assert payload["options"] == ["批准", "否决"]
        # 经理批准 → 完成
        final = engine.resume("批准", thread_id="chain1")
        assert final["vars"]["msg"] == "两级审批通过：T-9"
        assert final["vars"]["lead_decision"] == "同意"
        assert final["vars"]["manager_decision"] == "批准"

    # 验证“`chain``rejected``at``second``stage`”场景下的预期行为。
    def test_chain_rejected_at_second_stage(self):
        engine = WorkflowEngine(load_spec(CHAIN_SPEC), MockLLM())
        engine.run({"ticket": "T-8"}, thread_id="chain2")
        engine.resume("同意", thread_id="chain2")
        final = engine.resume("否决", thread_id="chain2")
        assert final["vars"]["msg"] == "审批未通过：T-8"

    # 验证“`chain``rejected``at`首个`stage``skips``manager`”场景下的预期行为。
    def test_chain_rejected_at_first_stage_skips_manager(self):
        engine = WorkflowEngine(load_spec(CHAIN_SPEC), MockLLM())
        engine.run({"ticket": "T-7"}, thread_id="chain3")
        final = engine.resume("驳回", thread_id="chain3")
        assert final["vars"]["msg"] == "审批未通过：T-7"
        assert "manager_decision" not in final["vars"]
        assert final["loop_counts"].get("manager_approve", 0) == 0
