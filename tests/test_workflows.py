"""端到端测试：运行仓库自带的示例工作流配置（纯离线 Mock）。"""

import copy
from pathlib import Path

import pytest

from flowagent import WorkflowEngine, load_spec_file
from flowagent.llm import MockLLM

CS = "configs/customer_service.yaml"
RW = "configs/report_writer.yaml"
RA = "configs/refund_approval.yaml"


# 处理`cs``spec`相关逻辑。
@pytest.fixture
def cs_spec():
    return load_spec_file(CS)


# 处理`engine`工厂方法相关逻辑。
@pytest.fixture
def engine_factory(cs_spec):
    # 执行构建对应的流程。
    def _make() -> WorkflowEngine:
        return WorkflowEngine(copy.deepcopy(cs_spec), MockLLM())

    return _make


class TestCustomerService:
    # 验证“`refund``branch`”场景下的预期行为。
    def test_refund_branch(self, engine_factory):
        final = engine_factory().run({"ticket": "商品坏了，我要退款"})
        assert final["vars"]["category"] == "退款"
        assert "退款处理" in final["vars"]["answer"]
        assert "评分 9/10" in final["vars"]["final_reply"]
        assert [s["node"] for s in final["trace"]] == [
            "classify", "refund", "review", "finalize",
        ]

    # 验证“`tech``branch``uses`工具”场景下的预期行为。
    def test_tech_branch_uses_tool(self, engine_factory):
        spec = engine_factory().spec
        spec.nodes["classify"].mock_response = "技术"
        final = WorkflowEngine(spec, MockLLM()).run({"ticket": "软件无法登录怎么办"})
        assert final["vars"]["category"] == "技术"
        assert "登录" in final["vars"]["answer"]  # 知识库命中
        assert "diagnose" in [s["node"] for s in final["trace"]]

    # 验证“`complaint``branch``parallel``fanout`”场景下的预期行为。
    def test_complaint_branch_parallel_fanout(self, engine_factory):
        spec = engine_factory().spec
        spec.nodes["classify"].mock_response = "我要投诉"
        final = WorkflowEngine(spec, MockLLM()).run({"ticket": "服务态度太差"})
        vars_ = final["vars"]
        assert vars_["category"] == "投诉"
        assert vars_["sentiment"] == "负面"
        assert "补偿方案" in vars_["answer"]
        assert "内部通知" in vars_["notify"]
        nodes_run = [s["node"] for s in final["trace"]]
        assert set(["compensation", "notify_team", "review"]).issubset(set(nodes_run))

    # 验证“`faq`默认项`branch`”场景下的预期行为。
    def test_faq_default_branch(self, engine_factory):
        spec = engine_factory().spec
        spec.nodes["classify"].mock_response = "这是一个咨询问题"
        final = WorkflowEngine(spec, MockLLM()).run({"ticket": "你们几点上班"})
        assert final["vars"]["category"] == "咨询"
        assert "咨询答复" in final["vars"]["answer"]
        assert "faq" in [s["node"] for s in final["trace"]]

    # 验证“`review``loop``triggers`当`score``low`”场景下的预期行为。
    def test_review_loop_triggers_when_score_low(self, cs_spec):
        spec = copy.deepcopy(cs_spec)
        spec.nodes["classify"].mock_response = "退款"
        spec.nodes["review"].mock_response = '{"score": 5, "reason": "不合格"}'
        final = WorkflowEngine(spec, MockLLM()).run({"ticket": "退货"})
        # 循环护栏：revise 最多 2 次，之后 loop_exhaust 放行
        assert final["loop_counts"]["revise"] == 2
        assert final["loop_counts"]["review"] == 3
        assert [s["node"] for s in final["trace"]] == [
            "classify", "refund", "review", "revise", "review", "revise", "review", "finalize",
        ]


class TestReportWriter:
    # 验证“`full`运行包含`loop``guard`”场景下的预期行为。
    def test_full_run_with_loop_guard(self):
        spec = load_spec_file(RW)
        engine = WorkflowEngine(spec, MockLLM())
        final = engine.run({"topic": "智能体框架选型"})
        # mock 评审固定 6 分：polish 执行 2 次后被循环护栏放行
        assert final["loop_counts"]["polish"] == 2
        assert final["loop_counts"]["review"] == 3
        assert [s["node"] for s in final["trace"]] == [
            "plan", "search", "draft", "review", "polish", "review", "polish", "review", "publish",
        ]
        assert "final_report" in final["vars"]
        assert "主题调研报告" in final["vars"]["final_report"]

    # 验证“`high``score``skips``loop`”场景下的预期行为。
    def test_high_score_skips_loop(self):
        spec = load_spec_file(RW)
        spec.nodes["review"].mock_response = '{"score": 9, "reason": "优秀"}'
        final = WorkflowEngine(spec, MockLLM()).run({"topic": "T"})
        assert [s["node"] for s in final["trace"]] == [
            "plan", "search", "draft", "review", "publish",
        ]


RA = "configs/refund_approval.yaml"

HIGH_RISK_CHILD = """\
workflow: {name: risk_assessment, start: assess}
nodes:
  assess:
    type: llm
    system: 你是风控评估器，只输出：高风险 或 低风险。
    prompt: "评估：{summary}"
    parser: classifier
    labels: [高风险, 低风险]
    output_key: risk_level
    mock_response: 高风险
    next: END
"""


class TestRefundApproval:
    """综合示例（子图+HITL+重试+质检循环）的端到端验证。"""

    # 验证“`low`风险`auto`路径”场景下的预期行为。
    def test_low_risk_auto_path(self):
        engine = WorkflowEngine(load_spec_file(RA), MockLLM())
        final = engine.run({"ticket": "T-1", "amount": 50})
        assert final["vars"]["risk"]["risk_level"] == "低风险"
        assert [s["node"] for s in final["trace"]] == [
            "intake", "risk_check", "process", "qa", "notify",
        ]
        assert "退款处理完成" in final["vars"]["final_notice"]

    # 验证“`high`风险`approved``via``human`”场景下的预期行为。
    def test_high_risk_approved_via_human(self, tmp_path):
        child = tmp_path / "risk_high.yaml"
        child.write_text(HIGH_RISK_CHILD, encoding="utf-8")
        parent = tmp_path / "ra.yaml"
        parent.write_text(
            Path(RA).read_text(encoding="utf-8").replace(
                "workflow_file: subflows/risk.yaml", f"workflow_file: {child.as_posix()}"
            ),
            encoding="utf-8",
        )
        engine = WorkflowEngine(load_spec_file(parent), MockLLM())
        paused = engine.run({"ticket": "T-2", "amount": 99999}, thread_id="ra-h")
        assert WorkflowEngine.waiting_on(paused)["options"] == ["批准", "拒绝"]
        final = engine.resume("批准", thread_id="ra-h")
        nodes = [s["node"] for s in final["trace"]]
        assert nodes == ["intake", "risk_check", "approval", "process", "qa", "notify"]
        assert "退款处理完成" in final["vars"]["final_notice"]

    # 验证“`high`风险`rejected``via``human`”场景下的预期行为。
    def test_high_risk_rejected_via_human(self, tmp_path):
        child = tmp_path / "risk_high.yaml"
        child.write_text(HIGH_RISK_CHILD, encoding="utf-8")
        parent = tmp_path / "ra.yaml"
        parent.write_text(
            Path(RA).read_text(encoding="utf-8").replace(
                "workflow_file: subflows/risk.yaml", f"workflow_file: {child.as_posix()}"
            ),
            encoding="utf-8",
        )
        engine = WorkflowEngine(load_spec_file(parent), MockLLM())
        engine.run({"ticket": "T-3", "amount": 99999}, thread_id="ra-r")
        final = engine.resume("拒绝", thread_id="ra-r")
        assert [s["node"] for s in final["trace"]] == ["intake", "risk_check", "approval", "rejected"]
        assert "已驳回" in final["vars"]["final_notice"]

    # 验证“`qa`修订版本`loop``pipeline`”场景下的预期行为。
    def test_qa_revision_loop_in_pipeline(self):
        spec = load_spec_file(RA)
        spec.nodes["qa"].mock_response = '{"score": 5, "reason": "缺少到账时间"}'
        final = WorkflowEngine(spec, MockLLM()).run({"ticket": "T-4", "amount": 30})
        assert final["loop_counts"]["fix"] == 1  # 质检循环护栏 max_loops=1
        assert "补充说明" in final["vars"]["receipt"]


class TestApprovalNegativeSemantics:
    """4.1：否定式答复绝不进入批准分支（双层防护：options 强校验 + eq 路由）。"""

    # 验证“`bu``pizhun``blocked`按`options``then`拒绝”场景下的预期行为。
    def test_bu_pizhun_blocked_by_options_then_reject(self, project_root):
        # 第一层：approval.yaml 声明了 options，未知值被引擎拦截重新询问
        spec = load_spec_file(project_root / "configs" / "approval.yaml")
        eng = WorkflowEngine(spec, MockLLM())
        eng.run({"ticket": "大额退款", "amount": 9}, thread_id="neg1")
        again = eng.resume("不批准", thread_id="neg1")
        assert WorkflowEngine.waiting_on(again) is not None  # 仍在等待，未放行
        final = eng.resume("拒绝", thread_id="neg1")
        assert "已驳回" in final["vars"]["final"]
        assert "已批准" not in final["vars"]["final"]

    # 验证“`bu``pizhun`路由拒绝不包含`options`”场景下的预期行为。
    def test_bu_pizhun_routes_to_reject_without_options(self, project_root):
        # 第二层：即使未声明 options（自由文本），eq 也不匹配“不批准”，落入安全侧 default
        spec = load_spec_file(project_root / "configs" / "approval.yaml")
        spec.nodes["ask"].options = []  # 模拟未声明 options 的自由文本场景
        eng = WorkflowEngine(spec, MockLLM())
        eng.run({"ticket": "大额退款", "amount": 9}, thread_id="neg2")
        final = eng.resume("不批准", thread_id="neg2")
        assert "已驳回" in final["vars"]["final"]
        assert "已批准" not in final["vars"]["final"]
