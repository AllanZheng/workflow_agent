"""可选集成测试：真实 LLM（配置 OPENAI_API_KEY 后自动启用，否则跳过）。

用法：
    OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4 \
    LLM_MODEL=glm-4.6 pytest tests/test_real_llm.py -v
"""

import os

import pytest

from flowagent import WorkflowEngine, load_spec_file
from flowagent.llm import create_llm

requires_real_llm = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="需要 OPENAI_API_KEY（可选真实 LLM 集成测试）"
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@requires_real_llm
class TestRealLLM:
    # 验证“`customer`服务分类与完成”场景下的预期行为。
    def test_customer_service_classifies_and_completes(self):
        engine = WorkflowEngine(load_spec_file(os.path.join(ROOT, "configs", "customer_service.yaml")))
        final = engine.run({"ticket": "收到的商品屏幕碎了，要求退款"})
        assert final["error"] is None
        assert final["vars"]["category"] in ("退款", "技术", "投诉", "咨询")
        assert "final_reply" in final["vars"]
        # 质检节点必须产出合法 JSON（score 为数值）
        review = final["vars"]["review"]
        assert isinstance(review, dict) and 1 <= float(review["score"]) <= 10

    # 验证“报告`writer``reaches`发布”场景下的预期行为。
    def test_report_writer_reaches_publish(self):
        engine = WorkflowEngine(load_spec_file(os.path.join(ROOT, "configs", "report_writer.yaml")))
        final = engine.run({"topic": "LLM 工作流引擎设计"})
        assert "final_report" in final["vars"]
        # 循环护栏必须生效：polish 至多执行 2 次
        assert final["loop_counts"].get("polish", 0) <= 2

    # 验证“工厂方法`returns``real`客户端”场景下的预期行为。
    def test_factory_returns_real_client(self):
        assert getattr(create_llm("auto"), "name", "") != "mock" or os.getenv("OPENAI_API_KEY") is None
