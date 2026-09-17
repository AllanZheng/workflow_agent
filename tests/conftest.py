from pathlib import Path

import pytest

from flowagent.llm import MockLLM


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _suite_tool_allowlist(monkeypatch):
    """测试套件默认白名单：内置工具 + tests 辅助模块。

    工具默认拒绝（评审 4.9）后，tests.flaky:* 等测试辅助工具需要显式放行；
    默认拒绝行为本身由 test_tools.py 用 monkeypatch.delenv 覆盖验证。
    """
    monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "flowagent.tools,tests")


@pytest.fixture(autouse=True)
def _suite_mock_llm(request, monkeypatch):
    """测试套件默认 Mock 模式：屏蔽外部 LLM 环境变量（OPENAI_API_KEY / FLOWAGENT_LLM）。

    功能测试必须离线确定——若宿主机导出了真实端点变量，跑真实网络的用例
    （如 TestInputHints 走完整工作流断言退出码）会因 LLM 输出不确定而偶发变红。
    真实 LLM 语义验证集中于 test_real_llm.py（显式豁免本 fixture，
    配置 OPENAI_API_KEY 后启用；其 skipif 在收集期读取环境，不受此影响）。
    """
    if Path(request.path).name == "test_real_llm.py":
        return
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FLOWAGENT_LLM", "auto")
