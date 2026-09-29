"""单元测试：LLM 工厂与 MockLLM。"""

import pytest

from flowagent.errors import ConfigError
from flowagent.llm import MockLLM, create_llm


class TestMockLLM:
    # 验证“`invoke``returns``aimessage`”场景下的预期行为。
    # English: Verify invoke returns aimessage.
    def test_invoke_returns_aimessage(self, mock_llm):
        resp = mock_llm.invoke([object()])
        assert resp.content == "MOCK-OK"

    # 验证“`custom`默认项”场景下的预期行为。
    # English: Verify custom default.
    def test_custom_default(self):
        assert MockLLM(default="X").invoke([]).content == "X"

    # 验证“`named``mock`”场景下的预期行为。
    # English: Verify named mock.
    def test_named_mock(self):
        assert MockLLM().name == "mock"


class TestFactory:
    # 验证“`explicit``mock`”场景下的预期行为。
    # English: Verify explicit mock.
    def test_explicit_mock(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert isinstance(create_llm("mock"), MockLLM)

    # 验证“`auto`不包含密钥`falls``back``mock`”场景下的预期行为。
    # English: Verify auto without key falls back to mock.
    def test_auto_without_key_falls_back_to_mock(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert isinstance(create_llm("auto"), MockLLM)

    # 验证“`auto`包含密钥`uses``openai``compatible`”场景下的预期行为。
    # English: Verify auto with key uses openai compatible.
    def test_auto_with_key_uses_openai_compatible(self, monkeypatch):
        from langchain_openai import ChatOpenAI

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        monkeypatch.setenv("LLM_MODEL", "glm-4.6")
        llm = create_llm("auto")
        assert isinstance(llm, ChatOpenAI)

    # 验证“`real`不包含密钥抛出”场景下的预期行为。
    # English: Verify real without key raises.
    def test_real_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError):
            create_llm("real")

    # 验证“未知提供方抛出”场景下的预期行为。
    # English: Verify unknown provider raises.
    def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError):
            create_llm("claude")

    # 验证“环境变量默认项判断是否`auto`”场景下的预期行为。
    # English: Verify env default is auto.
    def test_env_default_is_auto(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_LLM", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert isinstance(create_llm(), MockLLM)
