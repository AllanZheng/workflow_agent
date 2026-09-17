"""单元测试：LLM 工厂与 MockLLM。"""

import pytest

from flowagent.errors import ConfigError
from flowagent.llm import MockLLM, create_llm


class TestMockLLM:
    def test_invoke_returns_aimessage(self, mock_llm):
        resp = mock_llm.invoke([object()])
        assert resp.content == "MOCK-OK"

    def test_custom_default(self):
        assert MockLLM(default="X").invoke([]).content == "X"

    def test_named_mock(self):
        assert MockLLM().name == "mock"


class TestFactory:
    def test_explicit_mock(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert isinstance(create_llm("mock"), MockLLM)

    def test_auto_without_key_falls_back_to_mock(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert isinstance(create_llm("auto"), MockLLM)

    def test_auto_with_key_uses_openai_compatible(self, monkeypatch):
        from langchain_openai import ChatOpenAI

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        monkeypatch.setenv("LLM_MODEL", "glm-4.6")
        llm = create_llm("auto")
        assert isinstance(llm, ChatOpenAI)

    def test_real_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError):
            create_llm("real")

    def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError):
            create_llm("claude")

    def test_env_default_is_auto(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_LLM", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert isinstance(create_llm(), MockLLM)
