"""LLM 工厂：OpenAI 兼容接口优先，无 key 自动降级为 Mock。

环境变量约定：
- FLOWAGENT_LLM: auto（默认）/ mock / real
- OPENAI_API_KEY: API 密钥（GLM/DeepSeek/OpenAI/网关 等任一兼容服务）
- OPENAI_BASE_URL: 兼容端点，如 https://open.bigmodel.cn/api/paas/v4
- LLM_MODEL: 模型名，如 glm-4.6 / deepseek-chat / gpt-4o-mini
"""

import os

from langchain_core.messages import AIMessage, BaseMessage

from .errors import ConfigError

MOCK_DEFAULT_REPLY = "MOCK-OK"


class MockLLM:
    """确定性离线 LLM 替身。

    真实内容由节点级 mock_response 决定（见 nodes.execute_node），
    本类只负责兜底返回固定文案，保证整条工作流可离线跑通、测试可复现。
    """

    name = "mock"

    # 初始化对象状态并建立所需依赖。
    def __init__(self, default: str = MOCK_DEFAULT_REPLY):
        self.default = default

    # 处理`invoke`相关逻辑。
    def invoke(self, messages: list[BaseMessage], config=None, **kwargs) -> AIMessage:
        return AIMessage(content=self.default)


def create_llm(provider: str | None = None):
    """根据 provider/环境变量创建 LLM 客户端。

    - mock:  永远返回 MockLLM
    - auto:  有 OPENAI_API_KEY 则走真实 OpenAI 兼容端点，否则降级 MockLLM
    - real:  强制真实端点，缺 key 直接报错
    """
    provider = (provider or os.getenv("FLOWAGENT_LLM", "auto")).strip().lower()
    if provider == "mock":
        return MockLLM()

    if provider not in ("auto", "real"):
        raise ConfigError(f"未知 FLOWAGENT_LLM provider: {provider}（可选 auto/mock/real）")

    api_key = os.getenv("OPENAI_API_KEY")
    if provider == "auto" and not api_key:
        return MockLLM()
    if not api_key:
        raise ConfigError("provider=real 需要环境变量 OPENAI_API_KEY")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:  # pragma: no cover - requirements 已包含
        raise ConfigError("真实 LLM 需要安装 langchain-openai") from e

    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )
