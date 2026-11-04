"""flowagent：业务工作流智能体（V3 原生内核）。

包入口保持轻量：不载入 API/SDK/任何旧执行模块（技术方案 §5.1）。
"""

__version__ = "0.11.0"  # 版本单一来源：pyproject 动态读取此值

from .errors import (  # noqa: E402 - 仅异常类型
    ConfigError,
    FlowAgentError,
    OutputParseError,
    TemplateError,
    ToolError,
)

__all__ = [
    "ConfigError", "FlowAgentError", "OutputParseError", "TemplateError", "ToolError",
    "__version__",
]
