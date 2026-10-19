"""flowagent：业务工作流智能体（V3 原生内核 + 旧 YAML 引擎并存）。

包入口必须轻量：`import flowagent` 不载入旧引擎/API/SDK（技术方案 §5.1）。
旧引擎导出经 PEP 562 惰性解析，保持既有调用方兼容；M4 清退后移除。
"""

__version__ = "0.10.0"  # 版本单一来源：pyproject 动态读取此值

from .errors import (  # noqa: E402 - 轻量模块，仅异常类型
    ConfigError,
    FlowAgentError,
    OutputParseError,
    TemplateError,
    ToolError,
)

# 旧引擎/配置导出的惰性映射（首次访问才 import 对应模块）。
_LEGACY_EXPORTS = {
    "WorkflowSpec": (".config", "WorkflowSpec"),
    "load_spec": (".config", "load_spec"),
    "load_spec_file": (".config", "load_spec_file"),
    "WorkflowEngine": (".engine", "WorkflowEngine"),
    "build_graph": (".engine", "build_graph"),
    "MockLLM": (".llm", "MockLLM"),
    "create_llm": (".llm", "create_llm"),
    "WFState": (".state", "WFState"),
}

__all__ = [
    "WorkflowSpec", "load_spec", "load_spec_file", "WorkflowEngine", "build_graph",
    "MockLLM", "create_llm", "WFState",
    "ConfigError", "FlowAgentError", "OutputParseError", "TemplateError", "ToolError",
    "__version__",
]


def __getattr__(name):  # PEP 562：惰性解析旧导出
    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
