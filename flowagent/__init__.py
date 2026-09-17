"""flowagent：配置驱动的 LangGraph 工作流智能体引擎。"""

from .config import WorkflowSpec, load_spec, load_spec_file
from .engine import WorkflowEngine, build_graph
from .errors import ConfigError, FlowAgentError, OutputParseError, TemplateError, ToolError
from .llm import MockLLM, create_llm
from .state import WFState

__version__ = "0.9.0"  # 版本单一来源：pyproject 动态读取此值（评审 4.8）

__all__ = [
    "WorkflowSpec",
    "load_spec",
    "load_spec_file",
    "WorkflowEngine",
    "build_graph",
    "MockLLM",
    "create_llm",
    "WFState",
    "ConfigError",
    "FlowAgentError",
    "OutputParseError",
    "TemplateError",
    "ToolError",
    "__version__",
]
