"""flowagent 统一异常类型。"""


class FlowAgentError(Exception):
    """flowagent 所有自定义异常的基类。"""


class ConfigError(FlowAgentError):
    """工作流配置非法（YAML 解析失败、引用不存在的节点、字段缺失等）。"""


class TemplateError(FlowAgentError):
    """模板渲染失败。"""


class OutputParseError(FlowAgentError):
    """LLM 输出解析失败（JSON 不合法、分类标签未命中等）。"""


class ToolError(FlowAgentError):
    """工具节点执行失败。"""
