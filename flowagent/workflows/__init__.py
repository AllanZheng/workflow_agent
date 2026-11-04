"""业务工作流：普通 Python 函数注册表项（不引入节点基类或通用引擎）。

M1 范围：chat 问答（共享 Agent 循环 + 可选 Skill/MCP）。
群分析/会商/知识在 M2 落地；终端业务工具包装在 M2-3 接入 chat。
"""
from flowagent.workflows.chat import DEFAULT_CHAT_SYSTEM_PROMPT, build_chat_workflow

__all__ = ["DEFAULT_CHAT_SYSTEM_PROMPT", "build_chat_workflow"]
