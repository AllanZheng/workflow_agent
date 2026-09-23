"""im-tools-mcp：im 群消息只读 MCP 薄适配服务。

职责（技术方案 §3，D15）：
  - REST -> MCP 适配 + 游标翻页 + 时间窗过滤 + 语料格式化 + 上限保护；
  - 无 LLM、无状态、只读；
  - 调 im 现有 REST 接口（零改 im 代码），凭据由 MCP Edge 按请求注入。

对外两个工具（契约单一事实源见 tools.py）：
  - find_group(name_query) -> 候选群列表（最多 10 条）；
  - fetch_group_messages(conversation_id, start/end/days, segment_hours,
    max_chars, segment_index) -> 分段格式化语料 + 覆盖率。
"""

from .app import build_app
from .config import Settings
from .tools import FETCH_GROUP_MESSAGES_INPUT_SCHEMA, FIND_GROUP_INPUT_SCHEMA

__all__ = [
    "build_app",
    "Settings",
    "FIND_GROUP_INPUT_SCHEMA",
    "FETCH_GROUP_MESSAGES_INPUT_SCHEMA",
]
