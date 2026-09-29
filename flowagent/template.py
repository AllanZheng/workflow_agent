"""提示词/模板渲染与变量取值。

模板语法：``{变量名}`` 或 ``{a.b.c}``（点路径取值）。
只替换上下文中真实存在的变量，其余花括号原样保留——
因此提示词里可以直接内嵌 JSON 示例而不会被误当变量解析。
"""

import json
import re

from .errors import TemplateError

_TOKEN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\}")
_MISSING = object()


# English: Return path.
def get_path(obj, path: str, default=None):
    """按点路径从嵌套 dict 取值，取不到返回 None。"""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


# English: Handle render template operations.
def render_template(text: str, ctx: dict) -> str:
    """将模板中的 {var}/{a.b} 替换为上下文取值；未知占位符原样保留。"""

    # 处理`repl`相关逻辑。
    # English: Handle repl operations.
    def repl(m: re.Match) -> str:
        key = m.group(1)
        val = get_path(ctx, key, _MISSING)
        if val is _MISSING:
            return m.group(0)
        if val is None:
            return ""
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)

    try:
        return _TOKEN.sub(repl, text)
    except Exception as e:  # pragma: no cover - re.sub 对 str 输入基本不会抛
        raise TemplateError(f"模板渲染失败: {e}") from e
