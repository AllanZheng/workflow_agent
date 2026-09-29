"""LLM 输出解析器：text / json / lines / classifier。"""

import json
import re

from .errors import OutputParseError


# 解析文本。
def parse_text(content: str) -> str:
    return content.strip() if isinstance(content, str) else str(content)


# 解析`lines`。
def parse_lines(content: str) -> list:
    text = parse_text(content)
    return [line.strip() for line in text.splitlines() if line.strip()]


# 处理`strip``code``fence`相关逻辑。
def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*[ \t]*\n?", "", text)
        text = re.sub(r"\n?[ \t]*```\s*$", "", text)
    return text


def parse_json(content: str):
    """从 LLM 输出中提取第一个合法的 JSON 对象/数组。

    容忍 markdown 代码围栏与前后缀说明文字。
    """
    text = _strip_code_fence(parse_text(content))
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise OutputParseError(f"无法从LLM输出解析JSON: {content[:200]!r}")


def parse_classifier(content: str, labels: list) -> str:
    """在输出文本中查找分类标签（长标签优先，避免前缀误命中）。"""
    text = parse_text(content)
    for label in sorted(labels, key=len, reverse=True):
        if label in text:
            return label
    raise OutputParseError(f"输出中未找到任何标签 {labels}: {text[:200]!r}")


def validate_schema(value, schema: dict) -> None:
    """按 output_schema 校验已解析的 JSON 输出（失败关闭）。

    只支持轻量字段规范（type/min/max/enum/optional，见 config.OutputFieldSpec），
    不实现完整 JSON Schema——配置驱动的 YAML 里够用且可读。
    校验失败抛 OutputParseError：让节点走 retries/on_error/失败路径，
    绝不允许畸形输出静默落入“默认通过”分支（评审 4.2）。
    """
    if not isinstance(value, dict):
        raise OutputParseError(f"output_schema 要求顶层 JSON 对象，实际为 {type(value).__name__}")
    for key, rule in schema.items():
        item = value.get(key)
        if item is None:
            if getattr(rule, "optional", False):
                continue
            raise OutputParseError(f"输出缺少必填字段 {key!r}")
        rtype = getattr(rule, "type", "string")
        # 注意：bool 是 int 的子类，数值类型必须显式排除布尔
        if rtype == "string" and not isinstance(item, str):
            raise OutputParseError(f"字段 {key!r} 应为 string，实际为 {type(item).__name__}")
        if rtype == "boolean" and not isinstance(item, bool):
            raise OutputParseError(f"字段 {key!r} 应为 boolean，实际为 {type(item).__name__}")
        if rtype == "integer" and (isinstance(item, bool) or not isinstance(item, int)):
            raise OutputParseError(f"字段 {key!r} 应为 integer，实际为 {type(item).__name__}")
        if rtype == "number" and (isinstance(item, bool) or not isinstance(item, (int, float))):
            raise OutputParseError(f"字段 {key!r} 应为 number，实际为 {type(item).__name__}")
        is_num = isinstance(item, (int, float)) and not isinstance(item, bool)
        rmin, rmax = getattr(rule, "min", None), getattr(rule, "max", None)
        if is_num and rmin is not None and item < rmin:
            raise OutputParseError(f"字段 {key!r}={item!r} 低于下限 {rmin}")
        if is_num and rmax is not None and item > rmax:
            raise OutputParseError(f"字段 {key!r}={item!r} 超过上限 {rmax}")
        renum = getattr(rule, "enum", None)
        if renum is not None and item not in renum:
            raise OutputParseError(f"字段 {key!r}={item!r} 不在枚举 {renum} 内")


def parse_output(content: str, parser: str, labels: list | None = None):
    """按解析器名称分发。"""
    labels = labels or []
    if parser == "text":
        return parse_text(content)
    if parser == "json":
        return parse_json(content)
    if parser == "lines":
        return parse_lines(content)
    if parser == "classifier":
        return parse_classifier(content, labels)
    raise OutputParseError(f"未知解析器: {parser}")
