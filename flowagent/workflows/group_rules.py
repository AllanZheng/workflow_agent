"""群取数控制规则：目标/时间/分类由代码决定，模型参数仅进诊断。

落点（技术方案 §4.1–§4.3）：
- 原问题 = 真实用户输入（或群助手透传的 original_user_message）；
  不扫描平台附加材料/Skill 正文来决定控制参数。
- 群：可信目标优先；点名其他群须解析一致；歧义/无权限/未找到 → needs_input。
- 时间：显式表达解析为 [start, end_exclusive)，最长 31 天；无表达 → 配置默认窗；
  模型给的 days/日期只进诊断，不采用（有意改变 #1）。
- 分类：用户明确 2–8 类（别名归一）+ 追加"未归类"；越界/解析失败 → needs_input；
  无表达 → 配置默认口径。
- 澄清：最多两次；continuation v1 往返与逐字段合并；耗尽 → CLARIFICATION_EXHAUSTED。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from flowagent.core.contracts import (
    Clarification,
    ErrorCode,
    FlowAgentCoreError,
    Status,
    build_continuation,
    merge_continuation,
)

DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_TAXONOMY = ("问题", "通知", "讨论", "闲聊", "其他")
UNCLASSIFIED = "未归类"
MAX_WINDOW_DAYS = 31
DEFAULT_BATCH_MESSAGES = 100
DEFAULT_BATCH_CHARS = 12_000

_THIS_GROUP = re.compile(r"本群|这个群|我们群|群里|该群|当前群")
# 用户明确分类口径："按 X类、Y类 统计/分类/分析"。最少 2 类、最多 8 类。
_TAXONOMY_PHRASE = re.compile(
    r"(?:按|按照|分为|分成|划分)(?P<cats>[^。；，\n]{2,60}?)(?:分类|统计|分析|归纳|汇总)"
)
_CATEGORY_SPLIT = re.compile(r"[、,，/／\s和与]+")
_CATEGORY_TOKEN = re.compile(r"^[^\d]{1,12}$")


@dataclass
class GroupControl:
    """一次群分析的有效控制参数（全部由代码确定）。"""

    conversation_id: int | None = None
    group_name: str = ""
    start: str | None = None            # ISO 带时区；None=待定（需澄清或默认窗）
    end_exclusive: str | None = None
    taxonomy: tuple[str, ...] = ()
    timezone: str = DEFAULT_TIMEZONE
    reference_time: str = ""
    # 诊断（有界；不含凭据或群材料）。
    diagnostics: dict[str, Any] = field(default_factory=dict)


class NeedsClarification(FlowAgentCoreError):
    """缺必要条件：携带已构造的 Clarification，Runner 终结为 needs_input。"""

    def __init__(self, clarification: Clarification) -> None:
        super().__init__(ErrorCode.INVALID_INPUT, clarification.reason)
        self.clarification = clarification


# ---------------------------------------------------------------- 群目标

def resolve_group_target(
    *,
    trusted_target: Any,
    original_question: str,
    call_tool: Any = None,  # async (name, args) -> Any；平台解析入口
) -> tuple[int | None, str, dict[str, Any]]:
    """确定目标群：可信目标 > 本群指代 > 名称解析；歧义/未找到交给澄清。

    返回 (conversation_id, group_name, diagnostics)。call_tool 为 None 时
    名称解析不可用——除可信目标/本群指代外一律 needs_input。
    """
    diag: dict[str, Any] = {"source": "trusted_target" if trusted_target else "none"}
    if trusted_target is not None and getattr(trusted_target, "conversation_id", None):
        cid = int(trusted_target.conversation_id)
        if _named_other_group(original_question, cid):
            # 点名其他群：不静默分析当前群，须澄清确认（§4.2）。
            raise _clarify_group(original_question)
        return cid, "", diag
    if _named_other_group_probe(original_question):
        # 点名了某个群但无可信目标：必须解析；解析不了就澄清。
        raise _clarify_group(original_question)
    return None, "", diag


def _named_other_group_probe(question: str) -> bool:
    """出现"XX群"形态的点名（且不是纯本群指代）。"""
    text = _THIS_GROUP.sub("", question)
    return bool(re.search(r"[一-鿿\w]{1,20}群(里|内|中)?", text))


def _named_other_group(question: str, trusted: int) -> bool:
    """有可信目标但用户点名了别的群——不静默改目标，需澄清确认。"""
    return _named_other_group_probe(question) and not _THIS_GROUP.search(question)


def _clarify_group(question: str) -> NeedsClarification:
    return NeedsClarification(Clarification(
        reason="group_unresolved",
        missing_fields=["目标群"],
        hints=["请说明要分析的群名称或群 ID；如指本群请写\"本群\""],
    ))


# ---------------------------------------------------------------- 时间窗

def resolve_window(
    *,
    original_question: str,
    reference_time: datetime,
    timezone_name: str = DEFAULT_TIMEZONE,
    default_days: int = 7,
    supplement: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """解析时间窗 → (start, end_exclusive)，均 ISO 带时区。

    - 显式表达：绝对区间 / 相对（今天/昨天/前天/近N天/近N小时/本周/上周/本月/上月）。
    - 非法、倒序、>31 天、疑似但无法解析 → 抛 NeedsClarification（不截断、不采模型值）。
    - 无时间表达 → 配置默认窗（近 N 天滚动区间），source=default。
    - supplement（续接补充）里的显式表达优先于原始问题。
    """
    diag: dict[str, Any] = {}
    for source_name, text in (("supplement", str((supplement or {}).get("time") or "")),
                              ("question", original_question)):
        if not text:
            continue
        window = _parse_explicit_window(text, reference_time)
        if window is None:
            continue  # 该文本无显式时间表达：看下一来源/默认窗
        diag["window_source"] = source_name
        if isinstance(window, NeedsClarification):
            if source_name == "question":
                raise window
            continue
        start, end, detail = window
        diag.update(detail)
        return start, end, diag
    # 无显式表达：配置默认窗（有意改变 #1：模型 days 不采用）。
    end = reference_time
    start = reference_time - timedelta(days=default_days)
    diag.update({"window_source": "default", "default_days": default_days})
    return _iso(start), _iso(end), diag


def _parse_explicit_window(
    text: str, now: datetime
) -> tuple[str, str, dict[str, Any]] | NeedsClarification:
    """显式窗口解析；返回 (start, end_exclusive, detail) 或澄清。"""
    # 绝对日期/时间区间（复用既有语义：至/到/~；无时区按配置时区）。
    m = re.search(
        r"(?P<sd>\d{4}(?:-\d{1,2}-\d{1,2}|年\d{1,2}月\d{1,2}日))\s*"
        r"(?:(?P<sc>\d{1,2}[:：]\d{2}))?\s*(?:至|到|~|～|—)\s*"
        r"(?:(?P<ed>\d{4}(?:-\d{1,2}-\d{1,2}|年\d{1,2}月\d{1,2}日))\s*)?"
        r"(?:(?P<ec>\d{1,2}[:：]\d{2}))?", text)
    if m and (m["sc"] or m["ec"]):
        try:
            start = _dt(m["sd"], m["sc"] or "00:00", now.tzinfo)
            end_date = m["ed"] or m["sd"]
            if not m["ec"]:
                # 纯日期区间：结束按次日零点（开区间）。
                end = _dt(end_date, "00:00", now.tzinfo) + timedelta(days=1)
            else:
                end = _dt(end_date, m["ec"], now.tzinfo)
        except ValueError:
            return _clarify_time("日期无法解析")
        return _checked(start, end, {"expr": "absolute"})
    # 相对表达。
    if (mm := re.search(r"(?:最近|过去|近)\s*(\d+)\s*天", text)):
        days = int(mm.group(1))
        if not 1 <= days <= MAX_WINDOW_DAYS:
            return _clarify_time(f"天数超出 1..{MAX_WINDOW_DAYS}")
        end = now
        return _checked(now - timedelta(days=days), end, {"expr": "relative_days", "days": days})
    if (mm := re.search(r"(?:最近|过去|近)\s*(\d+)\s*(?:个?小时|h|H)", text)):
        hours = int(mm.group(1))
        if not 1 <= hours <= MAX_WINDOW_DAYS * 24:
            return _clarify_time("小时数超出范围")
        return _checked(now - timedelta(hours=hours), now, {"expr": "relative_hours"})
    for phrase, start_fn in (
        ("今天", lambda: now.replace(hour=0, minute=0, second=0, microsecond=0)),
        ("昨天", lambda: now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)),
        ("前天", lambda: now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)),
    ):
        if phrase in text:
            day_start = start_fn()
            return _checked(day_start, day_start + timedelta(days=1), {"expr": phrase})
    if "本周" in text:
        monday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now.weekday())
        return _checked(monday, now, {"expr": "this_week"})
    if "上周" in text:
        monday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now.weekday() + 7)
        return _checked(monday, monday + timedelta(days=7), {"expr": "last_week"})
    if "本月" in text:
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return _checked(month_start, now, {"expr": "this_month"})
    if "上月" in text:
        first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_end = first - timedelta(microseconds=1)
        last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return _checked(last_month_start, first, {"expr": "last_month"})
    # 疑似时间但未命中任何规则（如"国庆期间"）：澄清而非猜默认。
    if re.search(r"\d+\s*[天日小时]|周|月|号|点|至今|以来|期间", text):
        return _clarify_time("疑似时间表达但无法解析")
    return None  # 无时间表达


def _checked(start: datetime, end: datetime, detail: dict[str, Any]):
    if end <= start:
        return _clarify_time("时间倒序或为零长度")
    if end - start > timedelta(days=MAX_WINDOW_DAYS):
        return _clarify_time(f"窗口超过 {MAX_WINDOW_DAYS} 天")
    detail["span_days"] = round((end - start).total_seconds() / 86400, 2)
    return _iso(start), _iso(end), detail


def _clarify_time(reason: str) -> NeedsClarification:
    return NeedsClarification(Clarification(
        reason="time_window_invalid",
        missing_fields=["时间范围"],
        hints=["请给出明确的起止（如\"2026年10月16日00:00至10月18日00:00\"或\"最近7天\"）；"
               f"窗口最长 {MAX_WINDOW_DAYS} 天"],
    ))


def _dt(date: str, clock: str, tzinfo) -> datetime:
    y, mo, d = map(int, re.findall(r"\d+", date))
    h, mi = map(int, re.split(r"[:：]", clock)[:2])
    return datetime(y, mo, d, h, mi, tzinfo=tzinfo)


def _iso(value: datetime) -> str:
    return value.isoformat()


# ---------------------------------------------------------------- 分类口径

def resolve_taxonomy(
    *,
    original_question: str,
    default: tuple[str, ...] = DEFAULT_TAXONOMY,
    aliases: dict[str, str] | None = None,
    supplement: dict[str, Any] | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """分类口径：用户明确 2–8 类（去重+别名归一）+ 未归类；否则配置默认。

    "按时间顺序总结""按人汇总""分成两部分讲"不是类别列表，不触发分类澄清。
    """
    diag: dict[str, Any] = {}
    for source_name, text in (("supplement", str((supplement or {}).get("taxonomy") or "")),
                              ("question", original_question)):
        if not text:
            continue
        found = _taxonomy_from_text(text)
        if isinstance(found, NeedsClarification):
            if source_name == "question":
                raise found
            continue
        if found is not None:
            categories, normalized = found
            diag.update({"taxonomy_source": source_name, "requested": normalized,
                         "alias_applied": bool(aliases and normalized)})
            if aliases:
                categories = tuple(aliases.get(c, c) for c in categories)
                categories = _dedup(categories)
            return (*categories, UNCLASSIFIED), diag
    diag.update({"taxonomy_source": "default"})
    # 默认口径按配置原样（不加未归类）；仅用户自定义口径追加未归类。
    return tuple(default), diag


def _taxonomy_from_text(text: str):
    """提取类别列表；负样例（非类别短语）返回 None；越界/非法返回澄清。"""
    m = _TAXONOMY_PHRASE.search(text)
    if not m:
        return None
    raw = m["cats"]
    # 排除明显不是类别列表的表达。
    if re.search(r"时间顺序|按人|时间|人员|两部分|重点|顺序", raw):
        return None
    parts = [p.strip() for p in _CATEGORY_SPLIT.split(raw) if p.strip()]
    normalized: list[str] = []
    for part in parts:
        # 类别名保持原文（"风险类"含"类"字是名称的一部分；别名归一交给配置）。
        if not _CATEGORY_TOKEN.match(part):
            return _clarify_taxonomy()
        normalized.append(part)
    normalized = _dedup(normalized)
    if not 2 <= len(normalized) <= 8:
        return _clarify_taxonomy()
    return tuple(normalized), normalized


def _clarify_taxonomy() -> NeedsClarification:
    return NeedsClarification(Clarification(
        reason="taxonomy_invalid",
        missing_fields=["分类口径"],
        hints=["请明确列出 2 到 8 个类别，如\"按风险类、建议类统计\""],
    ))


def _dedup(items) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------- 控制参数总装

def build_control(
    *,
    original_question: str,
    trusted_target: Any,
    reference_time: datetime,
    config: dict[str, Any] | None = None,
    supplement: dict[str, Any] | None = None,
    model_arguments: dict[str, Any] | None = None,
) -> GroupControl:
    """组装一次群分析的有效控制参数；任一必要条件缺失抛 NeedsClarification。"""
    config = config or {}
    tz_name = config.get("timezone", DEFAULT_TIMEZONE)
    conversation_id, group_name, group_diag = resolve_group_target(
        trusted_target=trusted_target, original_question=original_question)
    start, end, time_diag = resolve_window(
        original_question=original_question, reference_time=reference_time,
        timezone_name=tz_name, default_days=int(config.get("default_days", 7)),
        supplement=supplement)
    taxonomy, tax_diag = resolve_taxonomy(
        original_question=original_question,
        default=tuple(config.get("taxonomy") or DEFAULT_TAXONOMY),
        aliases=config.get("taxonomy_aliases") or None,
        supplement=supplement)
    control = GroupControl(
        conversation_id=conversation_id, group_name=group_name,
        start=start, end_exclusive=end, taxonomy=taxonomy, timezone=tz_name,
        reference_time=_iso(reference_time),
        diagnostics={
            "group": group_diag, "window": time_diag, "taxonomy": tax_diag,
            # 模型参数只进诊断（有意改变 #1）。
            "model_arguments": model_arguments or None,
        },
    )
    return control


def control_to_fetch_arguments(control: GroupControl) -> dict[str, Any]:
    """有效控制参数 → fetch_group_analysis_input 调用参数。"""
    return {
        "conversation_id": control.conversation_id,
        "start": control.start,
        "end": control.end_exclusive,
    }


def supplement_from_user_answer(answer: str) -> dict[str, Any]:
    """用户补充文本 → 续接字段（时间/分类/群原文透传，由解析规则消费）。"""
    return {"time": answer, "taxonomy": answer}
