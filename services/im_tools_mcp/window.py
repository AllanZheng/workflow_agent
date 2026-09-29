"""分析时间窗解析：把 LLM 透传的时间记号解析为 UTC 时间窗。

背景：flowagent 引擎没有时钟（初始变量仅 message/history/system_prompt/skills），
无法在受管工作流内计算"当前时间"。因此约定由本服务（有时钟）解析时间记号，
parse_args 节点只做"透传/规范化"，不做日期运算。

支持的记号（大小写不敏感，允许首尾空白）：
  - 相对：``now``、``now-7d``、``now-12h``（N 为正整数，单位 d/h）；
  - ISO：``2026-09-01``、``2026-09-01 08:00``、``2026-09-01T08:00:30``；
  - 缺年：``09-01``、``9月1日``、``9月1日 08:00`` —— 年份取"距离现在最近的过去
    日期"（如现在是 2026-09-23，``9月1日`` -> 2026-09-01，``12月30日`` -> 2025-12-30）。

时区策略：不带时区的记号视为服务器本地时间（部署时区应与用户一致），统一
转换为 aware UTC 后参与比较；im 消息的 created_at 为 UTC ISO，直接可比。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# 相对记号：now / now-<N>d / now-<N>h
_RELATIVE_RE = re.compile(r"^now(?:-(\d+)([dh]))?$", re.IGNORECASE)
# 缺年数字日期：MM-DD（不允许纯数字歧义格式如 20260901）
_MD_RE = re.compile(r"^(\d{1,2})-(\d{1,2})$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
# 中文日期：[YYYY年]M月D日[ HH:mm[:ss]]
_CN_RE = re.compile(
    r"^(?:(\d{4})\s*年)?\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
    r"(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$"
)
ANALYSIS_TIMEZONE = ZoneInfo("Asia/Shanghai")


class WindowError(ValueError):
    """时间窗非法。code 用于 MCP 错误码：window_invalid / window_too_large。"""

    # 初始化对象状态并建立所需依赖。
    # English: Initialize object state and required dependencies.
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TimeWindow:
    start: datetime  # aware UTC，含端点
    end: datetime    # aware UTC，含端点

    # 处理`hours`相关逻辑。
    # English: Handle hours operations.
    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    # English: Handle segments total operations.
    def segments_total(self, segment_hours: int) -> int:
        """按段长切分的总段数（至少 1 段）。"""
        if segment_hours < 1:
            raise WindowError("window_invalid", "segment_hours 必须为正整数")
        total = 1
        remaining = (self.end - self.start).total_seconds()
        seg_seconds = segment_hours * 3600
        if remaining > seg_seconds:
            total = int(-(-remaining // seg_seconds))  # 向上取整，避免浮点误差
        return total

    # English: Handle segment bounds operations.
    def segment_bounds(self, segment_hours: int, index: int) -> tuple[datetime, datetime]:
        """第 index 段（从 0 计）的 [起, 止) 边界；末段止点为窗口 end（含）。"""
        total = self.segments_total(segment_hours)
        if index < 0 or index >= total:
            raise WindowError(
                "segment_out_of_range",
                f"segment_index 必须在 0..{total - 1}",
            )
        seg_start = self.start + timedelta(hours=segment_hours * index)
        seg_end = min(self.start + timedelta(hours=segment_hours * (index + 1)), self.end)
        return seg_start, seg_end


# English: Parse year inferred.
def _parse_year_inferred(month: int, day: int, now_local: datetime) -> datetime:
    """缺年日期 -> 距离现在最近的过去日期（今年已过用今年，否则用去年）。"""
    candidate = datetime(now_local.year, month, day)
    if candidate > now_local.replace(tzinfo=None):
        candidate = datetime(now_local.year - 1, month, day)
    return candidate


# English: Parse time token.
def parse_time_token(token: str | None, *, now: datetime | None = None) -> datetime | None:
    """解析单个时间记号为 aware UTC datetime；None/空 -> None。

    now 仅用于测试注入；默认取当前时间（服务器本地时区）。
    无法识别的记号抛 WindowError(window_invalid)。
    """
    if token is None:
        return None
    text = str(token).strip()
    if not text or text.lower() in {"null", "none", "未指定", "默认"}:
        return None
    now = now or datetime.now(ANALYSIS_TIMEZONE)
    now_local = now if now.tzinfo else now.replace(tzinfo=ANALYSIS_TIMEZONE)

    relative = _RELATIVE_RE.match(text)
    if relative:
        if relative.group(1) is None:
            return now_local.astimezone(timezone.utc)
        amount = int(relative.group(1))
        unit = relative.group(2).lower()
        delta = timedelta(days=amount) if unit == "d" else timedelta(hours=amount)
        return (now_local - delta).astimezone(timezone.utc)

    iso_like = text.replace("T", " ").replace("/", "-")
    cn = _CN_RE.match(text)
    md = _MD_RE.match(iso_like)
    try:
        if cn:
            year = int(cn.group(1)) if cn.group(1) else None
            month, day = int(cn.group(2)), int(cn.group(3))
            hour = int(cn.group(4) or 0)
            minute = int(cn.group(5) or 0)
            second = int(cn.group(6) or 0)
            base = (
                datetime(year, month, day, hour, minute, second)
                if year is not None
                else _parse_year_inferred(month, day, now_local).replace(
                    hour=hour, minute=minute, second=second
                )
            )
        elif md:
            month, day = int(md.group(1)), int(md.group(2))
            base = _parse_year_inferred(month, day, now_local)
        else:
            base = datetime.fromisoformat(iso_like)
    except ValueError as exc:
        raise WindowError(
            "window_invalid", f"无法识别的时间记号: {token!r}"
        ) from exc
    if base.tzinfo is not None:
        return base.astimezone(timezone.utc)
    return base.replace(tzinfo=now_local.tzinfo).astimezone(timezone.utc)


# English: Resolve window.
def resolve_window(
    start: str | None,
    end: str | None,
    days: int | None,
    *,
    now: datetime | None = None,
    max_days: int = 31,
) -> TimeWindow:
    """把 (start, end, days) 组合解析为受护栏保护的 UTC 时间窗。

    组合规则（宽恕式，缺项补默认）：
      - start 与 end 都给 -> 绝对窗口；
      - 只给一端 -> 另一端取 now（start 缺 -> now-7d，end 缺 -> now）；
      - 都缺 -> days（默认 7）相对窗口 [now-days 00:00 起整点对齐不强制, now]。

    护栏：start >= end -> window_invalid；跨度 > max_days 天 -> window_too_large。
    """
    reference_now = now or datetime.now(ANALYSIS_TIMEZONE)
    resolved_end = parse_time_token(end, now=reference_now) or reference_now.astimezone(
        timezone.utc
    )
    if end and (_ISO_DATE_RE.fullmatch(end.strip()) or _MD_RE.fullmatch(end.strip())
                or (_CN_RE.fullmatch(end.strip()) and not _CN_RE.fullmatch(end.strip()).group(4))):
        # A bare end date includes that whole local calendar day.  Exclusive
        # next-midnight avoids losing sub-second messages at 23:59:59.
        local_end = resolved_end.astimezone(reference_now.tzinfo or ANALYSIS_TIMEZONE)
        resolved_end = (local_end + timedelta(days=1)).astimezone(timezone.utc)
    if start is None and not str(end or "").strip() and days is None:
        days = 7
    if days is not None and parse_time_token(start, now=now) is None and not str(end or "").strip():
        # 纯相对窗口：days 天前的同一时刻 -> now
        now_utc = reference_now.astimezone(timezone.utc)
        resolved_start = now_utc - timedelta(days=days)
        resolved_end = now_utc
    else:
        parsed_start = parse_time_token(start, now=now)
        if parsed_start is None:
            parsed_start = resolved_end - timedelta(days=7)
        resolved_start, resolved_end = parsed_start, resolved_end

    if days is not None and days < 1:
        raise WindowError("window_invalid", "days 必须为正整数")
    if resolved_start >= resolved_end:
        raise WindowError(
            "window_invalid", "窗口起点必须早于终点（start < end）"
        )
    span_days = (resolved_end - resolved_start).total_seconds() / 86400.0
    if span_days > max_days:
        raise WindowError(
            "window_too_large", f"分析窗口不能超过 {max_days} 天，请缩小范围"
        )
    return TimeWindow(start=resolved_start, end=resolved_end)
