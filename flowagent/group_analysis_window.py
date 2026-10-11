"""保留用户明确指定的群分析时间，避免模型漏传后静默使用默认范围。"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_DATE = r"\d{4}(?:-\d{1,2}-\d{1,2}|年\d{1,2}月\d{1,2}日)"
_CLOCK = r"\d{1,2}[:：]\d{2}(?:[:：]\d{2})?"
_ZONE = r"Z|[+-]\d{2}:\d{2}"
_GAP = r"\s*(?:T|北京时间)?\s*"
_WINDOW = re.compile(
    rf"(?P<start_date>{_DATE}){_GAP}(?P<start_clock>{_CLOCK})"
    rf"(?P<start_zone>{_ZONE})?\s*(?:至|到|~|～|—)\s*"
    rf"(?:(?P<end_date>{_DATE}){_GAP})?(?P<end_clock>{_CLOCK})"
    rf"(?P<end_zone>{_ZONE})?"
)
_BEIJING = timezone(timedelta(hours=8))


def _timestamp(date: str, clock: str, zone: str | None) -> datetime:
    year, month, day = map(int, re.findall(r"\d+", date))
    parts = list(map(int, re.split(r"[:：]", clock)))
    hour, minute, second = (*parts, 0) if len(parts) == 2 else parts
    local = datetime(year, month, day, hour, minute, second)
    # 未写时区的业务时间按北京时间；显式 ISO 时区保持原意。
    if zone:
        return datetime.fromisoformat(local.isoformat() + zone.replace("Z", "+00:00"))
    return local.replace(tzinfo=_BEIJING)


def apply_requested_window(parsed: dict, message: str) -> dict:
    """仅覆盖明确的日期时段或最近 N 天，其余表达继续使用模型解析。"""
    result = dict(parsed)
    windows = list(_WINDOW.finditer(message))
    if windows:
        normalized = set()
        for match in windows:
            start = _timestamp(match["start_date"], match["start_clock"], match["start_zone"])
            end = _timestamp(match["end_date"] or match["start_date"], match["end_clock"],
                             match["end_zone"] or match["start_zone"])
            if not timedelta(0) < end - start <= timedelta(days=31):
                raise ValueError("invalid_requested_window")
            normalized.add((start.isoformat(), end.isoformat()))
        if len(normalized) != 1:
            raise ValueError("ambiguous_requested_window")
        result["start"], result["end"] = normalized.pop()
        result.pop("days", None)
    else:
        days = {int(value) for value in re.findall(r"(?:最近|过去|近)\s*(\d+)\s*天", message)}
        if days:
            if len(days) != 1 or not 1 <= next(iter(days)) <= 31:
                raise ValueError("invalid_requested_days")
            result["days"] = days.pop()
            result.pop("start", None)
            result.pop("end", None)
    return result
