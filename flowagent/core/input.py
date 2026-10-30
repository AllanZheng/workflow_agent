"""统一原生输入，保留显式空值的优先级。"""
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flowagent.core.contracts import RunInput


def continuation_clock(value: dict) -> datetime:
    """续接时钟必须可解析且具有可信时区，不能猜测缺失信息。"""
    if value.get("workflow") != "group_analysis":
        raise ValueError("invalid continuation workflow")
    raw_time, zone = value.get("reference_time"), value.get("timezone")
    if not isinstance(raw_time, str) or not isinstance(zone, str) or not zone:
        raise ValueError("missing continuation reference time or timezone")
    try:
        reference = datetime.fromisoformat(raw_time)
        if reference.tzinfo is None or reference.utcoffset() is None:
            raise ValueError("continuation reference time must contain timezone")
        # 同一时刻换到声明的时区；由 IANA 规则决定夏令时和日期。
        reference = reference.astimezone(ZoneInfo(zone))
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("invalid continuation reference time or timezone") from exc
    return reference


def normalize_input(request: RunInput) -> RunInput:
    # 顶层字段显式出现时覆盖旧 inputs；复制避免工作流修改调用者数据。
    inputs = deepcopy(request.inputs)
    history = deepcopy(request.history if request._history_supplied or request.history
                       else inputs.get("history") or [])
    continuation = deepcopy(request.continuation if request._continuation_supplied
                            else inputs.get("continuation"))
    inputs["history"] = history
    if continuation is None:
        inputs.pop("continuation", None)
    else:
        inputs["continuation"] = continuation
    return RunInput(workflow=request.workflow, inputs=inputs, history=history,
                    correlation_id=request.correlation_id, continuation=continuation)
