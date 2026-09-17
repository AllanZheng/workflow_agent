"""测试专用工具：不稳定工具（重试测试）与计数器。"""

import time

from flowagent.errors import ToolError

_CALLS: dict = {}


def reset() -> None:
    _CALLS.clear()


def calls(key: str = "default") -> int:
    return _CALLS.get(key, 0)


def flaky_tool(key: str = "default", fail_times: int = 1) -> str:
    """前 fail_times 次调用抛异常，之后成功——用于验证节点重试。"""
    _CALLS[key] = _CALLS.get(key, 0) + 1
    if _CALLS[key] <= fail_times:
        raise ToolError(f"第{_CALLS[key]}次调用失败")
    return f"ok after {_CALLS[key]} calls"


def slow_counting_tool(key: str = "slow", seconds: float = 0.6) -> str:
    """慢工具（带副作用计数）：验证超时按时返回且副作用不重复（评审 4.3）。"""
    _CALLS[key] = _CALLS.get(key, 0) + 1
    time.sleep(float(seconds))
    return "done"
