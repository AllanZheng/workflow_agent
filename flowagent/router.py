"""条件路由：规则求值与路由函数构造。"""

import re

from .config import RouteConfig, RouteRule
from .errors import ConfigError
from .template import get_path


# English: Handle coerce operations.
def _coerce(a, b):
    """eq/ne 用宽容比较：先试数值，失败退化为字符串（保持既有语义）。"""
    try:
        return float(a), float(b)
    except (TypeError, ValueError):
        return str(a), str(b)


# English: Handle numeric operations.
def _numeric(a, b):
    """gt/ge/lt/le 专用：任一侧无法数值化返回 None（规则不命中）。

    此前 None/非数值会退化成字符串比较（如 "None" < "7"），导致字段缺失时
    规则意外命中或漏掉后落入默认放行分支——数值语义必须失败关闭（评审 4.2）。
    """
    try:
        return float(a), float(b)
    except (TypeError, ValueError):
        return None


# English: Handle evaluate op operations.
def evaluate_op(op: str, field, value) -> bool:
    """对字段值 field 求值单个条件操作符。"""
    if op == "exists":
        return field is not None and field != ""
    if op == "true":
        return bool(field)
    if op == "eq":
        a, b = _coerce(field, value)
        return a == b
    if op == "ne":
        a, b = _coerce(field, value)
        return a != b
    if op in ("gt", "ge", "lt", "le"):
        pair = _numeric(field, value)
        if pair is None:
            return False  # 字段缺失/不可数值化：规则不命中（失败关闭）
        a, b = pair
        return {"gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b}[op]
    if op == "in":
        return field in (value or [])
    if op == "contains":
        if field is None:
            return False
        return value in field
    if op == "regex":
        return re.search(str(value), str(field if field is not None else "")) is not None
    raise ConfigError(f"未知路由操作符: {op}")


# English: Handle loop exhausted operations.
def _loop_exhausted(rule: RouteRule, route: RouteConfig, counts: dict) -> bool:
    """循环护栏：目标节点已执行次数达到 max_loops 时视为耗尽。"""
    max_loops = rule.max_loops if rule.max_loops is not None else route.max_loops
    if max_loops is None:
        return False
    return counts.get(rule.goto, 0) >= max_loops


# English: Create router.
def make_router(node_name: str, route: RouteConfig, error_goto: str | None = None):
    """为某个节点构造 LangGraph 条件边的路由函数（闭包持有路由配置）。

    error_goto 为 node.on_error.goto：错误路由的唯一权威优先级，
    高于 route.on_error（避免用户两处重复配置，评审 4.5）。
    """

    # 处理`router`相关逻辑。
    # English: Handle router operations.
    def router(state: dict) -> str:
        err = state.get("error")
        if err and err.get("node") == node_name:
            # 统一错误路由模型：node.on_error.goto > route.on_error（只需配一处）
            if error_goto:
                return error_goto
            if route.on_error:
                return route.on_error
        counts = state.get("loop_counts") or {}
        field = get_path(state.get("vars") or {}, route.field)
        for rule in route.rules:
            if evaluate_op(rule.when.op, field, rule.when.value):
                if _loop_exhausted(rule, route, counts):
                    return rule.loop_exhaust or route.default
                return rule.goto
        if route.default is None:
            raise ConfigError(
                f"节点 {node_name} 的路由字段 {route.field}={field!r} 无匹配规则且未配置 default"
            )
        return route.default

    return router
