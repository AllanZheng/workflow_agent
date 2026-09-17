"""单元测试：条件路由规则求值与路由函数。"""

import pytest

from flowagent.config import RouteConfig
from flowagent.errors import ConfigError
from flowagent.router import evaluate_op, make_router


class TestEvaluateOp:
    @pytest.mark.parametrize(
        ("op", "field", "value", "expected"),
        [
            ("eq", 7, 7, True),
            ("eq", "7", 7, True),          # 数值弱类型
            ("eq", "abc", "abc", True),
            ("eq", "abc", "abd", False),
            ("ne", 1, 2, True),
            ("gt", 8, 7, True),
            ("gt", "8", 7, True),
            ("gt", "a", "b", False),
            ("ge", 7, 7, True),
            ("lt", 6, 7, True),
            ("le", 7, 7, True),
            ("in", "退款", ["退款", "技术"], True),
            ("in", "咨询", ["退款", "技术"], False),
            ("contains", "包含退款字样", "退款", True),
            ("contains", None, "退款", False),
            ("regex", "工单#123", r"#\d+", True),
            ("regex", "无编号", r"#\d+", False),
            ("exists", "x", None, True),
            ("exists", "", None, False),
            ("exists", None, None, False),
            ("true", "x", None, True),
            ("true", 0, None, False),
        ],
    )
    def test_ops(self, op, field, value, expected):
        assert evaluate_op(op, field, value) is expected

    def test_unknown_op(self):
        with pytest.raises(ConfigError):
            evaluate_op("nope", 1, 1)


def _route(**kwargs) -> RouteConfig:
    base = {
        "field": "category",
        "rules": [{"when": {"op": "eq", "value": "退款"}, "goto": "refund"}],
        "default": "faq",
    }
    base.update(kwargs)
    return RouteConfig.model_validate(base)


class TestMakeRouter:
    def test_rule_hit(self):
        router = make_router("classify", _route())
        assert router({"vars": {"category": "退款"}}) == "refund"

    def test_default(self):
        router = make_router("classify", _route())
        assert router({"vars": {"category": "其他"}}) == "faq"

    def test_missing_field_uses_default(self):
        router = make_router("classify", _route())
        assert router({"vars": {}}) == "faq"

    def test_nested_field_path(self):
        route = RouteConfig.model_validate(
            {
                "field": "review.score",
                "rules": [{"when": {"op": "lt", "value": 7}, "goto": "revise"}],
                "default": "finalize",
            }
        )
        router = make_router("review", route)
        assert router({"vars": {"review": {"score": 5}}}) == "revise"
        assert router({"vars": {"review": {"score": 9}}}) == "finalize"

    def test_loop_guard_exhausted(self):
        route = _route(
            rules=[
                {
                    "when": {"op": "eq", "value": "退款"},
                    "goto": "refund",
                    "max_loops": 2,
                    "loop_exhaust": "giveup",
                }
            ]
        )
        router = make_router("n", route)
        assert router({"vars": {"category": "退款"}, "loop_counts": {"refund": 2}}) == "giveup"
        assert router({"vars": {"category": "退款"}, "loop_counts": {"refund": 1}}) == "refund"

    def test_route_level_max_loops(self):
        route = _route(max_loops=3, rules=[
            {"when": {"op": "eq", "value": "退款"}, "goto": "refund", "loop_exhaust": "giveup"}
        ])
        router = make_router("n", route)
        assert router({"vars": {"category": "退款"}, "loop_counts": {"refund": 3}}) == "giveup"

    def test_no_loop_guard(self):
        router = make_router("n", _route())
        # 未配置 max_loops 时，同一目标可无限进入
        assert router({"vars": {"category": "退款"}, "loop_counts": {"refund": 99}}) == "refund"

    def test_on_error_routes_to_fallback(self):
        route = _route(on_error="manual")
        router = make_router("broken", route)
        state = {"vars": {"category": "退款"}, "error": {"node": "broken"}}
        assert router(state) == "manual"

    def test_on_error_ignored_for_other_nodes(self):
        route = _route(on_error="manual")
        router = make_router("broken", route)
        state = {"vars": {"category": "退款"}, "error": {"node": "other"}}
        assert router(state) == "refund"

    def test_no_match_no_default_raises(self):
        route = RouteConfig.model_validate(
            {"field": "category", "rules": [{"when": {"op": "eq", "value": "x"}, "goto": "END"}]}
        )
        router = make_router("n", route)
        with pytest.raises(ConfigError):
            router({"vars": {"category": "y"}})

    def test_route_requires_rules_or_default(self):
        from flowagent.config import load_spec

        spec = {
            "workflow": {"name": "t", "start": "a"},
            "nodes": {
                "a": {
                    "type": "template",
                    "template": "t",
                    "output_key": "o",
                    "next": {"field": "x", "rules": []},
                }
            },
        }
        with pytest.raises(ConfigError):
            load_spec(spec)


class TestRouterNumericHardening:
    """4.2：数值操作符遇 None/不可数值化字段时规则不命中（禁止退化字符串比较）。"""

    def test_lt_with_none_field_no_match(self):
        assert evaluate_op("lt", None, 7) is False
        assert evaluate_op("ge", None, 7) is False

    def test_lt_with_non_numeric_string_no_match(self):
        assert evaluate_op("lt", "abc", 7) is False

    def test_lt_numeric_still_works(self):
        assert evaluate_op("lt", 5, 7) is True
        assert evaluate_op("lt", "5", 7) is True  # 可数值化字符串保持宽容
