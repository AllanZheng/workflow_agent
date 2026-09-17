"""单元测试：LLM 输出解析器。"""

import pytest

from flowagent.errors import OutputParseError
from flowagent.parsers import (
    parse_classifier,
    parse_json,
    parse_lines,
    parse_output,
    parse_text,
)


class TestText:
    def test_strip(self):
        assert parse_text("  hello \n") == "hello"

    def test_non_string_coerced(self):
        assert parse_text(123) == "123"


class TestLines:
    def test_basic(self):
        assert parse_lines("a\n\n b \nc") == ["a", "b", "c"]

    def test_empty(self):
        assert parse_lines("  \n") == []


class TestJson:
    def test_plain_object(self):
        assert parse_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        text = "```json\n{\"a\": 1}\n```"
        assert parse_json(text) == {"a": 1}

    def test_with_prefix_text(self):
        assert parse_json('结果如下：{"score": 7, "reason": "x"}') == {"score": 7, "reason": "x"}

    def test_array(self):
        assert parse_json("前缀 [1, 2, 3]") == [1, 2, 3]

    def test_skips_invalid_candidate_then_parses(self):
        # 第一个 { 候选解析失败应继续向后扫描
        assert parse_json('坏数据 {not json} 之后 {"ok": 1}') == {"ok": 1}

    def test_failure_raises(self):
        with pytest.raises(OutputParseError):
            parse_json("没有任何JSON")


class TestClassifier:
    def test_hit(self):
        assert parse_classifier("我认为这是：退款", ["退款", "技术"]) == "退款"

    def test_longest_label_first(self):
        # “投诉处理”包含“投诉”，长标签优先应命中“投诉处理”
        assert parse_classifier("投诉处理", ["投诉", "投诉处理"]) == "投诉处理"

    def test_miss_raises(self):
        with pytest.raises(OutputParseError):
            parse_classifier("不知道", ["退款", "技术"])


class TestDispatch:
    def test_parse_output_dispatch(self):
        assert parse_output(" x ", "text") == "x"
        assert parse_output("{\"a\":1}", "json") == {"a": 1}
        assert parse_output("a\nb", "lines") == ["a", "b"]
        assert parse_output("退款", "classifier", ["退款"]) == "退款"

    def test_unknown_parser(self):
        with pytest.raises(OutputParseError):
            parse_output("x", "nope")


class TestValidateSchema:
    """4.2：output_schema 校验——缺字段/类型错/越界/枚举外都抛 OutputParseError。"""

    def _rules(self, **kw):
        # 轻量规则对象：与 config.OutputFieldSpec 同名属性（string/integer/...）
        from types import SimpleNamespace
        base = {"type": "string", "min": None, "max": None, "enum": None, "optional": False}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_missing_required_field(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"reason": "x"}, {"score": self._rules()})

    def test_optional_field_missing_ok(self):
        from flowagent.parsers import validate_schema
        validate_schema({"score": 8}, {"score": self._rules(type="integer"),
                                       "reason": self._rules(optional=True)})

    def test_wrong_type(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"score": "8"}, {"score": self._rules(type="integer")})

    def test_out_of_range(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"score": 11}, {"score": self._rules(type="integer", max=10)})
        with pytest.raises(OutputParseError):
            validate_schema({"score": 0}, {"score": self._rules(type="integer", min=1)})

    def test_enum(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"level": "中"}, {"level": self._rules(enum=["高", "低"])})

    def test_top_level_not_object(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema([1, 2], {"score": self._rules()})

    def test_boolean_not_integer(self):
        # Python 的 bool 是 int 子类，必须显式排除
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"ok": True}, {"ok": self._rules(type="integer")})

    def test_valid_object_passes(self):
        from flowagent.parsers import validate_schema
        validate_schema(
            {"score": 8, "reason": "ok", "extra": 1},  # 额外字段不报错（宽容）
            {"score": self._rules(type="integer", min=1, max=10),
             "reason": self._rules(type="string")},
        )
