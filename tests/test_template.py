"""单元测试：模板渲染与点路径取值。"""

import json

import pytest

from flowagent.errors import TemplateError
from flowagent.template import get_path, render_template


class TestGetPath:
    def test_flat_key(self):
        assert get_path({"a": 1}, "a") == 1

    def test_nested_path(self):
        data = {"review": {"score": 9, "reason": "ok"}}
        assert get_path(data, "review.score") == 9
        assert get_path(data, "review.reason") == "ok"

    def test_missing_returns_none(self):
        assert get_path({"a": 1}, "b") is None
        assert get_path({"a": {"b": 1}}, "a.c") is None
        assert get_path({"a": 1}, "a.b") is None  # 非dict中段
        assert get_path(None, "a") is None


class TestRenderTemplate:
    def test_replaces_known_var(self):
        assert render_template("你好 {name}", {"name": "张三"}) == "你好 张三"

    def test_dotted_var(self):
        ctx = {"review": {"score": 8}}
        assert render_template("评分 {review.score}/10", ctx) == "评分 8/10"

    def test_unknown_placeholder_kept(self):
        text = "保持 {unknown} 原样"
        assert render_template(text, {"a": 1}) == text

    def test_json_braces_kept(self):
        """提示词中的 JSON 花括号不应被误当变量。"""
        text = '输出JSON：{"score": 1-10整数, "reason": "理由"}'
        assert render_template(text, {}) == text

    def test_dict_value_rendered_as_json(self):
        ctx = {"outline": {"title": "T"}}
        out = render_template("大纲：{outline}", ctx)
        assert json.loads(out.split("：", 1)[1]) == {"title": "T"}

    def test_none_value_kept(self):
        assert render_template("值 {x}", {"x": None}) == "值 {x}"

    def test_invalid_input_raises_template_error(self):
        with pytest.raises(TemplateError):
            render_template(None, {})  # type: ignore[arg-type]
