"""单元测试：模板渲染与点路径取值。"""

import json

import pytest

from flowagent.errors import TemplateError
from flowagent.template import get_path, render_template


class TestGetPath:
    # 验证“`flat`密钥”场景下的预期行为。
    # English: Verify flat key.
    def test_flat_key(self):
        assert get_path({"a": 1}, "a") == 1

    # 验证“`nested`路径”场景下的预期行为。
    # English: Verify nested path.
    def test_nested_path(self):
        data = {"review": {"score": 9, "reason": "ok"}}
        assert get_path(data, "review.score") == 9
        assert get_path(data, "review.reason") == "ok"

    # 验证“缺失`returns`无”场景下的预期行为。
    # English: Verify missing returns none.
    def test_missing_returns_none(self):
        assert get_path({"a": 1}, "b") is None
        assert get_path({"a": {"b": 1}}, "a.c") is None
        assert get_path({"a": 1}, "a.b") is None  # 非dict中段
        assert get_path(None, "a") is None


class TestRenderTemplate:
    # 验证“`replaces``known``var`”场景下的预期行为。
    # English: Verify replaces known var.
    def test_replaces_known_var(self):
        assert render_template("你好 {name}", {"name": "张三"}) == "你好 张三"

    # 验证“`dotted``var`”场景下的预期行为。
    # English: Verify dotted var.
    def test_dotted_var(self):
        ctx = {"review": {"score": 8}}
        assert render_template("评分 {review.score}/10", ctx) == "评分 8/10"

    # 验证“未知`placeholder``kept`”场景下的预期行为。
    # English: Verify unknown placeholder kept.
    def test_unknown_placeholder_kept(self):
        text = "保持 {unknown} 原样"
        assert render_template(text, {"a": 1}) == text

    # English: Verify JSON braces kept.
    def test_json_braces_kept(self):
        """提示词中的 JSON 花括号不应被误当变量。"""
        text = '输出JSON：{"score": 1-10整数, "reason": "理由"}'
        assert render_template(text, {}) == text

    # 验证“`dict`值`rendered``as`JSON”场景下的预期行为。
    # English: Verify dict value rendered as JSON.
    def test_dict_value_rendered_as_json(self):
        ctx = {"outline": {"title": "T"}}
        out = render_template("大纲：{outline}", ctx)
        assert json.loads(out.split("：", 1)[1]) == {"title": "T"}

    # 验证“无值渲染空值”场景下的预期行为。
    # English: Verify none value renders empty.
    def test_none_value_renders_empty(self):
        assert render_template("值 {x}", {"x": None}) == "值 "

    # 验证“无效输入抛出模板错误”场景下的预期行为。
    # English: Verify invalid input raises template error.
    def test_invalid_input_raises_template_error(self):
        with pytest.raises(TemplateError):
            render_template(None, {})  # type: ignore[arg-type]
