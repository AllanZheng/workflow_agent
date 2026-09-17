"""单元测试：状态 reducer 与内置工具。"""

import pytest

from flowagent.errors import ConfigError, ToolError
from flowagent.state import merge_dict
from flowagent.tools import (
    bump_counter,
    knowledge_base,
    load_tool,
    unreliable_tool,
)


class TestMergeDict:
    def test_merge_overwrites_right(self):
        assert merge_dict({"a": 1, "b": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_left_none(self):
        assert merge_dict(None, {"a": 1}) == {"a": 1}

    def test_right_none(self):
        assert merge_dict({"a": 1}, None) == {"a": 1}

    def test_both_none(self):
        assert merge_dict(None, None) == {}


class TestBuiltinTools:
    def test_knowledge_base_hit(self):
        assert "7天" in knowledge_base("请问退款流程是什么")

    def test_knowledge_base_miss(self):
        assert "未命中" in knowledge_base("随便聊聊天气")

    def test_bump_counter(self):
        assert bump_counter(4) == 5
        assert bump_counter(None) == 1

    def test_bump_counter_garbage_input(self):
        assert bump_counter("abc") == 1

    def test_unreliable_tool_ok(self):
        assert unreliable_tool("正常") == "processed:正常"

    def test_unreliable_tool_raises(self):
        with pytest.raises(ToolError):
            unreliable_tool("请触发故障")


class TestLoadTool:
    def test_load_builtin(self):
        assert callable(load_tool("flowagent.tools:knowledge_base"))

    def test_bad_ref_format(self):
        with pytest.raises(ConfigError):
            load_tool("no_colon_here")

    def test_module_not_found(self):
        with pytest.raises(ConfigError):
            load_tool("no.such.module:fn")

    def test_not_callable(self):
        with pytest.raises(ConfigError):
            load_tool("flowagent.tools:kb_x_nothing")


class TestToolDefaultDeny:
    """4.9：默认只放行内置工具；外部模块需显式白名单；* 显式全放行。"""

    def test_external_module_denied_by_default(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_TOOL_MODULES", raising=False)
        with pytest.raises(ConfigError):
            load_tool("shutil:rmtree")

    def test_builtin_tools_allowed_by_default(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_TOOL_MODULES", raising=False)
        assert callable(load_tool("flowagent.tools:knowledge_base"))

    def test_star_allows_everything(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "*")
        assert callable(load_tool("shutil:rmtree"))  # 仅验证可加载，不执行

    def test_explicit_prefix_still_works(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "tests")
        assert callable(load_tool("tests.flaky:flaky_tool"))
        with pytest.raises(ConfigError):
            load_tool("shutil:rmtree")
