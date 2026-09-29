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
    # 验证“合并`overwrites``right`”场景下的预期行为。
    def test_merge_overwrites_right(self):
        assert merge_dict({"a": 1, "b": 1}, {"b": 2}) == {"a": 1, "b": 2}

    # 验证“`left`无”场景下的预期行为。
    def test_left_none(self):
        assert merge_dict(None, {"a": 1}) == {"a": 1}

    # 验证“`right`无”场景下的预期行为。
    def test_right_none(self):
        assert merge_dict({"a": 1}, None) == {"a": 1}

    # 验证“`both`无”场景下的预期行为。
    def test_both_none(self):
        assert merge_dict(None, None) == {}


class TestBuiltinTools:
    # 验证“`knowledge``base``hit`”场景下的预期行为。
    def test_knowledge_base_hit(self):
        assert "7天" in knowledge_base("请问退款流程是什么")

    # 验证“`knowledge``base``miss`”场景下的预期行为。
    def test_knowledge_base_miss(self):
        assert "未命中" in knowledge_base("随便聊聊天气")

    # 验证“`bump``counter`”场景下的预期行为。
    def test_bump_counter(self):
        assert bump_counter(4) == 5
        assert bump_counter(None) == 1

    # 验证“`bump``counter``garbage`输入”场景下的预期行为。
    def test_bump_counter_garbage_input(self):
        assert bump_counter("abc") == 1

    # 验证“`unreliable`工具`ok`”场景下的预期行为。
    def test_unreliable_tool_ok(self):
        assert unreliable_tool("正常") == "processed:正常"

    # 验证“`unreliable`工具抛出”场景下的预期行为。
    def test_unreliable_tool_raises(self):
        with pytest.raises(ToolError):
            unreliable_tool("请触发故障")


class TestLoadTool:
    # 验证“加载`builtin`”场景下的预期行为。
    def test_load_builtin(self):
        assert callable(load_tool("flowagent.tools:knowledge_base"))

    # 验证“`bad``ref`格式化”场景下的预期行为。
    def test_bad_ref_format(self):
        with pytest.raises(ConfigError):
            load_tool("no_colon_here")

    # 验证“`module`不`found`”场景下的预期行为。
    def test_module_not_found(self):
        with pytest.raises(ConfigError):
            load_tool("no.such.module:fn")

    # 验证“不`callable`”场景下的预期行为。
    def test_not_callable(self):
        with pytest.raises(ConfigError):
            load_tool("flowagent.tools:kb_x_nothing")


class TestToolDefaultDeny:
    """4.9：默认只放行内置工具；外部模块需显式白名单；* 显式全放行。"""

    # 验证“`external``module``denied`按默认项”场景下的预期行为。
    def test_external_module_denied_by_default(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_TOOL_MODULES", raising=False)
        with pytest.raises(ConfigError):
            load_tool("shutil:rmtree")

    # 验证“`builtin`工具`allowed`按默认项”场景下的预期行为。
    def test_builtin_tools_allowed_by_default(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_TOOL_MODULES", raising=False)
        assert callable(load_tool("flowagent.tools:knowledge_base"))

    # 验证“`star`允许`everything`”场景下的预期行为。
    def test_star_allows_everything(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "*")
        assert callable(load_tool("shutil:rmtree"))  # 仅验证可加载，不执行

    # 验证“`explicit``prefix``still``works`”场景下的预期行为。
    def test_explicit_prefix_still_works(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "tests")
        assert callable(load_tool("tests.flaky:flaky_tool"))
        with pytest.raises(ConfigError):
            load_tool("shutil:rmtree")
