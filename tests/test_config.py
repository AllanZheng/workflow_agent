"""单元测试：配置模型校验与 YAML 加载。"""

import pytest

from flowagent.config import load_spec, load_spec_file
from flowagent.errors import ConfigError

VALID = {
    "workflow": {"name": "t", "start": "a"},
    "nodes": {
        "a": {
            "type": "template",
            "template": "hi {x}",
            "output_key": "msg",
            "next": "END",
        }
    },
}


# 处理`override`相关逻辑。
def _override(**changes) -> dict:
    import copy

    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


class TestValidSpec:
    # 验证“`minimal`”场景下的预期行为。
    def test_minimal(self):
        spec = load_spec(VALID)
        assert spec.workflow.name == "t"
        assert spec.workflow.start == "a"
        assert spec.nodes["a"].type == "template"

    # 验证“大语言模型`full`”场景下的预期行为。
    def test_llm_full(self):
        data = {
            "workflow": {"name": "t", "start": "c", "recursion_limit": 10},
            "nodes": {
                "c": {
                    "type": "llm",
                    "prompt": "p {x}",
                    "parser": "classifier",
                    "labels": ["a", "b"],
                    "output_key": "cat",
                    "mock_response": "a",
                    "next": "END",
                }
            },
        }
        spec = load_spec(data)
        assert spec.workflow.recursion_limit == 10
        assert spec.nodes["c"].labels == ["a", "b"]

    # 验证“列出`next`”场景下的预期行为。
    def test_list_next(self):
        data = _override(
            nodes={
                "a": {"type": "template", "template": "t", "output_key": "o", "next": ["END"]}
            }
        )
        spec = load_spec(data)
        assert spec.nodes["a"].next_targets == ["END"]


class TestInvalidSpec:
    # 验证“空值节点”场景下的预期行为。
    def test_empty_nodes(self):
        with pytest.raises(ConfigError):
            load_spec({"workflow": {"name": "t", "start": "a"}, "nodes": {}})

    # 验证“未知启动”场景下的预期行为。
    def test_unknown_start(self):
        with pytest.raises(ConfigError):
            load_spec(_override(workflow={"name": "t", "start": "zzz"}))

    # 验证“未知`next`目标”场景下的预期行为。
    def test_unknown_next_target(self):
        data = _override(
            nodes={
                "a": {"type": "template", "template": "t", "output_key": "o", "next": "ghost"}
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“未知节点类型”场景下的预期行为。
    def test_unknown_node_type(self):
        data = _override(
            nodes={
                "a": {"type": "weird", "template": "t", "output_key": "o", "next": "END"}
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“大语言模型不包含提示词”场景下的预期行为。
    def test_llm_without_prompt(self):
        data = _override(
            nodes={
                "a": {"type": "llm", "output_key": "o", "next": "END"}
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“工具不包含工具`ref`”场景下的预期行为。
    def test_tool_without_tool_ref(self):
        data = _override(
            nodes={
                "a": {"type": "tool", "output_key": "o", "next": "END"}
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“模板不包含模板”场景下的预期行为。
    def test_template_without_template(self):
        data = _override(
            nodes={
                "a": {"type": "template", "output_key": "o", "next": "END"}
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“缺失输出密钥”场景下的预期行为。
    def test_missing_output_key(self):
        data = _override(
            nodes={
                "a": {"type": "llm", "prompt": "p", "next": "END"}
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“`classifier`不包含`labels`”场景下的预期行为。
    def test_classifier_without_labels(self):
        data = _override(
            nodes={
                "a": {
                    "type": "llm",
                    "prompt": "p",
                    "parser": "classifier",
                    "output_key": "o",
                    "next": "END",
                }
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“路由不包含`rules`与默认项”场景下的预期行为。
    def test_route_without_rules_and_default(self):
        data = _override(
            nodes={
                "a": {
                    "type": "template",
                    "template": "t",
                    "output_key": "o",
                    "next": {"field": "x", "rules": []},
                }
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“路由未知`goto`”场景下的预期行为。
    def test_route_unknown_goto(self):
        data = _override(
            nodes={
                "a": {
                    "type": "template",
                    "template": "t",
                    "output_key": "o",
                    "next": {
                        "field": "x",
                        "rules": [{"when": {"op": "eq", "value": 1}, "goto": "ghost"}],
                        "default": "END",
                    },
                }
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“`extra`字段`forbidden`”场景下的预期行为。
    def test_extra_field_forbidden(self):
        data = _override()
        data["oops"] = 1
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“`negative`重试`rejected`”场景下的预期行为。
    def test_negative_retries_rejected(self):
        data = _override(
            nodes={
                "a": {
                    "type": "template",
                    "template": "t",
                    "output_key": "o",
                    "retries": -1,
                    "next": "END",
                }
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“`negative`重试`delay``rejected`”场景下的预期行为。
    def test_negative_retry_delay_rejected(self):
        data = _override(
            nodes={
                "a": {
                    "type": "template",
                    "template": "t",
                    "output_key": "o",
                    "retry_delay": -0.5,
                    "next": "END",
                }
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“`non`正数`timeout``rejected`”场景下的预期行为。
    def test_non_positive_timeout_rejected(self):
        data = _override(
            nodes={
                "a": {
                    "type": "template",
                    "template": "t",
                    "output_key": "o",
                    "timeout": 0,
                    "next": "END",
                }
            }
        )
        with pytest.raises(ConfigError):
            load_spec(data)

    # 验证“`non``dict``top`”场景下的预期行为。
    def test_non_dict_top(self):
        with pytest.raises(ConfigError):
            load_spec([1, 2])  # type: ignore[arg-type]


class TestLoadFile:
    # 验证“`yaml`文件”场景下的预期行为。
    def test_yaml_file(self, tmp_path):
        f = tmp_path / "wf.yaml"
        f.write_text(
            "workflow:\n  name: t\n  start: a\n"
            "nodes:\n  a:\n    type: template\n    template: 'x'\n"
            "    output_key: o\n    next: END\n",
            encoding="utf-8",
        )
        spec = load_spec_file(f)
        assert spec.workflow.name == "t"

    # 验证“缺失文件”场景下的预期行为。
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError):
            load_spec_file(tmp_path / "nope.yaml")

    # 验证“`bad``yaml`”场景下的预期行为。
    def test_bad_yaml(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("workflow: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_spec_file(f)

    # 验证“`non``mapping``yaml`”场景下的预期行为。
    def test_non_mapping_yaml(self, tmp_path):
        f = tmp_path / "list.yaml"
        f.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_spec_file(f)


class TestStrictValidation:
    """4.7：加载期拒绝必然在运行期失败的值。"""

    # 执行加载对应的流程。
    def _load(self, spec):
        from flowagent import load_spec
        from flowagent.errors import ConfigError
        with pytest.raises(ConfigError):
            load_spec(spec)

    # 验证“`recursion`数量上限`zero`”场景下的预期行为。
    def test_recursion_limit_zero(self):
        self._load({"workflow": {"name": "v", "start": "n", "recursion_limit": 0},
                    "nodes": {"n": {"type": "template", "template": "x",
                                    "output_key": "o", "next": "END"}}})

    # 验证“`max``loops``zero`”场景下的预期行为。
    def test_max_loops_zero(self):
        self._load({"workflow": {"name": "v", "start": "n"},
                    "nodes": {"n": {"type": "template", "template": "x", "output_key": "o",
                                    "next": {"field": "o",
                                             "rules": [{"when": {"op": "exists"}, "goto": "n",
                                                        "max_loops": 0}],
                                             "default": "END"}}}})

    # 验证“`max``loops`不包含`exhaust`或默认项”场景下的预期行为。
    def test_max_loops_without_exhaust_or_default(self):
        self._load({"workflow": {"name": "v", "start": "n"},
                    "nodes": {"n": {"type": "template", "template": "x", "output_key": "o",
                                    "next": {"field": "o",
                                             "rules": [{"when": {"op": "exists"}, "goto": "n",
                                                        "max_loops": 2}]}}}})

    # 验证“`bad``regex`”场景下的预期行为。
    def test_bad_regex(self):
        self._load({"workflow": {"name": "v", "start": "n"},
                    "nodes": {"n": {"type": "template", "template": "x", "output_key": "o",
                                    "next": {"field": "o",
                                             "rules": [{"when": {"op": "regex", "value": "("},
                                                        "goto": "END"}],
                                             "default": "END"}}}})

    # 验证“空值`parallel`列出”场景下的预期行为。
    def test_empty_parallel_list(self):
        self._load({"workflow": {"name": "v", "start": "n"},
                    "nodes": {"n": {"type": "template", "template": "x",
                                    "output_key": "o", "next": []}}})
