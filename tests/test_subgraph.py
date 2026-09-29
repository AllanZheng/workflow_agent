"""集成测试：workflow 子图节点（YAML 嵌套编排 + 循环引用防护）。"""

import pytest

from flowagent import WorkflowEngine, load_spec, load_spec_file
from flowagent.errors import ConfigError
from flowagent.llm import MockLLM

CHILD_YAML = """\
workflow: {name: child, start: c1}
nodes:
  c1:
    type: template
    template: '子图处理了:{x}'
    output_key: child_out
    next: END
"""


# 执行写入对应的流程。
def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


class TestSubgraph:
    # 验证“`subgraph`输出`wrapped`输出密钥”场景下的预期行为。
    def test_subgraph_output_wrapped_in_output_key(self, tmp_path):
        child = _write(tmp_path / "child.yaml", CHILD_YAML)
        spec = load_spec(
            {
                "workflow": {"name": "p", "start": "p1"},
                "nodes": {
                    "p1": {
                        "type": "workflow",
                        "workflow_file": str(child),
                        "output_key": "sub",
                        "next": "END",
                    }
                },
            }
        )
        final = WorkflowEngine(spec, MockLLM()).run({"x": "任务A"})
        # 子图 vars（含输入回传）整体包在 output_key 下
        assert final["vars"]["sub"]["child_out"] == "子图处理了:任务A"

    # 验证“`subgraph``vars``merged`不包含输出密钥”场景下的预期行为。
    def test_subgraph_vars_merged_without_output_key(self, tmp_path):
        child = _write(tmp_path / "child.yaml", CHILD_YAML)
        spec = load_spec(
            {
                "workflow": {"name": "p", "start": "p1"},
                "nodes": {
                    "p1": {
                        "type": "workflow",
                        "workflow_file": str(child),
                        "next": "END",
                    }
                },
            }
        )
        final = WorkflowEngine(spec, MockLLM()).run({"x": "B"})
        assert final["vars"]["child_out"] == "子图处理了:B"
        assert final["vars"]["x"] == "B"  # 子图输入回传合并

    # 验证“`parent`消费`subgraph`输出”场景下的预期行为。
    def test_parent_consumes_subgraph_output(self, tmp_path):
        child = _write(tmp_path / "child.yaml", CHILD_YAML)
        spec = load_spec(
            {
                "workflow": {"name": "p", "start": "p1"},
                "nodes": {
                    "p1": {
                        "type": "workflow",
                        "workflow_file": str(child),
                        "output_key": "sub",
                        "next": "p2",
                    },
                    "p2": {
                        "type": "template",
                        "template": "父级收到:{sub.child_out}",
                        "output_key": "final",
                        "next": "END",
                    },
                },
            }
        )
        final = WorkflowEngine(spec, MockLLM()).run({"x": "C"})
        assert final["vars"]["final"] == "父级收到:子图处理了:C"

    def test_subgraph_gets_llm_from_global_factory(self, tmp_path):
        """子图引擎未显式传 LLM，自动走 create_llm()（无 key → Mock）。"""
        child_yaml = (
            "workflow: {name: c, start: c1}\n"
            "nodes:\n"
            "  c1:\n"
            "    type: llm\n"
            "    prompt: 'p'\n"
            "    output_key: out\n"
            "    mock_response: '子图LLM输出'\n"
            "    next: END\n"
        )
        child = _write(tmp_path / "child2.yaml", child_yaml)
        spec = load_spec(
            {
                "workflow": {"name": "p", "start": "p1"},
                "nodes": {
                    "p1": {
                        "type": "workflow",
                        "workflow_file": str(child),
                        "output_key": "sub",
                        "next": "END",
                    }
                },
            }
        )
        final = WorkflowEngine(spec, MockLLM()).run({})
        assert final["vars"]["sub"]["out"] == "子图LLM输出"

    # 验证“`yaml`文件加载解析`relative`路径”场景下的预期行为。
    def test_yaml_file_load_resolves_relative_path(self, tmp_path):
        _write(tmp_path / "child.yaml", CHILD_YAML)
        parent = _write(
            tmp_path / "parent.yaml",
            "workflow: {name: p, start: p1}\n"
            "nodes:\n"
            "  p1:\n"
            "    type: workflow\n"
            "    workflow_file: child.yaml\n"
            "    output_key: sub\n"
            "    next: END\n",
        )
        spec = load_spec_file(parent)  # 相对路径应被改写为绝对路径
        assert (tmp_path / "child.yaml").resolve().__str__() in [
            str(n.workflow_file) for n in spec.nodes.values()
        ]
        final = WorkflowEngine(spec, MockLLM()).run({"x": "相对"})
        assert final["vars"]["sub"]["child_out"] == "子图处理了:相对"


class TestCycleProtection:
    # 处理`cycle`相关逻辑。
    def _cycle(self, tmp_path, a_text, b_text):
        _write(tmp_path / "a.yaml", a_text)
        _write(tmp_path / "b.yaml", b_text)
        with pytest.raises(ConfigError):
            load_spec_file(tmp_path / "a.yaml")

    # 验证“`self``reference`”场景下的预期行为。
    def test_self_reference(self, tmp_path):
        self._cycle(
            tmp_path,
            "workflow: {name: a, start: n}\n"
            "nodes:\n"
            "  n:\n"
            "    type: workflow\n"
            "    workflow_file: a.yaml\n"
            "    next: END\n",
            "",
        )

    # 验证“`mutual``reference`”场景下的预期行为。
    def test_mutual_reference(self, tmp_path):
        wf = "workflow: {{name: {n}, start: n}}\nnodes:\n  n:\n    type: workflow\n    workflow_file: {other}.yaml\n    next: END\n"
        self._cycle(tmp_path, wf.format(n="a", other="b"), wf.format(n="b", other="a"))

    # 验证“缺失`child`文件”场景下的预期行为。
    def test_missing_child_file(self, tmp_path):
        parent = _write(
            tmp_path / "p.yaml",
            "workflow: {name: p, start: n}\n"
            "nodes:\n"
            "  n:\n"
            "    type: workflow\n"
            "    workflow_file: ghost.yaml\n"
            "    next: END\n",
        )
        with pytest.raises(ConfigError):
            load_spec_file(parent)

    # 验证“工作流节点校验必需的工作流文件”场景下的预期行为。
    def test_workflow_node_requires_workflow_file(self):
        with pytest.raises(ConfigError):
            load_spec(
                {
                    "workflow": {"name": "p", "start": "n"},
                    "nodes": {"n": {"type": "workflow", "next": "END"}},
                }
            )


CHILD_WITH_HUMAN = """\
workflow: {name: child_hitl, start: ask}
nodes:
  ask:
    type: human
    message: '子图人工确认：{x}'
    options: [批准, 拒绝]
    output_key: decision
    next:
      field: decision
      rules:
        - when: {op: eq, value: 批准}
          goto: done
      default: denied
  done:
    type: template
    template: '子图批准:{x}'
    output_key: child_out
    next: END
  denied:
    type: template
    template: '子图拒绝:{x}'
    output_key: child_out
    next: END
"""


class TestSubgraphHumanPropagation:
    """4.4：子图 human 中断提升到父图，同一父 thread_id 完成恢复。"""

    # 处理`parent``spec`相关逻辑。
    def _parent_spec(self, child_path):
        return load_spec({
            "workflow": {"name": "parent_hitl", "start": "sub"},
            "nodes": {
                "sub": {
                    "type": "workflow",
                    "workflow_file": str(child_path),
                    "output_key": "sub",
                    "next": "END",
                }
            },
        })

    # 验证“`subgraph``human``pauses``parent`”场景下的预期行为。
    def test_subgraph_human_pauses_parent(self, tmp_path):
        child = _write(tmp_path / "child_hitl.yaml", CHILD_WITH_HUMAN)
        eng = WorkflowEngine(self._parent_spec(child), MockLLM())
        final = eng.run({"x": "任务B"}, thread_id="ph1")
        payload = WorkflowEngine.waiting_on(final)
        # 父图必须暂停（而非静默完成），且能看到子图的问题与上下文
        assert payload is not None
        assert "子图人工确认" in payload["question"]
        assert payload.get("workflow_node") == "sub"

    # 验证“`resume`包含相同`parent``thread`完成”场景下的预期行为。
    def test_resume_with_same_parent_thread_completes(self, tmp_path):
        child = _write(tmp_path / "child_hitl.yaml", CHILD_WITH_HUMAN)
        eng = WorkflowEngine(self._parent_spec(child), MockLLM())
        eng.run({"x": "任务C"}, thread_id="ph2")
        final = eng.resume("批准", thread_id="ph2")
        # 父图完成，子图输出按 output_key 包装
        assert final["vars"]["sub"]["child_out"] == "子图批准:任务C"

    # 验证“`resume`拒绝路由`subgraph`”场景下的预期行为。
    def test_resume_reject_routes_in_subgraph(self, tmp_path):
        child = _write(tmp_path / "child_hitl.yaml", CHILD_WITH_HUMAN)
        eng = WorkflowEngine(self._parent_spec(child), MockLLM())
        eng.run({"x": "任务D"}, thread_id="ph3")
        final = eng.resume("拒绝", thread_id="ph3")
        assert final["vars"]["sub"]["child_out"] == "子图拒绝:任务D"

    def test_invalid_answer_reasks_then_completes(self, tmp_path):
        """options 强校验在子图传播链路上同样生效。"""
        child = _write(tmp_path / "child_hitl.yaml", CHILD_WITH_HUMAN)
        eng = WorkflowEngine(self._parent_spec(child), MockLLM())
        eng.run({"x": "任务E"}, thread_id="ph4")
        again = eng.resume("不批准", thread_id="ph4")
        assert WorkflowEngine.waiting_on(again) is not None  # 重新询问
        final = eng.resume("批准", thread_id="ph4")
        assert final["vars"]["sub"]["child_out"] == "子图批准:任务E"
