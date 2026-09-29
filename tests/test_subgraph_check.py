"""子图输入交叉检查测试：父级上下文需满足子图声明的 inputs。"""

from flowagent import WorkflowEngine, load_spec, load_spec_file
from flowagent.config import check_spec
from flowagent.llm import MockLLM

GOOD_PARENT = """\
workflow: {{name: gp, start: p1, inputs: [x]}}
nodes:
  p1:
    type: workflow
    workflow_file: {child}
    output_key: sub
    next: END
"""

GOOD_CHILD = """\
workflow: {name: gc, start: c1, inputs: [x]}
nodes:
  c1:
    type: template
    template: "子图收到:{x}"
    output_key: out
    next: END
"""


# 执行写入对应的流程。
def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestSubgraphInputCrossCheck:
    # 验证“`satisfied`输入无`warning`”场景下的预期行为。
    def test_satisfied_inputs_no_warning(self, tmp_path):
        child = _write(tmp_path / "c.yaml", GOOD_CHILD)
        parent = _write(tmp_path / "p.yaml", GOOD_PARENT.format(child=child))
        assert check_spec(load_spec_file(parent)) == []

    # 验证“缺失输入`reported`”场景下的预期行为。
    def test_missing_input_reported(self, tmp_path):
        child = _write(tmp_path / "c.yaml", GOOD_CHILD)
        bad_parent = GOOD_PARENT.format(child=child).replace(
            "inputs: [x]", "inputs: []"
        )
        parent = _write(tmp_path / "p.yaml", bad_parent)
        warnings = check_spec(load_spec_file(parent))
        assert any("x" in w and "未提供" in w for w in warnings)

    # 验证“`cli`检查`surfaces``subgraph``warning`”场景下的预期行为。
    def test_cli_check_surfaces_subgraph_warning(self, tmp_path, capsys):
        child = _write(tmp_path / "c.yaml", GOOD_CHILD)
        bad_parent = GOOD_PARENT.format(child=child).replace(
            "inputs: [x]", "inputs: []"
        )
        parent = _write(tmp_path / "p.yaml", bad_parent)
        from flowagent.cli import main

        code = main(["check", "--strict", parent])
        out = capsys.readouterr().out
        assert code == 1
        assert "未提供" in out

    def test_broken_child_reference_skips_crosscheck(self, tmp_path):
        """子图文件缺失时由加载流程报错，交叉检查不重复崩溃。"""
        spec = load_spec(
            {
                "workflow": {"name": "p", "start": "p1"},
                "nodes": {
                    "p1": {
                        "type": "workflow",
                        "workflow_file": "ghost.yaml",
                        "next": "END",
                    }
                },
            }
        )
        assert check_spec(spec) == []

    # 验证“`warning``does`不`block`执行”场景下的预期行为。
    def test_warning_does_not_block_execution(self, tmp_path):
        child = _write(tmp_path / "c.yaml", GOOD_CHILD)
        parent = _write(tmp_path / "p.yaml", GOOD_PARENT.format(child=child))
        engine = WorkflowEngine(load_spec_file(parent), MockLLM())
        final = engine.run({"x": "值"})
        assert final["vars"]["sub"]["out"] == "子图收到:值"


LEAF = """\
workflow: {{name: leaf, start: n, inputs: [{need}]}}
nodes:
  n:
    type: template
    template: "叶子收到:{{{need}}}"
    output_key: leaf_out
    next: END
"""

# 中间层：消费 x，产出 mid，并把 leaf 作为更深层子图（leaf 需要变量 deep）
MIDDLE_BAD = """\
workflow: {{name: mid, start: m1, inputs: [x]}}
nodes:
  m1:
    type: template
    template: "中间层处理:{{x}}"
    output_key: mid
    next: w2
  w2:
    type: workflow
    workflow_file: {leaf}
    output_key: sub2
    next: END
"""

MIDDLE_GOOD = """\
workflow: {{name: mid, start: m1, inputs: [x]}}
nodes:
  m1:
    type: template
    template: "中间层处理:{{x}}，并产出 deep 内容"
    output_key: mid
    next: w2
  deep_src:
    type: template
    template: "deep 值"
    output_key: deep
    next: w2
  w2:
    type: workflow
    workflow_file: {leaf}
    output_key: sub2
    next: END
"""


class TestRecursiveChainCheck:
    # 处理`three``level`相关逻辑。
    def _three_level(self, tmp_path, middle_text):
        leaf = _write(tmp_path / "leaf.yaml", LEAF.format(need="deep"))
        _write(tmp_path / "middle.yaml", middle_text.format(leaf=leaf))
        parent = _write(
            tmp_path / "p.yaml",
            "workflow: {name: tp, start: p1, inputs: [x]}\n"
            "nodes:\n"
            "  p1:\n"
            "    type: workflow\n"
            "    workflow_file: middle.yaml\n"
            "    output_key: out\n"
            "    next: END\n",
        )
        return load_spec_file(parent)

    # 验证“`deep``chain``break`判断是否`reported`包含路径”场景下的预期行为。
    def test_deep_chain_break_is_reported_with_path(self, tmp_path):
        spec = self._three_level(tmp_path, MIDDLE_BAD)
        warnings = check_spec(spec)
        # leaf 需要的 deep 由更深层供给——应被递归发现且标注链路
        assert any("deep" in w and "链路" in w for w in warnings)

    # 验证“`satisfied``deep``chain`检查是否包含无`warning`”场景下的预期行为。
    def test_satisfied_deep_chain_has_no_warning(self, tmp_path):
        spec = self._three_level(tmp_path, MIDDLE_GOOD)
        assert check_spec(spec) == []

    # 验证“`cli``strict``catches``deep``break`”场景下的预期行为。
    def test_cli_strict_catches_deep_break(self, tmp_path, capsys):
        from flowagent.cli import main

        spec = self._three_level(tmp_path, MIDDLE_BAD)
        import yaml

        cfg = tmp_path / "chain.yaml"
        cfg.write_text(
            yaml.safe_dump(spec.model_dump(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        code = main(["check", "--strict", str(cfg)])
        out = capsys.readouterr().out
        assert code == 1
        assert "deep" in out
