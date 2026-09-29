"""R9 特性测试：tool.args $引用静态分析 + CLI run 输入键提示。"""

import json
from pathlib import Path

from flowagent.cli import main
from flowagent.config import analyze_references, load_spec

TOOL_SPEC = {
    "workflow": {"name": "t", "start": "a", "inputs": ["ticket"]},
    "nodes": {
        "a": {
            "type": "tool",
            "tool": "flowagent.tools:knowledge_base",
            "args": {"query": "$ticket", "limit": 3},
            "output_key": "answer",
            "next": "END",
        }
    },
}

TOOL_SPEC_BAD = {
    "workflow": {"name": "t", "start": "a", "inputs": []},
    "nodes": {
        "a": {
            "type": "tool",
            "tool": "flowagent.tools:knowledge_base",
            "args": {"query": "$tickit"},
            "output_key": "answer",
            "next": "END",
        }
    },
}


class TestToolArgReferences:
    # 验证“`dollar``ref``declared``passes`”场景下的预期行为。
    def test_dollar_ref_declared_passes(self):
        assert analyze_references(load_spec(TOOL_SPEC)) == []

    # 验证“`dollar``ref``undeclared``reported`”场景下的预期行为。
    def test_dollar_ref_undeclared_reported(self):
        assert analyze_references(load_spec(TOOL_SPEC_BAD)) == ["tickit"]

    # 验证“`dollar``ref``dotted``uses``root`”场景下的预期行为。
    def test_dollar_ref_dotted_uses_root(self):
        spec = {
            "workflow": {"name": "t", "start": "a", "inputs": ["ticket"]},
            "nodes": {
                "a": {
                    "type": "tool",
                    "tool": "flowagent.tools:knowledge_base",
                    "args": {"query": "$meta.ticket"},
                    "output_key": "answer",
                    "next": "END",
                }
            },
        }
        # $meta.ticket 的根段是 meta——需要声明 meta 而非 ticket
        assert analyze_references(load_spec(spec)) == ["meta"]
        spec["workflow"]["inputs"] = ["meta"]
        assert analyze_references(load_spec(spec)) == []


CS_CFG = "configs/customer_service.yaml"


class TestInputHints:
    # 验证“`matching`输入无`hint`”场景下的预期行为。
    def test_matching_inputs_no_hint(self, project_root, capsys):
        cfg = str(Path(project_root) / CS_CFG)
        code = main(["run", cfg, "--input", json.dumps({"ticket": "退款"})])
        err = capsys.readouterr().err
        assert code == 0
        assert "输入提示" not in err

    # 验证“`typo`密钥`warns``but``proceeds`”场景下的预期行为。
    def test_typo_key_warns_but_proceeds(self, project_root, capsys):
        cfg = str(Path(project_root) / CS_CFG)
        code = main(["run", cfg, "--input", json.dumps({"tickit": "退款"})])
        err = capsys.readouterr().err
        assert code == 0
        assert "未在 workflow.inputs 中声明" in err
        assert "tickit" in err

    # 验证“缺失`declared`密钥`warns`”场景下的预期行为。
    def test_missing_declared_key_warns(self, project_root, capsys):
        cfg = str(Path(project_root) / CS_CFG)
        code = main(["run", cfg, "--input", json.dumps({})])
        err = capsys.readouterr().err
        assert code == 0
        assert "已声明但本次未提供" in err
        assert "ticket" in err

    # 验证“无`hints`当输入`undeclared`”场景下的预期行为。
    def test_no_hints_when_inputs_undeclared(self, tmp_path, capsys):
        cfg = tmp_path / "noinputs.yaml"
        cfg.write_text(
            "workflow: {name: ni, start: a}\n"
            "nodes:\n"
            "  a:\n"
            "    type: template\n"
            "    template: '任意:{whatever}'\n"
            "    output_key: o\n"
            "    next: END\n",
            encoding="utf-8",
        )
        code = main(["run", str(cfg), "--input", json.dumps({"whatever": "x"})])
        err = capsys.readouterr().err
        assert code == 0
        assert "输入提示" not in err
