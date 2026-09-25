"""R5 特性测试：静态引用分析（check --strict）与 --output json。"""

import json
from pathlib import Path


from flowagent.cli import main
from flowagent.config import analyze_references, load_spec, load_spec_file

ROOT = Path(__file__).resolve().parents[1]

GOOD_SPEC = {
    "workflow": {"name": "g", "start": "a", "inputs": ["ticket"]},
    "nodes": {
        "a": {
            "type": "template",
            "template": "处理:{ticket}",
            "output_key": "summary",
            "next": "b",
        },
        "b": {
            "type": "template",
            "template": "摘要:{summary}",
            "output_key": "msg",
            "next": "END",
        },
    },
}

BAD_SPEC = {
    "workflow": {"name": "b", "start": "a", "inputs": []},
    "nodes": {
        "a": {
            "type": "template",
            "template": "处理:{ticket}的{unknown_var}",
            "output_key": "summary",
            "next": "END",
        }
    },
}


class TestAnalyzeReferences:
    def test_no_unresolved(self):
        assert analyze_references(load_spec(GOOD_SPEC)) == []

    def test_detects_unresolved(self):
        assert analyze_references(load_spec(BAD_SPEC)) == ["ticket", "unknown_var"]

    def test_on_error_set_counts_as_producer(self):
        spec = {
            "workflow": {"name": "x", "start": "a", "inputs": ["ticket"]},
            "nodes": {
                "a": {
                    "type": "template",
                    "template": "{ticket}",
                    "output_key": "summary",
                    "on_error": {"goto": "END", "set": {"fallback": "x"}},
                    "next": "END",
                }
            },
        }
        assert analyze_references(load_spec(spec)) == []

    def test_route_field_counted_as_reference(self):
        spec = {
            "workflow": {"name": "x", "start": "a", "inputs": ["ticket"]},
            "nodes": {
                "a": {
                    "type": "template",
                    "template": "{ticket}",
                    "output_key": "summary",
                    "next": {"field": "ghost", "rules": [], "default": "END"},
                }
            },
        }
        assert analyze_references(load_spec(spec)) == ["ghost"]

    def test_dotted_reference_uses_root(self):
        spec = {
            "workflow": {"name": "x", "start": "a", "inputs": []},
            "nodes": {
                "a": {
                    "type": "llm",
                    "prompt": "评分 {review.score}",
                    "parser": "json",
                    "output_key": "review",
                    "next": "END",
                }
            },
        }
        assert analyze_references(load_spec(spec)) == []

    def test_real_example_configs_pass_strict(self):
        for cfg in (
            "configs/customer_service.yaml",
            "configs/report_writer.yaml",
            "configs/approval.yaml",
            "configs/refund_approval.yaml",
            "configs/subflows/risk.yaml",
        ):
            assert analyze_references(load_spec_file(ROOT / cfg)) == [], cfg


class TestCheckStrictCli:
    def _write(self, tmp_path, name, data):
        import yaml

        cfg = tmp_path / name
        cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return str(cfg)

    def test_strict_passes_when_declared(self, tmp_path, capsys):
        code = main(["check", "--strict", self._write(tmp_path, "g.yaml", GOOD_SPEC)])
        assert code == 0
        assert "引用分析通过" in capsys.readouterr().out

    def test_strict_fails_on_unresolved(self, tmp_path, capsys):
        code = main(["check", "--strict", self._write(tmp_path, "b.yaml", BAD_SPEC)])
        assert code == 1
        assert "未解析引用" in capsys.readouterr().err

    def test_non_strict_warns_only(self, tmp_path, capsys):
        code = main(["check", self._write(tmp_path, "b2.yaml", BAD_SPEC)])
        out = capsys.readouterr().out
        assert code == 0
        assert "unknown_var" in out
        assert "⚠ 引用分析" in out


class TestJsonOutput:
    def test_run_output_json_finished(self, project_root, capsys):
        cfg = str(project_root / "configs" / "customer_service.yaml")
        code = main(
            ["run", cfg, "--output", "json", "--input", json.dumps({"ticket": "退款"})]
        )
        out = capsys.readouterr().out
        assert code == 0
        payload = json.loads(out[out.index("{"):])
        assert payload["status"] == "finished"
        assert payload["vars"]["category"] == "退款"
        assert payload["trace"][0]["node"] == "classify"

    def test_run_output_json_waiting_human(self, project_root, tmp_path, capsys):
        cfg = str(project_root / "configs" / "approval.yaml")
        with __import__("tests.test_cli", fromlist=["_with_postgres_env"])._with_postgres_env(
            tmp_path / "j.postgres"
        ):
            code = main(
                ["run", cfg, "--output", "json", "--thread-id", "j1",
                 "--input", json.dumps({"ticket": "T", "amount": 1})]
            )
        out = capsys.readouterr().out
        assert code == 3
        payload = json.loads(out[out.index("{"):])
        assert payload["status"] == "waiting_human"
        assert "请审批" in payload["pending"]["question"]
