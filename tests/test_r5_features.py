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
    # 验证“无`unresolved`”场景下的预期行为。
    # English: Verify no unresolved.
    def test_no_unresolved(self):
        assert analyze_references(load_spec(GOOD_SPEC)) == []

    # 验证“`detects``unresolved`”场景下的预期行为。
    # English: Verify detects unresolved.
    def test_detects_unresolved(self):
        assert analyze_references(load_spec(BAD_SPEC)) == ["ticket", "unknown_var"]

    # 验证“启用错误设置数量`as``producer`”场景下的预期行为。
    # English: Verify on error set counts as producer.
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

    # 验证“路由字段`counted``as``reference`”场景下的预期行为。
    # English: Verify route field counted as reference.
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

    # 验证“`dotted``reference``uses``root`”场景下的预期行为。
    # English: Verify dotted reference uses root.
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

    # 验证“`real``example`配置`pass``strict`”场景下的预期行为。
    # English: Verify real example configs pass strict.
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
    # 执行写入对应的流程。
    # English: Write the current operation.
    def _write(self, tmp_path, name, data):
        import yaml

        cfg = tmp_path / name
        cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return str(cfg)

    # 验证“`strict``passes`当`declared`”场景下的预期行为。
    # English: Verify strict passes when declared.
    def test_strict_passes_when_declared(self, tmp_path, capsys):
        code = main(["check", "--strict", self._write(tmp_path, "g.yaml", GOOD_SPEC)])
        assert code == 0
        assert "引用分析通过" in capsys.readouterr().out

    # 验证“`strict`标记失败启用`unresolved`”场景下的预期行为。
    # English: Verify strict fails on unresolved.
    def test_strict_fails_on_unresolved(self, tmp_path, capsys):
        code = main(["check", "--strict", self._write(tmp_path, "b.yaml", BAD_SPEC)])
        assert code == 1
        assert "未解析引用" in capsys.readouterr().err

    # 验证“`non``strict``warns`仅”场景下的预期行为。
    # English: Verify non strict warns only.
    def test_non_strict_warns_only(self, tmp_path, capsys):
        code = main(["check", self._write(tmp_path, "b2.yaml", BAD_SPEC)])
        out = capsys.readouterr().out
        assert code == 0
        assert "unknown_var" in out
        assert "⚠ 引用分析" in out


class TestJsonOutput:
    # 验证“运行输出JSON`finished`”场景下的预期行为。
    # English: Verify run output JSON finished.
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

    # 验证“运行输出JSON`waiting``human`”场景下的预期行为。
    # English: Verify run output JSON waiting human.
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
