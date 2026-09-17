"""脚手架命令与入门示例测试。"""

import json
import subprocess
import sys

from flowagent.cli import main


class TestScaffold:
    def test_new_then_check_strict_then_run(self, tmp_path, capsys):
        cfg = tmp_path / "my.yaml"
        assert main(["new", str(cfg), "--name", "my_flow"]) == 0
        out = capsys.readouterr().out
        assert "已生成工作流脚手架" in out
        # 生成的配置应通过严格校验
        assert main(["check", "--strict", str(cfg)]) == 0
        # 且可直接离线运行
        code = main(["run", str(cfg), "--input", json.dumps({"text": "今天真开心"})])
        assert code == 0
        out = capsys.readouterr().out
        assert "reply" in out

    def test_new_refuses_overwrite(self, tmp_path, capsys):
        cfg = tmp_path / "exists.yaml"
        cfg.write_text("x", encoding="utf-8")
        assert main(["new", str(cfg)]) == 2
        assert "已存在" in capsys.readouterr().err


def test_basic_usage_example_runs(project_root):
    result = subprocess.run(
        [sys.executable, str(project_root / "examples" / "basic_usage.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=None,
        cwd=str(project_root),
    )
    assert result.returncode == 0, result.stderr
    assert "Mermaid" in result.stdout
    assert "类别: 退款" in result.stdout


class TestVersionAndPackaging:
    """4.8：版本单一来源 + 包元数据完整。"""

    def test_installed_metadata_matches_runtime_version(self):
        from importlib.metadata import version
        import flowagent
        assert version("flowagent") == flowagent.__version__

    def test_console_script_registered(self):
        from importlib.metadata import entry_points
        eps = entry_points(group="console_scripts")
        assert any(ep.name == "flowagent" and ep.value.startswith("flowagent")
                   for ep in eps)

    def test_version_is_090_or_newer(self):
        import flowagent
        major, minor = (int(x) for x in flowagent.__version__.split(".")[:2])
        assert (major, minor) >= (0, 9)


def test_template_error_exported():
    """4.10：docs/API.md 声称可从顶层导入 TemplateError，必须真实导出。"""
    import flowagent
    assert flowagent.TemplateError is not None
    assert "TemplateError" in flowagent.__all__
