"""CLI 端到端测试：run / show / 错误码。"""

import json

import pytest

from flowagent.cli import main

CS = "configs/customer_service.yaml"


@pytest.fixture
def cs_path(project_root):
    return str(project_root / "configs" / "customer_service.yaml")


def test_run_success(cs_path, capsys):
    code = main(["run", cs_path, "--input", json.dumps({"ticket": "我要退款"}, ensure_ascii=False)])
    out = capsys.readouterr().out
    assert code == 0
    assert "客服处理结果" in out
    assert "MockLLM" in out


def test_run_with_input_file(cs_path, tmp_path, capsys):
    f = tmp_path / "input.json"
    f.write_text(json.dumps({"ticket": "退款"}, ensure_ascii=False), encoding="utf-8")
    code = main(["run", cs_path, "--input-file", str(f)])
    assert code == 0
    assert "执行轨迹" in capsys.readouterr().out


def test_show_prints_structure_and_mermaid(cs_path, capsys):
    code = main(["show", cs_path])
    out = capsys.readouterr().out
    assert code == 0
    assert "节点与流转" in out
    assert "Mermaid" in out
    assert "finalize" in out


def test_missing_config_returns_2(capsys):
    code = main(["run", "no/such/file.yaml"])
    assert code == 2
    assert "配置错误" in capsys.readouterr().err


def test_bad_input_json_returns_2(cs_path, capsys):
    code = main(["run", cs_path, "--input", "{broken json"])
    assert code == 2
    assert "输入错误" in capsys.readouterr().err


def test_non_object_input_returns_2(cs_path, capsys):
    code = main(["run", cs_path, "--input", "[1,2]"])
    assert code == 2


def test_runtime_failure_returns_1(project_root, tmp_path, capsys):
    cfg = tmp_path / "boom.yaml"
    cfg.write_text(
        "workflow:\n"
        "  name: boom\n"
        "  start: break\n"
        "nodes:\n"
        "  break:\n"
        "    type: tool\n"
        "    tool: flowagent.tools:unreliable_tool\n"
        "    args:\n"
        "      text: $ticket\n"
        "    output_key: r\n"
        "    next: END\n",
        encoding="utf-8",
    )
    code = main(["run", str(cfg), "--input", json.dumps({"ticket": "触发故障"}, ensure_ascii=False)])
    assert code == 1
    assert "运行失败" in capsys.readouterr().err


def test_error_degrade_prints_warning(project_root, tmp_path, capsys):
    cfg = tmp_path / "degrade.yaml"
    cfg.write_text(
        "workflow:\n"
        "  name: degrade\n"
        "  start: break\n"
        "nodes:\n"
        "  break:\n"
        "    type: tool\n"
        "    tool: flowagent.tools:unreliable_tool\n"
        "    args:\n"
        "      text: $ticket\n"
        "    output_key: r\n"
        "    on_error:\n"
        "      goto: done\n"
        "      set:\n"
        "        r: 已降级\n"
        "    next:\n"
        "      field: r\n"
        "      rules: []\n"
        "      default: done\n"
        "      on_error: done\n"
        "  done:\n"
        "    type: template\n"
        "    template: 结束:{r}\n"
        "    output_key: msg\n"
        "    next: END\n",
        encoding="utf-8",
    )
    code = main(["run", str(cfg), "--input", json.dumps({"ticket": "触发故障"}, ensure_ascii=False)])
    out = capsys.readouterr().out
    assert code == 0
    assert "已按 on_error 降级" in out


def test_python_dash_m_entry(project_root):
    """`python -m flowagent` 入口（__main__.py）可执行。"""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "flowagent", "show", "configs/customer_service.yaml"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=None,
    )
    assert result.returncode == 0
    assert "Mermaid" in result.stdout


# ---------------- v2 新命令 ----------------


def test_check_valid_config(project_root, capsys):
    cfg = str(project_root / "configs" / "approval.yaml")
    code = main(["check", cfg])
    out = capsys.readouterr().out
    assert code == 0
    assert "配置合法" in out
    assert "[ human  ]" in out


def test_check_invalid_config(project_root, tmp_path, capsys):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("workflow: {name: x, start: ghost}\nnodes: {}\n", encoding="utf-8")
    code = main(["check", str(cfg)])
    assert code == 2
    assert "配置错误" in capsys.readouterr().err


def test_run_stream_prints_progressive_output(project_root, capsys):
    cfg = str(project_root / "configs" / "customer_service.yaml")
    code = main(["run", cfg, "--stream", "--input", json.dumps({"ticket": "退款"})])
    out = capsys.readouterr().out
    assert code == 0
    assert "实时执行" in out
    assert "▸ classify" in out
    assert "输出变量" in out


def test_show_bad_config_returns_2(project_root, capsys):
    code = main(["show", "no/such/file.yaml"])
    assert code == 2
    assert "配置错误" in capsys.readouterr().err


def test_resume_bad_config_returns_2(project_root, capsys):
    code = main(["resume", "no/such/file.yaml", "--thread-id", "t", "--value", "x"])
    assert code == 2
    assert "配置错误" in capsys.readouterr().err


def test_run_stream_shows_error_degrade(project_root, tmp_path, capsys):
    """流式模式下 on_error 降级应实时打印错误条目。"""
    cfg = tmp_path / "degrade_stream.yaml"
    cfg.write_text(
        "workflow: {name: ds, start: break}\n"
        "nodes:\n"
        "  break:\n"
        "    type: tool\n"
        "    tool: flowagent.tools:unreliable_tool\n"
        "    args: {text: $ticket}\n"
        "    output_key: r\n"
        "    on_error: {goto: done, set: {r: 已降级}}\n"
        "    next: {field: r, rules: [], default: done, on_error: done}\n"
        "  done:\n"
        "    type: template\n"
        "    template: '结束:{r}'\n"
        "    output_key: msg\n"
        "    next: END\n",
        encoding="utf-8",
    )
    code = main(["run", str(cfg), "--stream", "--input", json.dumps({"ticket": "触发故障"})])
    out = capsys.readouterr().out
    assert code == 0
    assert "▸ [错误] break" in out
    assert "已按 on_error 降级" in out


def _with_sqlite_env(db_path):
    import os
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        keys = ("FLOWAGENT_CHECKPOINT", "FLOWAGENT_CHECKPOINT_PATH")
        old = {k: os.environ.get(k) for k in keys}
        os.environ["FLOWAGENT_CHECKPOINT"] = "sqlite"
        os.environ["FLOWAGENT_CHECKPOINT_PATH"] = str(db_path)
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _ctx()


class TestResumeInProcess:
    def test_resume_unknown_thread(self, project_root, tmp_path, capsys):
        cfg = tmp_path / "ap.yaml"
        cfg.write_text(APPROVAL_YAML, encoding="utf-8")
        with _with_sqlite_env(tmp_path / "u.sqlite"):
            code = main(["resume", str(cfg), "--thread-id", "ghost", "--value", "批准"])
        assert code == 1
        assert "未找到" in capsys.readouterr().err

    def test_resume_completed_thread(self, project_root, tmp_path, capsys):
        cfg = tmp_path / "ap.yaml"
        cfg.write_text(APPROVAL_YAML, encoding="utf-8")
        with _with_sqlite_env(tmp_path / "c.sqlite"):
            assert main(["run", str(cfg), "--thread-id", "t", "--input", '{"ticket": "B"}']) == 3
            assert main(["resume", str(cfg), "--thread-id", "t", "--value", "批准"]) == 0
            code = main(["resume", str(cfg), "--thread-id", "t", "--value", "批准"])
        assert code == 1
        assert "已结束" in capsys.readouterr().err

    def test_resume_that_raises_flowagent_error(self, project_root, tmp_path, capsys):
        """恢复后走到的节点抛业务异常（无 on_error）→ 退出码1。"""
        cfg = tmp_path / "boom_ap.yaml"
        cfg.write_text(
            "workflow: {name: ba, start: ask}\n"
            "nodes:\n"
            "  ask:\n"
            "    type: human\n"
            "    message: '批?{ticket}'\n"
            "    output_key: decision\n"
            "    next: boom\n"
            "  boom:\n"
            "    type: tool\n"
            "    tool: flowagent.tools:unreliable_tool\n"
            "    args: {text: $ticket}\n"
            "    output_key: r\n"
            "    next: END\n",
            encoding="utf-8",
        )
        with _with_sqlite_env(tmp_path / "b.sqlite"):
            assert main(["run", str(cfg), "--thread-id", "tb", "--input", '{"ticket": "触发故障"}']) == 3
            code = main(["resume", str(cfg), "--thread-id", "tb", "--value", "批准"])
        assert code == 1
        assert "恢复失败" in capsys.readouterr().err


APPROVAL_YAML = """\
workflow: {name: ap, start: ask}
nodes:
  ask:
    type: human
    message: "请审批：{ticket}"
    output_key: decision
    next:
      field: decision
      rules:
        - {when: {op: contains, value: 批准}, goto: END}
      default: END
"""


def _subprocess_env(db_path):
    import os

    env = dict(os.environ)
    env["FLOWAGENT_CHECKPOINT"] = "sqlite"
    env["FLOWAGENT_CHECKPOINT_PATH"] = str(db_path)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class TestHumanRoundtripCli:
    def test_pause_then_resume_across_processes(self, project_root, tmp_path):
        import subprocess
        import sys

        cfg = tmp_path / "ap.yaml"
        cfg.write_text(APPROVAL_YAML, encoding="utf-8")
        db = tmp_path / "hitl.sqlite"
        py = sys.executable
        env = _subprocess_env(db)

        r1 = subprocess.run(
            [py, "-m", "flowagent", "run", str(cfg), "--thread-id", "t1",
             "--input", json.dumps({"ticket": "工单A"})],
            capture_output=True, text=True, encoding="utf-8", timeout=120, env=env,
        )
        assert r1.returncode == 3, r1.stdout + r1.stderr
        assert "等待人工输入" in r1.stdout
        assert "请审批：工单A" in r1.stdout

        r2 = subprocess.run(
            [py, "-m", "flowagent", "resume", str(cfg), "--thread-id", "t1",
             "--value", "批准"],
            capture_output=True, text=True, encoding="utf-8", timeout=120, env=env,
        )
        assert r2.returncode == 0, r2.stdout + r2.stderr
        assert "decision = 批准" in r2.stdout

    def test_resume_unknown_thread_returns_1(self, project_root, tmp_path):
        import subprocess
        import sys

        cfg = tmp_path / "ap.yaml"
        cfg.write_text(APPROVAL_YAML, encoding="utf-8")
        db = tmp_path / "empty.sqlite"
        r = subprocess.run(
            [sys.executable, "-m", "flowagent", "resume", str(cfg),
             "--thread-id", "ghost", "--value", "批准"],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
            env=_subprocess_env(db),
        )
        assert r.returncode == 1
        assert "未找到" in r.stderr

    def test_resume_completed_thread_returns_1(self, project_root, tmp_path):
        import subprocess
        import sys

        cfg = tmp_path / "ap.yaml"
        cfg.write_text(APPROVAL_YAML, encoding="utf-8")
        db = tmp_path / "done.sqlite"
        env = _subprocess_env(db)
        # 第一次运行暂停在 human 节点
        subprocess.run(
            [sys.executable, "-m", "flowagent", "run", str(cfg), "--thread-id", "t9",
             "--input", json.dumps({"ticket": "B"})],
            capture_output=True, text=True, encoding="utf-8", timeout=120, env=env,
        )
        # 第一次恢复使流程完整结束
        r_done = subprocess.run(
            [sys.executable, "-m", "flowagent", "resume", str(cfg),
             "--thread-id", "t9", "--value", "批准"],
            capture_output=True, text=True, encoding="utf-8", timeout=120, env=env,
        )
        assert r_done.returncode == 0
        # 对已结束的线程再次恢复应被拦截
        r = subprocess.run(
            [sys.executable, "-m", "flowagent", "resume", str(cfg),
             "--thread-id", "t9", "--value", "批准"],
            capture_output=True, text=True, encoding="utf-8", timeout=120, env=env,
        )
        assert r.returncode == 1
        assert "已结束" in r.stderr

    def test_in_process_resume_after_pause(self, project_root, tmp_path, capsys):
        """进程内同引擎暂停→恢复（sqlite 不必需）。"""
        cfg = tmp_path / "ap.yaml"
        cfg.write_text(APPROVAL_YAML, encoding="utf-8")
        db = tmp_path / "mem.sqlite"
        import os

        old = {k: os.environ.get(k) for k in ("FLOWAGENT_CHECKPOINT", "FLOWAGENT_CHECKPOINT_PATH")}
        os.environ["FLOWAGENT_CHECKPOINT"] = "sqlite"
        os.environ["FLOWAGENT_CHECKPOINT_PATH"] = str(db)
        try:
            code1 = main(["run", str(cfg), "--thread-id", "m1", "--input", '{"ticket": "X"}'])
            assert code1 == 3
            code2 = main(["resume", str(cfg), "--thread-id", "m1", "--value", "拒绝"])
            out = capsys.readouterr().out
            assert code2 == 0
            assert "decision = 拒绝" in out
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
