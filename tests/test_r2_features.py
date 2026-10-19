"""第二轮优化特性测试：human 选项、交互式恢复、轨迹导出、钩子、白名单、子图透传。"""

import json

import pytest
import yaml
from langchain_core.messages import AIMessage

from flowagent import WorkflowEngine, load_spec
from flowagent.cli import main
from flowagent.errors import ConfigError
from flowagent.llm import MockLLM
from flowagent.nodes import NodeHooks
from flowagent.tools import load_tool

from .test_cli import APPROVAL_YAML, _with_postgres_env

APPROVAL_WITH_OPTIONS = """\
workflow: {name: apo, start: ask}
nodes:
  ask:
    type: human
    message: "请审批：{ticket}"
    options: [批准, 拒绝, 转交]
    output_key: decision
    next:
      field: decision
      rules:
        - {when: {op: contains, value: 批准}, goto: END}
      default: END
"""


class TestHumanOptions:
    # 验证“`options``included``interrupt`载荷”场景下的预期行为。
    # English: Verify options included in interrupt payload.
    def test_options_included_in_interrupt_payload(self):
        spec = load_spec(yaml.safe_load(APPROVAL_WITH_OPTIONS))
        engine = WorkflowEngine(spec, MockLLM())
        paused = engine.run({"ticket": "T"}, thread_id="opt1")
        payload = WorkflowEngine.waiting_on(paused)
        assert payload["options"] == ["批准", "拒绝", "转交"]
        # 检查点侧同样可读取（供 CLI 交互式恢复使用）
        snap = engine.get_state("opt1")
        assert WorkflowEngine.pending_payload(snap) == payload

    # 验证“待处理载荷无针对未知`thread`”场景下的预期行为。
    # English: Verify pending payload none for unknown thread.
    def test_pending_payload_none_for_unknown_thread(self):
        engine = WorkflowEngine(
            load_spec(
                {
                    "workflow": {"name": "n", "start": "a"},
                    "nodes": {
                        "a": {
                            "type": "template",
                            "template": "x",
                            "output_key": "o",
                            "next": "END",
                        }
                    },
                }
            ),
            MockLLM(),
        )
        engine.run({})
        assert WorkflowEngine.pending_payload(engine.get_state("whatever")) is None


class TestInteractiveResume:
    # 验证“`resume`不包含值`uses``interactive`输入”场景下的预期行为。
    # English: Verify resume without value uses interactive input.
    def test_resume_without_value_uses_interactive_input(self, tmp_path, capsys, monkeypatch):
        cfg = tmp_path / "ap.yaml"
        cfg.write_text(APPROVAL_YAML, encoding="utf-8")
        with _with_postgres_env(tmp_path / "i.postgres"):
            assert main(["run", str(cfg), "--thread-id", "it", "--input", '{"ticket": "C"}']) == 3
            answers = iter(["批准"])
            monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
            code = main(["resume", str(cfg), "--thread-id", "it"])
        out = capsys.readouterr().out
        assert code == 0
        assert "请输入" in out or "等待人工输入" in out
        assert "decision = 批准" in out

    # English: Verify value outside options reasks until valid.
    def test_value_outside_options_reasks_until_valid(self, tmp_path, capsys):
        """4.1 语义更新：超范围答复失败关闭——重新询问而非放行。"""
        cfg = tmp_path / "apo.yaml"
        cfg.write_text(APPROVAL_WITH_OPTIONS, encoding="utf-8")
        with _with_postgres_env(tmp_path / "w.postgres"):
            assert main(["run", str(cfg), "--thread-id", "w1", "--input", '{"ticket": "D"}']) == 3
            code = main(["resume", str(cfg), "--thread-id", "w1", "--value", "随便写"])
            assert code == 3  # 无效值 → 仍等待人工（重新询问），不放行
            code = main(["resume", str(cfg), "--thread-id", "w1", "--value", "批准"])
            assert code == 0  # 有效值 → 正常完成
        captured = capsys.readouterr()
        assert "不在选项" in captured.err
        assert "无效输入" in captured.out or "工作流已暂停" in captured.out


class TestTraceOut:
    # 验证“追踪导出`jsonl`”场景下的预期行为。
    # English: Verify trace export jsonl.
    def test_trace_export_jsonl(self, project_root, tmp_path, capsys):
        cfg = str(project_root / "configs" / "customer_service.yaml")
        trace_file = tmp_path / "trace.jsonl"
        code = main(
            ["run", cfg, "--input", '{"ticket": "退款"}', "--trace-out", str(trace_file)]
        )
        assert code == 0
        assert f"[轨迹已导出] {trace_file}" in capsys.readouterr().out
        lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
        nodes = [json.loads(line)["node"] for line in lines]
        assert nodes == ["classify", "refund", "review", "finalize"]


class CountingLLM:
    """非 Mock 计数 LLM：验证子图透传（复用父级实例而非重新创建）。"""

    name = "counting"

    # 初始化对象状态并建立所需依赖。
    # English: Initialize object state and required dependencies.
    def __init__(self):
        self.calls = 0

    # 处理`invoke`相关逻辑。
    # English: Handle invoke operations.
    def invoke(self, messages, config=None, **kwargs) -> AIMessage:
        self.calls += 1
        return AIMessage(content="计数LLM输出")


class TestSubgraphPassthrough:
    # 验证“`subgraph``reuses``parent`大语言模型`instance`”场景下的预期行为。
    # English: Verify subgraph reuses parent LLM instance.
    def test_subgraph_reuses_parent_llm_instance(self, tmp_path):
        child = tmp_path / "child.yaml"
        child.write_text(
            "workflow: {name: c, start: c1}\n"
            "nodes:\n"
            "  c1:\n"
            "    type: llm\n"
            "    prompt: 'p'\n"
            "    output_key: out\n"
            "    next: END\n",
            encoding="utf-8",
        )
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
        llm = CountingLLM()
        final = WorkflowEngine(spec, llm).run({})
        assert llm.calls == 1  # 子图节点用的是父级同一个 LLM 实例
        assert final["vars"]["sub"]["out"] == "计数LLM输出"


class TestToolAllowlist:
    # English: Verify default denies external modules.
    def test_default_denies_external_modules(self, monkeypatch):
        """4.9 语义更新：未配置白名单时默认只放行内置工具（失败关闭）。"""
        monkeypatch.delenv("FLOWAGENT_TOOL_MODULES", raising=False)
        with pytest.raises(ConfigError):
            load_tool("tests.flaky:flaky_tool")

    # 验证“`module``allowlist`”场景下的预期行为。
    # English: Verify module in allowlist.
    def test_module_in_allowlist(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "flowagent.tools")
        assert callable(load_tool("flowagent.tools:knowledge_base"))

    # 验证“`module``outside``allowlist``rejected`”场景下的预期行为。
    # English: Verify module outside allowlist rejected.
    def test_module_outside_allowlist_rejected(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "flowagent.tools")
        with pytest.raises(ConfigError):
            load_tool("tests.flaky:flaky_tool")

    # 验证“`allowlist``prefix`匹配”场景下的预期行为。
    # English: Verify allowlist prefix match.
    def test_allowlist_prefix_match(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "flowagent")
        assert callable(load_tool("flowagent.tools:knowledge_base"))


class TestNodeHooks:
    # 处理`spec`相关逻辑。
    # English: Handle spec operations.
    def _spec(self):
        return load_spec(
            {
                "workflow": {"name": "hk", "start": "a"},
                "nodes": {
                    "a": {
                        "type": "template",
                        "template": "hi",
                        "output_key": "o",
                        "next": "END",
                    }
                },
            }
        )

    # 验证“`hooks`接收启动与`end`”场景下的预期行为。
    # English: Verify hooks receive start and end.
    def test_hooks_receive_start_and_end(self):
        events = []
        hooks = NodeHooks(
            on_node_start=lambda name, state: events.append(("start", name)),
            on_node_end=lambda name, result, error: events.append(("end", name, error)),
        )
        final = WorkflowEngine(self._spec(), MockLLM(), hooks=hooks).run({})
        assert events == [("start", "a"), ("end", "a", None)]
        assert final["vars"]["o"] == "hi"

    # 验证“`broken``hook``does`不`break`工作流”场景下的预期行为。
    # English: Verify broken hook does not break workflow.
    def test_broken_hook_does_not_break_workflow(self, caplog):
        # 处理`bad``hook`相关逻辑。
        # English: Handle bad hook operations.
        def bad_hook(*args, **kwargs):
            raise RuntimeError("钩子炸了")

        hooks = NodeHooks(on_node_start=bad_hook, on_node_end=bad_hook)
        final = WorkflowEngine(self._spec(), MockLLM(), hooks=hooks).run({})
        assert final["vars"]["o"] == "hi"
        # 钩子错误改为结构化日志（flowagent.hook_failed），不再 print 到 stderr
        records = [r for r in caplog.records
                   if getattr(r, "event", None) == "flowagent.hook_failed"]
        assert len(records) == 2  # start + end 各一条
        assert all(r.exc_info for r in records)  # 带异常堆栈
