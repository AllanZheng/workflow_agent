"""递归上限耗尽的友好报错与 --version 测试。"""

import pytest

from flowagent import FlowAgentError, WorkflowEngine, load_spec
from flowagent.cli import main
from flowagent.llm import MockLLM

pytestmark = pytest.mark.anyio


# 处理`anyio``backend`相关逻辑。
# English: Handle anyio backend operations.
@pytest.fixture
def anyio_backend():
    return "asyncio"

UNGUARDED_LOOP = {
    "workflow": {"name": "rl", "start": "bump", "recursion_limit": 5},
    "nodes": {
        "bump": {
            "type": "tool",
            "tool": "flowagent.tools:bump_counter",
            "args": {"count": "$count"},
            "output_key": "count",
            "next": {
                "field": "count",
                "rules": [
                    {
                        "when": {"op": "lt", "value": 9999},
                        "goto": "bump",
                        "max_loops": 100,
                        "loop_exhaust": "done",
                    }
                ],
                "default": "done",
            },
        },
        "done": {
            "type": "template",
            "template": "done",
            "output_key": "msg",
            "next": "END",
        },
    },
}


# 处理`engine`相关逻辑。
# English: Handle engine operations.
@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine(load_spec(UNGUARDED_LOOP), MockLLM())


class TestRecursionFriendlyError:
    # 验证“运行抛出`flowagent`错误包含`hint`”场景下的预期行为。
    # English: Verify run raises flowagent error with hint.
    def test_run_raises_flowagent_error_with_hint(self, engine):
        with pytest.raises(FlowAgentError) as exc_info:
            engine.run({})
        message = str(exc_info.value)
        assert "recursion_limit=5" in message
        assert "max_loops" in message  # 包含修复建议

    # 验证“`original``cause``preserved`”场景下的预期行为。
    # English: Verify original cause preserved.
    def test_original_cause_preserved(self, engine):
        from langgraph.errors import GraphRecursionError

        with pytest.raises(FlowAgentError) as exc_info:
            engine.run({})
        assert isinstance(exc_info.value.__cause__, GraphRecursionError)

    # 验证“流式处理抛出`flowagent`错误”场景下的预期行为。
    # English: Verify stream raises flowagent error.
    def test_stream_raises_flowagent_error(self, engine):
        with pytest.raises(FlowAgentError):
            list(engine.stream({}))

    # 验证“`arun`抛出`flowagent`错误”场景下的预期行为。
    # English: Verify arun raises flowagent error.
    async def test_arun_raises_flowagent_error(self, engine):
        with pytest.raises(FlowAgentError):
            await engine.arun({})

    # 验证“`cli`报告`failure`不包含`traceback`”场景下的预期行为。
    # English: Verify cli reports failure without traceback.
    def test_cli_reports_failure_without_traceback(self, tmp_path, capsys):
        import yaml

        cfg = tmp_path / "rl.yaml"
        cfg.write_text(yaml.safe_dump(UNGUARDED_LOOP, allow_unicode=True), encoding="utf-8")
        assert main(["run", str(cfg), "--input", "{}"]) == 1
        err = capsys.readouterr().err
        assert "运行失败" in err
        assert "recursion_limit=5" in err
        assert "Traceback" not in err


class TestVersionFlag:
    # 验证“版本`flag`”场景下的预期行为。
    # English: Verify version flag.
    def test_version_flag(self, capsys):
        import flowagent

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        assert f"flowagent {flowagent.__version__}" in capsys.readouterr().out
