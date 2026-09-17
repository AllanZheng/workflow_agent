"""递归上限耗尽的友好报错与 --version 测试。"""

import pytest

from flowagent import FlowAgentError, WorkflowEngine, load_spec
from flowagent.cli import main
from flowagent.llm import MockLLM

pytestmark = pytest.mark.anyio


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


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine(load_spec(UNGUARDED_LOOP), MockLLM())


class TestRecursionFriendlyError:
    def test_run_raises_flowagent_error_with_hint(self, engine):
        with pytest.raises(FlowAgentError) as exc_info:
            engine.run({})
        message = str(exc_info.value)
        assert "recursion_limit=5" in message
        assert "max_loops" in message  # 包含修复建议

    def test_original_cause_preserved(self, engine):
        from langgraph.errors import GraphRecursionError

        with pytest.raises(FlowAgentError) as exc_info:
            engine.run({})
        assert isinstance(exc_info.value.__cause__, GraphRecursionError)

    def test_stream_raises_flowagent_error(self, engine):
        with pytest.raises(FlowAgentError):
            list(engine.stream({}))

    async def test_arun_raises_flowagent_error(self, engine):
        with pytest.raises(FlowAgentError):
            await engine.arun({})

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
    def test_version_flag(self, capsys):
        import flowagent

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        assert f"flowagent {flowagent.__version__}" in capsys.readouterr().out
