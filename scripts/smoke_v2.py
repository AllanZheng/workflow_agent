"""新特性冒烟脚本：human HITL / 子图 / 重试 / 超时 / sqlite / 流式。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3
import tempfile

from langgraph.checkpoint.sqlite import SqliteSaver

from flowagent import WorkflowEngine, load_spec
from flowagent.llm import MockLLM

tmp = Path(tempfile.mkdtemp())

# ---------- 1) human 节点（interrupt + resume） ----------
hitl_spec = load_spec(
    {
        "workflow": {"name": "hitl", "start": "ask"},
        "nodes": {
            "ask": {
                "type": "human",
                "message": "请审批工单：{ticket}（批准/拒绝）",
                "output_key": "decision",
                "next": {
                    "field": "decision",
                    "rules": [{"when": {"op": "eq", "value": "批准"}, "goto": "END"}],
                    "default": "redo",
                },
            },
            "redo": {
                "type": "template",
                "template": "已驳回：{ticket}",
                "output_key": "msg",
                "next": "END",
            },
        },
    }
)
engine = WorkflowEngine(hitl_spec, MockLLM())
paused = engine.run({"ticket": "退款申请#42"}, thread_id="h1")
payload = WorkflowEngine.waiting_on(paused)
print("1) HITL 暂停:", payload)
assert payload["question"].startswith("请审批工单")
resumed = engine.resume("拒绝", thread_id="h1")
print("1) HITL 恢复:", resumed["vars"]["decision"], "->", resumed["vars"]["msg"])
assert resumed["vars"]["decision"] == "拒绝"
assert resumed["loop_counts"]["ask"] == 1  # interrupt 未提交，恢复后完整执行一次

# ---------- 2) workflow 子图 ----------
child = tmp / "child.yaml"
child.write_text(
    "workflow: {name: child, start: c1}\n"
    "nodes:\n"
    "  c1:\n"
    "    type: template\n"
    "    template: '子图处理了:{x}'\n"
    "    output_key: child_out\n"
    "    next: END\n",
    encoding="utf-8",
)
parent_spec = load_spec(
    {
        "workflow": {"name": "parent", "start": "p1"},
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
sub_final = WorkflowEngine(parent_spec, MockLLM()).run({"x": "任务A"})
print("2) 子图:", sub_final["vars"]["final"])
assert "子图处理了:任务A" in sub_final["vars"]["final"]

# ---------- 3) 重试 ----------
retry_spec = load_spec(
    {
        "workflow": {"name": "retry", "start": "flaky"},
        "nodes": {
            "flaky": {
                "type": "tool",
                "tool": "tests.flaky:flaky_tool",
                "args": {"key": "r1", "fail_times": 2},
                "retries": 3,
                "retry_delay": 0,
                "output_key": "r",
                "next": "END",
            }
        },
    }
)
final = WorkflowEngine(retry_spec, MockLLM()).run({})
print("3) 重试:", final["vars"]["r"], "调用次数=", end=" ")
import tests.flaky as flaky  # noqa: E402

print(flaky.calls("r1"))
assert final["vars"]["r"] == "ok after 3 calls"

# ---------- 4) 超时 + on_error ----------
timeout_spec = load_spec(
    {
        "workflow": {"name": "timeout", "start": "slow"},
        "nodes": {
            "slow": {
                "type": "tool",
                "tool": "flowagent.tools:slow_tool",
                "args": {"seconds": 2},
                "timeout": 0.3,
                "output_key": "r",
                "on_error": {"goto": "fallback", "set": {"r": "已超时降级"}},
                "next": {"field": "r", "rules": [], "default": "fallback", "on_error": "fallback"},
            },
            "fallback": {
                "type": "template",
                "template": "降级完成:{r}",
                "output_key": "msg",
                "next": "END",
            },
        },
    }
)
final = WorkflowEngine(timeout_spec, MockLLM()).run({})
print("4) 超时降级:", final["error"]["type"], "->", final["vars"]["msg"])
assert final["error"]["type"] == "TimeoutError"

# ---------- 5) sqlite checkpointer 跨实例持久化 ----------
db = tmp / "ck.sqlite"
import os  # noqa: E402

os.environ["FLOWAGENT_CHECKPOINT"] = "sqlite"
os.environ["FLOWAGENT_CHECKPOINT_PATH"] = str(db)
from flowagent.engine import build_graph  # noqa: E402

build_graph(retry_spec, MockLLM())  # 环境变量生效
from flowagent.engine import make_checkpointer as mk  # noqa: E402

cp = mk()
print("5) checkpointer:", type(cp).__name__)
assert type(cp).__name__ == "SqliteSaver"
del os.environ["FLOWAGENT_CHECKPOINT"]

engine2 = WorkflowEngine(retry_spec, MockLLM(), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
e2 = engine2.run({"hello": "world"}, thread_id="persist-1")
engine3 = WorkflowEngine(retry_spec, MockLLM(), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
snap = engine3.get_state("persist-1").values
print("5) 跨实例持久化: hello =", snap["vars"].get("hello"))
assert snap["vars"]["hello"] == "world"

# ---------- 6) 流式 ----------
chunks = list(WorkflowEngine(parent_spec, MockLLM()).stream({"x": "流"}, thread_id="s1"))
print("6) 流式块数:", len(chunks), "末块含 final =", "final" in chunks[-1]["vars"])
assert len(chunks) >= 2 and "final" in chunks[-1]["vars"]

print("\nSMOKE-ALL-OK")
