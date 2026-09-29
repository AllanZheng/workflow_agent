"""异步 API（arun/astream）与节点耗时观测测试。"""

import pytest

from flowagent import WorkflowEngine, load_spec

pytestmark = pytest.mark.anyio

SPEC = {
    "workflow": {"name": "ax", "start": "a"},
    "nodes": {
        "a": {
            "type": "template",
            "template": "你好 {x}",
            "output_key": "msg",
            "next": "END",
        }
    },
}


# 处理`anyio``backend`相关逻辑。
# English: Handle anyio backend operations.
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 验证“`arun``returns``final`状态”场景下的预期行为。
# English: Verify arun returns final state.
async def test_arun_returns_final_state():
    engine = WorkflowEngine(load_spec(SPEC))
    final = await engine.arun({"x": "异步"}, thread_id="ax1")
    assert final["vars"]["msg"] == "你好 异步"
    snap = engine.get_state("ax1").values
    assert snap["vars"]["msg"] == "你好 异步"


# 验证“`astream``yields``steps`”场景下的预期行为。
# English: Verify astream yields steps.
async def test_astream_yields_steps():
    engine = WorkflowEngine(load_spec(SPEC))
    chunks = [chunk async for chunk in engine.astream({"x": "流"}, thread_id="ax2")]
    assert len(chunks) >= 1
    assert chunks[-1]["vars"]["msg"] == "你好 流"


# 验证“追踪纳入`duration``ms`”场景下的预期行为。
# English: Verify trace includes duration ms.
def test_trace_includes_duration_ms():
    engine = WorkflowEngine(load_spec(SPEC))
    final = engine.run({"x": "计时"})
    entry = final["trace"][0]
    assert isinstance(entry["duration_ms"], int)
    assert entry["duration_ms"] >= 0
