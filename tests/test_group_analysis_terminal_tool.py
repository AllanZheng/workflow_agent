"""群分析终端工具（GAT-01 ~ GAT-06）：报告原样作为回复，计数由程序计算。"""
import asyncio
import json

import pytest

from flowagent.agent_runtime import ManagedAgentRuntime
from flowagent.group_analysis import run_group_analysis
from flowagent.local_tools import GROUP_ANALYSIS_TOOL_NAME, build_group_analysis_tool
from flowagent.execution.planner import select_flow

from tests.af_helpers import PinnedMcp, ScriptedLlm, group_input, tool_call
from flowagent.function_stage import FunctionTask


class _Target:
    def __init__(self, conversation_id):
        self.conversation_id = conversation_id


def _runtime(llm, mcp, target=None):
    tool = build_group_analysis_tool(llm, mcp, target)
    return ManagedAgentRuntime(llm, mcp, local_tools=[tool])


def _run(coro):
    return asyncio.run(coro)


# GAT-01：只有 Edge 列出 fetch_group_analysis_input 时才注册终端工具。
def test_gat01_tool_registered_only_when_fetch_tool_is_visible():
    llm = ScriptedLlm([])
    visible = PinnedMcp(["fetch_group_analysis_input"])
    # 远端工具对本次执行不可见时，本地工具也不暴露。
    runtime_hidden = _runtime(llm, PinnedMcp(["fetch_group_analysis_input"], visible=[]), _Target(5))
    _, schemas, _ = _run(runtime_hidden._setup("hi", "", None))
    assert GROUP_ANALYSIS_TOOL_NAME not in {item["function"]["name"] for item in schemas}
    runtime_shown = _runtime(llm, visible, _Target(5))
    _, schemas, _ = _run(runtime_shown._setup("hi", "", None))
    assert GROUP_ANALYSIS_TOOL_NAME in {item["function"]["name"] for item in schemas}


# GAT-02 / GAT-03：模型调用工具后循环结束，报告与程序统计一致，统计不会被模型改写。
def test_gat02_gat03_report_is_returned_verbatim_with_program_counts():
    mcp = PinnedMcp(["fetch_group_analysis_input"],
                    {"fetch_group_analysis_input": group_input(5, 4)})
    # 报告文本里模型“声称”的数字与程序统计不同，最终统计块必须以程序为准。
    llm = ScriptedLlm(
        [tool_call(GROUP_ANALYSIS_TOOL_NAME, {"days": 1})],
        report_text="## 总结\n模型声称问题类有 99 条。\n\n## 分类统计\n- 问题：99 条\n\n## 汇报要点\n- 要点",
    )
    runtime = _runtime(llm, mcp, _Target(5))
    reply = _run(runtime.run("分析本群", "", 8))
    assert llm.bound_calls == 1                      # 调用工具后不再回到模型
    assert "- 问题：2 条（50.0%）" in reply and "- 通知：2 条（50.0%）" in reply
    assert "- 问题：99 条" not in reply
    result = runtime.terminal_metadata["analysis_result"]
    assert result["conversation_id"] == 5 and result["status"] == "complete"
    by_name = {item["name"]: item["count"] for item in result["statistics"]["categories"]}
    assert by_name["问题"] == 2 and by_name["通知"] == 2
    assert result["statistics"]["total_messages"] == 4


# GAT-04：流式只发一个文本增量（报告本身）。
def test_gat04_stream_emits_single_delta():
    mcp = PinnedMcp(["fetch_group_analysis_input"],
                    {"fetch_group_analysis_input": group_input(5, 2)})
    llm = ScriptedLlm([tool_call(GROUP_ANALYSIS_TOOL_NAME, {})])
    runtime = _runtime(llm, mcp, _Target(5))

    async def collect():
        return [piece async for piece in runtime.stream("分析本群", "", 8)]

    pieces = _run(collect())
    assert len(pieces) == 1 and pieces[0].startswith("## 总结")
    assert runtime.terminal_metadata["analysis_result"]["conversation_id"] == 5


# GAT-05：已锁定目标群时，工具参数里的其他群 ID 被忽略，仍只读取锁定的群。
def test_gat05_locked_target_ignores_other_conversation_id():
    mcp = PinnedMcp(["fetch_group_analysis_input"],
                    {"fetch_group_analysis_input": group_input(5, 2)})
    llm = ScriptedLlm([tool_call(GROUP_ANALYSIS_TOOL_NAME, {"conversation_id": 99})])
    _run(_runtime(llm, mcp, _Target(5)).run("分析", "", 8))
    fetches = [args for remote, args in mcp.calls if remote == "fetch_group_analysis_input"]
    assert fetches and all(args["conversation_id"] == 5 for args in fetches)


# 未锁定目标群且没有给出群信息时，提示选择群而不是猜测。
def test_no_target_and_no_group_asks_user_to_choose():
    mcp = PinnedMcp(["fetch_group_analysis_input"], {})
    llm = ScriptedLlm([tool_call(GROUP_ANALYSIS_TOOL_NAME, {})])
    runtime = _runtime(llm, mcp, None)
    reply = _run(runtime.run("分析群", "", 8))
    assert "请选择要分析的群" in reply
    assert runtime.terminal_metadata == {"analysis_result": None}
    assert mcp.calls == []


# 非法参数直接拒绝，不做任何远端调用。
def test_unknown_arguments_are_rejected():
    mcp = PinnedMcp(["fetch_group_analysis_input"], {})
    llm = ScriptedLlm([tool_call(GROUP_ANALYSIS_TOOL_NAME, {"evil": 1})])
    with pytest.raises(ValueError, match="invalid_group_analysis_arguments"):
        _run(_runtime(llm, mcp, _Target(5)).run("分析", "", 8))
    assert mcp.calls == []


# GAT-06 回归：直接调用共享流水线（V2 同步模型路径）结果与异步路径一致。
def test_gat06_shared_pipeline_sync_and_async_agree():
    class SyncLlm(ScriptedLlm):
        def invoke(self, messages):
            return asyncio.run(self.ainvoke(messages))

    async def call_tool(name, arguments):
        return group_input(5, 4)

    parsed = {"group_query": "", "group_id": None, "start": "", "end": "", "days": None,
              "taxonomy_override": ""}
    sync_reply, sync_meta = _run(run_group_analysis(
        parsed, _Target(5), SyncLlm([]), call_tool, lambda: "sys", use_async=False))
    async_reply, async_meta = _run(run_group_analysis(
        parsed, _Target(5), ScriptedLlm([]), call_tool, lambda: "sys", use_async=True))
    assert sync_meta["analysis_result"]["statistics"] == async_meta["analysis_result"]["statistics"]
    assert json.dumps(sync_meta, sort_keys=True) == json.dumps(async_meta, sort_keys=True)


# FST-01（路由部分）：function_task 优先于 V3；V3 无 function_task 走 qa；旧路由不变。
def test_fst01_function_task_routes_before_v3():
    class V3Config:
        snapshot_v3 = object()

    class V2Spec:
        class workflow:
            group_analysis_mode = "deterministic"

    task = FunctionTask(function_key="risk_consultation", stage="draft",
                        output_schema="consultation_document_v1")
    assert select_flow(V3Config(), function_task=task) == "function_stage"
    assert select_flow(V3Config()) == "qa"
    empty = type("C", (), {"snapshot_v3": None})()
    assert select_flow(empty, V2Spec()) == "group_analysis"
    assert select_flow(empty, None) == "managed_yaml"
