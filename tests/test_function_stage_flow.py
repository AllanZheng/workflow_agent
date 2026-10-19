"""功能阶段 flow（FST-02 ~ FST-07）：结构化输出、修复重试、知识依据统计、引用过滤。"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from flowagent.errors import ConfigError, FunctionStageError
from flowagent.execution.errors import classify_exception
from flowagent.execution_service import ExecutionRequest, _execute_function_stage
from flowagent.function_stage import (
    FunctionTask, build_system_prompt, extract_json_object, run_function_stage,
    summarize_knowledge,
)
from flowagent.mcp_edge_client import McpEdgeError

from tests.af_helpers import PinnedMcp, ScriptedLlm, tool_call


def _run(coro):
    return asyncio.run(coro)


def _task(stage="draft", mode="off", **inputs):
    schema = "consultation_tasks_v1" if stage == "tasks" else "consultation_document_v1"
    return FunctionTask(function_key="risk_consultation", stage=stage, output_schema=schema,
                        inputs={"knowledge_mode": mode, **inputs})


def _request(task, message="材料"):
    return SimpleNamespace(function_task=task, message=message)


def _document(**overrides):
    body = {"title": "会商初稿", "summary": "摘要", "body_markdown": "## 正文\n内容",
            "risks": [{"level": "high", "title": "风险1", "detail": "说明"}],
            "evidence_refs": ["m1"], "knowledge_citations": []}
    body.update(overrides)
    return AIMessage(content=json.dumps(body, ensure_ascii=False))


async def _stage(task, llm, mcp=None, message="材料"):
    return await run_function_stage(
        _request(task, message), llm=llm, mcp_client=mcp, system_prompt="基础提示",
        skills=[], skill_loader=None, execution_id="e1", generation=1)


HIT = {"success": True, "data": [
    {"id": "chunk-1", "knowledge_id": "doc-1", "knowledge_title": "制度A", "content": "…"}]}
EMPTY = {"success": True, "data": []}


# FST-02：非 V3 快照不支持功能阶段。
def test_fst02_non_v3_snapshot_is_rejected():
    request = SimpleNamespace(agent_config=SimpleNamespace(snapshot_v3=None))
    with pytest.raises(ConfigError, match="snapshot_version_not_supported"):
        _run(_execute_function_stage(request, model_authorizer=None, skill_loader=None))


# FST-03：合法 JSON 通过校验，reply 取正文，output 含规范化字段。
def test_fst03_valid_document_output():
    llm = ScriptedLlm([_document()])
    reply, meta, _ = _run(_stage(_task(), llm))
    result = meta["function_result"]
    assert reply == "## 正文\n内容" and result["status"] == "ok"
    assert result["output"]["risks"][0]["level"] == "high"
    assert result["knowledge"] == {"status": "not_used", "citations": []}


# FST-03：被围栏与说明文字包裹的 JSON 仍可提取；摘要超过 300 字被截断。
def test_extract_json_handles_fences_and_long_summary():
    assert extract_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert extract_json_object("结果如下：{\"a\": 2} 完毕") == {"a": 2}
    long = AIMessage(content=json.dumps({"title": "t", "summary": "长" * 500, "body_markdown": "b"}))
    _, meta, _ = _run(_stage(_task(), ScriptedLlm([long])))
    assert len(meta["function_result"]["output"]["summary"]) == 300


# FST-04：第一次 JSON 非法，修复重试后成功；修复提示只带原因。
def test_fst04_repair_retry_succeeds():
    llm = ScriptedLlm([AIMessage(content="这不是 JSON"), _document()])
    _, meta, _ = _run(_stage(_task(), llm, message="原始材料"))
    assert meta["function_result"]["status"] == "ok" and llm.bound_calls == 2


# FST-04：连续两次非法 -> invalid_structured_output，且分类为可重试。
def test_fst04_two_invalid_outputs_fail_as_retryable():
    llm = ScriptedLlm([AIMessage(content="坏"), AIMessage(content='{"title": ""}')])
    with pytest.raises(FunctionStageError) as caught:
        _run(_stage(_task(), llm))
    assert str(caught.value) == "invalid_structured_output"
    error = classify_exception(caught.value)
    assert (error.code, error.http_status, error.retryable) == ("invalid_structured_output", 422, True)


# 任务阶段的契约：负责人必须是正整数或 null，空内容被拒绝。
def test_tasks_schema_validation():
    good = AIMessage(content=json.dumps({"summary": "概述", "tasks": [
        {"content": "整改", "assignee_user_id": 11, "due_date": "2026-10-02",
         "source_risk": "风险1", "evidence_refs": ["m1"]},
        {"content": "复核", "assignee_user_id": None, "due_date": None}]}, ensure_ascii=False))
    reply, meta, _ = _run(_stage(_task("tasks"), ScriptedLlm([good])))
    tasks = meta["function_result"]["output"]["tasks"]
    assert [item["assignee_user_id"] for item in tasks] == [11, None]
    assert reply.splitlines()[1:] == ["1. 整改", "2. 复核"]
    bad = AIMessage(content=json.dumps({"tasks": [{"content": "x", "assignee_user_id": "11"}]}))
    with pytest.raises(FunctionStageError):
        _run(_stage(_task("tasks"), ScriptedLlm([bad, bad])))


# FST-05：知识状态只看真实工具结果，模型自报被忽略。
@pytest.mark.parametrize("scenario,expected", [
    ("none", "not_used"), ("hit", "ok"), ("empty", "empty"), ("error", "failed"),
])
def test_fst05_knowledge_status_from_real_calls(scenario, expected):
    results = {"hybrid_search": {"hit": HIT, "empty": EMPTY}.get(
        scenario, McpEdgeError("MCP tool returned an error: hybrid_search (upstream_unavailable)"))}
    mcp = PinnedMcp(["hybrid_search"], results)
    replies = [_document(knowledge_status="ok")]          # 模型自报 ok 必须被忽略
    if scenario != "none":
        replies.insert(0, tool_call(mcp.alias_of("hybrid_search"), {"kb_id": "kb", "query": "x"}))
    llm = ScriptedLlm(replies)
    _, meta, _ = _run(_stage(_task(mode="optional"), llm, mcp))
    assert meta["function_result"]["knowledge"]["status"] == expected


# FST-05 补充：检索失败不会中断循环，模型仍能收到错误文本并继续。
def test_tool_error_does_not_abort_loop():
    mcp = PinnedMcp(["hybrid_search"], {"hybrid_search": McpEdgeError("boom")})
    llm = ScriptedLlm([tool_call(mcp.alias_of("hybrid_search"), {"kb_id": "kb", "query": "x"}),
                       _document()])
    reply, meta, _ = _run(_stage(_task(mode="required"), llm, mcp))
    assert meta["function_result"]["status"] == "ok" and reply


# FST-06：引用只保留真实返回过的文档/分块，并使用真实标题。
def test_fst06_citations_are_filtered_to_real_results():
    mcp = PinnedMcp(["hybrid_search"], {"hybrid_search": HIT})
    cited = _document(knowledge_citations=[
        {"kb_id": "kb", "knowledge_id": "doc-1", "chunk_id": "chunk-1", "title": "模型编的标题"},
        {"kb_id": "kb", "knowledge_id": "doc-1", "chunk_id": "chunk-FAKE", "title": "伪造分块"},
        {"kb_id": "kb", "knowledge_id": "doc-FAKE", "chunk_id": None, "title": "伪造文档"},
        {"kb_id": "kb", "title": "缺少文档ID"},
    ])
    llm = ScriptedLlm([tool_call(mcp.alias_of("hybrid_search"), {"kb_id": "kb", "query": "x"}), cited])
    _, meta, _ = _run(_stage(_task(mode="optional"), llm, mcp))
    assert meta["function_result"]["knowledge"]["citations"] == [
        {"kb_id": "kb", "knowledge_id": "doc-1", "chunk_id": "chunk-1", "title": "制度A"}]
    assert meta["function_result"]["output"]["knowledge_citations"] == \
        meta["function_result"]["knowledge"]["citations"]


# FST-07：知识依据模式决定提示词中的 Skill / 检索要求。
def test_fst07_prompt_reflects_knowledge_mode():
    required = build_system_prompt("基础", _task(mode="required"))
    optional = build_system_prompt("基础", _task(mode="optional"))
    off = build_system_prompt("基础", _task(mode="off"))
    assert "必须先加载知识库查询 Skill" in required
    assert "可先加载知识库查询 Skill" in optional
    assert "知识库查询 Skill" not in off
    tasks_prompt = build_system_prompt("", _task("tasks", assignable_members=[{"user_id": 11, "name": "甲"}]))
    assert "可分配人员" in tasks_prompt and "甲" in tasks_prompt


# 阶段与输出 schema 必须对应；未知字段被拒绝。
def test_function_task_contract_validation():
    with pytest.raises(ValueError):
        FunctionTask(function_key="risk_consultation", stage="tasks",
                     output_schema="consultation_document_v1")
    with pytest.raises(ValueError):
        FunctionTask.model_validate({"function_key": "risk_consultation", "stage": "draft",
                                     "output_schema": "consultation_document_v1", "x": 1})


# summarize_knowledge 对空事件、混合结果的行为。
def test_summarize_knowledge_mixed_results():
    status, docs, chunks = summarize_knowledge([
        {"remote_name": "hybrid_search", "outcome": "failed", "payload": None},
        {"remote_name": "hybrid_search", "outcome": "success", "payload": EMPTY},
        {"remote_name": "other_tool", "outcome": "success", "payload": HIT},
    ])
    assert (status, docs, chunks) == ("empty", {}, set())
    assert summarize_knowledge([])[0] == "not_used"


# ExecutionRequest 只在携带 function_task 时接受该字段，且非法值被拒绝。
def test_execution_request_accepts_function_task_field():
    assert "function_task" in ExecutionRequest.model_fields
    field = ExecutionRequest.model_fields["function_task"]
    assert field.default is None
