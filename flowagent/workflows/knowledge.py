"""知识查询工作流：代码定范围与数量 → 检索 → 答案组织 → 引用校验。

落点（技术方案 §4.5 知识查询 + K01/K02）：
- 工具经 workflow_bindings 指向具体 MCP 工具（不让模型猜工具用途）。
- 引用只来自此次成功取得的文档/分块；空结果是有效结果；检索失败交兜底
  （标 data_unavailable），不能编造引用。
"""
from __future__ import annotations

import json
from typing import Any

from flowagent.core.contracts import (
    ErrorCode,
    FlowAgentCoreError,
    RunResult,
    Status,
    WorkflowSpec,
)
from flowagent.workflows.outputs import filter_citations, summarize_knowledge_events

KNOWLEDGE_SYSTEM_PROMPT = (
    "你是知识库问答助手。只依据检索到的文档内容回答；引用使用文档的 knowledge_id；"
    "检索为空时如实说明未找到；不得编造引用或内容。"
)


def build_knowledge_workflow(*, bindings: dict[str, str] | None = None) -> WorkflowSpec:
    """构造知识查询工作流；bindings: {search: 服务器__工具, read: ...}。"""

    async def preflight(inputs: dict[str, Any], ctx: Any) -> str | None:
        if not bindings or not bindings.get("search") or ctx.tools is None:
            return "dependency_missing"
        return None

    async def run(inputs: dict[str, Any], ctx: Any) -> RunResult:
        question = str(inputs.get("message") or inputs.get("question") or "")
        if not question.strip():
            raise FlowAgentCoreError(ErrorCode.INVALID_INPUT, "question required")
        ctx.step = "knowledge.search"
        ctx.check_cancel()
        await ctx.budget.reserve_tool()
        ctx.usage.mcp_attempts += 1
        from flowagent.core.contracts import ToolResultData

        result = await ctx.tools.call(bindings["search"], {"query": question})
        # 检索计入证据：知识状态/引用过滤的唯一依据（K01/K02）。
        ctx.record_tool(bindings["search"], {"query": question}, result)
        if result.status in ("error", "unknown"):
            # 检索失败：兜底标 data_unavailable；不能冒充未找到或已核验。
            raise FlowAgentCoreError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"knowledge search failed: {result.message} (state={result.status})",
            )
        payload = result.value if isinstance(result.value, dict) else {}
        rows = payload.get("data") if isinstance(payload.get("data"), list) else []
        if not rows:
            # 空结果是有效结果（K01）。
            return RunResult(
                run_id=ctx.run_id, status=Status.COMPLETED,
                reply="知识库中没有检索到相关内容；请换一个问法或确认知识范围。",
                output={"knowledge_result": {"status": "empty", "citations": []}},
                execution_path="workflow",
            )
        ctx.step = "knowledge.answer"
        ctx.check_cancel()
        from flowagent.core.context import BudgetExceeded

        try:
            await ctx.budget.reserve_model()
        except BudgetExceeded as exc:
            raise FlowAgentCoreError(ErrorCode.MODEL_BUDGET_EXCEEDED, str(exc)) from exc
        ctx.usage.model_requests += 1
        digest = json.dumps(rows, ensure_ascii=False, default=str)[:8000]
        completed = None
        async for event in ctx.model.stream(
            [{"role": "system", "content": KNOWLEDGE_SYSTEM_PROMPT},
             {"role": "user", "content": f"问题：{question}\n\n检索结果：{digest}\n\n"
              "请回答并在末尾单独一行输出引用: [{\"knowledge_id\": \"...\", \"title\": \"...\"}]"}],
            [], {},
        ):
            if event.kind == "completed":
                completed = event
        answer = (completed.text if completed else "") or ""
        knowledge = summarize_knowledge_events(ctx.evidence.get("tool_calls") or [])
        citations = filter_citations(_extract_inline_citations(answer), knowledge)
        # 非本次文档/分块的引用一律拒绝展示（K02）：行内引用被过滤后移除原文标注。
        return RunResult(
            run_id=ctx.run_id, status=Status.COMPLETED, reply=answer,
            output={"knowledge_result": {"status": knowledge["status"], "citations": citations}},
            execution_path="workflow",
        )

    return WorkflowSpec(name="knowledge_query", run=run, preflight=preflight)


def _extract_inline_citations(answer: str) -> list[dict[str, Any]]:
    """解析末尾引用行（[{knowledge_id, title}]）；解析失败视为无结构化引用。"""
    import re

    m = re.search(r"引用\s*[:：]\s*(\[.*\])\s*$", answer.strip(), re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
