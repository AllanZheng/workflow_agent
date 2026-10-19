"""风险会商阶段工作流：draft/final/tasks × 知识模式，最多一次无工具格式修复。

落点（技术方案 §4.5 + 测试方案 §5.8 冻结表）：
- 输入校验：stage 与 schema 匹配；非法 knowledge_mode 归一 off。
- 生成：共享 Agent 循环（知识模式要求时可用知识工具）。
- 校验：outputs.validate_*；失败用同一材料做一次无工具修复（不重新检索/写工具）。
- 知识状态来自真实工具轨迹（ctx.evidence），模型自报不采信。
- required + failed/not_used → 业务失败（knowledge_required_unavailable）。
"""
from __future__ import annotations

import json
from typing import Any

from flowagent.core.agent import run_agent
from flowagent.core.contracts import (
    ErrorCode,
    FlowAgentCoreError,
    RunResult,
    Status,
    WorkflowSpec,
)
from flowagent.workflows.outputs import (
    OutputInvalid,
    filter_citations,
    summarize_knowledge_events,
    validate_consultation_document,
    validate_consultation_tasks,
)

_STAGE_SCHEMA = {"draft": "consultation_document_v1", "final": "consultation_document_v1",
                 "tasks": "consultation_tasks_v1"}

_DOCUMENT_SCHEMA_TEXT = (
    '{"title":"标题","summary":"不超过300字的摘要","body_markdown":"Markdown 正文",'
    '"risks":[{"level":"high|medium|low","title":"风险","detail":"说明"}],'
    '"evidence_refs":["m123"],'
    '"knowledge_citations":[{"kb_id":"库ID","knowledge_id":"文档ID","chunk_id":"分块ID","title":"标题"}]}'
)
_TASKS_SCHEMA_TEXT = (
    '{"summary":"任务概述","tasks":[{"content":"任务内容","assignee_user_id":11,'
    '"due_date":"2026-10-01","source_risk":"来源风险","evidence_refs":["m123"]}]}'
)


class _WikiOnlyTools:
    """把 ctx.tools 过滤为知识工具（会商阶段不暴露其他业务工具）。

    名称判定：优先 tool_bindings 的 remote_name（V3 固定别名场景），
    否则用 spec.name 本身（原生命名 server__tool 或裸名）。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def _remote_name(self, spec: Any) -> str:
        bindings = getattr(self._inner, "tool_bindings", None) or {}
        pinned = bindings.get(spec.name) or {}
        remote = pinned.get("remote_name")
        return str(remote or spec.name)

    async def list(self):
        from flowagent.workflows.outputs import WIKI_TOOL_NAMES

        specs = await self._inner.list()
        return [s for s in specs
                if self._remote_name(s).split("__", 1)[-1] in WIKI_TOOL_NAMES]

    async def call(self, name: str, arguments: dict[str, Any]):
        return await self._inner.call(name, arguments)


def build_consultation_workflow(*, system_base: str = "你是风险会商助手，依据给定材料产出结构化结论。") -> WorkflowSpec:
    """构造会商阶段工作流；inputs 携带 function_task 与材料。"""

    async def run(inputs: dict[str, Any], ctx: Any) -> RunResult:
        task = inputs.get("function_task") or {}
        stage = task.get("stage")
        output_schema = task.get("output_schema")
        if stage not in _STAGE_SCHEMA or _STAGE_SCHEMA.get(stage) != output_schema:
            raise FlowAgentCoreError(
                ErrorCode.INVALID_INPUT,
                f"stage/schema mismatch: {stage}/{output_schema}")
        task_inputs = task.get("inputs") or {}
        knowledge_mode = task_inputs.get("knowledge_mode", "off")
        if knowledge_mode not in ("off", "optional", "required"):
            knowledge_mode = "off"
        materials = str(inputs.get("message") or inputs.get("materials") or "")

        # ---- 生成（共享循环；知识模式非 off 时提供知识工具）----
        from flowagent.core.agent import ToolRegistry

        tools = _WikiOnlyTools(ctx.tools) if (ctx.tools is not None and knowledge_mode != "off") else None
        retrieved_text = ""
        if knowledge_mode == "required" and tools is not None:
            # required：检索是流程步骤（代码发起），不依赖模型自觉；结果进生成提示。
            retrieved_text = await _code_driven_search(ctx, tools, materials)
        registry = ToolRegistry(remote=tools)
        prompt = _stage_prompt(system_base, stage, task_inputs, materials)
        if retrieved_text:
            prompt += "\n\n知识库检索结果（只依据这些内容引用）：\n" + retrieved_text[:6000]
        outcome = await run_agent(
            ctx, system_prompt=_stage_system(stage, knowledge_mode),
            user_message=prompt, history=[], registry=registry, duplicate_guard=False,
        )
        draft_text = outcome.text or ""

        # ---- 结构校验 + 一次无工具修复 ----
        output, repaired = await _validate_with_repair(ctx, stage, draft_text)
        knowledge = summarize_knowledge_events(ctx.evidence.get("tool_calls") or [])
        citations = filter_citations(output.get("knowledge_citations") or [], knowledge)
        if output_schema == "consultation_document_v1":
            output["knowledge_citations"] = citations

        # ---- required 门禁：状态来自真实轨迹 ----
        if knowledge_mode == "required" and knowledge["status"] in ("failed", "not_used"):
            raise FlowAgentCoreError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"knowledge_required_unavailable: {knowledge['status']}",
            )

        reply = _render_reply(stage, output)
        return RunResult(
            run_id=ctx.run_id, status=Status.COMPLETED, reply=reply,
            output={"function_result": {
                "status": "ok", "output": output,
                "knowledge": {"status": knowledge["status"], "citations": citations},
                "format_repair_attempted": repaired,
            }},
            execution_path="workflow",
        )

    return WorkflowSpec(name="consultation_stage", run=run)


async def _code_driven_search(ctx: Any, tools: "_WikiOnlyTools", materials: str) -> str:
    """required 模式的代码发起检索：取第一个知识工具，成功结果进证据与提示。"""
    from flowagent.core.context import BudgetExceeded

    specs = await tools.list()
    if not specs:
        return ""
    ctx.check_cancel()
    try:
        await ctx.budget.reserve_tool()
    except BudgetExceeded:
        return ""
    ctx.usage.mcp_attempts += 1
    result = await tools.call(specs[0].name, {"query": materials[:120]})
    ctx.record_tool(specs[0].name, {"query": materials[:120]}, result,
                    remote_name=tools._remote_name(specs[0]))
    if result.status != "success" or not isinstance(result.value, dict):
        return ""
    import json as _json

    return _json.dumps(result.value, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- 提示与校验

def _stage_system(stage: str, knowledge_mode: str) -> str:
    base = (
        "你是风险会商助手。严格按给定 JSON schema 输出单个 JSON 对象，不输出其他文字。"
    )
    if knowledge_mode == "required":
        return base + "必须先通过知识库工具检索依据，再输出结论。"
    if knowledge_mode == "optional":
        return base + "可按需使用知识库工具检索依据。"
    return base


def _stage_prompt(base: str, stage: str, task_inputs: dict[str, Any], materials: str) -> str:
    schema_text = _DOCUMENT_SCHEMA_TEXT if stage in ("draft", "final") else _TASKS_SCHEMA_TEXT
    parts = [f"阶段：{stage}", f"输出 JSON schema：{schema_text}", f"材料：\n{materials}"]
    if stage == "tasks" and task_inputs.get("assignable_members"):
        members = json.dumps(task_inputs["assignable_members"], ensure_ascii=False)
        parts.append(f"可分配负责人（assignee_user_id 只能取其中 user_id 或 null）：{members}")
    return "\n\n".join(parts)


async def _validate_with_repair(ctx: Any, stage: str, draft_text: str):
    """校验输出；失败用同一材料做一次无工具修复（不重新检索/写工具）。"""
    validator = validate_consultation_document if stage in ("draft", "final") else validate_consultation_tasks
    try:
        return validator(_extract_json(draft_text)), False
    except (OutputInvalid, ValueError):
        pass
    # 修复轮：无工具、同一材料、附错误反馈。
    ctx.check_cancel()
    from flowagent.core.context import BudgetExceeded

    try:
        await ctx.budget.reserve_model()
    except BudgetExceeded as exc:
        raise FlowAgentCoreError(ErrorCode.MODEL_BUDGET_EXCEEDED, str(exc)) from exc
    ctx.usage.model_requests += 1
    repaired = None
    feedback = _validate_with_repair.__doc__ or ""
    async for event in ctx.model.stream(
        [{"role": "system", "content": _stage_system(stage, "off") + " 你上一版输出未通过校验，请修正。"},
         {"role": "user", "content": f"上一版输出：\n{draft_text[:4000]}\n\n请重新输出符合 schema 的单个 JSON 对象。"}],
        [], {},
    ):
        if event.kind == "completed":
            repaired = event
    text = (repaired.text if repaired else "") or ""
    try:
        return validator(_extract_json(text)), True
    except (OutputInvalid, ValueError) as exc:
        raise FlowAgentCoreError(
            ErrorCode.OUTPUT_INVALID,
            f"consultation output invalid after one repair: {exc}",
        ) from exc


def _extract_json(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise OutputInvalid("输出中没有 JSON 对象") from None
        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            raise OutputInvalid("输出不是合法 JSON") from None


def _render_reply(stage: str, output: dict[str, Any]) -> str:
    if stage == "tasks":
        if not output.get("tasks"):
            return f"{output.get('summary') or ''}\n本次无任务。".strip()
        lines = [output.get("summary") or ""]
        lines += [f"{i}. {task['content']}" for i, task in enumerate(output["tasks"], 1)]
        return "\n".join(lines).strip()
    return str(output.get("body_markdown") or "")
