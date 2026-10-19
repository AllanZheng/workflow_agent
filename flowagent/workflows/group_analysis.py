"""群分析工作流：取数 → 分批分类 → 程序统计 → 报告引用校验 → 终态发布。

有意改变行为（相对旧引擎，技术方案 §1.3）：
- 无时间表达 → 配置默认窗（模型 days 只进诊断，不采用）。
- 全部分类失败 → 业务未完成，交唯一兜底（不再返回固定提示 partial）。
- 单条超长 → 计 unprocessed(message_too_large)，不再使整次分析抛错。

覆盖口径：received_unique = classified + failed + unprocessed；
上游 fetched/included/partial_reasons 原样另存，不混入内部等式。
"""
from __future__ import annotations

import json
from typing import Any

from flowagent.core.contracts import (
    Clarification,
    ErrorCode,
    FlowAgentCoreError,
    RunResult,
    Status,
    WorkflowSpec,
    build_continuation,
)
from flowagent.workflows.group_rules import (
    DEFAULT_BATCH_CHARS,
    DEFAULT_BATCH_MESSAGES,
    GroupControl,
    NeedsClarification,
    build_control,
    control_to_fetch_arguments,
)

FETCH_TOOL = "fetch_group_analysis_input"
RESOLVE_TOOL = "platform_resolve_group"

_CLASSIFIER_SYSTEM = "你是群消息分类器，不编造消息引用。"
_REPORT_SYSTEM = (
    "你是群分析报告撰写人。基于给定的分类统计与消息摘录输出中文 Markdown 报告，"
    "只包含 ## 总结、## 分类统计、## 汇报要点 三节；引用消息时使用其 message_ref；"
    "不得改写统计数字，不得引用给定范围之外的消息。"
)


def build_group_analysis_workflow(*, config: dict[str, Any] | None = None) -> WorkflowSpec:
    """构造群分析工作流；config 为 profile 的 group_analysis 段。"""
    cfg = dict(config or {})

    async def preflight(inputs: dict[str, Any], ctx: Any) -> str | None:
        if ctx.tools is None:
            return "dependency_missing"
        return None

    async def run(inputs: dict[str, Any], ctx: Any) -> RunResult:
        return await _run_group_analysis(inputs, ctx, cfg)

    return WorkflowSpec(name="group_analysis", run=run, preflight=preflight)


async def _run_group_analysis(inputs: dict[str, Any], ctx: Any, cfg: dict[str, Any]) -> RunResult:
    # ---- 原问题与续接（§4.1/§4.3）----
    continuation = inputs.get("continuation")
    user_message = str(inputs.get("message") or "")
    metadata = inputs.get("request_metadata") or {}
    original_question = str(
        (metadata.get("original_user_message") if isinstance(metadata, dict) else "") or user_message
    )
    clarify_count = 0
    supplement: dict[str, Any] | None = None
    if isinstance(continuation, dict) and continuation.get("version") == 1:
        original_question = str(continuation.get("original_request") or original_question)
        clarify_count = int(continuation.get("clarify_count") or 0)
        supplement = dict(continuation.get("resolved_fields") or {})
        # 本次用户补充逐字段覆盖（补充文本同时供时间/分类解析）。
        supplement["time"] = user_message
        supplement["taxonomy"] = user_message
    ctx.original_question = original_question

    import datetime as _dt

    reference_time = _dt.datetime.fromisoformat(
        ctx.reference_time or _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).isoformat()
    )
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(
            tzinfo=_dt.timezone(_dt.timedelta(hours=8)))

    # ---- 控制参数（代码决定；缺条件 → 澄清/耗尽）----
    try:
        control = build_control(
            original_question=original_question,
            trusted_target=inputs.get("analysis_target"),
            reference_time=reference_time,
            config=cfg,
            supplement=supplement,
            model_arguments=inputs.get("model_arguments"),
        )
    except NeedsClarification as pending:
        if clarify_count >= 2:
            # 两轮澄清后仍缺条件：唯一一次直连只说明原因（不取数、不生成统计）。
            raise FlowAgentCoreError(
                ErrorCode.CLARIFICATION_EXHAUSTED,
                f"clarification exhausted; missing={pending.clarification.missing_fields}",
            ) from pending
        return _needs_input_result(ctx, pending.clarification, original_question, cfg,
                                    clarify_count + 1, control_hint=None)
    if control.conversation_id is None:
        # 无可信目标且未点名：需要群。
        return _needs_input_result(
            ctx, Clarification(
                reason="group_unresolved", missing_fields=["目标群"],
                hints=['请说明要分析的群（如"测试群"或"本群"）']),
            original_question, cfg, clarify_count + 1, control_hint=control)

    # ---- 取数（有效控制参数由代码生成；被服务拒绝属契约错误）----
    ctx.step = "group.fetch"
    ctx.check_cancel()
    await ctx.budget.reserve_tool()
    ctx.usage.mcp_attempts += 1
    fetch_arguments = control_to_fetch_arguments(control)
    fetch_result = await ctx.tools.call(FETCH_TOOL, fetch_arguments)
    # 取数计入证据（回执/诊断依据）；远端名解析兼容固定别名。
    resolver = getattr(ctx.tools, "remote_name", None)
    remote_name = (resolver(FETCH_TOOL) if callable(resolver) else FETCH_TOOL) or FETCH_TOOL
    ctx.record_tool(FETCH_TOOL, fetch_arguments, fetch_result, remote_name=remote_name)
    if fetch_result.status in ("error", "unknown"):
        if fetch_result.error_code == ErrorCode.TOOL_ARGUMENT_INVALID.value:
            raise FlowAgentCoreError(
                ErrorCode.TOOL_ARGUMENT_INVALID,
                f"fetch rejected code-controlled arguments: {fetch_result.message}",
            )
        raise FlowAgentCoreError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"group fetch failed: {fetch_result.message} (state={fetch_result.status})",
        )
    raw = fetch_result.value if isinstance(fetch_result.value, dict) else None
    if not isinstance(raw, dict) or "messages" not in raw:
        raise FlowAgentCoreError(
            ErrorCode.OUTPUT_INVALID, "fetch payload missing messages")

    upstream_partial = bool(raw.get("status") == "partial") or bool(raw.get("partial_reasons"))
    messages = list(raw.get("messages") or [])
    if not messages:
        if upstream_partial:
            # 上游 partial 且未返回可用消息：执行错误，交兜底（不是"无消息"）。
            raise FlowAgentCoreError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"upstream partial with no usable messages: {raw.get('partial_reasons')}")
        return _completed_no_messages(ctx, control, raw)

    # ---- 分批与分类（每批最多一次修复；超长单条计 unprocessed）----
    ctx.step = "group.classify"
    batches, overlong = _make_batches(
        messages, int(cfg.get("batch_messages", DEFAULT_BATCH_MESSAGES)),
        int(cfg.get("batch_chars", DEFAULT_BATCH_CHARS)))
    assignments: dict[str, str] = {}
    failed_refs: list[str] = []
    for batch in batches:
        batch_assignments = await _classify_batch(ctx, batch, control.taxonomy)
        for ref, category in batch_assignments.items():
            if category is None:
                failed_refs.append(ref)
            else:
                assignments[ref] = category
    unprocessed = [{"message_ref": ref, "reason": "message_too_large"} for ref in overlong]
    classified = len(assignments)
    received_unique = classified + len(failed_refs) + len(unprocessed)

    if classified == 0:
        # 有消息但无可信分类：业务未完成（有意改变 #2），交唯一兜底。
        raise FlowAgentCoreError(
            ErrorCode.OUTPUT_INVALID,
            f"no trusted classification: failed={len(failed_refs)} "
            f"unprocessed={len(unprocessed)}",
        )

    # ---- 程序统计（模型不能改写）----
    stats = _compute_statistics(assignments, control.taxonomy, messages)

    # ---- 报告与引用校验（最多一次同证据修复）----
    ctx.step = "group.report"
    report, report_refs, report_repaired = await _build_report(
        ctx, control, stats, messages, sorted(assignments))
    invalid_refs = [ref for ref in report_refs if ref not in assignments]

    partial_reasons: list[str] = list(raw.get("partial_reasons") or [])
    if failed_refs:
        partial_reasons.append("classification_failed")
    if unprocessed:
        partial_reasons.append("message_too_large")
    if invalid_refs:
        # 引用修复仍无效：不展示无效引用，降级程序 partial 报告（有意行为）。
        partial_reasons.append("report_refs_invalid")
        report, report_refs = _program_partial_report(control, stats, failed_refs, unprocessed), []

    status = Status.PARTIAL if (partial_reasons or upstream_partial) else Status.COMPLETED
    analysis = {
        "conversation_id": control.conversation_id,
        "requested_window": {"start": control.start, "end_exclusive": control.end_exclusive},
        "effective_window": raw.get("window") or {},
        "requested_taxonomy": {"text": original_question},
        "effective_taxonomy": {"categories": list(control.taxonomy)},
        "statistics": stats,
        "coverage": {
            "received_unique": received_unique,
            "classified": classified,
            "failed": len(failed_refs),
            "unprocessed": len(unprocessed),
            "unprocessed_detail": unprocessed,
            "upstream": raw.get("counts") or {},
        },
        "report_refs": report_refs,
        "report_repair_attempted": report_repaired,
        "status": status.value,
        "partial_reasons": sorted(set(partial_reasons)),
    }
    reply = report if status is Status.COMPLETED else _with_coverage_note(report, analysis)
    return RunResult(
        run_id=ctx.run_id, status=status, reply=reply,
        output={"analysis_result": analysis},
        execution_path="workflow",
        recovery={"requested_workflow": "group_analysis"},
    )


async def run_group_analysis_via_tool(
    arguments: dict[str, Any], ctx: Any, cfg: dict[str, Any]
) -> RunResult:
    """chat 终端工具入口：控制参数仍由代码从原问题确定，模型参数仅进诊断。"""
    chat_inputs = (ctx.diagnostics.get("_inputs") or {})
    inputs = {
        "message": ctx.original_question or str(arguments.get("message") or ""),
        "analysis_target": chat_inputs.get("analysis_target"),
        "request_metadata": chat_inputs.get("request_metadata") or {},
        "continuation": chat_inputs.get("continuation"),
        "model_arguments": arguments,
    }
    return await _run_group_analysis(inputs, ctx, cfg)


# ---------------------------------------------------------------- 澄清终态

def _needs_input_result(
    ctx: Any, clarification: Clarification, original_question: str,
    cfg: dict[str, Any], next_count: int, *, control_hint: GroupControl | None,
) -> RunResult:
    continuation = build_continuation(
        workflow="group_analysis", original_request=original_question,
        reference_time=ctx.reference_time or "",
        timezone=cfg.get("timezone", "Asia/Shanghai"),
        resolved_fields={},
        clarify_count=next_count,
    )
    clarification.continuation = continuation
    clarification.confirmed_fields = {}
    return RunResult(
        run_id=ctx.run_id, status=Status.NEEDS_INPUT,
        reply=clarification.render_body(), clarification=clarification,
        execution_path="workflow",
        recovery={"requested_workflow": "group_analysis", "clarify_count": next_count},
    )


def _completed_no_messages(ctx: Any, control: GroupControl, raw: dict[str, Any]) -> RunResult:
    analysis = {
        "conversation_id": control.conversation_id,
        "effective_window": raw.get("window") or {},
        "effective_taxonomy": {"categories": list(control.taxonomy)},
        "statistics": {"total_messages": 0, "categories": []},
        "coverage": {"received_unique": 0, "classified": 0, "failed": 0,
                     "unprocessed": 0, "upstream": raw.get("counts") or {}},
        "status": "completed", "partial_reasons": [],
    }
    span = f"{control.start} ~ {control.end_exclusive}"
    return RunResult(
        run_id=ctx.run_id, status=Status.COMPLETED,
        reply=f"窗口内没有群消息（{span}）。如需其他时间范围请补充说明。",
        output={"analysis_result": analysis}, execution_path="workflow",
    )


# ---------------------------------------------------------------- 分批

def _make_batches(
    messages: list[dict[str, Any]], max_count: int, max_chars: int
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """按条数与字符预算分批；单条超限不截断，计 unprocessed（有意改变 #3）。"""
    batches: list[list[dict[str, Any]]] = []
    overlong: list[str] = []
    current: list[dict[str, Any]] = []
    used = 0
    for message in messages:
        text = str(message.get("text") or "")
        ref = str(message.get("message_ref") or "")
        if len(text) > max_chars:
            overlong.append(ref)
            continue
        if current and (len(current) >= max_count or used + len(text) > max_chars):
            batches.append(current)
            current, used = [], 0
        current.append(message)
        used += len(text)
    if current:
        batches.append(current)
    return batches, overlong


# ---------------------------------------------------------------- 分类（两 attempts）

async def _llm_json(ctx: Any, system: str, prompt: str) -> dict[str, Any]:
    """一次模型 JSON 调用（计入预算）；解析失败抛 MODEL_INVALID_RESPONSE。"""
    ctx.check_cancel()
    try:
        await ctx.budget.reserve_model()
    except Exception as exc:  # BudgetExceeded
        from flowagent.core.context import BudgetExceeded
        if isinstance(exc, BudgetExceeded):
            raise FlowAgentCoreError(ErrorCode.MODEL_BUDGET_EXCEEDED, str(exc)) from exc
        raise
    ctx.usage.model_requests += 1
    completed = None
    async for event in ctx.model.stream(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        [], {},
    ):
        if event.kind == "completed":
            completed = event
    text = (completed.text if completed else "") or ""
    try:
        parsed = json.loads(_strip_fence(text))
    except json.JSONDecodeError as exc:
        raise FlowAgentCoreError(
            ErrorCode.MODEL_INVALID_RESPONSE, "classification output not json") from exc
    if not isinstance(parsed, dict):
        raise FlowAgentCoreError(
            ErrorCode.MODEL_INVALID_RESPONSE, "classification output not object")
    return parsed


def _strip_fence(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    return candidate.strip()


async def _classify_batch(
    ctx: Any, batch: list[dict[str, Any]], taxonomy: tuple[str, ...]
) -> dict[str, str]:
    """分类一批消息；失败修复最多一次（同材料，不重新取数）；无效计 None。"""
    lines = [
        {"message_ref": str(m.get("message_ref") or ""), "text": str(m.get("text") or "")[:500]}
        for m in batch
    ]
    expected = [item["message_ref"] for item in lines]
    allowed = list(taxonomy)
    prompt = (
        f"请把以下每条消息归入且仅归入一个类别。\n"
        f"允许类别：{json.dumps(allowed, ensure_ascii=False)}\n"
        f"消息：{json.dumps(lines, ensure_ascii=False)}\n"
        '输出 JSON：{"assignments":[{"message_ref":"...","category":"..."}],"summary":"事项"}'
    )
    feedback = ""
    for _attempt in range(2):
        try:
            result = await _llm_json(ctx, _CLASSIFIER_SYSTEM, prompt + feedback)
            assignments = result.get("assignments")
            if not isinstance(assignments, list):
                raise ValueError("assignments not list")
            mapping: dict[str, str] = {}
            for item in assignments:
                ref, category = item.get("message_ref"), item.get("category")
                if ref in expected and category in allowed and ref not in mapping:
                    mapping[str(ref)] = str(category)
            if len(mapping) == len(expected):
                return mapping
        except (ValueError, TypeError, FlowAgentCoreError):
            pass
        # 修复提示（同证据）：给出精确约束后重试一次。
        feedback = (
            f"\n约束：assignments 必须恰好覆盖这些 message_ref 各一次：{json.dumps(expected, ensure_ascii=False)}；"
            f"category 只能取 {json.dumps(allowed, ensure_ascii=False)}。"
        )
    return {ref: None for ref in expected}


# ---------------------------------------------------------------- 统计与报告

def _compute_statistics(
    assignments: dict[str, str], taxonomy: tuple[str, ...], messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """程序统计：分母 = classified；模型不能改写。"""
    text_by_ref = {str(m.get("message_ref") or ""): str(m.get("text") or "") for m in messages}
    total = len(assignments)
    categories = []
    for name in taxonomy:
        count = sum(1 for c in assignments.values() if c == name)
        if count == 0:
            continue
        quotes = [text_by_ref[r] for r, c in assignments.items() if c == name][:2]
        categories.append({
            "name": name, "count": count,
            "ratio": round(count * 100 / total, 1) if total else 0.0,
            "quotes": [q[:60] for q in quotes],
        })
    return {"total_messages": total, "categories": categories}


async def _build_report(
    ctx: Any, control: GroupControl, stats: dict[str, Any],
    messages: list[dict[str, Any]], valid_refs: list[str],
) -> tuple[str, list[str], bool]:
    """生成报告并校验引用；无效引用最多同证据修复一次。"""
    digest = json.dumps(
        [{"message_ref": str(m.get("message_ref") or ""), "text": str(m.get("text") or "")[:160]}
         for m in messages], ensure_ascii=False)
    prompt = (
        f"时间窗：{control.start} ~ {control.end_exclusive}\n"
        f"分类统计：{json.dumps(stats, ensure_ascii=False)}\n"
        f"消息（引用只能使用其中的 message_ref）：{digest}\n"
        "末尾单独一行输出引用列表：引用: [m1, m2]"
    )
    ctx.check_cancel()
    try:
        await ctx.budget.reserve_model()
    except Exception as exc:
        from flowagent.core.context import BudgetExceeded
        if isinstance(exc, BudgetExceeded):
            return _program_partial_report(control, stats, [], []), [], False
        raise
    ctx.usage.model_requests += 1
    report, refs = "", []
    for attempt in range(2):
        completed = None
        async for event in ctx.model.stream(
            [{"role": "system", "content": _REPORT_SYSTEM},
             {"role": "user", "content": prompt if attempt == 0 else prompt + _REF_FEEDBACK(valid_refs)}],
            [], {},
        ):
            if event.kind == "completed":
                completed = event
        report = (completed.text if completed else "") or ""
        refs = _extract_report_refs(report)
        if refs and all(r in valid_refs for r in refs):
            return report, refs, attempt > 0
        if not refs:
            # 模型未给引用列表：按零引用处理，不再修复。
            return report, [], False
    return report, refs, True


_REF_LINE = __import__("re").compile(r"引用\s*[:：]\s*\[([^\]]*)\]")


def _extract_report_refs(report: str) -> list[str]:
    m = _REF_LINE.search(report)
    if not m:
        return []
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def _REF_FEEDBACK(valid_refs: list[str]) -> str:
    return f"\n引用列表只能包含这些 message_ref：{json.dumps(valid_refs, ensure_ascii=False)}；请修正后重发。"


def _program_partial_report(
    control: GroupControl, stats: dict[str, Any],
    failed_refs: list[str], unprocessed: list[dict[str, Any]],
) -> str:
    """程序 partial 报告：只用已校验分类与统计，说明未完成原因；不保留/补造引用。"""
    lines = ["## 总结", f"窗口 {control.start} ~ {control.end_exclusive} 的群消息分析报告（程序生成）。"]
    lines.append("## 分类统计")
    for category in stats["categories"]:
        lines.append(f"- {category['name']}：{category['count']} 条（{category['ratio']}%）")
    lines.append("## 汇报要点")
    if failed_refs or unprocessed:
        lines.append(f"- 覆盖说明：{len(failed_refs)} 条分类失败、{len(unprocessed)} 条未处理（超长）。")
    lines.append("- 报告由程序依据已校验统计生成：模型报告引用未通过校验。")
    return "\n".join(lines)


def _with_coverage_note(report: str, analysis: dict[str, Any]) -> str:
    coverage = analysis["coverage"]
    note = (f"⚠ 覆盖情况：共接收 {coverage['received_unique']} 条，可信分类 {coverage['classified']} 条，"
            f"失败 {coverage['failed']} 条，未处理 {coverage['unprocessed']} 条；"
            f"原因：{', '.join(analysis['partial_reasons'])}。")
    return f"{report}\n\n{note}"
