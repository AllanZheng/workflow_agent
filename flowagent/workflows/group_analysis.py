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
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from flowagent.core.input import continuation_clock

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
    "你是群分析报告撰写人。基于消息摘录撰写中文定性摘要和建议；"
    "程序会生成分类统计，你的正文不得包含数量、比例、排序或数量比较；"
    "若需提及时间，必须逐字来自末尾引用的消息；"
    "不要单独编写分类统计。不得引用给定范围之外的消息。"
)


def build_group_analysis_workflow(*, config: dict[str, Any] | None = None) -> WorkflowSpec:
    """构造群分析工作流；config 为 profile 的 group_analysis 段。"""
    cfg = dict(config or {})

    async def preflight(inputs: dict[str, Any], ctx: Any) -> str | None:
        # 固定群任务不可在依赖缺失时改走自主规划；run 中统一失败。
        return None

    async def run(inputs: dict[str, Any], ctx: Any) -> RunResult:
        return await _run_group_analysis(inputs, ctx, cfg)

    return WorkflowSpec(name="group_analysis", run=run, preflight=preflight)


async def _run_group_analysis(inputs: dict[str, Any], ctx: Any, cfg: dict[str, Any]) -> RunResult:
    fetch_tool = await _resolve_fetch_tool(ctx, cfg)
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
        previous = continuation.get("_supplements")
        previous = previous if isinstance(previous, list) else []
        # 新补充优先，同时保留历史补充中未被本次覆盖的条件。
        texts = [user_message] + [str(item) for item in reversed(previous)]
        for key in ("time", "taxonomy"):
            old_value = supplement.get(key)
            supplement[key] = texts + ([old_value] if isinstance(old_value, str) else [])
    ctx.original_question = original_question

    import datetime as _dt

    control_cfg = cfg
    if isinstance(continuation, dict) and continuation.get("version") == 1:
        try:
            reference_time = continuation_clock(continuation)
        except ValueError as exc:
            raise FlowAgentCoreError(ErrorCode.INVALID_INPUT, str(exc)) from exc
        # 保留原始文本以便跨轮机器标记一致，运算使用转换后的时区对象。
        ctx.reference_time = continuation["reference_time"]
        control_cfg = {**cfg, "timezone": continuation["timezone"]}
    else:
        reference_time = _dt.datetime.fromisoformat(
            ctx.reference_time or _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).isoformat()
        )
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(
                tzinfo=_dt.timezone(_dt.timedelta(hours=8)))
        ctx.reference_time = reference_time.isoformat()

    # ---- 控制参数（代码决定；缺条件 → 澄清/耗尽）----
    try:
        control = build_control(
            original_question=original_question,
            trusted_target=inputs.get("analysis_target"),
            reference_time=reference_time,
            config=control_cfg,
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
                                    clarify_count + 1, control_hint=None,
                                    previous=continuation, answer=user_message)
    if control.conversation_id is None:
        # 无可信目标且未点名：需要群。
        return _needs_input_result(
            ctx, Clarification(
                reason="group_unresolved", missing_fields=["目标群"],
                hints=['请说明要分析的群（如"测试群"或"本群"）']),
            original_question, cfg, clarify_count + 1, control_hint=control,
            previous=continuation, answer=user_message)

    # ---- 取数（有效控制参数由代码生成；被服务拒绝属契约错误）----
    ctx.step = "group.fetch"
    ctx.check_cancel()
    from flowagent.core.context import BudgetExceeded
    try:
        await ctx.budget.reserve_tool()
    except BudgetExceeded as exc:
        raise FlowAgentCoreError(ErrorCode.TOOL_BUDGET_EXCEEDED, str(exc)) from exc
    fetch_arguments = control_to_fetch_arguments(control)
    from flowagent.core.context import await_remote_tool
    fetch_result = await await_remote_tool(ctx, lambda: ctx.tools.call(fetch_tool, fetch_arguments))
    # 取数计入证据（回执/诊断依据）；远端名解析兼容固定别名。
    resolver = getattr(ctx.tools, "remote_name", None)
    remote_name = (resolver(fetch_tool) if callable(resolver) else fetch_tool) or fetch_tool
    ctx.record_tool(fetch_tool, fetch_arguments, fetch_result, remote_name=remote_name)
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
    report, report_refs, report_repaired, report_issue, metadata_removed = await _build_report(
        ctx, control, stats, messages, sorted(assignments))

    partial_reasons: list[str] = list(raw.get("partial_reasons") or [])
    if failed_refs:
        partial_reasons.append("classification_failed")
    if unprocessed:
        partial_reasons.append("message_too_large")
    if report_issue:
        # 两次摘要校验失败沿用原有 partial 契约；不外发未经校验的模型正文。
        partial_reasons.append(report_issue)
        report, report_refs = _program_partial_report(control, stats, failed_refs, unprocessed), []

    status = Status.PARTIAL if (partial_reasons or upstream_partial) else Status.COMPLETED
    window = raw.get("window") if isinstance(raw.get("window"), dict) else {}
    analysis = {
        "conversation_id": control.conversation_id,
        "requested_window": {"start": control.start, "end_exclusive": control.end_exclusive},
        "effective_window": window,
        # 平台持久化读取实际取数窗口的顶层字段，不以请求窗口冒充上游结果。
        "window_start": window.get("start"),
        "window_end_exclusive": window.get("end_exclusive"),
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
        "report_source_metadata_removed": metadata_removed,
        "status": "complete" if status is Status.COMPLETED else "partial",
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


async def _resolve_fetch_tool(ctx: Any, cfg: dict[str, Any]) -> str:
    """平台从冻结 remote_name 定位唯一别名，本地则使用 profile 绑定。"""
    if ctx.tools is None:
        raise FlowAgentCoreError(ErrorCode.CAPABILITY_UNAVAILABLE, "group fetch tool is unavailable")
    bindings = getattr(ctx.tools, "tool_bindings", None)
    if bindings is not None:
        matches = [alias for alias, item in bindings.items() if item.get("remote_name") == FETCH_TOOL]
        if len(matches) != 1:
            raise FlowAgentCoreError(ErrorCode.CAPABILITY_UNAVAILABLE,
                                     "group fetch tool requires one frozen binding")
        alias = matches[0]
    else:
        alias = str(cfg.get("fetch_tool") or FETCH_TOOL)
    # 别名必须在当前 MCP 目录中可见；快照绑定本身不是调用许可。
    from flowagent.core.context import await_capability
    available = await await_capability(ctx, ctx.tools.list())
    if not any(item.name == alias for item in available):
        raise FlowAgentCoreError(ErrorCode.CAPABILITY_UNAVAILABLE, f"group fetch tool not visible: {alias}")
    return alias


# ---------------------------------------------------------------- 澄清终态

def _needs_input_result(
    ctx: Any, clarification: Clarification, original_question: str,
    cfg: dict[str, Any], next_count: int, *, control_hint: GroupControl | None,
    previous: Any = None, answer: str = "",
) -> RunResult:
    continuation = build_continuation(
        workflow="group_analysis", original_request=original_question,
        reference_time=ctx.reference_time or "",
        timezone=(previous.get("timezone") if isinstance(previous, dict)
                  else cfg.get("timezone", "Asia/Shanghai")),
        resolved_fields=dict(previous.get("resolved_fields") or {}) if isinstance(previous, dict) else {},
        clarify_count=next_count,
    )
    if isinstance(previous, dict):
        prior = previous.get("_supplements")
        continuation["_supplements"] = (list(prior) if isinstance(prior, list) else []) + [answer]
    clarification.continuation = continuation
    clarification.confirmed_fields = {}
    return RunResult(
        run_id=ctx.run_id, status=Status.NEEDS_INPUT,
        reply=clarification.render_body(), clarification=clarification,
        execution_path="workflow",
        recovery={"requested_workflow": "group_analysis", "clarify_count": next_count},
    )


def _completed_no_messages(ctx: Any, control: GroupControl, raw: dict[str, Any]) -> RunResult:
    window = raw.get("window") if isinstance(raw.get("window"), dict) else {}
    analysis = {
        "conversation_id": control.conversation_id,
        "requested_window": {"start": control.start, "end_exclusive": control.end_exclusive},
        "effective_window": window,
        "window_start": window.get("start"),
        "window_end_exclusive": window.get("end_exclusive"),
        "requested_taxonomy": {"text": ctx.original_question},
        "effective_taxonomy": {"categories": list(control.taxonomy)},
        "statistics": {"total_messages": 0, "categories": []},
        "coverage": {"received_unique": 0, "classified": 0, "failed": 0,
                     "unprocessed": 0, "upstream": raw.get("counts") or {}},
        "status": "complete", "partial_reasons": [],
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
    completed = None
    async for event in ctx.stream_model(
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
        except FlowAgentCoreError as exc:
            # 授权、取消、预算等终止条件不能伪装为分类格式失败并继续请求。
            if exc.code in (ErrorCode.AUTHORIZATION_DENIED, ErrorCode.CANCELLED_BY_USER,
                            ErrorCode.DEADLINE_EXCEEDED, ErrorCode.MAIN_DEADLINE_EXCEEDED,
                            ErrorCode.MODEL_BUDGET_EXCEEDED,
                            ErrorCode.CONTEXT_LIMIT_EXCEEDED):
                raise
        except (ValueError, TypeError):
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
) -> tuple[str, list[str], bool, str | None, int]:
    """仅发布校验通过的定性摘要，统计始终由程序生成。"""
    digest = json.dumps(
        [{"message_ref": str(m.get("message_ref") or ""), "text": str(m.get("text") or "")[:160]}
         for m in messages], ensure_ascii=False)
    prompt = (
        f"时间窗：{control.start} ~ {control.end_exclusive}\n"
        f"消息（引用只能使用其中的 message_ref）：{digest}\n"
        "只描述事件及建议，不写数量、比例、排序或数量比较；时间仅可逐字引用所列消息；"
        "如需引用，在末尾单独一行输出引用列表：引用: [m1, m2]"
    )
    report, refs, issue = "", [], None
    for attempt in range(2):
        ctx.check_cancel()
        try:
            await ctx.budget.reserve_model()
        except Exception as exc:
            from flowagent.core.context import BudgetExceeded
            if isinstance(exc, BudgetExceeded):
                # 预算耗尽不能伪装成已完成的程序报告，交唯一兜底所有者。
                raise FlowAgentCoreError(ErrorCode.MODEL_BUDGET_EXCEEDED, str(exc)) from exc
            raise
        completed = None
        async for event in ctx.stream_model(
            [{"role": "system", "content": _REPORT_SYSTEM},
             {"role": "user", "content": prompt if attempt == 0 else prompt + _REF_FEEDBACK(valid_refs)}],
            [], {},
        ):
            if event.kind == "completed":
                completed = event
        # 截断、过滤或工具调用不是完整报告，不能将其正文当作成功发布。
        if completed is None or (completed.finish_reason or "stop") != "stop" or completed.tool_calls:
            issue = "report_summary_invalid"
            continue
        report = (completed.text if completed else "") or ""
        summary, refs, issue, metadata_removed = _validate_report_summary(report, valid_refs, messages)
        if issue is None:
            return _render_report(stats, summary, refs, metadata_removed), refs, attempt > 0, None, metadata_removed
    return "", [], True, issue, 0


_REF_LINE = re.compile(r"引用\s*[:：]\s*\[([^\]]*)\]\s*")
_CHINESE_NUMBER = r"[零〇一二两三四五六七八九十百千万亿兆壹贰叁肆伍陆柒捌玖拾佰仟萬億]+"
_CATEGORY_WORD = r"(?:风险|问题|故障|事件|消息|事故|建议|隐患|异常)"
_CATEGORY_ABSENCE = r"(?:没有发生|没有出现|未出现|未发生|没有|未见|不存在|未发现|无)"
_QUANTITY = re.compile(
    rf"(?:{_CHINESE_NUMBER}\s*(?:多|余|几)?\s*(?:条|个|则|人|次|起|项|例|件|宗|消息|事件|事故|成|倍|名|份|类|种|台|天|周|月|年|小时|分钟|秒|%|以上|以下|左右|余|分之))"
    rf"|(?:第\s*{_CHINESE_NUMBER})|(?:百分|千分)(?:之)?\s*{_CHINESE_NUMBER}"
    r"|[一二三四五六七八九壹贰叁肆伍陆柒捌玖][十百千万亿兆拾佰仟萬億][零〇一二两三四五六七八九十百千万亿兆壹贰叁肆伍陆柒捌玖拾佰仟萬億]*"
    rf"|(?:为|是|有|达|约)\s*{_CHINESE_NUMBER}(?=[\s，。；：,.!！？]|$)"
    rf"|(?:总数|总计|合计|共|数量|数目|共有|总共)\s*[:：]?\s*{_CHINESE_NUMBER}(?![\u4e00-\u9fff])"
    rf"|(?<![\u4e00-\u9fff]){_CHINESE_NUMBER}(?=[\s，。；：,.!！？]|$)"
    r"|(?:比[^。；，\n]{0,20}(?:多|少|高|低))"
    # 类别前后出现的绝对否定可能与程序统计直接冲突，宁可触发修复也不冒充已验证。
    rf"|(?:{_CATEGORY_ABSENCE}(?:任何|明显)?{_CATEGORY_WORD})"
    rf"|(?:{_CATEGORY_WORD}(?:类|消息|事件)?(?:并|尚)?{_CATEGORY_ABSENCE})"
    r"|(?:多数|少数|半数|一半|大半|最多|最少|较多|较少|偏多|偏少|更多|更少|更高|更低|多于|少于|高于|低于|远超|超过|超出|领先|落后|不及|多过|少过|占比|比例|比率|占据|占主导|大多数|大部分|绝大部分|排名|首位|所有|全部|每条|每个|无一例外)"
    r"|\b(?:exceed(?:s|ed|ing)?|outnumber(?:s|ed|ing)?|surpass(?:es|ed|ing)?|higher|lower|greater|fewer)\b"
    r"|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|dozen|percent|majority|minority|most|more|less|several|few)\b",
    re.IGNORECASE,
)
_SUMMARY_HEADING = re.compile(r"^\s*#{1,6}\s*(?:总结|分类统计|汇报要点)\s*$")
_ORDINARY_NUMERAL_WORDS = ("进一步", "下一步", "一旦", "一律", "一致", "统一", "一起", "一直", "一线", "零件", "参会", "参加", "参与")
# 仅承认与有时区 timestamp 交叉验证的 Edge 头部，不能把内容中的方括号当可信时间。
_UTC_STAMP = r"\[(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]) (?:[01][0-9]|2[0-3]):[0-5][0-9]\]"
_EDGE_PREFIX = re.compile(rf"^(?P<stamp>{_UTC_STAMP})(?=\s)")
_EDGE_MENTION = re.compile(rf"(?m)(?P<lead>^|(?<=[。！？!?]))[ \t]*在[ \t]*(?P<stamp>{_UTC_STAMP})[ \t]*[，,][ \t]*")
# 范围必须整体匹配同一引用消息；不能用不同消息的两个时点拼造一个范围。
_RELATIVE_DAY = r"(?:(?:前天|昨天|今天|今日|明天|后天|本周|上周|下周|周[一二三四五六日天]|星期[一二三四五六日天])\s*)?"
_CLOCK_HOUR = r"(?:1[0-2]|0?[0-9]|十[一二]?|[零一二两三四五六七八九])"
_CLOCK_MINUTE = r"(?:半|(?:[0-5]?[0-9]|[零一二两三四五六七八九十]{1,3})分)?"
_CLOCK = rf"{_RELATIVE_DAY}(?:凌晨|早上|上午|中午|下午|傍晚|晚上)\s*{_CLOCK_HOUR}\s*(?:点|时){_CLOCK_MINUTE}"
_CLOCK_RANGE_END = rf"(?:{_CLOCK}|{_CLOCK_HOUR}\s*(?:点|时){_CLOCK_MINUTE})"
_CLOCK_24 = rf"{_RELATIVE_DAY}(?<![0-9])(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?![0-9])"
_TIME_BOUNDARY_PREFIX = r"(?:(?:不早于|不晚于|早于|晚于|截至|截止到|不超过|至少|最早|最晚)\s*)?"
_TIME_BOUNDARY_SUFFIX = r"(?:之前|之后|以前|以后|前后|左右|附近|前|后)?"
_SOURCE_TIME = re.compile(
    rf"{_TIME_BOUNDARY_PREFIX}(?:{_CLOCK}|{_CLOCK_24})"
    rf"(?:\s*(?:至|到|~|～|—|–|-)\s*(?:{_CLOCK_RANGE_END}|{_CLOCK_24}))?{_TIME_BOUNDARY_SUFFIX}"
    r"(?!\s*(?:多|余)?\s*(?:条|个|次|项|例|件|宗|类|种|份|%))"
)


def _extract_report_refs(report: str) -> list[str]:
    m = _REF_LINE.fullmatch(report.strip().splitlines()[-1]) if report.strip() else None
    if not m:
        return []
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def _REF_FEEDBACK(valid_refs: list[str]) -> str:
    return (f"\n引用列表只能包含这些 message_ref：{json.dumps(valid_refs, ensure_ascii=False)}；"
            "时间须逐字来自所引用消息；请删除其他数字、数量、比例、排序及数量比较后重发。")


def _source_body_and_stamp(item: dict[str, Any]) -> tuple[str, str | None]:
    text = str(item.get("text") or "")
    prefix = _EDGE_PREFIX.match(text)
    if not prefix:
        return text, None
    # 即使前缀没有可信时间字段，也不能把其内 UTC 时刻冒充业务原文时间。
    source_body = text[prefix.end():]
    raw_timestamp = item.get("timestamp")
    if not isinstance(raw_timestamp, str):
        return source_body, None
    try:
        parsed = datetime.fromisoformat(raw_timestamp.removesuffix("Z") + "+00:00"
                                        if raw_timestamp.endswith("Z") else raw_timestamp)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return source_body, None
        expected = parsed.astimezone(timezone.utc).strftime("[%m-%d %H:%M]")
    except (ValueError, OverflowError):
        return source_body, None
    return source_body, expected if expected == prefix.group("stamp") else None


def _sanitize_source_metadata(summary: str, messages: list[dict[str, Any]], refs: list[str]) -> tuple[str, int]:
    # 每个可清洗标记都必须有独立的已引用来源，且只能删除句首“在 [UTC 标记]，”。
    cited: dict[str, dict[str, Any]] = {}
    for item in messages:
        ref = str(item.get("message_ref") or "")
        if ref in refs:
            cited.setdefault(ref, item)
    stamps = Counter(stamp for item in cited.values() for _, stamp in [_source_body_and_stamp(item)] if stamp)
    removed = 0

    def strip(match: re.Match[str]) -> str:
        nonlocal removed
        stamp = match.group("stamp")
        if stamps[stamp] <= 0:
            return match.group()
        stamps[stamp] -= 1
        removed += 1
        return match.group("lead")

    return _EDGE_MENTION.sub(strip, summary), removed


def _validate_report_summary(
    report: str, valid_refs: list[str], messages: list[dict[str, Any]],
) -> tuple[str, list[str], str | None, int]:
    """校验完整模型正文；末尾引用仅作标识符，不能参与数值判断。"""
    lines = report.strip().splitlines()
    refs = _extract_report_refs(report)
    if lines and _REF_LINE.fullmatch(lines[-1]):
        lines = lines[:-1]
    summary = "\n".join(lines).strip()
    if any(ref not in valid_refs for ref in refs):
        return "", [], "report_refs_invalid", 0
    summary, metadata_removed = _sanitize_source_metadata(summary, messages, refs)
    lines = summary.splitlines()
    normalized = unicodedata.normalize("NFKC", summary)
    # 数字或引用标识只能留在已验证的末行，避免正文冒充可信引用。
    if (any(ch in normalized for ch in "[]【】《》")
            or re.search(r"引用\s*:|(?:参见|依据|来源|引用|参考|出自)\s*[【《]", normalized)):
        return "", [], "report_refs_invalid", 0
    if any(re.match(r"^\s*#{1,6}\s*", line) and not _SUMMARY_HEADING.fullmatch(line) for line in lines):
        return "", [], "report_summary_invalid", 0
    if "<" in normalized or ">" in normalized:
        return "", [], "report_summary_invalid", 0
    if not any(char.isalpha() for line in lines if not _SUMMARY_HEADING.fullmatch(line) for char in line):
        return "", [], "report_summary_invalid", 0
    source_texts = [_source_body_and_stamp(item)[0] for item in messages
                    if str(item.get("message_ref") or "") in refs]
    # 两侧使用相同规则识别完整时间 token，避免截掉“半/以后/范围”等限定。
    # 方括号是另一种未授权引用格式，不得从其中单独抽出裸时间作为业务来源。
    unbracketed = [re.sub(r"\[[^\]]*\]|【[^】]*】|《[^》]*》", "", source) for source in source_texts]
    source_times = {match.group() for source in unbracketed for match in _SOURCE_TIME.finditer(source)}
    time_matches = list(_SOURCE_TIME.finditer(summary))
    for quantity in _QUANTITY.finditer(summary):
        if not any(match.start() <= quantity.start() and quantity.end() <= match.end() for match in time_matches):
            return "", [], "report_summary_invalid", 0
    summary_without_time = summary
    for match in reversed(time_matches):
        token = match.group()
        if token not in source_times:
            return "", [], "report_summary_invalid", 0
        summary_without_time = summary_without_time[:match.start()] + summary_without_time[match.end():]
    normalized_without_time = unicodedata.normalize("NFKC", summary_without_time)
    # 已核验时间以外仍检查数量语境，再去除有限普通词，防止放开统计数字。
    if _QUANTITY.search(normalized_without_time):
        return "", [], "report_summary_invalid", 0
    numeral_text = normalized_without_time
    for word in _ORDINARY_NUMERAL_WORDS:
        numeral_text = numeral_text.replace(word, "")
    if any(char.isnumeric() or char == "两" for char in numeral_text):
        return "", [], "report_summary_invalid", 0
    return summary, refs, None, metadata_removed


def _render_report(stats: dict[str, Any], summary: str, refs: list[str], metadata_removed: int = 0) -> str:
    # 分类总数、计数及比例直接渲染自程序统计，不复用模型产出的统计段。
    lines = ["## 分类统计", f"可信分类：{stats['total_messages']} 条"]
    for category in stats["categories"]:
        lines.append(f"- {category['name']}：{category['count']} 条（{category['ratio']}%）")
    lines.append("## 汇报要点")
    if metadata_removed:
        lines.append("（已省略来源系统的 UTC 时间标签）")
    lines.extend(line for line in summary.splitlines() if not _SUMMARY_HEADING.fullmatch(line))
    if refs:
        lines.append(f"引用: [{', '.join(refs)}]")
    return "\n".join(lines)


def _program_partial_report(
    control: GroupControl, stats: dict[str, Any],
    failed_refs: list[str], unprocessed: list[dict[str, Any]],
) -> str:
    """程序 partial 报告：只用已校验分类与统计，说明未完成原因；不保留/补造引用。"""
    lines = ["## 总结", f"窗口 {control.start} ~ {control.end_exclusive} 的群消息分析报告（程序生成）。"]
    lines.append("## 分类统计")
    lines.append(f"可信分类：{stats['total_messages']} 条")
    for category in stats["categories"]:
        lines.append(f"- {category['name']}：{category['count']} 条（{category['ratio']}%）")
    lines.append("## 汇报要点")
    if failed_refs or unprocessed:
        lines.append(f"- 覆盖说明：{len(failed_refs)} 条分类失败、{len(unprocessed)} 条未处理（超长）。")
    lines.append("- 报告由程序依据已校验统计生成：模型报告摘要或引用未通过校验。")
    return "\n".join(lines)


def _with_coverage_note(report: str, analysis: dict[str, Any]) -> str:
    coverage = analysis["coverage"]
    note = (f"⚠ 覆盖情况：共接收 {coverage['received_unique']} 条，可信分类 {coverage['classified']} 条，"
            f"失败 {coverage['failed']} 条，未处理 {coverage['unprocessed']} 条；"
            f"原因：{', '.join(analysis['partial_reasons'])}。")
    return f"{report}\n\n{note}"
