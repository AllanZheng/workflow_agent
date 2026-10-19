"""M2 业务工作流测试：群分析（目标/时间/口径/分类/覆盖/报告/澄清）、风险会商、知识查询。

行为编号（对应测试方案 §5.6–§5.8）：
- G01 群目标；G02 时间与口径；G03 分类修复；G04 覆盖与状态；G05 统计复算；
  G06 报告引用；G07 澄清续接；G08 上游边界。
- H01 会商 schema；H02 格式修复；H03 知识状态矩阵。
- K01 知识查询；K02 引用拒绝；APP：chat 终端工具随 group_analysis 开关注入。

约定：全程离线（模型/工具替身来自 tests.native_helpers），reference_time 显式给定；
与规范不符且确认属内核缺陷的行为，以 xfail(strict=False) 标注并在 reason 中说明。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from flowagent.app import build_runner
from flowagent.core.contracts import (
    ErrorCode,
    FlowAgentCoreError,
    Status,
    ToolResultData,
)
from flowagent.workflows.consultation import build_consultation_workflow
from flowagent.workflows.group_analysis import (
    FETCH_TOOL,
    build_group_analysis_workflow,
    run_group_analysis_via_tool,
)
from flowagent.workflows.group_rules import (
    NeedsClarification,
    build_control,
    control_to_fetch_arguments,
    resolve_taxonomy,
)
from flowagent.workflows.knowledge import build_knowledge_workflow
from flowagent.workflows.outputs import (
    OutputInvalid,
    filter_citations,
    summarize_knowledge_events,
    validate_consultation_document,
    validate_consultation_tasks,
)
from tests.native_helpers import FakeModel, FakeTools, make_context

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------- 公共脚手架

REF = datetime(2026, 10, 19, 10, 0, tzinfo=timezone(timedelta(hours=8)))
REF_ISO = "2026-10-19T10:00:00+08:00"            # 参考时钟：2026-10-19（周一）10:00
TARGET = SimpleNamespace(conversation_id=42)     # 可信群目标
Q_3D = "分析本群最近3天，按风险类、建议类统计"    # 标准群分析问题（3天窗+两类口径）


def _ctx(model: Any, tools: Any, *, reference_time: str = REF_ISO):
    """组装上下文：工作流消费 ISO 字符串参考时钟。"""
    ctx = make_context(model=model, tools=tools)
    ctx.reference_time = reference_time
    return ctx


def _group_inputs(message, *, target=None, continuation=None, metadata=None, model_args=None):
    """群分析 inputs；analysis_target 为平台可信目标（非模型参数）。"""
    return {"message": message, "analysis_target": target,
            "request_metadata": metadata or {}, "continuation": continuation,
            "model_arguments": model_args}


def _messages(*pairs):
    """群消息载荷行 [{message_ref, text}]。"""
    return [{"message_ref": ref, "text": text} for ref, text in pairs]


def _payload(messages, *, counts=None, partial_reasons=None):
    """取数工具成功载荷；partial_reasons 非空时视作上游 partial。"""
    raw: dict[str, Any] = {"messages": messages,
                           "window": {"start": REF_ISO, "end_exclusive": REF_ISO},
                           "counts": counts if counts is not None else
                           {"fetched": len(messages), "included": len(messages)}}
    if partial_reasons:
        raw["partial_reasons"] = list(partial_reasons)
        raw["status"] = "partial"
    return raw


def _cls_raw(assignments):
    """分类器脚本轮：任意 assignments JSON。"""
    return FakeModel.text_turn(
        json.dumps({"assignments": assignments, "summary": "设备巡检与风险"}, ensure_ascii=False))


def _cls(mapping):
    """合法分类轮：{message_ref: category} 全覆盖。"""
    return _cls_raw([{"message_ref": r, "category": c} for r, c in mapping.items()])


def _bad_cls():
    """非法分类轮：重复引用 + 群外引用 + 未授权类别（触发一次修复）。"""
    return _cls_raw([{"message_ref": "m1", "category": "风险类"},
                     {"message_ref": "m1", "category": "风险类"},
                     {"message_ref": "mX", "category": "八卦类"}])


def _rep_text(body: str):
    """报告脚本轮：原始正文。"""
    return FakeModel.text_turn(body)


def _rep(refs):
    """报告脚本轮：末尾带引用行。"""
    return _rep_text("## 总结\n设备风险类消息需跟进。\n## 分类统计\n见统计表。\n"
                     "## 汇报要点\n关注设备老化风险。\n引用: [" + ", ".join(refs) + "]")


async def _run_group(model_turns, inputs, *, payload=None, fetch_result=None, cfg=None, ctx=None):
    """跑一次群分析工作流；返回 (RunResult, 模型替身, 工具替身)。"""
    results = {FETCH_TOOL: fetch_result if fetch_result is not None
               else (payload if payload is not None else {"messages": []})}
    tools = FakeTools(catalog=[{"name": FETCH_TOOL, "description": "群消息取数",
                                "input_schema": {"type": "object"}}], results=results)
    model = FakeModel(model_turns)
    if ctx is None:
        ctx = _ctx(model, tools)
    spec = build_group_analysis_workflow(config=cfg)
    return await spec.run(inputs, ctx), model, tools


# ---------------------------------------------------------------- G01 群目标

async def test_g01_this_group_uses_trusted_target():
    """G01 本群指代+可信目标 → 目标会话即取数对象；取数参数=控制参数。"""
    control = build_control(original_question=Q_3D, trusted_target=TARGET, reference_time=REF)
    assert control.conversation_id == 42
    assert control_to_fetch_arguments(control) == {
        "conversation_id": 42,
        "start": "2026-10-16T10:00:00+08:00",
        "end": "2026-10-19T10:00:00+08:00",
    }


async def test_g01_named_other_group_with_trusted_target_requires_clarification():
    """G01 有可信目标但点名其他群 → 应 NeedsClarification（group_unresolved）。"""
    with pytest.raises(NeedsClarification) as exc:
        build_control(original_question="分析生产环境群的消息，按风险类、建议类统计",
                      trusted_target=TARGET, reference_time=REF)
    assert exc.value.clarification.reason == "group_unresolved"


async def test_g01_no_target_no_naming_leaves_group_open():
    """G01 无可信目标且未点名任何群 → conversation_id 为 None（工作流转 needs_input）。"""
    control = build_control(original_question="分析最近3天的消息，按风险类、建议类统计",
                            trusted_target=None, reference_time=REF)
    assert control.conversation_id is None
    assert control.group_name == ""


async def test_g01_trusted_target_beats_model_arguments():
    """G01 可信目标优先；模型给的 conversation_id/days 只进诊断、不采用。"""
    control = build_control(
        original_question=Q_3D, trusted_target=TARGET, reference_time=REF,
        model_arguments={"conversation_id": 999, "days": 30})
    assert control.conversation_id == 42
    assert control.start == "2026-10-16T10:00:00+08:00"  # days=30 未采用，窗口仍 3 天
    assert control.diagnostics["model_arguments"] == {"conversation_id": 999, "days": 30}


async def test_g01_terminal_tool_entry_uses_original_question():
    """G01 终端工具入口：控制参数取 ctx.original_question 与 chat 暂存的 _inputs。"""
    tools = FakeTools(results={FETCH_TOOL: _payload(_messages(("m1", "设备报警需要处理")))})
    model = FakeModel([_cls({"m1": "风险类"}), _rep(["m1"])])
    ctx = _ctx(model, tools)
    ctx.original_question = Q_3D
    ctx.diagnostics["_inputs"] = {"analysis_target": TARGET, "request_metadata": {}}
    result = await run_group_analysis_via_tool({"days": 30}, ctx, {})
    assert result.status is Status.COMPLETED
    assert tools.calls[0] == (FETCH_TOOL, {"conversation_id": 42,
                                           "start": "2026-10-16T10:00:00+08:00",
                                           "end": "2026-10-19T10:00:00+08:00"})


async def test_g01_original_user_message_metadata_wins():
    """G01 request_metadata.original_user_message 才是真实用户输入（控制参数来源）。"""
    tools = FakeTools(results={FETCH_TOOL: _payload(_messages(("m1", "会议室空调报修")))})
    model = FakeModel([_cls({"m1": "问题"}), _rep(["m1"])])
    ctx = _ctx(model, tools)
    inputs = _group_inputs("帮我看看", target=TARGET,
                           metadata={"original_user_message": "分析本群最近2天，按问题、通知分类"})
    result = await build_group_analysis_workflow().run(inputs, ctx)
    assert result.status is Status.COMPLETED
    assert tools.calls[0][1]["start"] == "2026-10-17T10:00:00+08:00"  # 最近2天
    taxonomy = result.output["analysis_result"]["effective_taxonomy"]["categories"]
    assert taxonomy == ["问题", "通知", "未归类"]


# ---------------------------------------------------------------- G02 时间与口径

async def test_g02_recent_days_window_and_taxonomy():
    """G02 “最近3天”→ 滚动 3 天窗；两类口径追加未归类。"""
    control = build_control(original_question=Q_3D, trusted_target=TARGET, reference_time=REF)
    assert (control.start, control.end_exclusive) == (
        "2026-10-16T10:00:00+08:00", "2026-10-19T10:00:00+08:00")
    assert control.taxonomy == ("风险类", "建议类", "未归类")
    assert control.diagnostics["window"]["expr"] == "relative_days"


async def test_g02_absolute_window_valid_two_categories():
    """G02 绝对区间（带时刻）→ [start, end) 原样采用；口径同步解析。"""
    control = build_control(
        original_question="2026年10月16日00:00至2026年10月18日00:00 按风险类、建议类统计",
        trusted_target=TARGET, reference_time=REF)
    assert (control.start, control.end_exclusive) == (
        "2026-10-16T00:00:00+08:00", "2026-10-18T00:00:00+08:00")
    assert control.taxonomy == ("风险类", "建议类", "未归类")
    assert control.diagnostics["window"]["expr"] == "absolute"


@pytest.mark.parametrize("question,reason", [
    ("2026年9月1日00:00至2026年10月18日00:00 按风险类、建议类统计", "time_window_invalid"),
    ("2026年10月18日00:00至2026年10月16日00:00 按风险类、建议类统计", "time_window_invalid"),
    ("分析本群国庆期间的发言，按风险类、建议类统计", "time_window_invalid"),
    ("2026年10月16日00:00至2026年10月18日00:00 按风险类统计", "taxonomy_invalid"),
], ids=["over-31d", "reversed", "suspect", "single-cat-abs"])
async def test_g02_invalid_conditions_require_clarification(question, reason):
    """G02 疑似不可解析/绝对区间单类/超31天/倒序 → NeedsClarification。"""
    with pytest.raises(NeedsClarification) as exc:
        build_control(original_question=question, trusted_target=TARGET, reference_time=REF)
    assert exc.value.clarification.reason == reason
    assert exc.value.clarification.missing_fields


@pytest.mark.parametrize("question", [
    "按风险类统计",                      # 只有 1 类
    "按技术类、产品类、运营类、市场类、财务类、人事类、安全类、质量类、客户类统计",  # 9 类
], ids=["one-cat", "nine-cats"])
async def test_g02_category_count_bounds_require_clarification(question):
    """G02 口径越界（1类/9类）→ resolve_taxonomy 抛 NeedsClarification（不回默认）。"""
    with pytest.raises(NeedsClarification) as exc:
        resolve_taxonomy(original_question=question, default=("问题", "通知", "讨论", "闲聊", "其他"))
    assert exc.value.clarification.reason == "taxonomy_invalid"


@pytest.mark.parametrize("question,start,end", [
    ("分析本群上周的发言，按风险类、建议类统计",
     "2026-10-12T00:00:00+08:00", "2026-10-19T00:00:00+08:00"),
    ("分析本群近6小时的发言，按风险类、建议类统计",
     "2026-10-19T04:00:00+08:00", "2026-10-19T10:00:00+08:00"),
], ids=["last-week", "recent-hours"])
async def test_g02_relative_windows(question, start, end):
    """G02 “上周”=上周一零点至本周一零点；“近N小时”=滚动窗。"""
    control = build_control(original_question=question, trusted_target=TARGET, reference_time=REF)
    assert (control.start, control.end_exclusive) == (start, end)


async def test_g02_default_window_ignores_model_days():
    """G02 无时间表达 → 配置默认窗（source=default）；模型 days 只进诊断。"""
    control = build_control(
        original_question="分析本群的消息，按风险类、建议类分类",
        trusted_target=TARGET, reference_time=REF,
        config={"default_days": 7}, model_arguments={"days": 30})
    assert (control.start, control.end_exclusive) == (
        "2026-10-12T10:00:00+08:00", "2026-10-19T10:00:00+08:00")
    assert control.diagnostics["window"]["window_source"] == "default"
    assert control.diagnostics["model_arguments"] == {"days": 30}
    assert control.taxonomy == ("风险类", "建议类", "未归类")


@pytest.mark.parametrize("question", ["按时间顺序总结本群发言", "按人汇总本群发言"],
                         ids=["time-order", "by-person"])
async def test_g02_non_category_phrases_fall_to_default(question):
    """G02 “按时间顺序总结/按人汇总”不是类别列表 → 不澄清，回默认口径（配置原样）。"""
    taxonomy, diag = resolve_taxonomy(
        original_question=question, default=("问题", "通知", "讨论", "闲聊", "其他"))
    assert taxonomy == ("问题", "通知", "讨论", "闲聊", "其他")  # 默认口径不加未归类
    assert diag["taxonomy_source"] == "default"


async def test_g02_taxonomy_alias_mapping():
    """G02 别名归一：事故类→风险类 后去重，再追加未归类。"""
    control = build_control(
        original_question="分析本群最近3天，按事故类、建议类统计",
        trusted_target=TARGET, reference_time=REF,
        config={"taxonomy_aliases": {"事故类": "风险类"}})
    assert control.taxonomy == ("风险类", "建议类", "未归类")


# ---------------------------------------------------------------- G03 分类修复

async def test_g03_classification_repair_once_recovers():
    """G03 首轮分类非法（重复/群外引用/未授权类别）→ 一次同证据修复后全部可信。"""
    payload = _payload(_messages(("m1", "设备报警需要处理"), ("m2", "收到，已派单")))
    model_turns = [_bad_cls(), _cls({"m1": "风险类", "m2": "建议类"}), _rep(["m1"])]
    result, model, _ = await _run_group(model_turns, _group_inputs(Q_3D, target=TARGET),
                                        payload=payload)
    assert result.status is Status.COMPLETED
    cov = result.output["analysis_result"]["coverage"]
    assert (cov["classified"], cov["failed"]) == (2, 0)
    assert model.requests[0]["messages"][0]["content"].startswith("你是群消息分类器")
    assert "约束" not in model.requests[0]["messages"][1]["content"]
    assert "约束" in model.requests[1]["messages"][1]["content"]  # 修复轮带精确约束


async def test_g03_two_bad_attempts_counted_failed_partial():
    """G03 同批两次都失败 → 记 failed；分批隔离下另一批不受影响，整体 partial。"""
    payload = _payload(_messages(("m1", "一"), ("m2", "二"), ("m3", "三")))
    model_turns = [_cls({"m1": "风险类", "m2": "建议类"}),  # 第一批一次通过
                   _bad_cls(), _bad_cls(),                  # 第二批（m3）两次失败
                   _rep(["m1", "m2"])]
    result, model, _ = await _run_group(model_turns, _group_inputs(Q_3D, target=TARGET),
                                        payload=payload, cfg={"batch_messages": 2})
    assert result.status is Status.PARTIAL
    analysis = result.output["analysis_result"]
    cov = analysis["coverage"]
    assert (cov["classified"], cov["failed"]) == (2, 1)
    assert cov["received_unique"] == cov["classified"] + cov["failed"] + cov["unprocessed"] == 3
    assert "classification_failed" in analysis["partial_reasons"]
    assert "约束" in model.requests[2]["messages"][1]["content"]  # 第二批第二次尝试


# ---------------------------------------------------------------- G04 覆盖与状态

async def test_g04_no_messages_upstream_complete():
    """G04 上游完整但零消息 → completed“无消息”，不发模型请求。"""
    result, model, _ = await _run_group([], _group_inputs(Q_3D, target=TARGET),
                                        payload=_payload([]))
    assert result.status is Status.COMPLETED
    assert "没有群消息" in result.reply
    analysis = result.output["analysis_result"]
    assert analysis["statistics"] == {"total_messages": 0, "categories": []}
    assert analysis["coverage"]["received_unique"] == 0
    assert model.call_count == 0


async def test_g04_upstream_partial_reasons_passthrough():
    """G04 上游 partial（limit 截断）→ partial_reasons 原样透传，整体 partial。"""
    payload = _payload(_messages(("m1", "一"), ("m2", "二")),
                       counts={"fetched": 1001, "included": 1000}, partial_reasons=["limit"])
    model_turns = [_cls({"m1": "风险类", "m2": "建议类"}), _rep(["m1"])]
    result, _, _ = await _run_group(model_turns, _group_inputs(Q_3D, target=TARGET),
                                    payload=payload)
    analysis = result.output["analysis_result"]
    assert result.status is Status.PARTIAL
    assert analysis["partial_reasons"] == ["limit"]
    assert analysis["coverage"]["upstream"] == {"fetched": 1001, "included": 1000}
    assert analysis["coverage"]["received_unique"] == 2


async def test_g04_all_classification_failed_raises_output_invalid():
    """G04 有消息但全部分类失败 → 业务未完成，抛 OUTPUT_INVALID 交唯一兜底。"""
    payload = _payload(_messages(("m1", "一"), ("m2", "二")))
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_group([_bad_cls(), _bad_cls()], _group_inputs(Q_3D, target=TARGET),
                         payload=payload)
    assert exc.value.code is ErrorCode.OUTPUT_INVALID


async def test_g04_overlong_single_message_unprocessed_partial():
    """G04 单条超长 → unprocessed(message_too_large)，不炸整次分析，整体 partial。"""
    payload = _payload([{"message_ref": "m1", "text": "长" * 12001},
                        {"message_ref": "m2", "text": "正常消息"}])
    model_turns = [_cls({"m2": "风险类"}), _rep(["m2"])]
    result, _, _ = await _run_group(model_turns, _group_inputs(Q_3D, target=TARGET),
                                    payload=payload)
    analysis = result.output["analysis_result"]
    assert result.status is Status.PARTIAL
    cov = analysis["coverage"]
    assert cov["unprocessed"] == 1
    assert cov["unprocessed_detail"] == [{"message_ref": "m1", "reason": "message_too_large"}]
    assert cov["received_unique"] == cov["classified"] + cov["failed"] + cov["unprocessed"] == 2
    assert analysis["partial_reasons"] == ["message_too_large"]
    assert "覆盖情况" in result.reply


# ---------------------------------------------------------------- G05 统计复算

async def test_g05_statistics_recomputable_from_assignments():
    """G05 统计由程序复算：分母=classified；比例可复算；引用取原文且≤2条。"""
    texts = {"m1": "泵房异响", "m2": "消防通道被占", "m3": "好的收到", "m4": "明天例会改下午"}
    payload = _payload(_messages(*texts.items()))
    labels = {"m1": "风险类", "m2": "风险类", "m3": "建议类", "m4": "建议类"}
    result, _, _ = await _run_group([_cls(labels), _rep(["m1"])],
                                    _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.COMPLETED
    stats = result.output["analysis_result"]["statistics"]
    assert stats["total_messages"] == 4
    assert sum(c["count"] for c in stats["categories"]) == 4  # 分母=classified
    by_name = {c["name"]: c for c in stats["categories"]}
    assert by_name["风险类"]["count"] == 2
    assert by_name["风险类"]["ratio"] == round(2 * 100 / 4, 1) == 50.0
    assert by_name["建议类"]["quotes"] == [texts["m3"], texts["m4"]]  # 引用与原文一致
    assert all(len(q) <= 60 for c in stats["categories"] for q in c["quotes"])


# ---------------------------------------------------------------- G06 报告引用

async def test_g06_report_invalid_ref_repaired_once():
    """G06 报告引用越界(m9) → 同证据修复一次后可信，repair_attempted=True。"""
    payload = _payload(_messages(("m1", "一"), ("m2", "二")))
    model_turns = [_cls({"m1": "风险类", "m2": "建议类"}), _rep(["m9"]), _rep(["m1"])]
    result, model, _ = await _run_group(model_turns, _group_inputs(Q_3D, target=TARGET),
                                        payload=payload)
    analysis = result.output["analysis_result"]
    assert result.status is Status.COMPLETED
    assert analysis["report_refs"] == ["m1"]
    assert analysis["report_repair_attempted"] is True
    assert "引用列表只能包含" in model.requests[2]["messages"][1]["content"]


async def test_g06_report_repair_still_invalid_program_partial():
    """G06 修复后引用仍无效 → 降级程序 partial 报告，不展示无效引用。"""
    payload = _payload(_messages(("m1", "一"), ("m2", "二")))
    model_turns = [_cls({"m1": "风险类", "m2": "建议类"}), _rep(["m9"]), _rep(["m9"])]
    result, _, _ = await _run_group(model_turns, _group_inputs(Q_3D, target=TARGET),
                                    payload=payload)
    analysis = result.output["analysis_result"]
    assert result.status is Status.PARTIAL
    assert "report_refs_invalid" in analysis["partial_reasons"]
    assert analysis["report_refs"] == []
    assert "程序生成" in result.reply
    assert "m9" not in result.reply


async def test_g06_no_ref_line_means_no_repair():
    """G06 报告无引用行 → 按零引用处理，不再修复（模型共调用 2 次）。"""
    payload = _payload(_messages(("m1", "一")))
    model_turns = [_cls({"m1": "风险类"}),
                   _rep_text("## 总结\n正文没有引用行。\n## 分类统计\n略。\n## 汇报要点\n无。")]
    result, model, _ = await _run_group(model_turns, _group_inputs(Q_3D, target=TARGET),
                                        payload=payload)
    analysis = result.output["analysis_result"]
    assert result.status is Status.COMPLETED
    assert analysis["report_refs"] == []
    assert analysis["report_repair_attempted"] is False
    assert model.call_count == 2


# ---------------------------------------------------------------- G07 澄清续接

async def test_g07_first_run_needs_input_with_v1_continuation():
    """G07 首跑缺群目标 → needs_input；continuation v1 计数=1；正文带机器标记。"""
    result, model, _ = await _run_group(
        [], _group_inputs("分析最近3天的消息，按风险类、建议类统计"), payload=_payload([]))
    assert result.status is Status.NEEDS_INPUT
    assert result.clarification is not None
    assert result.clarification.reason == "group_unresolved"
    assert result.clarification.missing_fields == ["目标群"]
    cont = result.clarification.continuation
    assert cont["version"] == 1 and cont["workflow"] == "group_analysis"
    assert cont["clarify_count"] == 1
    assert cont["original_request"] == "分析最近3天的消息，按风险类、建议类统计"
    assert cont["reference_time"] == REF_ISO and cont["resolved_fields"] == {}
    assert "sample-agent:clarification:v1" in result.reply  # render_body 机器标记
    assert result.recovery["clarify_count"] == 1
    assert model.call_count == 0  # 澄清正文由代码模板生成，模型零调用


_CONT_V1 = {
    "version": 1, "workflow": "group_analysis",
    "original_request": "分析本群最近3天，按风险类、建议类统计",
    "reference_time": REF_ISO, "timezone": "Asia/Shanghai",
    "resolved_fields": {}, "clarify_count": 1,
}


async def test_g07_second_run_executes_with_original_taxonomy():
    """G07 续接：口径由原问题保留；补充文本解析失败不炸流程（回原问题窗口）。"""
    payload = _payload(_messages(("m1", "一"), ("m2", "二")))
    model_turns = [_cls({"m1": "风险类", "m2": "建议类"}), _rep(["m1"])]
    result, _, tools = await _run_group(
        model_turns,
        _group_inputs("2026年10月16日至10月18日", target=TARGET, continuation=dict(_CONT_V1)),
        payload=payload)
    assert result.status is Status.COMPLETED
    analysis = result.output["analysis_result"]
    assert analysis["effective_taxonomy"]["categories"] == ["风险类", "建议类", "未归类"]
    assert tools.calls[0][1]["conversation_id"] == 42
    # 补充时间缺时刻不可解析（supplement 源吞掉澄清）→ 回原问题的“最近3天”窗。
    assert tools.calls[0][1]["start"] == "2026-10-16T10:00:00+08:00"


async def test_g07_second_run_supplement_time_applied():
    """G07 续接补充可解析时间 → 窗口按补充执行；口径仍由原问题决定。"""
    payload = _payload(_messages(("m1", "一")))
    model_turns = [_cls({"m1": "风险类"}), _rep(["m1"])]
    result, _, tools = await _run_group(
        model_turns,
        _group_inputs("2026年10月16日00:00至2026年10月18日00:00", target=TARGET,
                      continuation=dict(_CONT_V1)),
        payload=payload)
    assert result.status is Status.COMPLETED
    assert tools.calls[0][1] == {"conversation_id": 42,
                                 "start": "2026-10-16T00:00:00+08:00",
                                 "end": "2026-10-18T00:00:00+08:00"}
    assert (result.output["analysis_result"]["effective_taxonomy"]["categories"]
            == ["风险类", "建议类", "未归类"])


async def test_g07_clarify_count_two_and_still_missing_raises_exhausted():
    """G07 已澄清两轮仍缺必要条件 → CLARIFICATION_EXHAUSTED（不取数、不生成）。"""
    cont = dict(_CONT_V1, clarify_count=2,
                original_request="分析生产环境群的消息，按风险类、建议类统计")
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_group([], _group_inputs("这次再说", continuation=cont), payload=_payload([]))
    assert exc.value.code is ErrorCode.CLARIFICATION_EXHAUSTED


async def test_g07_cross_midnight_yesterday_uses_original_reference():
    """G07 参考时钟固定 23:50，“昨天”按原参考解出前一整天（不漂移到今天）。"""
    tools = FakeTools(results={FETCH_TOOL: _payload(_messages(("m1", "昨夜值班记录")))})
    model = FakeModel([_cls({"m1": "风险类"}), _rep(["m1"])])
    ctx = _ctx(model, tools, reference_time="2026-10-19T23:50:00+08:00")
    result = await build_group_analysis_workflow().run(
        _group_inputs("分析本群昨天，按风险类、建议类统计", target=TARGET), ctx)
    assert result.status is Status.COMPLETED
    assert tools.calls[0][1] == {"conversation_id": 42,
                                 "start": "2026-10-18T00:00:00+08:00",
                                 "end": "2026-10-19T00:00:00+08:00"}


# ---------------------------------------------------------------- G08 上游边界

async def test_g08_fetch_rejects_code_arguments():
    """G08 取数工具以 TOOL_ARGUMENT_INVALID 拒绝代码参数 → 契约错误上抛。"""
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_group([], _group_inputs(Q_3D, target=TARGET),
                         fetch_result=ToolResultData.error("bad args", "TOOL_ARGUMENT_INVALID"))
    assert exc.value.code is ErrorCode.TOOL_ARGUMENT_INVALID


async def test_g08_fetch_unknown_state_is_capability_unavailable():
    """G08 取数 unknown/普通失败 → CAPABILITY_UNAVAILABLE 交唯一兜底。"""
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_group([], _group_inputs(Q_3D, target=TARGET),
                         fetch_result=ToolResultData.unknown("网关超时"))
    assert exc.value.code is ErrorCode.CAPABILITY_UNAVAILABLE


# ---------------------------------------------------------------- 会商脚手架

WIKI_CATALOG = [{"name": "wiki__hybrid_search", "description": "知识库混合检索",
                 "input_schema": {"type": "object"}}]
WIKI_HIT = {"data": [{"knowledge_id": "kb-1", "knowledge_title": "巡检制度",
                      "id": "c1", "content": "设备每月巡检一次", "score": 0.9}]}


def _doc_text(**overrides):
    """文档 JSON 脚本轮（draft/final 共用 schema）。"""
    doc = {"title": "十月风险会商纪要", "summary": "整体风险可控。",
           "body_markdown": "## 结论\n本月风险总体可控。",
           "risks": [{"level": "high", "title": "设备老化", "detail": "巡检发现老化隐患"}],
           "evidence_refs": ["m1"], "knowledge_citations": []}
    doc.update(overrides)
    return FakeModel.text_turn(json.dumps(doc, ensure_ascii=False))


def _tasks_text(tasks, summary="任务安排"):
    """任务 JSON 脚本轮。"""
    return FakeModel.text_turn(
        json.dumps({"summary": summary, "tasks": tasks}, ensure_ascii=False))


def _task(content="完成巡检整改", **overrides):
    item = {"content": content, "assignee_user_id": None, "due_date": None,
            "source_risk": "", "evidence_refs": []}
    item.update(overrides)
    return item


def _cons_inputs(stage, schema, *, mode="off", members=None):
    """会商 inputs：function_task 携带 stage/output_schema/inputs。"""
    return {"function_task": {"function_key": "risk_consultation", "stage": stage,
                              "output_schema": schema,
                              "inputs": {"knowledge_mode": mode,
                                         "assignable_members": members or []}},
            "message": "会议材料：十月巡检发现设备老化风险，需评估整改。"}


async def _run_consultation(model_turns, inputs, *, tools=None):
    """跑一次会商阶段工作流；返回 (RunResult, 模型替身)。"""
    model = FakeModel(model_turns)
    spec = build_consultation_workflow()
    return await spec.run(inputs, _ctx(model, tools)), model


# ---------------------------------------------------------------- H01 会商 schema

async def test_h01_draft_valid_document_completes():
    """H01 合法文档草稿 → 逐字段校验通过，reply=正文，无修复，知识 not_used。"""
    result, model = await _run_consultation(
        [_doc_text()], _cons_inputs("draft", "consultation_document_v1"))
    fr = result.output["function_result"]
    assert result.status is Status.COMPLETED
    assert fr["status"] == "ok" and fr["format_repair_attempted"] is False
    assert fr["output"]["title"] == "十月风险会商纪要"
    assert fr["output"]["risks"][0]["level"] == "high"
    assert fr["knowledge"]["status"] == "not_used"
    assert result.reply == "## 结论\n本月风险总体可控。"
    assert model.call_count == 1


async def test_h01_summary_truncated_to_300():
    """H01 summary 301 字 → 截断到 300，不触发修复。"""
    result, model = await _run_consultation(
        [_doc_text(summary="长" * 301)], _cons_inputs("draft", "consultation_document_v1"))
    assert result.status is Status.COMPLETED
    assert len(result.output["function_result"]["output"]["summary"]) == 300
    assert model.call_count == 1


async def test_h01_invalid_risk_level_repaired_once_without_tools():
    """H01 risk.level 非法 → 一次无工具修复后通过；修复轮提示带“未通过校验”。"""
    bad = _doc_text(risks=[{"level": "critical", "title": "设备老化", "detail": "d"}])
    result, model = await _run_consultation(
        [bad, _doc_text()], _cons_inputs("draft", "consultation_document_v1"))
    fr = result.output["function_result"]
    assert result.status is Status.COMPLETED
    assert fr["format_repair_attempted"] is True
    assert model.call_count == 2
    assert "未通过校验" in model.requests[1]["messages"][0]["content"]
    assert model.requests[1]["tools"] == []  # 修复轮不提供任何工具


@pytest.mark.parametrize("stage,schema", [
    ("draft", "consultation_tasks_v1"),
    ("tasks", "consultation_document_v1"),
    ("unknown", "consultation_document_v1"),
], ids=["doc-as-tasks", "tasks-as-doc", "unknown-stage"])
async def test_h01_stage_schema_mismatch_invalid_input(stage, schema):
    """H01 stage 与 output_schema 不匹配 → INVALID_INPUT（模型零调用）。"""
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_consultation([_doc_text()], _cons_inputs(stage, schema))
    assert exc.value.code is ErrorCode.INVALID_INPUT


async def test_h01_tasks_empty_is_valid_no_tasks_reply():
    """H01 空任务数组合法 → reply 含“本次无任务”。"""
    result, _ = await _run_consultation(
        [_tasks_text([], summary="无新任务")], _cons_inputs("tasks", "consultation_tasks_v1"))
    assert result.status is Status.COMPLETED
    assert "本次无任务" in result.reply
    assert result.output["function_result"]["output"]["tasks"] == []


async def test_h01_tasks_missing_content_repaired_once():
    """H01 任务缺 content → 一次修复后通过。"""
    result, model = await _run_consultation(
        [_tasks_text([_task(content="")]), _tasks_text([_task()])],
        _cons_inputs("tasks", "consultation_tasks_v1"))
    assert result.status is Status.COMPLETED
    assert result.output["function_result"]["format_repair_attempted"] is True
    assert model.call_count == 2


async def test_h01_tasks_assignee_invalid_repaired():
    """H01 assignee 布尔被拒 → 修复轮改为正整数后通过；reply 为编号列表。"""
    result, _ = await _run_consultation(
        [_tasks_text([_task(assignee_user_id=True)]),
         _tasks_text([_task(assignee_user_id=11)])],
        _cons_inputs("tasks", "consultation_tasks_v1"))
    fr = result.output["function_result"]
    assert result.status is Status.COMPLETED
    assert fr["format_repair_attempted"] is True
    assert fr["output"]["tasks"][0]["assignee_user_id"] == 11
    assert result.reply.splitlines()[-1] == "1. 完成巡检整改"


async def test_h01_due_date_unparseable_becomes_null():
    """H01 “明年”这类不可解析期限 → 归 null，不生成错误期限、不修复。"""
    result, model = await _run_consultation(
        [_tasks_text([_task(due_date="明年")])], _cons_inputs("tasks", "consultation_tasks_v1"))
    fr = result.output["function_result"]
    assert result.status is Status.COMPLETED
    assert fr["format_repair_attempted"] is False
    assert fr["output"]["tasks"][0]["due_date"] is None
    assert model.call_count == 1


# ---------------------------------------------------------------- H02 格式修复

async def test_h02_format_repair_once_exactly_two_calls():
    """H02 首轮非 JSON → 一次无工具修复；模型恰两次调用；修复轮无工具。"""
    result, model = await _run_consultation(
        [FakeModel.text_turn("抱歉，我暂时无法按格式输出。"), _doc_text()],
        _cons_inputs("draft", "consultation_document_v1"))
    fr = result.output["function_result"]
    assert result.status is Status.COMPLETED
    assert fr["format_repair_attempted"] is True
    assert model.call_count == 2
    assert model.requests[1]["tools"] == []
    assert "未通过校验" in model.requests[1]["messages"][0]["content"]


async def test_h02_both_turns_invalid_raises_output_invalid():
    """H02 两轮都非法 → OUTPUT_INVALID 交错误边界。"""
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_consultation(
            [FakeModel.text_turn("不是JSON"), FakeModel.text_turn("还不是JSON")],
            _cons_inputs("draft", "consultation_document_v1"))
    assert exc.value.code is ErrorCode.OUTPUT_INVALID


# ---------------------------------------------------------------- H03 知识状态矩阵

async def test_h03_off_without_wiki_calls_not_used_ok():
    """H03 off+零知识调用 → not_used，业务 ok；会商阶段不暴露非知识工具。"""
    tools = FakeTools(catalog=[{"name": "crm__query", "description": "客户查询",
                                "input_schema": {"type": "object"}}])
    result, model = await _run_consultation(
        [_doc_text()], _cons_inputs("draft", "consultation_document_v1", mode="off"),
        tools=tools)
    assert result.status is Status.COMPLETED
    assert result.output["function_result"]["knowledge"]["status"] == "not_used"
    assert model.requests[0]["tools"] == []  # wiki-only 过滤后无可暴露工具


async def test_h03_optional_called_empty_is_valid():
    """H03 optional + 检索成功零命中 → empty 是有效结果，业务 ok。"""
    tools = FakeTools(catalog=WIKI_CATALOG, results={"wiki__hybrid_search": {"data": []}})
    result, _ = await _run_consultation(
        [FakeModel.tool_turn({"id": "t1", "name": "wiki__hybrid_search",
                             "arguments": {"query": "老化"}}),
         _doc_text()],
        _cons_inputs("draft", "consultation_document_v1", mode="optional"), tools=tools)
    fr = result.output["function_result"]
    assert result.status is Status.COMPLETED
    assert fr["knowledge"]["status"] == "empty"
    assert fr["knowledge"]["citations"] == []


async def test_h03_required_ok_keeps_only_real_citations():
    """H03 required + 检索命中 → ok；引用仅保留真实证据支撑（标题用真实值覆盖）。"""
    cited = [{"kb_id": "库A", "knowledge_id": "kb-1", "chunk_id": "c1", "title": "伪造标题"},
             {"kb_id": "库A", "knowledge_id": "kb-ghost", "title": "幽灵文档"}]
    tools = FakeTools(catalog=WIKI_CATALOG, results={"wiki__hybrid_search": WIKI_HIT})
    result, _ = await _run_consultation(
        [FakeModel.tool_turn({"id": "t1", "name": "wiki__hybrid_search",
                             "arguments": {"query": "老化"}}),
         _doc_text(knowledge_citations=cited)],
        _cons_inputs("draft", "consultation_document_v1", mode="required"), tools=tools)
    fr = result.output["function_result"]
    assert result.status is Status.COMPLETED
    assert fr["knowledge"]["status"] == "ok"
    assert fr["knowledge"]["citations"] == [
        {"kb_id": "库A", "knowledge_id": "kb-1", "chunk_id": "c1", "title": "巡检制度"}]
    assert fr["output"]["knowledge_citations"] == fr["knowledge"]["citations"]


async def test_h03_required_empty_still_completes():
    """H03 required + 检索成功零命中 → empty 合法，不抛门禁。"""
    tools = FakeTools(catalog=WIKI_CATALOG, results={"wiki__hybrid_search": {"data": []}})
    result, _ = await _run_consultation(
        [FakeModel.tool_turn({"id": "t1", "name": "wiki__hybrid_search",
                             "arguments": {"query": "老化"}}),
         _doc_text()],
        _cons_inputs("draft", "consultation_document_v1", mode="required"), tools=tools)
    assert result.status is Status.COMPLETED
    assert result.output["function_result"]["knowledge"]["status"] == "empty"


async def test_h03_required_failed_tool_raises():
    """H03 required + 知识工具失败 → CAPABILITY_UNAVAILABLE(knowledge_required_unavailable)。"""
    tools = FakeTools(catalog=WIKI_CATALOG, results={
        "wiki__hybrid_search": ToolResultData.error("kb down", "MCP_TOOL_ERROR")})
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_consultation(
            [FakeModel.tool_turn({"id": "t1", "name": "wiki__hybrid_search",
                             "arguments": {"query": "老化"}}),
             _doc_text()],
            _cons_inputs("draft", "consultation_document_v1", mode="required"), tools=tools)
    assert exc.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert "knowledge_required_unavailable" in str(exc.value)


async def test_h03_required_not_used_raises():
    """H03 required + 知识工具完全不可用（代码发起检索无从发生）→ not_used 过不了门禁。"""
    tools = FakeTools(catalog=[{"name": "crm__query", "description": "客户查询",
                                "input_schema": {"type": "object"}}])
    with pytest.raises(FlowAgentCoreError) as exc:
        await _run_consultation(
            [_doc_text()],
            _cons_inputs("draft", "consultation_document_v1", mode="required"), tools=tools)
    assert exc.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert "not_used" in str(exc.value)


# ---------------------------------------------------------------- K01/K02 知识查询

_KBOUND = {"search": "wiki__hybrid_search"}


async def test_k01_preflight_dependency_missing():
    """K01 未绑定检索工具或无工具能力 → preflight=dependency_missing；齐备为 None。"""
    assert await build_knowledge_workflow(bindings=None).preflight({}, make_context()) \
        == "dependency_missing"
    assert await build_knowledge_workflow(bindings=_KBOUND).preflight({}, make_context()) \
        == "dependency_missing"
    assert await build_knowledge_workflow(bindings=_KBOUND).preflight(
        {}, _ctx(FakeModel(), FakeTools())) is None


async def test_k01_empty_data_is_valid_not_found():
    """K01 检索零命中 → completed“未找到”，status=empty 且无引用，不发模型请求。"""
    tools = FakeTools(catalog=WIKI_CATALOG, results={"wiki__hybrid_search": {"data": []}})
    model = FakeModel()
    result = await build_knowledge_workflow(bindings=_KBOUND).run(
        {"message": "巡检频率是多少？"}, _ctx(model, tools))
    assert result.status is Status.COMPLETED
    assert "没有检索到相关内容" in result.reply
    assert result.output["knowledge_result"] == {"status": "empty", "citations": []}
    assert model.call_count == 0


async def test_k01_search_error_is_capability_unavailable():
    """K01 检索失败（error）→ CAPABILITY_UNAVAILABLE，不冒充“未找到”。"""
    tools = FakeTools(results={
        "wiki__hybrid_search": ToolResultData.error("kb down", "MCP_TOOL_ERROR")})
    with pytest.raises(FlowAgentCoreError) as exc:
        await build_knowledge_workflow(bindings=_KBOUND).run(
            {"message": "巡检频率是多少？"}, _ctx(FakeModel(), tools))
    assert exc.value.code is ErrorCode.CAPABILITY_UNAVAILABLE


async def test_k01_hit_answers_with_status_ok_and_kept_citation():
    """K01 检索命中 → status=ok；答案引用与证据对得上则保留。"""
    tools = FakeTools(catalog=WIKI_CATALOG, results={"wiki__hybrid_search": WIKI_HIT})
    answer = ("根据《巡检制度》，设备每月巡检一次。\n"
              "引用: [{\"knowledge_id\": \"kb-1\", \"title\": \"巡检制度\"}]")
    ctx = _ctx(FakeModel([FakeModel.text_turn(answer)]), tools)
    result = await build_knowledge_workflow(bindings=_KBOUND).run(
        {"message": "巡检频率是多少？"}, ctx)
    assert result.status is Status.COMPLETED
    kr = result.output["knowledge_result"]
    assert kr["status"] == "ok"
    assert [c["knowledge_id"] for c in kr["citations"]] == ["kb-1"]
    assert kr["citations"][0]["title"] == "巡检制度"


async def test_k02_ghost_citation_dropped_real_kept():
    """K02 虚构引用(kb-ghost)拒绝；真实引用(kb-1)保留。"""
    tools = FakeTools(catalog=WIKI_CATALOG, results={"wiki__hybrid_search": WIKI_HIT})
    answer = ("每月巡检一次。\n"
              "引用: [{\"knowledge_id\": \"kb-1\"}, {\"knowledge_id\": \"kb-ghost\"}]")
    ctx = _ctx(FakeModel([FakeModel.text_turn(answer)]), tools)
    result = await build_knowledge_workflow(bindings=_KBOUND).run(
        {"message": "巡检频率是多少？"}, ctx)
    kr = result.output["knowledge_result"]
    assert [c["knowledge_id"] for c in kr["citations"]] == ["kb-1"]


# ---------------------------------------------------------------- outputs 单元

async def test_outputs_document_validator_rules():
    """outputs 文档校验：缺失必填抛错；summary 截断；risks 归一；引用列表容错。"""
    with pytest.raises(OutputInvalid):
        validate_consultation_document({"title": "", "summary": "s", "body_markdown": "b"})
    doc = validate_consultation_document({
        "title": "T", "summary": "长" * 301, "body_markdown": "B",
        "risks": [{"level": "low", "title": "R", "detail": "D"}],
        "evidence_refs": ["m1", None, 3], "knowledge_citations": "bad"})
    assert len(doc["summary"]) == 300
    assert doc["risks"] == [{"level": "low", "title": "R", "detail": "D"}]
    assert doc["evidence_refs"] == ["m1", "3"]
    assert doc["knowledge_citations"] == []


@pytest.mark.parametrize("raw,expected", [
    (None, None), (7, 7), ("9", 9),
    (True, OutputInvalid), (0, OutputInvalid), (-1, OutputInvalid), ("abc", OutputInvalid),
], ids=["null", "int", "digit-str", "bool", "zero", "negative", "text"])
async def test_outputs_assignee_rules(raw, expected):
    """outputs assignee 规则：正整数/数字串/None 通过；bool/0/负数/文本拒绝。"""
    if expected is OutputInvalid:
        with pytest.raises(OutputInvalid):
            validate_consultation_tasks({"tasks": [{"content": "c", "assignee_user_id": raw}]})
    else:
        out = validate_consultation_tasks({"tasks": [{"content": "c", "assignee_user_id": raw}]})
        assert out["tasks"][0]["assignee_user_id"] == expected


@pytest.mark.parametrize("raw,expected", [
    ("2026-10-01", "2026-10-01"), ("明年", None), ("", None), (None, None), ("2026/10/1", None),
], ids=["iso", "vague", "empty", "null", "slash"])
async def test_outputs_due_date_rules(raw, expected):
    """outputs due_date：仅 YYYY-MM-DD 保留，其余归 null。"""
    out = validate_consultation_tasks({"tasks": [{"content": "c", "due_date": raw}]})
    assert out["tasks"][0]["due_date"] == expected


async def test_outputs_summarize_knowledge_states():
    """outputs 知识状态矩阵：not_used/ok/empty/failed 与 docs/chunks 提取。"""
    assert summarize_knowledge_events([])["status"] == "not_used"
    ok = summarize_knowledge_events([{"name": "wiki__hybrid_search", "status": "success",
                                      "result_value": {"data": [
                                          {"knowledge_id": "k1", "knowledge_title": "T",
                                           "id": "c1"}]}}])
    assert ok["status"] == "ok" and ok["docs"] == {"k1": {"title": "T"}}
    assert ("k1", "c1") in ok["chunks"]
    empty = summarize_knowledge_events([{"name": "hybrid_search", "status": "success",
                                         "result_value": {"data": []}}])
    assert empty["status"] == "empty"
    failed = summarize_knowledge_events([{"name": "wiki__get_knowledge", "status": "error",
                                          "error_code": "X", "result_value": None}])
    assert failed["status"] == "failed"


async def test_outputs_filter_citations_rules():
    """outputs 引用过滤：未知文档丢弃；chunk 不匹配丢弃；标题以真实证据覆盖。"""
    knowledge = {"docs": {"k1": {"title": "真实标题"}}, "chunks": {("k1", "c1")}}
    kept = filter_citations([
        {"knowledge_id": "k1", "chunk_id": "c1", "title": "假", "kb_id": "kb"},
        {"knowledge_id": "k1", "chunk_id": "c9", "title": "chunk不对"},
        {"knowledge_id": "ghost", "title": "幽灵"},
        {"knowledge_id": "k1", "chunk_id": None, "title": None},
    ], knowledge)
    assert kept == [
        {"kb_id": "kb", "knowledge_id": "k1", "chunk_id": "c1", "title": "真实标题"},
        {"kb_id": None, "knowledge_id": "k1", "chunk_id": None, "title": "真实标题"},
    ]


# ---------------------------------------------------------------- APP 装配

_PROFILE = {"model": {"name": "mock-model", "base_url": "http://127.0.0.1:9", "api_key": "k"}}


def _chat_terminal_names(runner):
    """轻量内省：从 chat run 闭包读注入的终端工具名（不发模型请求）。"""
    spec = runner.workflows["chat"]
    cells = {name: cell.cell_contents for name, cell in
             zip(spec.run.__code__.co_freevars, spec.run.__closure__ or ())}
    return sorted(t.name for t in (cells.get("tools") or []))


async def test_app_registers_four_workflows_and_toggles_terminal_tool():
    """APP：注册四工作流；chat 终端工具随 group_analysis 开关注入/摘除。"""
    disabled = build_runner({**_PROFILE, "workflows": {"group_analysis": {"enabled": False}}})
    assert set(disabled.workflows) >= {"chat", "group_analysis",
                                       "consultation_stage", "knowledge_query"}
    assert disabled.policy.workflow_enabled["group_analysis"] is False
    assert disabled.policy.is_enabled("consultation_stage") is True  # 缺省启用
    assert _chat_terminal_names(disabled) == []  # 禁用 → 不暴露终端工具

    enabled = build_runner(dict(_PROFILE))
    assert "analyze_group_messages" in _chat_terminal_names(enabled)  # 默认启用 → 注入
