"""群分析报告统计由代码渲染，模型摘要不能偷偷改写统计口径。"""
import pytest

from flowagent.core.contracts import EventKind, RunInput, Status
from flowagent.core.runner import Runner
from flowagent.workflows.group_analysis import FETCH_TOOL, build_group_analysis_workflow
from tests.native_helpers import FakeModel, FakeTools
from tests.test_native_workflows import (
    Q_3D, TARGET, _cls, _ctx, _group_inputs, _messages, _payload, _rep_text, _run_group,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _fixture():
    # 分母与类别数不同，避免“值存在就算正确”的错误校验。
    payload = _payload(_messages(("m1", "泵房故障"), ("m2", "阀门故障"), ("m3", "建议复查")))
    classify = _cls({"m1": "风险类", "m2": "风险类", "m3": "建议类"})
    return payload, classify


async def test_report_renders_statistics_and_never_publishes_model_statistics():
    payload, classify = _fixture()
    report = "## 总结\n关注设备故障。\n## 分类统计\n共有999条。\n## 汇报要点\n建议跟进。\n引用: [m1]"
    fixed = "## 总结\n关注设备故障。\n## 汇报要点\n建议跟进。\n引用: [m1]"
    result, model, _ = await _run_group([classify, _rep_text(report), _rep_text(fixed)],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.COMPLETED
    stats = result.output["analysis_result"]["statistics"]
    assert stats["total_messages"] == 3
    assert [(x["name"], x["count"], x["ratio"]) for x in stats["categories"]] == [
        ("风险类", 2, 66.7), ("建议类", 1, 33.3)]
    assert "999" not in result.reply
    assert "3 条" in result.reply and "风险类：2 条（66.7%）" in result.reply
    assert "建议类：1 条（33.3%）" in result.reply
    assert "关注设备故障" in result.reply and "引用: [m1]" in result.reply
    assert result.output["analysis_result"]["report_repair_attempted"] is True
    assert model.call_count == 3


@pytest.mark.parametrize("claim", ["共有999条", "总数1条", "总数三", "总数为三", "各类都是三", "共三条", "九百", "三十多个",
                                   "约有三百", "总数为九消息", "风险有七宗事故", "风险有一宗事件", "占一半", "风险三百九十九", "共三件故障", "至少三分之一",
                                   "风险占大半", "百分百事故", "风险排名首位", "风险比建议多", "占５０％", "占多数", "风险最多", "半数", "一半", "没有风险",
                                   "建议远超风险", "风险超过建议", "建议领先风险", "风险多过建议", "风险少过建议",
                                   "风险不存在", "风险并不存在", "风险未发现", "未出现风险", "没有发生风险",
                                   "Risks exceed suggestions", "Suggestions outnumber risks",
                                   "There were a dozen incidents", "There were forty incidents", "Twenty-five faults"])
async def test_invalid_model_summary_is_repaired_without_leaking_it(claim):
    payload, classify = _fixture()
    bad = f"## 总结\n{claim}。\n## 汇报要点\n需关注。\n引用: [m1]"
    good = "## 总结\n设备故障需跟进。\n## 汇报要点\n建议复查。\n引用: [m1]"
    result, model, _ = await _run_group([classify, _rep_text(bad), _rep_text(good)],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.COMPLETED
    assert result.output["analysis_result"]["report_repair_attempted"] is True
    assert claim not in result.reply and "设备故障需跟进" in result.reply
    assert model.call_count == 3


async def test_invalid_summary_twice_is_explicit_partial_without_bad_text():
    payload, classify = _fixture()
    bad = "## 总结\n共有999条。\n## 汇报要点\n需处理。\n引用: [m1]"
    result, model, _ = await _run_group([classify, _rep_text(bad), _rep_text(bad)],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.PARTIAL
    assert "report_summary_invalid" in result.output["analysis_result"]["partial_reasons"]
    assert "999" not in result.reply and "3 条" in result.reply
    assert model.call_count == 3


async def test_forged_reference_is_not_accepted_without_terminal_reference_line():
    payload, classify = _fixture()
    bad = "## 总结\n设备故障 [m999]。\n## 汇报要点\n需处理。"
    result, _, _ = await _run_group([classify, _rep_text(bad), _rep_text(bad)],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.PARTIAL
    assert "m999" not in result.reply


@pytest.mark.parametrize("forged", ["引用: 不存在的消息", "参见【虚构消息】", "【虚构消息】", "依据《虚构群消息》"])
async def test_forged_unbracketed_reference_label_is_not_accepted(forged):
    payload, classify = _fixture()
    bad = "## 总结\n设备故障。\n" + forged
    result, _, _ = await _run_group([classify, _rep_text(bad), _rep_text(bad)],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.PARTIAL
    assert "report_refs_invalid" in result.output["analysis_result"]["partial_reasons"]
    assert forged not in result.reply


@pytest.mark.parametrize("empty", ["## 总结\n引用: [m1]", "---\n引用: [m1]", "\u200b\u200c\n引用: [m1]", "<!-- comment -->\n引用: [m1]"])
async def test_empty_summary_is_partial_after_repair(empty):
    payload, classify = _fixture()
    result, _, _ = await _run_group([classify, _rep_text(empty), _rep_text(empty)],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.PARTIAL
    assert "report_summary_invalid" in result.output["analysis_result"]["partial_reasons"]


async def test_unknown_heading_alone_is_not_a_summary():
    payload, classify = _fixture()
    empty = "## 新标题\n引用: [m1]"
    result, _, _ = await _run_group([classify, _rep_text(empty), _rep_text(empty)],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.PARTIAL
    assert "report_summary_invalid" in result.output["analysis_result"]["partial_reasons"]


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls"])
async def test_nonfinal_report_is_repaired_and_never_published(finish):
    payload, classify = _fixture()
    bad = FakeModel.text_turn("未完成的摘要。\n引用: [m1]", finish_reason=finish)
    good = _rep_text("设备故障需跟进。\n引用: [m1]")
    result, model, _ = await _run_group([classify, bad, good],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.COMPLETED and model.call_count == 3
    assert "未完成的摘要" not in result.reply and "设备故障需跟进" in result.reply


async def test_truncated_report_twice_is_partial():
    payload, classify = _fixture()
    bad = FakeModel.text_turn("未完成的摘要。\n引用: [m1]", finish_reason="length")
    result, _, _ = await _run_group([classify, bad, bad],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.PARTIAL
    assert "report_summary_invalid" in result.output["analysis_result"]["partial_reasons"]
    assert "未完成的摘要" not in result.reply


@pytest.mark.parametrize("wording", ["一旦出现故障，一律及时检查，保持记录一致",
    "建议进一步检查", "问题一直存在", "现场一线员工反馈故障", "建议下一步跟进"])
async def test_summary_accepts_ordinary_chinese_phrases_and_digit_reference_ids(wording):
    payload, classify = _fixture()
    valid = f"## 总结\n{wording}。\n引用: [m1]"
    result, model, _ = await _run_group([classify, _rep_text(valid)],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.COMPLETED
    assert wording in result.reply
    assert result.output["analysis_result"]["report_refs"] == ["m1"]
    assert model.call_count == 2


async def test_stream_only_emits_validated_program_report():
    payload, classify = _fixture()
    bad = "## 总结\n共有999条。\n引用: [m1]"
    good = "## 总结\n设备故障需跟进。\n引用: [m1]"
    model = FakeModel([classify, _rep_text(bad), _rep_text(good)])
    tools = FakeTools(catalog=[{"name": FETCH_TOOL}], results={FETCH_TOOL: payload})
    ctx = _ctx(model, tools)
    runner = Runner(workflows={"group_analysis": build_group_analysis_workflow()},
                    context_factory=lambda: ctx, fallback_owner="caller")
    events = [e async for e in runner.stream(RunInput(workflow="group_analysis",
              inputs=_group_inputs(Q_3D, target=TARGET)))]
    text = "".join(e.data["text"] for e in events if e.kind is EventKind.TEXT_DELTA)
    finished = [e for e in events if e.kind is EventKind.RUN_FINISHED]
    assert len(finished) == 1 and finished[0].data["result"]["status"] == "completed"
    assert text == finished[0].data["result"]["reply"]
    assert "999" not in text and "3 条" in text and "设备故障需跟进" in text
