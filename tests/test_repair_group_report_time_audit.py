"""报告时间只能来自实际引用的消息，不能借机放开统计数字。"""
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


def _model(text):
    return _cls({"m1": "风险类", "m2": "建议类"}), _rep_text(text)


@pytest.mark.parametrize("time_text", ["下午三点", "下午3点", "15:00", "下午三点至下午五点", "下午三点二十分"])
async def test_cited_exact_time_is_allowed_and_statistics_still_program_generated(time_text):
    payload = _payload(_messages(("m1", f"计划{time_text}现场巡检"), ("m2", "建议复查")))
    classify, report = _model(f"计划{time_text}现场巡检。\n引用: [m1]")
    result, model, _ = await _run_group([classify, report, report], _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.COMPLETED and model.call_count == 2
    assert time_text in result.reply and "可信分类：2 条" in result.reply
    assert result.output["analysis_result"]["report_refs"] == ["m1"]


@pytest.mark.parametrize("source,summary,refs", [
    ("计划下午五点现场巡检", "计划下午三点现场巡检。", "m1"),
    ("计划下午三点现场巡检", "计划下午三点现场巡检。", "m2"),
    ("计划昨天下午三点现场巡检", "计划明天下午三点现场巡检。", "m1"),
    ("计划今天15:00现场巡检", "计划明天15:00现场巡检。", "m1"),
    ("计划下午三点半现场巡检", "计划下午三点现场巡检。", "m1"),
    ("计划下午三点以后现场巡检", "计划下午三点现场巡检。", "m1"),
    ("计划下午三点至下午五点现场巡检", "计划下午三点现场巡检。", "m1"),
    ("计划下午3点现场巡检", "计划下午３点现场巡检。", "m1"),
    ("计划15:00现场巡检", "计划１５：００现场巡检。", "m1"),
    ("下午三点、下午五点均有安排", "计划下午三点至下午五点现场巡检。", "m1"),
    ("计划下午三点现场巡检", "计划下午三点现场巡检。", ""),
    ("计划下午三点现场巡检", "计划下午三点共三条故障。", "m1"),
    ("计划下午三点现场巡检", "计划下午三点共999条故障。", "m1"),
    ("计划下午三点条记录", "计划下午三点条记录。", "m1"),
    ("计划下午三点半数风险", "计划下午三点半数风险。", "m1"),
])
async def test_uncited_unsourced_or_mixed_statistics_are_not_published(source, summary, refs):
    payload = _payload(_messages(("m1", source), ("m2", "建议复查")))
    classify, bad = _model(summary + (f"\n引用: [{refs}]" if refs else ""))
    result, model, _ = await _run_group([classify, bad, bad], _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.PARTIAL and model.call_count == 3
    assert "report_summary_invalid" in result.output["analysis_result"]["partial_reasons"]
    assert summary not in result.reply and "999" not in result.reply
    assert "可信分类：2 条" in result.reply


@pytest.mark.parametrize("source,summary", [
    ("计划下午三点巡检", "计划下午三点以前巡检。"),
    ("计划下午三点巡检", "计划下午三点以后巡检。"),
    ("计划下午三点巡检", "计划下午三点前巡检。"),
    ("计划下午三点巡检", "计划下午三点后巡检。"),
    ("计划下午三点巡检", "计划下午三点左右巡检。"),
    ("计划下午三点巡检", "计划不早于下午三点巡检。"),
    ("计划下午三点巡检", "计划不晚于下午三点巡检。"),
    ("计划15:00巡检", "计划15:00以后巡检。"),
    ("计划15:00巡检", "计划15:00之前巡检。"),
])
async def test_time_boundary_not_in_cited_source_is_rejected(source, summary):
    payload = _payload(_messages(("m1", source), ("m2", "建议复查")))
    classify, bad = _model(summary + "\n引用: [m1]")
    result, _, _ = await _run_group([classify, bad, bad], _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.PARTIAL
    assert summary not in result.reply


@pytest.mark.parametrize("source,summary", [
    ("计划下午三点以前巡检", "计划下午三点以前巡检。"),
    ("计划不早于下午三点巡检", "计划不早于下午三点巡检。"),
    ("计划今天15:00巡检", "计划今天15:00巡检。"),
    ("计划下午三点巡检", "计划下午三点开始检修。"),
])
async def test_cited_boundary_or_ordinary_action_is_allowed(source, summary):
    payload = _payload(_messages(("m1", source), ("m2", "建议复查")))
    classify, report = _model(summary + "\n引用: [m1]")
    result, model, _ = await _run_group([classify, report, report],
        _group_inputs(Q_3D, target=TARGET), payload=payload)
    assert result.status is Status.COMPLETED and model.call_count == 2
    assert summary in result.reply


async def test_stream_never_publishes_unsourced_time_before_repair():
    payload = _payload(_messages(("m1", "计划下午三点现场巡检"), ("m2", "建议复查")))
    classify, bad = _model("计划下午五点现场巡检。\n引用: [m1]")
    good = _rep_text("计划下午三点现场巡检。\n引用: [m1]")
    model = FakeModel([classify, bad, good])
    tools = FakeTools(catalog=[{"name": FETCH_TOOL}], results={FETCH_TOOL: payload})
    ctx = _ctx(model, tools)
    runner = Runner(workflows={"group_analysis": build_group_analysis_workflow()},
                    context_factory=lambda: ctx, fallback_owner="caller")
    events = [e async for e in runner.stream(RunInput(workflow="group_analysis",
              inputs=_group_inputs(Q_3D, target=TARGET)))]
    text = "".join(e.data["text"] for e in events if e.kind is EventKind.TEXT_DELTA)
    terminal = [e for e in events if e.kind is EventKind.RUN_FINISHED]
    assert len(terminal) == 1 and terminal[0].data["result"]["status"] == "completed"
    assert text == terminal[0].data["result"]["reply"]
    assert "下午五点" not in text and "下午三点" in text
