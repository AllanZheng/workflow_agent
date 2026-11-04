"""Edge UTC 时间前缀是来源元数据，不应被当作用户消息时间发布。"""
import pytest

from flowagent.core.contracts import EventKind, RunInput, Status
from flowagent.core.runner import Runner
from flowagent.workflows.group_analysis import FETCH_TOOL, build_group_analysis_workflow
from tests.native_helpers import FakeModel, FakeTools
from tests.test_native_workflows import Q_3D, TARGET, _cls, _ctx, _group_inputs, _payload, _rep_text, _run_group

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _source():
    return [
        {"message_ref": "m1", "timestamp": "2026-11-01T08:00:00+08:00",
         "text": "[11-01 00:00] 测试成员: 设备发生故障，需要处理"},
        {"message_ref": "m2", "timestamp": "2026-11-01T00:01:00+00:00",
         "text": "[11-01 00:01] 测试成员: 下午三点召开巡检会议"},
    ]


def _classification():
    return _cls({"m1": "风险类", "m2": "建议类"})


_QWEN_REPORT = (
    "事件摘要：\n在 [11-01 00:00]，测试成员反馈设备发生故障，需要处理。"
    "在 [11-01 00:01]，测试成员通知下午三点召开巡检会议。\n\n"
    "工作建议：\n建议针对设备故障尽快安排维修处理，保障设备正常运行。"
    "建议参会人员按时参加下午三点的巡检会议，共同商讨故障解决方案及后续巡检安排。"
    "\n\n引用：[m1, m2]"
)


async def _run(report, messages=None):
    payload = _payload(_source() if messages is None else messages)
    # 无效摘要保留一次重试数据，以便断言进入有标识的 partial 而非脚本耗尽。
    return await _run_group([_classification(), _rep_text(report), _rep_text(report)],
                            _group_inputs(Q_3D, target=TARGET), payload=payload)


async def test_real_qwen_summary_strips_verified_utc_metadata_preserving_event_time():
    result, model, _ = await _run(_QWEN_REPORT)
    assert result.status is Status.COMPLETED and model.call_count == 2
    assert "[11-01 00:00]" not in result.reply and "[11-01 00:01]" not in result.reply
    assert "测试成员反馈设备发生故障" in result.reply
    assert "参会人员按时参加下午三点" in result.reply
    assert "可信分类：2 条" in result.reply
    analysis = result.output["analysis_result"]
    assert analysis["report_refs"] == ["m1", "m2"]
    assert analysis["report_source_metadata_removed"] == 2
    assert "UTC 时间标签" in result.reply


@pytest.mark.parametrize("text", ["建议参会人员按时参加会议。", "建议相关人员参与处理。"])
async def test_ordinary_chinese_can_use_ambiguous_numeric_character(text):
    result, model, _ = await _run(text + "\n引用: [m1]")
    assert result.status is Status.COMPLETED and model.call_count == 2
    assert text in result.reply


@pytest.mark.parametrize("bad", [
    "在 [11-01 00:01]，测试成员通知巡检。\n引用: [m1]",
    "在 [11-01 00:02]，测试成员通知巡检。\n引用: [m1, m2]",
    "在 [[11-01 00:00]]，测试成员通知巡检。\n引用: [m1]",
    "在 [m999]，测试成员通知巡检。\n引用: [m1]",
    "在 [11-01 00:00]，总数999条。\n引用: [m1]",
    "在 [11-01 00:00]，共有三条。\n引用: [m1]",
    "在 [11-01 00:00]，在 [11-01 00:00]，设备故障。\n引用: [m1]",
    "在 [11-01 00:00]，\n引用: [m1]",
    "设备在 00:00 故障。\n引用: [m1]",
    "参条消息需关注。\n引用: [m1]",
    "参佰消息需关注。\n引用: [m1]",
    "参会人员讨论三条故障。\n引用: [m1]",
])
async def test_invalid_citations_metadata_or_counts_never_become_complete(bad):
    result, model, _ = await _run(bad)
    assert result.status is Status.PARTIAL and model.call_count == 3
    assert bad not in result.reply and "999" not in result.reply
    assert result.output["analysis_result"].get("report_source_metadata_removed", 0) == 0


@pytest.mark.parametrize("broken", [
    {"message_ref": "m1", "text": "[11-01 00:00] 测试成员: 设备发生故障"},
    {"message_ref": "m1", "timestamp": "2026-11-01T00:00:00",
     "text": "[11-01 00:00] 测试成员: 设备发生故障"},
    {"message_ref": "m1", "timestamp": "2026-11-01T00:01:00+00:00",
     "text": "[11-01 00:00] 测试成员: 设备发生故障"},
    {"message_ref": "m1", "timestamp": "2026-11-01T00:00:00+00:00",
     "text": "用户内容中有 [11-01 00:00] 但不是元数据前缀"},
])
async def test_metadata_cannot_be_removed_without_validated_utc_prefix(broken):
    result, model, _ = await _run("在 [11-01 00:00]，设备故障。\n引用: [m1]", [broken, _source()[1]])
    assert result.status is Status.PARTIAL and model.call_count == 3
    assert "[11-01 00:00]" not in result.reply


async def test_noncanonical_bracketed_source_time_is_not_a_bare_business_time():
    broken = {"message_ref": "m1", "text": "[11-01 00:00]设备故障"}
    result, model, _ = await _run("设备在 00:00 故障。\n引用: [m1]", [broken, _source()[1]])
    assert result.status is Status.PARTIAL and model.call_count == 3
    assert "设备在 00:00 故障" not in result.reply


async def test_qwen_summary_stream_publishes_only_cleaned_final_reply():
    payload = _payload(_source())
    model = FakeModel([_classification(), _rep_text(_QWEN_REPORT)])
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
    assert "[11-01 00:00]" not in text and "[11-01 00:01]" not in text
    assert "参会人员按时参加下午三点" in text and "UTC 时间标签" in text
