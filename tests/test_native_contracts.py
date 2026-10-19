"""原生内核契约（C01）：数据对象、公开信封、澄清标记与枚举完整性。

覆盖行为 ID：
- C01: RunInput / RunResult / RunEvent / Clarification 数据对象与公开信封校验；
  RunResult.to_public_dict() 全量形状（error 为 None 序列化为 null）；
  Clarification.render_body() 与机器标记的渲染/解析/剥离往返；
  build_continuation / merge_continuation 的合并顺序与计数保护；
  ErrorCode（20 个）/ Status（6 个）/ EventKind（6 个）取值完整性；
  WorkflowSpec / ToolResultData（四态）构造。
"""
from __future__ import annotations

import json

import pytest

from flowagent.core.contracts import (
    Clarification,
    ErrorCode,
    EventKind,
    FlowAgentCoreError,
    RunEvent,
    RunInput,
    RunResult,
    Status,
    ToolCallSpec,
    ToolResultData,
    Usage,
    WorkflowSpec,
    build_continuation,
    merge_continuation,
    parse_clarification_marker,
    render_clarification_marker,
    strip_clarification_marker,
)

MARKER_PREFIX = "sample-agent:clarification:v1"


# ---------------------------------------------------------------- 枚举完整性


def test_status_enum_complete() -> None:
    """C01：6 个执行终态全部存在且取值稳定。"""
    expected = {
        "COMPLETED": "completed",
        "NEEDS_INPUT": "needs_input",
        "PARTIAL": "partial",
        "DEGRADED": "degraded",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
    }
    assert {member.name: member.value for member in Status} == expected
    assert len(Status) == 6
    # 字符串枚举可直接按取值反查，适配层无需额外映射表
    assert Status("completed") is Status.COMPLETED


def test_error_code_enum_complete() -> None:
    """C01：20 个稳定错误码全集存在，且 name 与 value 同名。"""
    expected = {
        "INVALID_INPUT",
        "CAPABILITY_UNAVAILABLE",
        "MODEL_TIMEOUT",
        "MODEL_UPSTREAM_ERROR",
        "MODEL_INVALID_RESPONSE",
        "TOOL_ARGUMENT_INVALID",
        "MCP_TOOL_ERROR",
        "SKILL_LOAD_FAILED",
        "SKILL_BINDING_MISMATCH",
        "MODEL_BUDGET_EXCEEDED",
        "TOOL_BUDGET_EXCEEDED",
        "CONTEXT_LIMIT_EXCEEDED",
        "DUPLICATE_TOOL_CALL",
        "CLARIFICATION_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "OUTPUT_INVALID",
        "INTERNAL_ERROR",
        "FALLBACK_FAILED",
        # 稳定终止码单独两类，不伪装成依赖错误
        "AUTHORIZATION_DENIED",
        "CANCELLED_BY_USER",
    }
    actual = {member.value for member in ErrorCode}
    assert actual == expected
    assert len(ErrorCode) == 20
    # 每个成员的 name 与 value 一致，日志/信封无需两套拼写
    assert {member.name for member in ErrorCode} == expected
    assert ErrorCode("MODEL_TIMEOUT") is ErrorCode.MODEL_TIMEOUT


def test_event_kind_values() -> None:
    """C01：原生事件 6 种 kind 的线上取值固定。"""
    expected = {
        "RUN_STARTED": "run.started",
        "STEP_STARTED": "step.started",
        "STEP_FINISHED": "step.finished",
        "TOOL_FINISHED": "tool.finished",
        "TEXT_DELTA": "text.delta",
        "RUN_FINISHED": "run.finished",
    }
    assert {member.name: member.value for member in EventKind} == expected
    assert len(EventKind) == 6


# ---------------------------------------------------------------- 数据对象默认值


def test_run_input_defaults_are_isolated() -> None:
    """C01：RunInput 默认容器互不共享（信封解析不会串请求）。"""
    first = RunInput(workflow="chat")
    second = RunInput(workflow="leave")
    assert first.inputs == {} and first.history == []
    assert first.correlation_id is None and first.continuation is None

    # 修改第一个实例的默认容器，不得影响第二个实例
    first.inputs["message"] = "你好"
    first.history.append({"role": "user", "content": "你好"})
    assert second.inputs == {}
    assert second.history == []


def test_run_result_defaults() -> None:
    """C01：RunResult 未显式赋值的字段遵循空值契约。"""
    result = RunResult(run_id="r-1", status=Status.COMPLETED)
    assert result.reply == ""
    assert result.output is None
    assert result.citations == []
    assert result.usage == Usage()  # 零消耗的 Usage 默认对象
    assert result.error is None
    assert result.execution_path is None
    assert result.recovery == {}
    assert result.clarification is None


def test_clarification_defaults_are_isolated() -> None:
    """C01：Clarification 的字典/列表默认值按实例隔离。"""
    first = Clarification(reason="missing_fields", missing_fields=["city"])
    second = Clarification(reason="missing_fields", missing_fields=["city"])
    first.confirmed_fields["day"] = "明天"
    first.hints.append("请使用 24 小时制")
    first.continuation["clarify_count"] = 1
    assert second.confirmed_fields == {}
    assert second.hints == []
    assert second.continuation == {}


def test_run_event_data_default_isolated() -> None:
    """C01：RunEvent.data 默认空字典按实例隔离。"""
    first = RunEvent(run_id="r", seq=0, kind=EventKind.RUN_STARTED, step=None)
    second = RunEvent(run_id="r", seq=1, kind=EventKind.STEP_STARTED, step="s1")
    assert first.data == {}
    first.data["k"] = "v"
    assert second.data == {}
    assert second.step == "s1"


# ---------------------------------------------------------------- 公开信封


def _sample_clarification() -> Clarification:
    """构造带完整 continuation 的澄清载荷（render/信封用例共用）。"""
    continuation = build_continuation(
        workflow="leave",
        original_request="我明天想请假",
        reference_time="2026-10-21 09:00",
        timezone="Asia/Shanghai",
        resolved_fields={"city": "成都"},
        clarify_count=2,
    )
    return Clarification(
        reason="missing_fields",
        missing_fields=["time_range", "duration"],
        confirmed_fields={"city": "成都"},
        hints=["时间请使用 24 小时制"],
        continuation=continuation,
    )


def test_to_public_dict_full_shape() -> None:
    """C01：to_public_dict 输出完整公开信封，usage 平铺、error 拆 code/message。"""
    clarification = _sample_clarification()
    result = RunResult(
        run_id="run-42",
        status=Status.NEEDS_INPUT,
        reply="还需要补充请假时间。",
        output={"draft": True},
        citations=[{"source": "kb", "title": "请假制度"}],
        usage=Usage(
            model_requests=2,
            model_retries=1,
            mcp_attempts=3,
            skill_loads=1,
            tokens_in=120,
            tokens_out=None,  # token 不可得时为 null
            elapsed_ms=45.5,
        ),
        error=FlowAgentCoreError(ErrorCode.INVALID_INPUT, "缺少时间范围"),
        execution_path="workflow",
        recovery={"fallback_attempted": False},
        clarification=clarification,
    )

    public = result.to_public_dict()
    # 信封顶层键必须与文档完全一致，不多不少（防止内部字段泄漏）
    assert set(public) == {
        "run_id",
        "status",
        "reply",
        "output",
        "citations",
        "usage",
        "error",
        "execution_path",
        "recovery",
        "clarification",
    }
    assert public == {
        "run_id": "run-42",
        "status": "needs_input",  # 枚举序列化为取值字符串
        "reply": "还需要补充请假时间。",
        "output": {"draft": True},
        "citations": [{"source": "kb", "title": "请假制度"}],
        "usage": {
            "model_requests": 2,
            "model_retries": 1,
            "mcp_attempts": 3,
            "skill_loads": 1,
            "tokens_in": 120,
            "tokens_out": None,
            "elapsed_ms": 45.5,
        },
        "error": {"code": "INVALID_INPUT", "message": "缺少时间范围"},
        "execution_path": "workflow",
        "recovery": {"fallback_attempted": False},
        "clarification": {
            "reason": "missing_fields",
            "missing_fields": ["time_range", "duration"],
            "confirmed_fields": {"city": "成都"},
            "hints": ["时间请使用 24 小时制"],
            "continuation": clarification.continuation,
        },
    }
    # HTTP 适配层必须能直接 json.dumps 该信封（中文不转义也可行）
    encoded = json.dumps(public, ensure_ascii=False)
    assert "还需要补充请假时间。" in encoded


def test_to_public_dict_none_error_and_clarification_serialize_null() -> None:
    """C01：error / clarification 为 None 时信封序列化为 null。"""
    result = RunResult(run_id="run-ok", status=Status.COMPLETED, reply="完成")
    public = result.to_public_dict()
    assert public["error"] is None
    assert public["clarification"] is None
    assert public["output"] is None
    assert public["execution_path"] is None
    encoded = json.dumps(public, ensure_ascii=False)
    assert '"error": null' in encoded
    assert '"clarification": null' in encoded


# ---------------------------------------------------------------- 澄清模板与机器标记


def test_clarification_render_body_contains_fields_and_marker() -> None:
    """C01：澄清正文含缺失字段、已确认条件、提示与版本化机器标记。"""
    clarification = _sample_clarification()
    body = clarification.render_body()

    # 缺失字段按顿号连接，出现在首行
    assert "还需要补充以下信息才能继续：time_range、duration。" in body
    # 已确认字段逐行展示
    assert "已确认条件：" in body
    assert "- city：成都" in body
    # 补充提示逐行展示
    assert "补充提示：" in body
    assert "- 时间请使用 24 小时制" in body
    # 尾部携带机器标记，展示端可据此识别续接
    assert MARKER_PREFIX in body
    assert body.rstrip().endswith("-->")


def test_render_body_omits_empty_sections() -> None:
    """C01：没有已确认字段/提示时正文不输出对应小节。"""
    clarification = Clarification(reason="missing_fields", missing_fields=["time_range"])
    body = clarification.render_body()
    assert "还需要补充以下信息才能继续：time_range。" in body
    assert "已确认条件" not in body
    assert "补充提示" not in body
    # 即使 continuation 为空也渲染 v1 标记，保证可解析
    assert parse_clarification_marker(body) == {
        "version": 1,
        "workflow": None,
        "reference_time": None,
        "timezone": None,
        "clarify_count": 0,
    }


def test_marker_roundtrip_restores_continuation_fields() -> None:
    """C01：render → parse 往返恢复 workflow/reference_time/timezone/clarify_count。"""
    continuation = build_continuation(
        workflow="leave",
        original_request="我明天想请假",
        reference_time="2026-10-21 09:00",
        timezone="Asia/Shanghai",
        resolved_fields={"city": "成都"},
        clarify_count=3,
    )
    parsed = parse_clarification_marker(render_clarification_marker(continuation))
    assert parsed == {
        "version": 1,
        "workflow": "leave",
        "reference_time": "2026-10-21 09:00",
        "timezone": "Asia/Shanghai",
        "clarify_count": 3,
    }


def test_parse_marker_works_on_any_text() -> None:
    """C01：用户把标记连同补充文字一起发回时，parse 仍能定位并解析。"""
    continuation = build_continuation(
        workflow="leave",
        original_request="我明天想请假",
        reference_time="2026-10-21 09:00",
        timezone="Asia/Shanghai",
        resolved_fields={},
        clarify_count=1,
    )
    user_text = (
        "补充：时间是上午 9 点到 11 点。\n"
        + render_clarification_marker(continuation)
        + "\n以上，谢谢。"
    )
    parsed = parse_clarification_marker(user_text)
    assert parsed is not None
    assert parsed["workflow"] == "leave"
    assert parsed["clarify_count"] == 1
    # 正文前后内容不影响标记识别
    assert "补充：时间是上午 9 点到 11 点。" in user_text


def test_parse_marker_rejects_unknown_version_and_invalid_json() -> None:
    """C01：版本不是 1、JSON 非法、缺少标记时均返回 None（视为新请求）。"""
    version_two = (
        f"<!-- {MARKER_PREFIX} "
        + json.dumps({"version": 2, "workflow": "leave", "clarify_count": 1})
        + " -->"
    )
    assert parse_clarification_marker(version_two) is None

    # 缺少 version 字段同样不识别
    missing_version = f"<!-- {MARKER_PREFIX} " + json.dumps({"workflow": "leave"}) + " -->"
    assert parse_clarification_marker(missing_version) is None

    # 标记内不是合法 JSON
    invalid_json = f"<!-- {MARKER_PREFIX} {{not-json}} -->"
    assert parse_clarification_marker(invalid_json) is None

    # 完全没有标记
    assert parse_clarification_marker("普通用户消息，没有任何标记") is None


def test_strip_clarification_marker_removes_marker_only() -> None:
    """C01：strip 只移除机器标记并保留面向用户的正文。"""
    clarification = _sample_clarification()
    body = clarification.render_body()
    stripped = strip_clarification_marker(body)

    assert MARKER_PREFIX not in stripped
    assert "<!--" not in stripped and "-->" not in stripped
    assert stripped == (
        "还需要补充以下信息才能继续：time_range、duration。\n"
        "已确认条件：\n"
        "- city：成都\n"
        "补充提示：\n"
        "- 时间请使用 24 小时制"
    )
    # 剥离后仍可解析原始正文（存储保留原文，提示注入前才剥离）
    assert parse_clarification_marker(body) is not None


def test_strip_without_marker_returns_trimmed_text() -> None:
    """C01：无标记文本 strip 后仅去除首尾空白，不改动正文。"""
    assert strip_clarification_marker("  你好，世界  ") == "你好，世界"


# ---------------------------------------------------------------- continuation 合并


def test_build_continuation_shape() -> None:
    """C01：build_continuation 产出 version=1 的完整 7 字段结构。"""
    resolved = {"city": "成都"}
    continuation = build_continuation(
        workflow="leave",
        original_request="我明天想请假",
        reference_time="2026-10-21 09:00",
        timezone="Asia/Shanghai",
        resolved_fields=resolved,
        clarify_count=1,
    )
    assert set(continuation) == {
        "version",
        "workflow",
        "original_request",
        "reference_time",
        "timezone",
        "resolved_fields",
        "clarify_count",
    }
    assert continuation["version"] == 1
    assert continuation["workflow"] == "leave"
    assert continuation["original_request"] == "我明天想请假"
    assert continuation["reference_time"] == "2026-10-21 09:00"
    assert continuation["timezone"] == "Asia/Shanghai"
    assert continuation["resolved_fields"] == {"city": "成都"}
    assert continuation["clarify_count"] == 1


def test_merge_continuation_supplement_overrides_field_by_field() -> None:
    """C01：合并顺序为 原始 → 已确认 → 本次补充逐字段覆盖。"""
    original = build_continuation(
        workflow="leave",
        original_request="我明天想请假",
        reference_time="2026-10-21 09:00",
        timezone="Asia/Shanghai",
        resolved_fields={"city": "成都", "day": "明天"},
        clarify_count=1,
    )
    merged = merge_continuation(original, {"day": "后天", "time_range": "上午"})

    # 未覆盖字段保留，覆盖字段取本次补充，新字段追加
    assert merged["resolved_fields"] == {
        "city": "成都",
        "day": "后天",
        "time_range": "上午",
    }
    # 原始请求上下文不变
    assert merged["version"] == 1
    assert merged["workflow"] == "leave"
    assert merged["original_request"] == "我明天想请假"
    assert merged["reference_time"] == "2026-10-21 09:00"
    assert merged["timezone"] == "Asia/Shanghai"
    # 合并不得就地修改传入的 continuation
    assert original["resolved_fields"] == {"city": "成都", "day": "明天"}


def test_merge_continuation_without_base() -> None:
    """C01：无既有 continuation 时，补充字段成为唯一已确认字段。"""
    merged = merge_continuation(None, {"city": "成都"})
    assert merged == {"resolved_fields": {"city": "成都"}}


def test_merge_continuation_clarify_count_not_overridable() -> None:
    """C01：clarify_count 由内核维护，用户补充不能覆盖顶层计数。"""
    original = build_continuation(
        workflow="leave",
        original_request="我明天想请假",
        reference_time="2026-10-21 09:00",
        timezone="Asia/Shanghai",
        resolved_fields={},
        clarify_count=2,
    )
    merged = merge_continuation(original, {"clarify_count": 99})
    # 顶层计数保持 continuation 中的 2，补充值只落在 resolved_fields 内
    assert merged["clarify_count"] == 2
    assert merged["resolved_fields"]["clarify_count"] == 99


# ---------------------------------------------------------------- WorkflowSpec / 工具四态


def test_workflow_spec_construction() -> None:
    """C01：WorkflowSpec 是普通函数注册项，preflight/validate 可省略。"""

    async def run(inputs, ctx):  # pragma: no cover - 仅校验可调用对象挂载
        return RunResult(run_id="r", status=Status.COMPLETED)

    async def preflight(inputs, ctx):  # pragma: no cover
        return None

    def validate(result, ctx):  # pragma: no cover
        return True

    bare = WorkflowSpec(name="chat", run=run)
    assert bare.name == "chat"
    assert bare.run is run
    assert bare.preflight is None
    assert bare.validate is None

    full = WorkflowSpec(name="leave", run=run, preflight=preflight, validate=validate)
    assert full.preflight is preflight
    assert full.validate is validate


def test_tool_result_data_four_states() -> None:
    """C01：ToolResultData 四态构造；terminal_result 携带原始 RunResult。"""
    success = ToolResultData.success({"ok": True})
    assert success.status == "success"
    assert success.value == {"ok": True}
    assert success.error_code is None and success.message is None and success.terminal is None

    error = ToolResultData.error("工具调用失败")
    assert error.status == "error"
    assert error.message == "工具调用失败"
    assert error.error_code == "MCP_TOOL_ERROR"  # 默认错误码

    custom = ToolResultData.error("参数非法", code="TOOL_ARGUMENT_INVALID")
    assert custom.error_code == "TOOL_ARGUMENT_INVALID"

    unknown = ToolResultData.unknown("结果未知")
    assert unknown.status == "unknown"
    assert unknown.message == "结果未知"

    result = RunResult(run_id="r-term", status=Status.NEEDS_INPUT, reply="请补充时间")
    terminal = ToolResultData.terminal_result(result)
    assert terminal.status == "terminal"
    assert terminal.terminal is result  # 必须携带同一 RunResult 对象

    # ToolCallSpec 目录项默认值
    spec = ToolCallSpec(name="echo")
    assert spec.description == ""
    assert spec.input_schema == {}


def test_core_error_coerces_code_and_preserves_cause() -> None:
    """C01：FlowAgentCoreError 接受字符串码并保留 cause（错误信封来源）。"""
    cause = ValueError("boom")
    error = FlowAgentCoreError("INVALID_INPUT", "输入非法", cause=cause)
    assert error.code is ErrorCode.INVALID_INPUT
    assert str(error) == "输入非法"
    assert error.__cause__ is cause


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
