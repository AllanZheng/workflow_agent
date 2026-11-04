"""原生内核兜底行为测试（F01–F05）。

覆盖测试方案行为编号：
- F01 错误来源矩阵：主阶段任一错误在本地所有者下恰好一次无工具直连；
  调用方所有者下零本地直连、仅交还主错误。
- F02 兜底成功 => DEGRADED（无业务成功输出）；兜底再失败 => FALLBACK_FAILED，
  recovery 同时保留 primary_error 与 fallback_error。
- F03 未知写结果不重放：兜底阶段不再调用工具，兜底请求不带工具。
- F04 取消 / 授权拒绝 / 总时限耗尽不进入兜底；兜底窗口边界（85s 可兜底，
  超过 90s 不再兜底）。
- F05 澄清耗尽 => 本地仅解释型兜底，output == {"data_unavailable": True}；
  调用方所有者下 FAILED 且零本地请求。

全部用例离线：FakeModel / FakeTools / ScriptedClock，无网络、无真实等待。
"""
from __future__ import annotations

import pytest

from flowagent.core.contracts import (
    ErrorCode,
    FlowAgentCoreError,
    RunInput,
    RunResult,
    Status,
    ToolResultData,
    WorkflowSpec,
)
from flowagent.core.runner import Runner
from flowagent.workflows.chat import build_chat_workflow
from tests.native_helpers import FakeModel, FakeTools, ScriptedClock, make_context

pytestmark = pytest.mark.anyio


# 处理 anyio backend 相关逻辑：与套件其他原生测试一致，固定 asyncio。
@pytest.fixture
def anyio_backend():
    return "asyncio"


QUESTION = "请给出本季度设备巡检的结论"
FALLBACK_TEXT = "原任务未能完成：执行中出现错误，请稍后重试或补充信息。"

ORIGINS = [
    "invalid_input", "mcp_raise", "model_upstream", "validate_false", "unknown_exception",
]


# ---------------------------------------------------------------- 公共装配


def _runner(workflows, *, owner: str, model, tools=None) -> Runner:
    """组装 Runner：每次 execute 新建上下文（默认预算 + 可控时钟）。"""
    return Runner(
        workflows=workflows,
        context_factory=lambda: make_context(model=model, tools=tools),
        fallback_owner=owner,
    )


def _scripted_runner(workflows, *, owner: str, model, clock: ScriptedClock, tools=None) -> Runner:
    """带共享 ScriptedClock 的 Runner：工作流可推进时钟模拟耗时。"""
    return Runner(
        workflows=workflows,
        context_factory=lambda: make_context(model=model, tools=tools, clock=clock),
        fallback_owner=owner,
    )


def _scenario(origin: str) -> dict:
    """按错误来源构造独立场景；每次调用全新实例，避免脚本串台。

    main_calls 为主阶段预计模型请求数（不含兜底请求），用于断言
    “模型总请求数 == 主阶段尝试数 + 1”。
    """
    if origin == "invalid_input":
        # chat 缺 message：工作流在发起任何模型请求前抛 INVALID_INPUT。
        return {
            "workflows": {"chat": build_chat_workflow()},
            "workflow": "chat",
            "inputs": {"message": ""},
            "model": FakeModel(turns=[FakeModel.text_turn(FALLBACK_TEXT)]),
            "tools": None,
            "primary": ErrorCode.INVALID_INPUT,
            "main_calls": 0,
        }
    if origin == "mcp_raise":
        # MCP 工具脚本抛错（不可恢复码，主阶段只发一次工具调用轮）。
        return {
            "workflows": {"chat": build_chat_workflow()},
            "workflow": "chat",
            "inputs": {"message": QUESTION},
            "model": FakeModel(turns=[
                FakeModel.tool_turn({"id": "c1", "name": "echo", "arguments": {"q": 1}}),
                FakeModel.text_turn(FALLBACK_TEXT),
            ]),
            "tools": FakeTools(results={
                "echo": FlowAgentCoreError(ErrorCode.MCP_TOOL_ERROR, "mcp boom")}),
            "primary": ErrorCode.MCP_TOOL_ERROR,
            "main_calls": 1,
        }
    if origin == "model_upstream":
        # 模型上游错误属可恢复码：单轮恢复重发一次，第二次仍失败才终止主阶段。
        return {
            "workflows": {"chat": build_chat_workflow()},
            "workflow": "chat",
            "inputs": {"message": QUESTION},
            "model": FakeModel(turns=[
                FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "upstream down"),
                FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "upstream down"),
                FakeModel.text_turn(FALLBACK_TEXT),
            ]),
            "tools": None,
            "primary": ErrorCode.MODEL_UPSTREAM_ERROR,
            "main_calls": 2,
        }
    if origin == "validate_false":
        # 工作流产出业务结果但校验失败：Runner 转成 OUTPUT_INVALID。
        async def run(inputs, ctx):
            return RunResult(run_id="", status=Status.COMPLETED, reply="业务结果")

        return {
            "workflows": {"biz": WorkflowSpec(
                name="biz", run=run, validate=lambda result, ctx: False)},
            "workflow": "biz",
            "inputs": {"message": QUESTION},
            "model": FakeModel(turns=[FakeModel.text_turn(FALLBACK_TEXT)]),
            "tools": None,
            "primary": ErrorCode.OUTPUT_INVALID,
            "main_calls": 0,
        }
    if origin == "unknown_exception":
        # 未知异常统一包装为 INTERNAL_ERROR（保留异常类型与消息）。
        async def run(inputs, ctx):
            raise ValueError("boom")

        return {
            "workflows": {"biz": WorkflowSpec(name="biz", run=run)},
            "workflow": "biz",
            "inputs": {"message": QUESTION},
            "model": FakeModel(turns=[FakeModel.text_turn(FALLBACK_TEXT)]),
            "tools": None,
            "primary": ErrorCode.INTERNAL_ERROR,
            "main_calls": 0,
        }
    raise AssertionError(f"unknown origin: {origin}")


def _prompt_text(request: dict) -> str:
    """兜底请求的全部消息正文（用于检查原问题与提示词口径）。"""
    return "\n".join(str(item.get("content", "")) for item in request["messages"])


# ---------------------------------------------------------------- F01


@pytest.mark.parametrize("origin", ORIGINS)
async def test_f01_local_owner_one_tool_free_fallback(origin):
    """F01（本地所有者）：任一错误来源 -> DEGRADED，且恰好一次无工具直连。"""
    case = _scenario(origin)
    runner = _runner(case["workflows"], owner="local",
                     model=case["model"], tools=case["tools"])
    result = await runner.run(RunInput(workflow=case["workflow"], inputs=case["inputs"]))

    assert result.status is Status.DEGRADED
    assert result.execution_path == "direct_llm"
    assert result.reply == FALLBACK_TEXT
    # 兜底是通用说明，不得伪装成业务成功输出。
    assert result.output is None
    assert result.error is not None and result.error.code is case["primary"]

    requests = case["model"].requests
    # 主阶段尝试数 + 恰好一次兜底请求。
    assert len(requests) == case["main_calls"] + 1
    fallback_request = requests[-1]
    # 兜底请求不带任何工具。
    assert fallback_request["tools"] == []
    prompt = _prompt_text(fallback_request)
    if case["inputs"].get("message"):
        assert QUESTION in prompt
    else:
        # 无原问题时提示词仍以“原始请求：”模板开场。
        assert "原始请求：" in prompt

    recovery = result.recovery
    assert recovery["requested_workflow"] == case["workflow"]
    assert recovery["route_reason"] == "registered"
    assert recovery["primary_error"]["code"] == case["primary"].value
    assert recovery["fallback_attempted"] is True
    assert recovery["fallback_error"] is None
    assert recovery["final_source"] == "direct_llm"


@pytest.mark.parametrize("origin", ORIGINS)
async def test_f01_caller_owner_zero_local_calls(origin):
    """F01（调用方所有者）：FAILED 交还主错误，本地零直连。"""
    case = _scenario(origin)
    runner = _runner(case["workflows"], owner="caller",
                     model=case["model"], tools=case["tools"])
    result = await runner.run(RunInput(workflow=case["workflow"], inputs=case["inputs"]))

    assert result.status is Status.FAILED
    assert result.error is not None and result.error.code is case["primary"]
    # 零额外模型调用：主阶段用掉几次就是几次。
    assert len(case["model"].requests) == case["main_calls"]
    assert result.recovery["fallback_attempted"] is False
    assert result.recovery["fallback_error"] is None
    assert result.recovery["final_source"] == "caller"


# ---------------------------------------------------------------- F02


async def test_f02_fallback_success_degrades_without_business_output():
    """F02：兜底成功 -> DEGRADED，reply 为兜底正文，output 为 None，recovery 完整。"""
    case = _scenario("unknown_exception")
    runner = _runner(case["workflows"], owner="local", model=case["model"])
    result = await runner.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result.status is Status.DEGRADED
    assert result.reply == FALLBACK_TEXT
    assert result.output is None
    # 主错误保留（未知异常被包装为 INTERNAL_ERROR，消息保留原异常类型）。
    assert result.error.code is ErrorCode.INTERNAL_ERROR
    recovery = result.recovery
    assert recovery["primary_error"] == {
        "code": "INTERNAL_ERROR", "message": "ValueError: boom"}
    assert recovery["fallback_attempted"] is True
    assert recovery["fallback_error"] is None
    assert recovery["final_source"] == "direct_llm"
    assert recovery["requested_workflow"] == "biz"
    assert len(case["model"].requests) == 1


async def test_f02_fallback_failure_keeps_both_errors():
    """F02：兜底再次失败 -> FAILED / FALLBACK_FAILED，两侧错误都保留。"""
    async def run(inputs, ctx):
        raise FlowAgentCoreError(ErrorCode.INTERNAL_ERROR, "primary boom")

    model = FakeModel(turns=[
        FlowAgentCoreError(ErrorCode.MODEL_TIMEOUT, "fallback down")])
    runner = _runner({"biz": WorkflowSpec(name="biz", run=run)},
                     owner="local", model=model)
    result = await runner.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result.status is Status.FAILED
    assert result.error is not None
    assert result.error.code is ErrorCode.FALLBACK_FAILED
    assert "INTERNAL_ERROR" in str(result.error) and "MODEL_TIMEOUT" in str(result.error)
    recovery = result.recovery
    assert recovery["primary_error"]["code"] == "INTERNAL_ERROR"
    assert recovery["primary_error"]["message"] == "primary boom"
    assert recovery["fallback_error"]["code"] == "MODEL_TIMEOUT"
    assert recovery["fallback_error"]["message"] == "fallback down"
    assert recovery["fallback_attempted"] is True
    assert recovery["final_source"] == "fallback_failed"
    assert len(model.requests) == 1


# ---------------------------------------------------------------- F03


async def test_f03_unknown_write_result_not_replayed():
    """F03：工具返回 unknown 后主阶段失败，兜底不重放任何工具调用。"""
    tools = FakeTools(
        catalog=[{"name": "write_note", "description": "写入",
                  "input_schema": {"type": "object"}}],
        results={"write_note": ToolResultData.unknown("写入结果不明")},
    )

    async def run(inputs, ctx):
        # 写操作结果未知：不得重放，直接终止主阶段进入兜底。
        tool_result = await ctx.tools.call("write_note", {"note": "x"})
        ctx.record_tool("write_note", {"note": "x"}, tool_result)
        raise FlowAgentCoreError(ErrorCode.MCP_TOOL_ERROR, "写结果不明，终止主阶段")

    model = FakeModel(turns=[FakeModel.text_turn(FALLBACK_TEXT)])
    runner = _runner({"biz": WorkflowSpec(name="biz", run=run)},
                     owner="local", model=model, tools=tools)
    result = await runner.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result.status is Status.DEGRADED
    assert result.recovery["primary_error"]["code"] == "MCP_TOOL_ERROR"
    # 兜底期间 FakeTools.calls 不增长（主阶段仅一次写入尝试）。
    assert tools.calls == [("write_note", {"note": "x"})]
    # 恰好一次兜底请求，且不带工具。
    assert len(model.requests) == 1
    assert model.requests[0]["tools"] == []


# ---------------------------------------------------------------- F04


async def test_f04_cancel_before_fallback():
    """F04：工作流内 ctx.cancel() 后失败 -> CANCELLED，零兜底模型调用。"""
    async def run(inputs, ctx):
        ctx.cancel()  # 用户取消标记
        raise FlowAgentCoreError(ErrorCode.INTERNAL_ERROR, "取消后收尾失败")

    model = FakeModel(turns=[FakeModel.text_turn("兜底不应出现")])
    runner = _runner({"biz": WorkflowSpec(name="biz", run=run)},
                     owner="local", model=model)
    result = await runner.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result.status is Status.CANCELLED
    assert result.error is not None
    assert result.error.code is ErrorCode.CANCELLED_BY_USER
    # 取消后不得发起新的模型调用（含兜底）。
    assert model.requests == []
    assert result.recovery["fallback_attempted"] is False


async def test_f04_authorization_denied_no_fallback():
    """F04：授权/归属拒绝 -> FAILED，零兜底模型调用。"""
    async def run(inputs, ctx):
        raise FlowAgentCoreError(ErrorCode.AUTHORIZATION_DENIED, "identity mismatch")

    model = FakeModel(turns=[FakeModel.text_turn("兜底不应出现")])
    runner = _runner({"biz": WorkflowSpec(name="biz", run=run)},
                     owner="local", model=model)
    result = await runner.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result.status is Status.FAILED
    assert result.error is not None
    assert result.error.code is ErrorCode.AUTHORIZATION_DENIED
    assert model.requests == []
    assert result.recovery["fallback_attempted"] is False
    assert result.recovery["final_source"] == "error"


async def test_f04_fallback_window_boundary():
    """F04：85s 时总剩余 5s -> 允许一次兜底；超过 90s -> 不再兜底。"""
    # 85s：窗口 = min(30s, 90s-85s) = 5s > 0，兜底仍可发起。
    clock = ScriptedClock()

    async def run_late(inputs, ctx):
        clock.advance_ms(85_000)
        raise FlowAgentCoreError(ErrorCode.INTERNAL_ERROR, "85 秒时失败")

    model = FakeModel(turns=[FakeModel.text_turn(FALLBACK_TEXT)])
    runner = _scripted_runner(
        {"biz": WorkflowSpec(name="biz", run=run_late)},
        owner="local", model=model, clock=clock,
    )
    result = await runner.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result.status is Status.DEGRADED
    assert len(model.requests) == 1
    # 兜底请求携带的时限不得超过剩余总时限（5s）。
    assert 0 < model.requests[0]["limits"]["timeout_ms"] <= 5_000
    assert result.recovery["fallback_attempted"] is True

    # 90s 之后：总时限耗尽，主错误为 DEADLINE_EXCEEDED -> 不进入兜底。
    clock2 = ScriptedClock()

    async def run_expired(inputs, ctx):
        clock2.advance_ms(91_000)
        raise FlowAgentCoreError(ErrorCode.DEADLINE_EXCEEDED, "总时限耗尽")

    model2 = FakeModel(turns=[FakeModel.text_turn("不应调用")])
    runner2 = _scripted_runner(
        {"biz": WorkflowSpec(name="biz", run=run_expired)},
        owner="local", model=model2, clock=clock2,
    )
    result2 = await runner2.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result2.status is Status.FAILED
    assert result2.error is not None
    assert result2.error.code is ErrorCode.DEADLINE_EXCEEDED
    assert model2.requests == []
    assert result2.recovery["fallback_attempted"] is False
    assert result2.recovery["final_source"] == "error"


# ---------------------------------------------------------------- F05


async def test_f05_clarification_exhausted_local_explain_only():
    """F05（本地所有者）：澄清耗尽 -> 仅解释型兜底，data_unavailable 标记。"""
    async def run(inputs, ctx):
        raise FlowAgentCoreError(ErrorCode.CLARIFICATION_EXHAUSTED, "澄清次数已用尽")

    model = FakeModel(turns=[FakeModel.text_turn("请补充设备编号后再试")])
    runner = _runner({"biz": WorkflowSpec(name="biz", run=run)},
                     owner="local", model=model)
    result = await runner.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result.status is Status.DEGRADED
    # 业务数据不可用标记：调用方据此不得展示任何统计结果。
    assert result.output == {"data_unavailable": True}
    assert result.reply == "请补充设备编号后再试"
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request["tools"] == []
    prompt = _prompt_text(request)
    assert QUESTION in prompt
    # 仅解释口径：说明未完成原因与需补充条件，明确禁止给出统计/结论。
    assert "未完成的原因" in prompt
    assert "需要补充的条件" in prompt
    assert "不要给出任何统计或结论" in prompt
    # 普通兜底才允许“给出一般性建议”，澄清耗尽场景不得出现该邀请。
    assert "如能给出安全的一般性建议" not in prompt
    assert result.recovery["primary_error"]["code"] == "CLARIFICATION_EXHAUSTED"
    assert result.recovery["final_source"] == "direct_llm"


async def test_f05_clarification_exhausted_caller_owner():
    """F05（调用方所有者）：FAILED 交还 CLARIFICATION_EXHAUSTED，零本地请求。"""
    async def run(inputs, ctx):
        raise FlowAgentCoreError(ErrorCode.CLARIFICATION_EXHAUSTED, "澄清次数已用尽")

    model = FakeModel(turns=[FakeModel.text_turn("不应调用")])
    runner = _runner({"biz": WorkflowSpec(name="biz", run=run)},
                     owner="caller", model=model)
    result = await runner.run(RunInput(workflow="biz", inputs={"message": QUESTION}))

    assert result.status is Status.FAILED
    assert result.error is not None
    assert result.error.code is ErrorCode.CLARIFICATION_EXHAUSTED
    assert model.requests == []
    assert result.recovery["fallback_attempted"] is False
    assert result.recovery["final_source"] == "caller"
