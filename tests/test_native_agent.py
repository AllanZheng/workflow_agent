"""原生 V3 内核共享 Agent 循环测试（A01-A09）。

覆盖行为编号：
- A01 单轮文本直接发布；模型仅一次请求且携带工具 schema。
- A02 Skill 懒加载：load_skill 目录枚举、正文回填、缓存不重复加载、空目录不暴露。
- A03 允许工具的轮次正文缓冲：首轮正文不发布，但进入下一轮上下文。
- A04 工具可用但模型直接作答：不强制工具调用。
- A05 批次校验：未知工具/终端混调/参数非法一律整批零执行并回填修正消息；
      修正轮次受模型额度约束。
- A06 终端工具独占一轮：结果直接成为终态，不再追加模型调用（含 needs_input）。
- A07 单轮恢复：可恢复错误重发一次；连续失败上抛；授权拒绝不重试；
      finish_reason=length 视为无效响应。
- A08 工具错误反馈：error/unknown 分别以 [TOOL ERROR]/[TOOL UNKNOWN] 回填；
      任意远程异常按当前契约上抛。
- A09 重复守卫：键序无关的相同有效参数第二次调用前终止；不同参数、load_skill
      重复、显式关闭守卫均放行。

全程离线：模型/工具/Skill 替身与受控时钟来自 tests.native_helpers。
"""
from __future__ import annotations  # noqa: I001  # 垫片须先于替身模块导入执行

import pytest

from flowagent.core.agent import LocalTool, ToolRegistry, run_agent
from flowagent.core.context import Budget
from flowagent.core.contracts import (
    Clarification,
    ErrorCode,
    FlowAgentCoreError,
    ModelEvent,
    RunResult,
    Status,
    ToolResultData,
)

# HELPER-BUG：tests/native_helpers.py 仍从 flowagent.core.contracts 导入 RunContext，
# 而内核已把它定义在 flowagent.core.context。为遵守“只写本文件”的约束，导入前补别名；
# 若共享替身模块被修复（contracts 已有该名），此兼容垫片自动跳过。
import flowagent.core.contracts as _contracts
from flowagent.core.context import RunContext as _RunContext

if not hasattr(_contracts, "RunContext"):
    _contracts.RunContext = _RunContext  # type: ignore[attr-defined]

from tests.native_helpers import (
    FakeModel,
    FakeSkills,
    FakeTools,
    ScriptedClock,
    make_context,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------- 测试夹具/助手


def _budget(**overrides) -> Budget:
    """默认口径账本（模型 24 / MCP 32 / 90-60-30s），可覆盖并注入受控时钟。"""
    clock = ScriptedClock()
    params = {
        "model_total": 24,
        "mcp_total": 32,
        "deadline_ms": 90_000,
        "main_deadline_ms": 60_000,
        "fallback_timeout_ms": 30_000,
        "clock": clock,
    }
    params.update(overrides)
    return Budget(**params)


async def _drive(
    *,
    turns=(),
    model=None,
    tools=None,
    skills=None,
    local=None,
    budget=None,
    duplicate_guard: bool = True,
):
    """跑一轮共享 Agent 循环；返回 (outcome, ctx, model) 便于断言请求与 usage。"""
    model = model if model is not None else FakeModel(list(turns))
    ctx = make_context(model=model, tools=tools, skills=skills, budget=budget)
    registry = ToolRegistry(remote=tools, local=local, skills=skills)
    outcome = await run_agent(
        ctx,
        system_prompt="系统提示",
        user_message="用户问题",
        history=[],
        registry=registry,
        duplicate_guard=duplicate_guard,
    )
    return outcome, ctx, model


def _tool_messages(request: dict) -> list[dict]:
    """模型请求中 role=tool 的消息（工具结果回填）。"""
    return [msg for msg in request["messages"] if msg.get("role") == "tool"]


def _function_names(request: dict) -> list[str]:
    return [schema["function"]["name"] for schema in request["tools"]]


def _terminal_tool(result: RunResult, seen: list[dict]) -> LocalTool:
    """构造返回给定 RunResult 的终端本地工具，并记录其入参。"""

    async def handler(arguments):
        seen.append(arguments)
        return ToolResultData.terminal_result(result)

    return LocalTool(
        name="finish", description="业务终态", input_schema={"type": "object"},
        handler=handler, terminal=True,
    )


# ---------------------------------------------------------------- A01 单轮文本


async def test_a01_single_text_turn_publishes_text():
    """A01：单轮文本即发布；无工具执行；模型恰好一次请求且带工具 schema。"""
    tools = FakeTools()
    model = FakeModel([FakeModel.text_turn("你好，我是助手")])
    outcome, ctx, _ = await _drive(model=model, tools=tools)

    assert outcome.text == "你好，我是助手"
    assert outcome.terminal is None
    assert tools.calls == []                     # 没有任何工具被执行
    assert len(model.requests) == 1              # 恰好一次模型请求
    assert _function_names(model.requests[0]) == ["echo"]  # schema 列表已下发
    assert ctx.usage.model_requests == 1


# ---------------------------------------------------------------- A02 Skill 加载


async def test_a02_load_skill_schema_body_and_usage():
    """A02：load_skill 目录枚举 s1；正文经工具消息回填；skill_loads 计 1。"""
    skills = FakeSkills()
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "load_skill", "arguments": {"skill_id": "s1"}}),
        FakeModel.text_turn("按巡检制度执行"),
    ])
    outcome, ctx, _ = await _drive(model=model, tools=FakeTools(), skills=skills)

    assert outcome.text == "按巡检制度执行"
    # 第一轮 schema：load_skill 的参数枚举只含目录中的 skill_id
    load_schema = next(
        s for s in model.requests[0]["tools"] if s["function"]["name"] == "load_skill")
    assert load_schema["function"]["parameters"]["properties"]["skill_id"]["enum"] == ["s1"]
    # 第二轮请求的 tool 消息携带 Skill 正文
    messages = _tool_messages(model.requests[1])
    assert len(messages) == 1
    assert skills.bodies["s1"] in messages[0]["content"]
    assert skills.load_calls == ["s1"]
    assert ctx.usage.skill_loads == 1


async def test_a02_second_load_hits_cache():
    """A02：同一 skill_id 第二次加载不再调用 Skills.load，但正文依旧回填。"""
    skills = FakeSkills()
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "load_skill", "arguments": {"skill_id": "s1"}}),
        FakeModel.tool_turn({"id": "c2", "name": "load_skill", "arguments": {"skill_id": "s1"}}),
        FakeModel.text_turn("完成"),
    ])
    outcome, _, _ = await _drive(model=model, tools=FakeTools(), skills=skills)

    assert outcome.text == "完成"
    assert skills.load_calls == ["s1"]           # 只真正加载过一次（缓存命中）
    for request in model.requests[1:3]:
        assert any(skills.bodies["s1"] in msg["content"] for msg in _tool_messages(request))


async def test_a02_empty_catalog_hides_load_skill():
    """A02：Skills 目录为空时，工具列表中不出现 load_skill。"""
    skills = FakeSkills()
    skills.items = []  # 替身构造器的空列表会回落到默认目录，这里显式清空

    registry = ToolRegistry(remote=FakeTools(), skills=skills)
    assert "load_skill" not in [spec.name for spec in await registry.list()]

    outcome, _, model = await _drive(
        model=FakeModel([FakeModel.text_turn("ok")]), tools=FakeTools(), skills=skills)
    assert outcome.text == "ok"
    assert "load_skill" not in _function_names(model.requests[0])


# ---------------------------------------------------------------- A03 正文缓冲


async def test_a03_tool_turn_text_is_buffered_not_published():
    """A03：带工具调用的轮次正文只作进度；最终只发布末轮正文，但首轮正文进入上下文。"""
    tools = FakeTools()
    first_turn = [
        ModelEvent(kind="text_delta", text="中间稿"),
        ModelEvent(
            kind="completed", text="中间稿", finish_reason="tool_calls",
            tool_calls=[{"id": "c1", "name": "echo", "arguments": '{"q": 1}'}],
        ),
    ]
    model = FakeModel([first_turn, FakeModel.text_turn("最终答复")])
    outcome, _, _ = await _drive(model=model, tools=tools)

    assert outcome.text == "最终答复"
    assert "中间稿" not in outcome.text           # 首轮正文未被发布
    assert len(tools.calls) == 1                 # 工具确实执行
    assistants = [m for m in model.requests[1]["messages"] if m.get("role") == "assistant"]
    assert assistants[0]["content"] == "中间稿"   # 缓冲正文仍在下一轮上下文中
    assert assistants[0]["tool_calls"][0]["function"]["name"] == "echo"


# ---------------------------------------------------------------- A04 无工具直接答


async def test_a04_direct_answer_without_forced_tool_call():
    """A04：工具 schema 非空时，模型仍可直接作答，不强制调用工具。"""
    tools = FakeTools()
    model = FakeModel([FakeModel.text_turn("直接回答")])
    outcome, _, _ = await _drive(model=model, tools=tools)

    assert outcome.text == "直接回答"
    assert model.requests[0]["tools"]            # schema 非空
    assert tools.calls == []


# ---------------------------------------------------------------- A05 批次校验


async def test_a05_unknown_tool_batch_runs_nothing():
    """A05(a)：批次含未知工具 → 整批零执行，两个调用都收到 [TOOL ERROR]，循环继续。"""
    tools = FakeTools()
    model = FakeModel([
        FakeModel.tool_turn(
            {"id": "c1", "name": "echo", "arguments": {"q": 1}},
            {"id": "c2", "name": "missing_tool", "arguments": {}},
        ),
        FakeModel.text_turn("已修正"),
    ])
    outcome, _, _ = await _drive(model=model, tools=tools)

    assert outcome.text == "已修正"               # 修正后循环继续
    assert tools.calls == []                     # 整批零部分执行
    messages = _tool_messages(model.requests[1])
    assert len(messages) == 2
    assert all(msg["content"].startswith("[TOOL ERROR]") for msg in messages)
    assert {msg["tool_call_id"] for msg in messages} == {"c1", "c2"}


async def test_a05_terminal_mixed_batch_runs_nothing():
    """A05(b)：终端工具与其他工具混调 → 零执行 + 修正消息。"""
    tools = FakeTools()
    seen: list[dict] = []
    finish = _terminal_tool(
        RunResult(run_id="r", status=Status.COMPLETED, reply="不应到达"), seen)
    model = FakeModel([
        FakeModel.tool_turn(
            {"id": "t1", "name": "finish", "arguments": {}},
            {"id": "c1", "name": "echo", "arguments": {}},
        ),
        FakeModel.text_turn("改为普通回答"),
    ])
    outcome, _, _ = await _drive(model=model, tools=tools, local=[finish])

    assert outcome.text == "改为普通回答"
    assert seen == []                            # 终端工具未执行
    assert tools.calls == []                     # 普通工具也未执行
    messages = _tool_messages(model.requests[1])
    assert len(messages) == 2
    assert all("[TOOL ERROR]" in msg["content"] for msg in messages)
    assert any("终端工具" in msg["content"] for msg in messages)


async def test_a05_invalid_arguments_json_runs_nothing():
    """A05(c)：参数不是合法 JSON → 整批零执行并回填修正消息。"""
    tools = FakeTools()
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "echo", "arguments": "{not-json"}),
        FakeModel.text_turn("修正后回答"),
    ])
    outcome, _, _ = await _drive(model=model, tools=tools)

    assert outcome.text == "修正后回答"
    assert tools.calls == []
    messages = _tool_messages(model.requests[1])
    assert len(messages) == 1
    assert "参数不是合法 JSON 对象" in messages[0]["content"]


async def test_a05_corrections_bounded_by_model_budget():
    """A05：修正轮次同样消耗模型额度；耗尽时按规格应抛 MODEL_BUDGET_EXCEEDED。

    model_total=2 => 主阶段上限 1（兜底保留 1 次）；第一次修正后即耗尽。
    """
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "missing_tool", "arguments": {}}),
        FakeModel.text_turn("不会被消费"),
    ])
    with pytest.raises(FlowAgentCoreError) as excinfo:
        await _drive(model=model, tools=FakeTools(), budget=_budget(model_total=2))
    assert excinfo.value.code is ErrorCode.MODEL_BUDGET_EXCEEDED


# ---------------------------------------------------------------- A06 终端工具


async def test_a06_terminal_tool_ends_loop_without_model_call():
    """A06：终端工具返回已校验 RunResult 后直接结束，不再追加模型调用。"""
    tools = FakeTools()
    terminal = RunResult(
        run_id="run-1", status=Status.COMPLETED, reply="终端完成", output={"n": 1})
    seen: list[dict] = []
    finish = _terminal_tool(terminal, seen)
    model = FakeModel([FakeModel.tool_turn(
        {"id": "t1", "name": "finish", "arguments": {"k": "v"}})])
    outcome, _, _ = await _drive(model=model, tools=tools, local=[finish])

    assert outcome.terminal is terminal          # 原对象直接成为终态
    assert outcome.text is None
    assert outcome.rounds == 1
    assert seen == [{"k": "v"}]                  # 入参已解析并传给处理器
    assert len(model.requests) == 1              # 模型只被请求一次
    assert tools.calls == []


async def test_a06_terminal_needs_input_returned_directly():
    """A06：终端工具返回 needs_input 的 RunResult 同样直接返回。"""
    terminal = RunResult(
        run_id="run-2", status=Status.NEEDS_INPUT, reply="请补充设备编号",
        clarification=Clarification(reason="missing", missing_fields=["device"]))
    seen: list[dict] = []
    model = FakeModel([FakeModel.tool_turn(
        {"id": "t1", "name": "finish", "arguments": {}})])
    outcome, _, _ = await _drive(model=model, local=[_terminal_tool(terminal, seen)])

    assert outcome.terminal is terminal
    assert outcome.terminal.status is Status.NEEDS_INPUT


# ---------------------------------------------------------------- A07 单轮恢复


async def test_a07_recoverable_error_retried_once():
    """A07：可恢复通信错误重发一次；重发计入额度与重试计数。"""
    model = FakeModel([
        FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "upstream boom"),
        FakeModel.text_turn("恢复成功"),
    ])
    outcome, ctx, _ = await _drive(model=model, tools=FakeTools())

    assert outcome.text == "恢复成功"
    assert len(model.requests) == 2              # 首次失败 + 一次重发
    assert ctx.usage.model_retries == 1
    assert ctx.usage.model_requests == 2


async def test_a07_second_consecutive_failure_raises():
    """A07：连续第二次可恢复失败不再重试，原错误码上抛。"""
    model = FakeModel([
        FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "boom-1"),
        FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "boom-2"),
    ])
    with pytest.raises(FlowAgentCoreError) as excinfo:
        await _drive(model=model, tools=FakeTools())
    assert excinfo.value.code is ErrorCode.MODEL_UPSTREAM_ERROR
    assert len(model.requests) == 2


async def test_a07_authorization_denied_never_retried():
    """A07：授权拒绝属于稳定终止码，立即上抛且不消耗重发配额。"""
    model = FakeModel([
        FlowAgentCoreError(ErrorCode.AUTHORIZATION_DENIED, "denied"),
        FakeModel.text_turn("不应被消费"),
    ])
    with pytest.raises(FlowAgentCoreError) as excinfo:
        await _drive(model=model, tools=FakeTools())

    assert excinfo.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert len(model.requests) == 1              # 没有重发
    assert len(model.turns) == 1                 # 脚本第二轮未被消费


async def test_a07_length_finish_is_invalid_response():
    """A07：finish_reason=length 的截断响应按无效响应处理（重发额度用尽后上抛）。"""
    model = FakeModel([
        FakeModel.text_turn("截断一", finish_reason="length"),
        FakeModel.text_turn("截断二", finish_reason="length"),
    ])
    with pytest.raises(FlowAgentCoreError) as excinfo:
        await _drive(model=model, tools=FakeTools())
    assert excinfo.value.code is ErrorCode.MODEL_INVALID_RESPONSE


async def test_a07_length_response_gets_one_retry():
    """A07：截断响应（未执行任何工具）应重发一次；重发成功则正常发布。"""
    model = FakeModel([
        FakeModel.text_turn("截断", finish_reason="length"),
        FakeModel.text_turn("重发后完整"),
    ])
    outcome, ctx, _ = await _drive(model=model, tools=FakeTools())

    assert outcome.text == "重发后完整"
    assert len(model.requests) == 2
    assert ctx.usage.model_retries == 1


# ---------------------------------------------------------------- A08 工具错误


async def test_a08_error_result_fed_back_and_loop_continues():
    """A08：工具 error 结果以 [TOOL ERROR] 回填，循环继续。"""
    tools = FakeTools(results={
        "echo": ToolResultData.error("后端拒绝", "MCP_TOOL_ERROR")})
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "echo", "arguments": {"q": 1}}),
        FakeModel.text_turn("已说明失败"),
    ])
    outcome, _, _ = await _drive(model=model, tools=tools)

    assert outcome.text == "已说明失败"
    message = _tool_messages(model.requests[1])[0]
    assert message["content"].startswith("[TOOL ERROR]")
    assert "后端拒绝" in message["content"]


async def test_a08_unknown_result_has_explicit_prefix():
    """A08：unknown 状态必须以 [TOOL UNKNOWN] 声明结果不明，防模型虚构成功。"""
    tools = FakeTools(results={"echo": ToolResultData.unknown("结果未知")})
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "echo", "arguments": {}}),
        FakeModel.text_turn("不能假设成功"),
    ])
    outcome, _, _ = await _drive(model=model, tools=tools)

    assert outcome.text == "不能假设成功"
    message = _tool_messages(model.requests[1])[0]
    assert message["content"].startswith("[TOOL UNKNOWN]")
    assert "结果未知" in message["content"]


async def test_a08_remote_exception_propagates():
    """A08：当前契约——Agent 不吞任意远程异常，原样上抛交由 Runner 错误边界。"""
    tools = FakeTools(results={"echo": ValueError("remote down")})
    model = FakeModel([FakeModel.tool_turn(
        {"id": "c1", "name": "echo", "arguments": {}})])
    with pytest.raises(ValueError, match="remote down"):
        await _drive(model=model, tools=tools)
    assert len(tools.calls) == 1                 # 尝试已在调用入口记录


# ---------------------------------------------------------------- A09 重复守卫


async def test_a09_duplicate_effective_args_blocked_before_second_execution():
    """A09：键序不同的相同有效参数视为重复，第二次执行前抛 DUPLICATE_TOOL_CALL。"""
    tools = FakeTools()
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "echo", "arguments": '{"a": 1, "b": 2}'}),
        FakeModel.tool_turn({"id": "c2", "name": "echo", "arguments": '{"b": 2, "a": 1}'}),
    ])
    with pytest.raises(FlowAgentCoreError) as excinfo:
        await _drive(model=model, tools=tools)

    assert excinfo.value.code is ErrorCode.DUPLICATE_TOOL_CALL
    assert len(tools.calls) == 1                 # 第二次未真正执行
    assert len(model.requests) == 2              # 第二次请求已到达模型，守卫在执行前拦截


async def test_a09_different_args_allowed():
    """A09：有效参数不同则放行。"""
    tools = FakeTools()
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "echo", "arguments": {"a": 1}}),
        FakeModel.tool_turn({"id": "c2", "name": "echo", "arguments": {"a": 2}}),
        FakeModel.text_turn("完成"),
    ])
    outcome, _, _ = await _drive(model=model, tools=tools)

    assert outcome.text == "完成"
    assert len(tools.calls) == 2


async def test_a09_load_skill_repeat_allowed():
    """A09：load_skill 相同参数重复调用不触发守卫（缓存命中）。"""
    skills = FakeSkills()
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "load_skill", "arguments": {"skill_id": "s1"}}),
        FakeModel.tool_turn({"id": "c2", "name": "load_skill", "arguments": {"skill_id": "s1"}}),
        FakeModel.text_turn("完成"),
    ])
    outcome, _, _ = await _drive(model=model, tools=FakeTools(), skills=skills)

    assert outcome.text == "完成"
    assert skills.load_calls == ["s1"]


async def test_a09_duplicate_guard_disabled_allows_repeat():
    """A09：duplicate_guard=False 时相同参数重复调用照常执行。"""
    tools = FakeTools()
    model = FakeModel([
        FakeModel.tool_turn({"id": "c1", "name": "echo", "arguments": {"a": 1}}),
        FakeModel.tool_turn({"id": "c2", "name": "echo", "arguments": {"a": 1}}),
        FakeModel.text_turn("完成"),
    ])
    outcome, _, _ = await _drive(model=model, tools=tools, duplicate_guard=False)

    assert outcome.text == "完成"
    assert len(tools.calls) == 2
