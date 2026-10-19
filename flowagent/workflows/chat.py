"""chat 问答工作流：共享 Agent 循环，可自动加载 Skill、调用 MCP、选择终端业务工具。

M1 形态：普通问答 + Skill/MCP；终端业务工具（群分析等）由装配层注入 local_tools，
其 RunResult 已校验，直接成为执行终态（不追加模型改写）。
"""
from __future__ import annotations

from typing import Any

from flowagent.core.agent import AgentOutcome, LocalTool, ToolRegistry, run_agent
from flowagent.core.contracts import ErrorCode, FlowAgentCoreError, RunResult, Status, WorkflowSpec

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "你是业务助手。回答用户问题；需要外部信息时使用提供的工具或加载 Skill；"
    "缺少必要条件时明确说明需要补充什么；不得虚构工具结果、数据或引用。"
)


def build_chat_workflow(
    *,
    system_prompt: str | None = None,
    terminal_tools: list[LocalTool] | None = None,
) -> WorkflowSpec:
    """构造 chat 工作流；terminal_tools 由装配层按可用性注入（不可用流程不暴露）。"""
    prompt = system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT
    tools = list(terminal_tools or [])

    async def preflight(inputs: dict[str, Any], ctx: Any) -> str | None:
        # chat 无绑定依赖；平台模式下能力可用性由请求级组装决定。
        return None

    async def run(inputs: dict[str, Any], ctx: Any) -> RunResult:
        message = str(inputs.get("message") or inputs.get("question") or "")
        if not message.strip():
            raise FlowAgentCoreError(ErrorCode.INVALID_INPUT, "chat requires message")
        history = [dict(item) for item in inputs.get("history") or []]
        registry = ToolRegistry(remote=ctx.tools, skills=ctx.skills, local=tools)
        outcome: AgentOutcome = await run_agent(
            ctx, system_prompt=prompt, user_message=message, history=history,
            registry=registry, duplicate_guard=False,
        )
        return _outcome_result(outcome)

    return WorkflowSpec(name="chat", run=run, preflight=preflight)


def _outcome_result(outcome: AgentOutcome) -> RunResult:
    if outcome.is_terminal:
        return outcome.terminal  # type: ignore[return-value]
    return RunResult(
        run_id="", status=Status.COMPLETED, reply=outcome.text or "",
        execution_path="workflow",
    )
