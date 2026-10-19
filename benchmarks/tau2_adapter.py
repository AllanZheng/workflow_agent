"""τ² telecom 套件：官方环境 + 原生工具调用适配。

基建全部复用工作区：vendor/tau2-bench（环境/任务/用户模拟器/确定性奖励）、
flowagent_suite/telecom_bridge.py（活跃环境桥 + 工具执行语义）。
被测引擎通过 Tau2Mcp 以原生 tool calling 操作 tau2 工具（区别于旧 YAML 适配器的
"JSON 决策"模式；历史 YAML 成绩仅作参考，不直接可比）。

跨轮上下文：V3 运行时 history 只保留 user/assistant 文本（与平台现行行为一致），
不注入 prior_tool_log；这是被测引擎的真实语义，适配器不做额外补偿。
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from benchmarks._compat import bootstrap_paths

bootstrap_paths(with_tau2=True)

from tau2.agent.base_agent import HalfDuplexAgent  # noqa: E402

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
EMPTY_REPLY = "（本轮回复为空，请问还有其他需要帮助的吗？）"

# 入口系统提示：与旧 JSON 决策适配器同域，但去掉 JSON 决策说明（原生工具调用）。
SYSTEM_SUFFIX = (
    "你是电信客服助手，通过工具调用为用户办理业务。"
    "每次只执行一个动作，工具结果会反馈给你；完成办理后用中文向用户确认结果。"
)


def _bridge():
    bootstrap_paths(with_tau2=True)
    import sys

    suite_dir = str(bootstrap_paths.__globals__["BENCH_WORKSPACE"] / "flowagent_suite")
    if suite_dir not in sys.path:
        sys.path.insert(0, suite_dir)
    import telecom_bridge

    return telecom_bridge


class Tau2Mcp:
    """tau2 环境工具 → V3 运行时 MCP 协议（进程内直调，无 HTTP）。"""

    def __init__(self, tools: list[Any], bridge: Any) -> None:
        self._tools = tools
        self._bridge = bridge
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_bindings: dict[str, dict[str, Any]] = {}

    async def list_tools(self) -> list[dict[str, Any]]:
        out = []
        for tool in self._tools:
            desc = getattr(tool, "long_desc", None) or getattr(tool, "short_desc", "") or ""
            try:
                schema = tool.params.model_json_schema()
            except Exception:  # noqa: BLE001 - 无参数模型兜底
                schema = {"type": "object", "properties": {}}
            out.append({"name": tool.name, "description": desc, "input_schema": schema})
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> str:
        self.calls.append((name, dict(arguments or {})))
        return self._bridge.dispatch(name, arguments)

    # 原生内核 Tools 协议（list/call）：与旧协议（list_tools/call_tool）并存。
    async def list(self) -> list[Any]:
        from flowagent.core.contracts import ToolCallSpec

        specs = []
        for tool in self._tools:
            desc = getattr(tool, "long_desc", None) or getattr(tool, "short_desc", "") or ""
            try:
                schema = tool.params.model_json_schema()
            except Exception:  # noqa: BLE001 - 无参数模型兜底
                schema = {"type": "object", "properties": {}}
            specs.append(ToolCallSpec(name=tool.name, description=desc, input_schema=schema))
        return specs

    async def call(self, name: str, arguments: dict[str, Any]):
        from flowagent.core.contracts import ToolResultData

        return ToolResultData.success(await self.call_tool(name, arguments))


class V3TelecomAgent(HalfDuplexAgent):
    """tau2 HalfDuplexAgent：每条用户消息触发一次 V3 运行时调用。"""

    def __init__(self, engine: Any, tools: list[Any], domain_policy: str, bridge: Any) -> None:
        super().__init__(tools=tools, domain_policy=f"{domain_policy}\n\n{SYSTEM_SUFFIX}")
        self._engine = engine
        self._mcp = Tau2Mcp(tools, bridge)
        self.call_count = 0
        self.last_error = ""
        self._transcript: list[dict[str, str]] = []

    # tau2 协议：无外部持久状态（state 仅存消息列表）
    def get_init_state(self, message_history=None) -> list:  # noqa: ARG002
        return []

    def _line(self, msg: Any) -> str:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") in (None, "text")
            )
        text = str(content or "").strip()
        if not text:
            return ""
        role = getattr(msg, "role", "user")
        return f"{'用户' if role == 'user' else '客服'}: {text}"

    def generate_next_message(self, message: Any, state: list) -> tuple[Any, list]:
        self.call_count += 1
        state.append(message)
        user_text = self._line(message).removeprefix("用户: ")
        self._transcript.append({"role": "user", "content": user_text})

        reply, error = "", ""
        try:
            result = asyncio.run(self._engine.chat(
                user_text,
                system_prompt=self.domain_policy,
                history=list(self._transcript[:-1]),
                mcp_client=self._mcp,
                max_tool_iterations=8,
            ))
            reply = (result.get("reply") or "").strip()
            if result.get("error"):
                error = str(result["error"])
                reply = ""
        except Exception as exc:  # noqa: BLE001 - tau2 要求 AssistantMessage 有 content
            error = f"{type(exc).__name__}: {exc}"
            reply = ""

        reply = _THINK_RE.sub("", reply).strip() or EMPTY_REPLY
        self.last_error = error
        if error:
            print(f"[tau2-agent] turn error: {error[:200]}", flush=True)
        self._transcript.append({"role": "assistant", "content": reply})
        from tau2.data_model.message import AssistantMessage

        assistant = AssistantMessage(role="assistant", content=reply)
        state.append(assistant)
        return assistant, state


async def run_suite(
    engine: Any,
    manifest: dict[str, Any],
    *,
    limit: int | None = None,
    case_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按官方 runner 逐任务执行；每任务独立环境，串行（桥只持一个活跃环境）。"""
    bootstrap_paths(with_tau2=True)
    bridge = _bridge()
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_environment, build_user
    from tau2.runner.helpers import get_tasks
    from tau2.runner.simulation import run_simulation

    suite_cfg = manifest["suites"].get("tau2", {})
    user_cfg = manifest.get("user_simulator_llm", {})
    max_steps = int(suite_cfg.get("max_steps", 40))
    max_errors = int(suite_cfg.get("max_errors", 8))
    seed = int(suite_cfg.get("seed", 42))
    evaluation_type = EvaluationType(str(suite_cfg.get("evaluation_type", "all")))

    tasks = get_tasks("telecom")
    if case_ids:
        wanted = set(case_ids)
        tasks = [t for t in tasks if t.id in wanted]
    if limit:
        tasks = tasks[:limit]

    records: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        # tau2 编排器是同步的且内部含同步 LLM 客户端；放到工作线程跑，
        # 使 generate_next_message 里的 asyncio.run 不与外层事件循环嵌套。
        record = await asyncio.to_thread(_run_one_task, engine, task, user_cfg,
                                         max_steps, max_errors, seed, evaluation_type, bridge)
        records.append(record)
        print(
            f"[tau2] {index + 1}/{len(tasks)} {task.id} reward={record['reward']} "
            f"{record['seconds']}s err={record['error']}",
            flush=True,
        )

    rewards = [r["reward"] for r in records if r.get("reward") is not None]
    summary = {
        "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "pass_count": sum(r.get("pass_count") or 0 for r in records),
        "num_tasks": len(records),
        "num_evaluated": len(rewards),
        "user_llm": f'{user_cfg.get("model")} @ {user_cfg.get("api_base")}',
        "max_steps": max_steps,
        "seed": seed,
        "evaluation_type": str(evaluation_type),
    }
    return records, summary


def _run_one_task(engine, task, user_cfg, max_steps, max_errors, seed, evaluation_type, bridge):
    """同步执行单个 tau2 任务（在工作线程中运行；异常记录不外抛）。"""
    import time as _time

    started = _time.perf_counter()
    record: dict[str, Any] = {"id": task.id, "engine": engine.name, "error": None}
    agent = None
    try:
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.runner.build import build_environment, build_user
        from tau2.runner.simulation import run_simulation

        env = build_environment("telecom")
        bridge.set_active_environment(env, task.id)
        agent = V3TelecomAgent(engine, env.get_tools(), env.get_policy(), bridge)
        user = build_user(
            "user_simulator", env, task,
            llm=user_cfg.get("model", "openai/Qwen3.6-35B"),
            llm_args={"api_base": user_cfg.get("api_base"), "api_key": user_cfg.get("api_key", "EMPTY")},
        )
        orchestrator = Orchestrator(
            domain="telecom", agent=agent, user=user, environment=env, task=task,
            max_steps=max_steps, max_errors=max_errors, seed=seed,
        )
        run = run_simulation(orchestrator, evaluation_type=evaluation_type)
        reward = run.reward_info.reward
        record.update({
            "reward": reward,
            "pass_count": 1 if reward == 1 else 0,
            "turns": len(run.messages),
            "agent_calls": agent.call_count,
            "agent_last_error": agent.last_error,
        })
    except Exception as exc:  # noqa: BLE001 - 单任务失败不终止整套
        record.update({"error": f"{type(exc).__name__}: {exc}", "reward": None, "pass_count": 0})
    record["seconds"] = round(_time.perf_counter() - started, 1)
    return record
