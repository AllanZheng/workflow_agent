"""原生内核引擎：新 Runner（chat / group_analysis / consultation）。

与 baseline_v3 同接口（engine.chat/group_analysis/consultation），
供候选对照与容量测量；usage 直接来自内核 RunResult.usage。
"""
from __future__ import annotations

import time
from typing import Any

from benchmarks._compat import bootstrap_paths, resolve_secret

bootstrap_paths()

from flowagent.core.contracts import RunInput, ToolCallSpec, ToolResultData  # noqa: E402
from flowagent.core.context import RunContext, default_budget  # noqa: E402
from flowagent.core.runner import Runner  # noqa: E402
from flowagent.workflows.chat import build_chat_workflow  # noqa: E402
from flowagent.workflows.consultation import build_consultation_workflow  # noqa: E402
from flowagent.workflows.group_analysis import build_group_analysis_workflow  # noqa: E402

ENTRY_SYSTEM_PROMPT = (
    "你是企业群助手。能力：日常问答；群消息分析与统计（需要群、时间范围、分类口径，"
    "缺失时向用户澄清）。可用的群分析工具会直接完成取数、分类统计与报告。"
)


class NativeEngine:
    """原生内核封装：Runner + 请求级脚本化工具。"""

    name = "native"

    def __init__(self, manifest: dict[str, Any]) -> None:
        model = manifest["model"]
        self.base_url = model["base_url"]
        self.model_name = model["model_name"]
        self.api_key = resolve_secret(model.get("api_key"), env_default="BENCH_MODEL_API_KEY") or "EMPTY"
        self.temperature = float(model.get("temperature", 0))

    def _make_context(self, tools: Any = None) -> RunContext:
        from flowagent.clients.model import OpenAICompatibleModel

        model = OpenAICompatibleModel(
            base_url=self.base_url, api_key=self.api_key,
            model_name=self.model_name, temperature=self.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return RunContext(model=model, tools=tools, budget=default_budget())

    def _runner(self, tools: Any, workflows: dict[str, Any], local_tools: list | None = None) -> Runner:
        def factory() -> RunContext:
            return self._make_context(tools)

        return Runner(workflows=workflows, context_factory=factory, fallback_owner="local")

    async def chat(
        self, message: str, *, system_prompt: str, history: list | None = None,
        mcp_client: Any = None, local_tools: list | None = None,
        max_tool_iterations: int = 8, temperature: float | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        from flowagent.core.agent import LocalTool

        started = time.perf_counter()
        terminal = local_tools or []
        runner = self._runner(
            mcp_client, {"chat": build_chat_workflow(system_prompt=system_prompt, terminal_tools=terminal)})
        result = await runner.run(RunInput(
            workflow="chat", inputs={"message": message, "history": list(history or [])},
        ))
        return {
            "reply": result.reply,
            "error": None if result.error is None else f"{result.error.code.value}: {result.error}",
            "status": result.status.value,
            "execution_path": result.execution_path,
            "terminal_metadata": (result.output or {}).get("analysis_result"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "model_calls": result.usage.model_requests,
            "tokens_in": result.usage.tokens_in,
            "tokens_out": result.usage.tokens_out,
        }

    async def group_analysis(
        self, *, message: str, parsed: dict[str, Any], target: Any,
        fetch_script: dict[str, Any], requested_message: str | None = None,  # noqa: ARG002
        report_system: str | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """原生群分析：控制参数由代码从原问题解析（parse 阶段不复刻旧 LLM parse）。"""
        from types import SimpleNamespace

        started = time.perf_counter()
        fetch = _ScriptTools(fetch_script)
        runner = self._runner(
            fetch, {"group_analysis": build_group_analysis_workflow(config={"default_days": 7})})
        inputs: dict[str, Any] = {"message": message}
        if target is not None:
            inputs["analysis_target"] = SimpleNamespace(conversation_id=int(target.conversation_id))
        result = await runner.run(RunInput(workflow="group_analysis", inputs=inputs))
        analysis = (result.output or {}).get("analysis_result")
        coverage = (analysis or {}).get("coverage") or {}
        return {
            "reply": result.reply, "analysis_result": analysis,
            "error": None if result.error is None else f"{result.error.code.value}: {result.error}",
            "status": result.status.value,
            "fetch_calls": [(name, args) for name, args in fetch.calls],
            "coverage": coverage,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "model_calls": result.usage.model_requests,
            "tokens_in": result.usage.tokens_in, "tokens_out": result.usage.tokens_out,
        }

    async def consultation(
        self, *, task: dict[str, Any], message: str, system_prompt: str, mcp_client: Any = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        started = time.perf_counter()
        runner = self._runner(mcp_client, {"consultation_stage": build_consultation_workflow()})
        result = await runner.run(RunInput(workflow="consultation_stage", inputs={
            "function_task": task, "message": message}))
        return {
            "reply": result.reply,
            "function_result": (result.output or {}).get("function_result"),
            "error": None if result.error is None else f"{result.error.code.value}: {result.error}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "model_calls": result.usage.model_requests,
            "tokens_in": result.usage.tokens_in, "tokens_out": result.usage.tokens_out,
        }


class _ScriptTools:
    """脚本化 Tools（原生协议）：list 返回声明目录；call 路由脚本。"""

    def __init__(self, script: dict[str, Any]) -> None:
        self._script = script
        self.calls: list[tuple[str, dict]] = []
        self._tools = {
            name: ToolCallSpec(name=name, description="bench", input_schema={"type": "object"})
            for name in (script.get("tools") or {})
        }

    async def list(self) -> list[ToolCallSpec]:
        return list(self._tools.values())

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        self.calls.append((name, dict(arguments)))
        handler = (self._script.get("tools") or {}).get(name)
        if handler is None:
            return ToolResultData.error(f"unknown tool: {name}")
        return ToolResultData.success(handler)
