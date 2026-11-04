"""旧 V3 基线引擎：驱动当前 ManagedAgentRuntime（M0 冻结用）。

测绘结论（2026-10-19）：
- 模型：langchain_openai.ChatOpenAI，max_retries=0（单轮恢复只有一个所有者），
  vLLM Qwen 模板需 extra_body.chat_template_kwargs.enable_thinking=False，且只允许一条 system。
- 聊天循环：flowagent.agent_runtime.ManagedAgentRuntime.run(message, system_prompt,
  max_tool_iterations, history) -> str；usage 在 run() 后读取（每次 run 重置）。
- 群分析：flowagent.group_analysis.run_group_analysis(parsed, target, llm, call_tool,
  report_system, use_async=True, requested_message=...) -> (reply, {"analysis_result": ...})。
- 会商：flowagent.function_stage.run_function_stage(req, llm=, mcp_client=, system_prompt=,
  skills=, skill_loader=, execution_id=, generation=) -> (reply, meta, runtime)。

本文件不 import 旧 YAML 引擎（WorkflowEngine/build_graph）；历史 YAML 成绩仅作参考。
JSONL 行协议支持子进程隔离：每行 {"op": ...} 请求 -> 每行结果。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from benchmarks._compat import SAMPLE_AGENT_ROOT, bootstrap_paths, resolve_secret

bootstrap_paths()  # flowagent 源码注入（幂等）

from langchain_openai import ChatOpenAI  # noqa: E402

from flowagent.agent_runtime import ManagedAgentRuntime  # noqa: E402
from flowagent.function_stage import FunctionTask, run_function_stage  # noqa: E402
from flowagent.group_analysis import run_group_analysis  # noqa: E402
from flowagent.local_tools import build_group_analysis_tool  # noqa: E402

DEFAULT_REPORT_SYSTEM = (
    "你是群分析报告撰写人。基于分段纪要与分类统计，输出中文 Markdown 报告，\n"
    "只包含以下三个二级标题段，不添加其他标题与开场白：\n"
    "## 总结 —— 本周期群内发生了什么：事项、讨论、结论；\n"
    "## 分类统计 —— 各类占比与 1-2 条代表消息引用（保留 [MM-DD HH:mm] 昵称 前缀）；\n"
    "## 汇报要点 —— 需要领导/值守者关注的事项：待办、风险、未决问题（条目化）。\n"
    "要求：数据只来自纪要与统计，不编造；人名只用群内昵称；\n"
    "若纪要声明了采样/截断/覆盖不全，必须在「总结」末尾加一行\"⚠ 覆盖率：…\"如实声明；\n"
    "若统计缺失，用一句话说明后跳过对应条目。"
)

# 生产 parse_args 节点系统提示（configs/managed/group_analysis.yaml 原文）：
# 基线与候选共用，保证控制参数解析的提示一致（首次对照不混入提示优化）。
PARSE_ARGS_SYSTEM = (
    "你是参数解析器，只输出一个 JSON 对象，不输出任何其他文字。\n"
    "解析规则：\n"
    "1. group_query：提取用户明确说出的群名称或关键词；不推测群名。请求已有目标时填空串。\n"
    "   group_id：仅当用户明确给出群会话 ID 时填写正整数，否则填 null。最终群 ID 仍由平台核验并绑定。\n"
    "   用户本轮只补充群名或群 ID 时，可结合此前对话理解分析意图。\n"
    "2. start / end：时间范围记号，保持机器可读的原文：\n"
    "   - 相对：now、now-7d、now-30d（\"最近N天/上周\"→ start=now-Nd, end=now）；\n"
    "   - 绝对：YYYY-MM-DD 或 M月D日（用户没说年份就不补年份，服务端会推断）；\n"
    "   - 未提及则输出空串 \"\"。\n"
    "3. days：用户表达了\"N 天\"且未给绝对日期时输出整数 N，否则输出 null。\n"
    "4. taxonomy_override：用户临时指定的分类口径原文（如\"按消息性质分\"），\n"
    "   没有则输出空串 \"\"。\n"
    "5. start/end/days 全部为空串/null 时，工具端默认按最近 7 天处理。"
)


class _TrackedBound:
    """bind_tools 结果的代理：记录每次调用的 token 用量（缺失为 None）。"""

    def __init__(self, inner: Any, tracker: "TrackedChatModel") -> None:
        self._inner = inner
        self._tracker = tracker

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = await self._inner.ainvoke(messages, **kwargs)
        self._tracker._record(result, time.perf_counter() - started)
        return result

    async def astream(self, messages: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        chunk_count = 0
        async for chunk in self._inner.astream(messages, **kwargs):
            chunk_count += 1
            yield chunk
        # 流式聚合 usage 不可得：记录调用次数与耗时，token 置 None。
        self._tracker._record(None, time.perf_counter() - started, streamed=True, chunks=chunk_count)


class TrackedChatModel:
    """包装 ChatOpenAI：捕获 token 用量；自身也可作为 _invoke_json 的 llm（ainvoke）。

    vLLM 兼容：空工具列表不调用 bind_tools（tools=[] 会被端点 400 拒绝）。
    """

    def __init__(self, inner: ChatOpenAI) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]]) -> "_TrackedBound | TrackedChatModel":
        if not tools:
            return self
        return _TrackedBound(self.inner.bind_tools(tools), self)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = await self.inner.ainvoke(messages, **kwargs)
        self._record(result, time.perf_counter() - started)
        return result

    async def astream(self, messages: Any, **kwargs: Any) -> Any:
        async for chunk in _TrackedBound(self.inner, self).astream(messages, **kwargs):
            yield chunk

    def _record(self, result: Any, elapsed: float, *, streamed: bool = False, chunks: int = 0) -> None:
        usage = None
        if result is not None:
            meta_usage = getattr(result, "usage_metadata", None)
            if isinstance(meta_usage, dict):
                usage = {
                    "input_tokens": meta_usage.get("input_tokens"),
                    "output_tokens": meta_usage.get("output_tokens"),
                }
            else:
                token_usage = (getattr(result, "response_metadata", None) or {}).get("token_usage")
                if isinstance(token_usage, dict):
                    usage = {
                        "input_tokens": token_usage.get("prompt_tokens"),
                        "output_tokens": token_usage.get("completion_tokens"),
                    }
        self.calls.append({
            "elapsed_ms": round(elapsed * 1000, 1), "usage": usage,
            "streamed": streamed, "chunks": chunks or None,
        })

    # ---- 汇总 ----
    def usage_summary(self) -> dict[str, Any]:
        ins = [c["usage"]["input_tokens"] for c in self.calls if c.get("usage")]
        outs = [c["usage"]["output_tokens"] for c in self.calls if c.get("usage")]
        return {
            "model_calls": len(self.calls),
            "tokens_in": sum(ins) if ins else None,
            "tokens_out": sum(outs) if outs else None,
            "detail": self.calls,
        }


class ScriptedFetch:
    """按用例脚本应答 fetch_group_analysis_input / platform_resolve_group；记录实际参数。"""

    def __init__(self, script: dict[str, Any]) -> None:
        self.script = script
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        handler = (self.script.get("tools") or {}).get(name)
        if handler is None:
            raise KeyError(name)
        if isinstance(handler, dict) and handler.get("__error__"):
            raise RuntimeError(handler["__error__"])
        if callable(handler):
            return handler(arguments)
        return handler


class BaselineV3Engine:
    """旧 V3 引擎封装：chat / group_analysis / consultation 三种驱动。"""

    name = "baseline-v3"

    def __init__(self, manifest: dict[str, Any]) -> None:
        model = manifest["model"]
        self.base_url = model["base_url"]
        self.model_name = model["model_name"]
        self.api_key = resolve_secret(model.get("api_key"), env_default="BENCH_MODEL_API_KEY") or "EMPTY"
        self.temperature = float(model.get("temperature", 0))

    # ---- 模型 ----
    def build_llm(self, *, temperature: float | None = None, thinking: bool = False) -> TrackedChatModel:
        kwargs: dict[str, Any] = {}
        if not thinking:
            # vLLM Qwen 官方模板：关思考；否则 max_tokens 小时会被思考截断。
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        inner = ChatOpenAI(
            model=self.model_name, api_key=self.api_key, base_url=self.base_url,
            temperature=self.temperature if temperature is None else temperature,
            max_retries=0, **kwargs,
        )
        return TrackedChatModel(inner)

    # ---- 聊天（可选工具/本地终端工具）----
    async def chat(
        self,
        message: str,
        *,
        system_prompt: str,
        history: list[dict[str, Any]] | None = None,
        mcp_client: Any = None,
        local_tools: list[Any] | None = None,
        max_tool_iterations: int = 8,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        llm = self.build_llm(temperature=temperature)
        runtime = ManagedAgentRuntime(
            llm, mcp_client, local_tools=local_tools,
            tolerate_tool_errors=True, record_tool_events=True,
        )
        error: str | None = None
        reply = ""
        try:
            reply = await runtime.run(message, system_prompt, max_tool_iterations, history)
        except Exception as exc:  # noqa: BLE001 - 失败记录原样保留
            error = f"{type(exc).__name__}: {exc}"
        record: dict[str, Any] = {
            "reply": reply, "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            **llm.usage_summary(),
            "loaded_skills": list(runtime.loaded_skills),
            "tool_events": list(runtime.usage.tool_events),
            "terminal_metadata": runtime.terminal_metadata,
        }
        if mcp_client is not None and hasattr(mcp_client, "calls"):
            record["mcp_calls"] = [(n, a) for n, a in mcp_client.calls]
        return record

    # ---- 群分析（直达 run_group_analysis；脚本化取数）----
    async def group_analysis(
        self,
        *,
        message: str,
        parsed: dict[str, Any],
        target: Any,
        fetch_script: dict[str, Any],
        requested_message: str | None = None,
        report_system: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        llm = self.build_llm()
        fetch = ScriptedFetch(fetch_script)
        error: str | None = None
        reply, meta = "", {}
        try:
            reply, meta = await run_group_analysis(
                parsed, target, llm, fetch,
                lambda: report_system or DEFAULT_REPORT_SYSTEM,
                use_async=True,
                requested_message=message if requested_message is None else requested_message,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        return {
            "reply": reply, "analysis_result": (meta or {}).get("analysis_result"),
            "error": error, "fetch_calls": fetch.calls,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            **llm.usage_summary(),
        }

    # ---- 会商阶段 ----
    async def consultation(
        self,
        *,
        task: dict[str, Any],
        message: str,
        system_prompt: str,
        mcp_client: Any = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        llm = self.build_llm()
        function_task = FunctionTask(**task)
        request = SimpleNamespace(function_task=function_task, message=message)
        error: str | None = None
        reply, meta = "", {}
        try:
            reply, meta, _runtime = await run_function_stage(
                request, llm=llm, mcp_client=mcp_client, system_prompt=system_prompt,
                skills=[], skill_loader=None, execution_id="bench-execution", generation=1,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        return {
            "reply": reply, "function_result": (meta or {}).get("function_result"),
            "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            **llm.usage_summary(),
        }


async def _jsonl_loop(stream: Any, out: Any) -> None:
    """子进程行协议：每行一个请求，输出一行结果。"""
    engine: BaselineV3Engine | None = None
    for line in stream:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if engine is None:
            engine = BaselineV3Engine(request["manifest"])
        op = request["op"]
        if op == "chat":
            result = await engine.chat(**request["args"])
        elif op == "group_analysis":
            result = await engine.group_analysis(**request["args"])
        elif op == "consultation":
            result = await engine.consultation(**request["args"])
        else:
            result = {"error": f"unknown op: {op}"}
        out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        out.flush()


def main() -> None:
    asyncio.run(_jsonl_loop(sys.stdin, sys.stdout))


if __name__ == "__main__":
    main()
