"""原生内核测试共用替身：脚本化模型/工具/Skill、可控时钟与事件捕获。

约定（对应测试方案 §2.2）：
- FakeModel 每轮按脚本吐 ModelEvent 或抛异常；记录请求与额度观察点。
- FakeTools 脚本化 list/call，支持 success/error/unknown/terminal 四态与异常注入。
- 可控时钟驱动 deadline 用例；Event/Barrier 控制竞争，不用任意 sleep。
"""
from __future__ import annotations

import asyncio
from typing import Any

from flowagent.core.contracts import (
    ErrorCode,
    FlowAgentCoreError,
    ModelEvent,
    Skills,
    ToolResultData,
    Tools,
)
from flowagent.core.context import RunContext  # RunContext 定义在 context 而非 contracts


class FakeModel:
    """脚本化模型：turns 依次消费；每项为事件列表（异步生成器产出）或异常。

    usage_hit: completed 事件可带 usage；缺省 None（token 缺失为 null 的口径）。
    """

    def __init__(self, turns: list[Any] = (), *, default_text: str | None = None) -> None:
        self.turns = list(turns)
        self.default_text = default_text
        self.requests: list[dict[str, Any]] = []

    async def stream(self, messages, tools, limits):
        self.requests.append({"messages": messages, "tools": list(tools), "limits": dict(limits or {})})
        if self.turns:
            turn = self.turns.pop(0)
        elif self.default_text is not None:
            turn = [ModelEvent(kind="completed", text=self.default_text, tool_calls=[],
                               finish_reason="stop")]
        else:
            turn = FlowAgentCoreError(ErrorCode.MODEL_INVALID_RESPONSE, "no scripted turns left")
        if isinstance(turn, BaseException):
            raise turn
        for event in turn:
            yield event

    @staticmethod
    def text_turn(text: str, *, finish_reason: str = "stop", usage: dict | None = None) -> list[ModelEvent]:
        return [ModelEvent(kind="text_delta", text=text),
                ModelEvent(kind="completed", text=text, tool_calls=[],
                           finish_reason=finish_reason, usage=usage)]

    @staticmethod
    def tool_turn(*calls: dict[str, Any], text: str = "", finish_reason: str = "tool_calls",
                  usage: dict | None = None) -> list[ModelEvent]:
        """calls: {id, name, arguments(JSON 字符串或 dict)}"""
        normalized = [
            {"id": call.get("id") or f"c{i}", "name": call["name"],
             "arguments": call["arguments"] if isinstance(call["arguments"], str)
             else __import__("json").dumps(call["arguments"], ensure_ascii=False)}
            for i, call in enumerate(calls)
        ]
        return [ModelEvent(kind="completed", text=text, tool_calls=normalized,
                           finish_reason=finish_reason, usage=usage)]

    @property
    def call_count(self) -> int:
        return len(self.requests)


class FakeTools(Tools):
    """脚本化工具集：catalog 固定；call 按名称路由脚本（值/异常/可调用）。"""

    def __init__(
        self,
        catalog: list[dict[str, Any]] | None = None,
        results: dict[str, Any] | None = None,
    ) -> None:
        self.catalog_entries = catalog or [{"name": "echo", "description": "回显",
                                            "input_schema": {"type": "object"}}]
        self.results = dict(results or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.barrier: asyncio.Barrier | None = None  # 并发竞争同步点

    async def list(self):
        from flowagent.core.contracts import ToolCallSpec
        return [ToolCallSpec(name=item["name"], description=item.get("description", ""),
                             input_schema=item.get("input_schema", {}))
                for item in self.catalog_entries]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResultData:
        self.calls.append((name, dict(arguments)))
        if self.barrier is not None:
            await self.barrier.wait()
        script = self.results.get(name)
        if isinstance(script, BaseException):
            raise script
        if callable(script):
            return script(arguments)
        if isinstance(script, ToolResultData):
            return script
        return ToolResultData.success(script if script is not None else {"ok": True, "tool": name})


class FakeSkills(Skills):
    """脚本化 Skill：catalog 列表 + load 正文（可注入异常）。"""

    def __init__(self, items: list[dict[str, Any]] | None = None,
                 bodies: dict[str, str] | None = None) -> None:
        self.items = items or [{"skill_id": "s1", "name": "巡检制度",
                                "summary": "设备巡检制度检索指南"}]
        self.bodies = dict(bodies or {"s1": "# 巡检制度\n检索关键词：巡检频率。"})
        self.load_calls: list[str] = []

    def catalog(self) -> list[dict[str, Any]]:
        return list(self.items)

    async def load(self, skill_id: str) -> str:
        self.load_calls.append(skill_id)
        body = self.bodies.get(skill_id)
        if body is None:
            raise FlowAgentCoreError(ErrorCode.SKILL_LOAD_FAILED, f"unknown skill: {skill_id}")
        return body


class ScriptedClock:
    """可控单调时钟：advance() 前进；deadline 用例不依赖真实等待。"""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


class EventSink:
    """事件捕获：按序记录全部 RunEvent。"""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def __call__(self, event) -> None:
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [event.kind.value for event in self.events]

    def of_kind(self, kind: str) -> list[Any]:
        return [event for event in self.events if event.kind.value == kind]


def make_context(
    *,
    model: FakeModel | Any = None,
    tools: FakeTools | Any = None,
    skills: FakeSkills | Any = None,
    budget: Any = None,
    sink: EventSink | None = None,
    clock: ScriptedClock | None = None,
) -> RunContext:
    """组装测试用 RunContext；默认预算 + 可控时钟（clock 贯穿 budget 与 ctx）。"""
    from flowagent.core.context import Budget

    clock = clock or ScriptedClock()
    if budget is None:
        budget = Budget(
            model_total=24, mcp_total=32, deadline_ms=90_000,
            main_deadline_ms=60_000, fallback_timeout_ms=30_000, clock=clock,
        )
    return RunContext(
        model=model or FakeModel(default_text="ok"),
        tools=tools, skills=skills, budget=budget,
        emit=sink or EventSink(), clock=clock,
    )


def completed_event(events: list[Any]) -> Any:
    """取唯一 run.finished 事件。"""
    finished = [e for e in events if e.kind.value == "run.finished"]
    assert len(finished) == 1, f"expected single run.finished, got {len(finished)}"
    return finished[0]
