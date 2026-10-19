"""内核契约：数据对象、能力接口、稳定错误码与澄清模板。

边界（技术方案 §5.1）：本文件只定义类型、错误与澄清模板渲染/解析；
不做 HTTP、不做业务字段解析、不 import 平台协议或 SDK。

消息结构约定：保留 role / content / tool_calls / tool_call_id，
工具往返不得压成纯文本历史。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Protocol, runtime_checkable

# ---------------------------------------------------------------- 状态与错误


class Status(str, Enum):
    """执行终态（技术方案 §3.2 唯一终态表）。"""

    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorCode(str, Enum):
    """稳定错误码（技术方案 §3.3 全集；授权/取消另有终止码不伪装依赖错误）。"""

    INVALID_INPUT = "INVALID_INPUT"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_UPSTREAM_ERROR = "MODEL_UPSTREAM_ERROR"
    MODEL_INVALID_RESPONSE = "MODEL_INVALID_RESPONSE"
    TOOL_ARGUMENT_INVALID = "TOOL_ARGUMENT_INVALID"
    MCP_TOOL_ERROR = "MCP_TOOL_ERROR"
    SKILL_LOAD_FAILED = "SKILL_LOAD_FAILED"
    SKILL_BINDING_MISMATCH = "SKILL_BINDING_MISMATCH"
    MODEL_BUDGET_EXCEEDED = "MODEL_BUDGET_EXCEEDED"
    TOOL_BUDGET_EXCEEDED = "TOOL_BUDGET_EXCEEDED"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"
    DUPLICATE_TOOL_CALL = "DUPLICATE_TOOL_CALL"
    CLARIFICATION_EXHAUSTED = "CLARIFICATION_EXHAUSTED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    OUTPUT_INVALID = "OUTPUT_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    FALLBACK_FAILED = "FALLBACK_FAILED"
    # 稳定终止码：授权/归属拒绝与用户取消，不进入普通恢复路径。
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    CANCELLED_BY_USER = "CANCELLED_BY_USER"


class FlowAgentCoreError(RuntimeError):
    """内核错误：code 稳定，message 面向日志；未知异常不得吞掉堆栈。"""

    def __init__(self, code: ErrorCode | str, message: str, *, cause: BaseException | None = None) -> None:
        self.code = ErrorCode(code)
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


# ---------------------------------------------------------------- 数据对象


@dataclass
class RunInput:
    """一次执行的输入；无凭据、无平台快照（平台字段在适配层消化）。"""

    workflow: str
    inputs: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    correlation_id: str | None = None
    continuation: dict[str, Any] | None = None


@dataclass
class Usage:
    """实际消耗：模型请求、MCP 尝试、Skill 加载、token 与耗时；不可得 token 为 None。"""

    model_requests: int = 0
    model_retries: int = 0
    mcp_attempts: int = 0
    skill_loads: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    elapsed_ms: float | None = None

    def merge(self, other: "Usage") -> None:
        self.model_requests += other.model_requests
        self.model_retries += other.model_retries
        self.mcp_attempts += other.mcp_attempts
        self.skill_loads += other.skill_loads
        for key in ("tokens_in", "tokens_out"):
            mine, theirs = getattr(self, key), getattr(other, key)
            setattr(self, key, (mine or 0) + theirs if (mine is not None or theirs is not None) else None)
        if self.elapsed_ms is not None and other.elapsed_ms is not None:
            self.elapsed_ms = round(self.elapsed_ms + other.elapsed_ms, 1)


@dataclass
class Clarification:
    """needs_input 的载荷：正文由代码模板生成，模型不参与。"""

    reason: str
    missing_fields: list[str]
    confirmed_fields: dict[str, Any] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)
    continuation: dict[str, Any] = field(default_factory=dict)

    def render_body(self) -> str:
        """面向用户的澄清正文（代码模板；尾附版本化机器标记）。"""
        missing = "、".join(self.missing_fields)
        confirmed = "\n".join(f"- {key}：{value}" for key, value in self.confirmed_fields.items())
        hints = "\n".join(f"- {hint}" for hint in self.hints)
        lines = [f"还需要补充以下信息才能继续：{missing}。"]
        if confirmed:
            lines.append(f"已确认条件：\n{confirmed}")
        if hints:
            lines.append(f"补充提示：\n{hints}")
        body = "\n".join(lines)
        return f"{body}\n{render_clarification_marker(self.continuation)}"


@dataclass
class RunResult:
    """执行终态：status/reply/output/citations/usage/error/execution_path/recovery。"""

    run_id: str
    status: Status
    reply: str = ""
    output: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: FlowAgentCoreError | None = None
    execution_path: str | None = None  # workflow / autonomous / direct_llm
    recovery: dict[str, Any] = field(default_factory=dict)
    clarification: Clarification | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """对外序列化：error 拆 code/message，usage 平铺；不包含内部字段。"""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "reply": self.reply,
            "output": self.output,
            "citations": self.citations,
            "usage": {
                "model_requests": self.usage.model_requests,
                "model_retries": self.usage.model_retries,
                "mcp_attempts": self.usage.mcp_attempts,
                "skill_loads": self.usage.skill_loads,
                "tokens_in": self.usage.tokens_in,
                "tokens_out": self.usage.tokens_out,
                "elapsed_ms": self.usage.elapsed_ms,
            },
            "error": None if self.error is None else {
                "code": self.error.code.value, "message": str(self.error)},
            "execution_path": self.execution_path,
            "recovery": self.recovery,
            "clarification": None if self.clarification is None else {
                "reason": self.clarification.reason,
                "missing_fields": self.clarification.missing_fields,
                "confirmed_fields": self.clarification.confirmed_fields,
                "hints": self.clarification.hints,
                "continuation": self.clarification.continuation,
            },
        }


# ---------------------------------------------------------------- 事件


class EventKind(str, Enum):
    RUN_STARTED = "run.started"
    STEP_STARTED = "step.started"
    STEP_FINISHED = "step.finished"
    TOOL_FINISHED = "tool.finished"
    TEXT_DELTA = "text.delta"
    RUN_FINISHED = "run.finished"


@dataclass
class RunEvent:
    """原生事件：seq 从 0 连续递增，唯一 run.finished 携带完整 RunResult。"""

    run_id: str
    seq: int
    kind: EventKind
    step: str | None
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 能力接口


@dataclass
class ModelEvent:
    """模型流事件：text_delta / tool_delta / completed（含完整正文、工具调用与 usage）。"""

    kind: str  # "text_delta" | "tool_delta" | "completed"
    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


@runtime_checkable
class Model(Protocol):
    """模型能力：stream(messages, tools, limits)；SDK 自动重试在客户端关闭。"""

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        limits: Mapping[str, Any],
    ) -> AsyncIterator[ModelEvent]: ...


@dataclass
class ToolCallSpec:
    """工具目录项：名称、描述、JSON Schema；模型不能通过参数扩大范围。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultData:
    """工具结果四态：success / error / unknown / terminal。

    terminal 携带已校验 RunResult（含 needs_input），独占一轮，直接成为执行终态。
    """

    status: str
    value: Any = None
    error_code: str | None = None
    message: str | None = None
    terminal: RunResult | None = None

    @classmethod
    def success(cls, value: Any) -> "ToolResultData":
        return cls(status="success", value=value)

    @classmethod
    def error(cls, message: str, code: str = "MCP_TOOL_ERROR") -> "ToolResultData":
        return cls(status="error", message=message, error_code=code)

    @classmethod
    def unknown(cls, message: str) -> "ToolResultData":
        return cls(status="unknown", message=message)

    @classmethod
    def terminal_result(cls, result: RunResult) -> "ToolResultData":
        return cls(status="terminal", terminal=result)


@runtime_checkable
class Tools(Protocol):
    """工具能力：list 返回目录；call 执行一次并计入尝试数。"""

    def list(self) -> Awaitable[list[ToolCallSpec]]: ...

    def call(self, name: str, arguments: dict[str, Any]) -> Awaitable[ToolResultData]: ...


@runtime_checkable
class Skills(Protocol):
    """Skill 能力：catalog 目录；load 正文（固定版本；目录不算使用）。"""

    def catalog(self) -> list[dict[str, Any]]: ...

    async def load(self, skill_id: str) -> str: ...


@dataclass
class WorkflowSpec:
    """业务工作流：普通 Python 函数注册表项，不引入节点基类或通用引擎。

    enabled 由运行策略提供，不挂在 spec 上（未注册/禁用/缺依赖走自主规划）。
    """

    name: str
    run: Callable[[dict[str, Any], "Any"], Awaitable[RunResult]]
    preflight: Callable[[dict[str, Any], "Any"], Awaitable[str | None]] | None = None
    validate: Callable[[RunResult, "Any"], bool] | None = None


# ---------------------------------------------------------------- 澄清机器标记

_MARKER_PREFIX = "sample-agent:clarification:v1"
_MARKER_RE = re.compile(
    r"<!--\s*" + re.escape(_MARKER_PREFIX) + r"\s+(\{.*?\})\s*-->", re.DOTALL
)


def render_clarification_marker(continuation: dict[str, Any]) -> str:
    """版本化标记：保存 reference_time/timezone/workflow/计数；不含凭据或群材料。

    展示端不展示；解析后从模型提示中移除；原生显式 continuation 的正文不附标记。
    """
    payload = {
        "version": continuation.get("version", 1),
        "workflow": continuation.get("workflow"),
        "reference_time": continuation.get("reference_time"),
        "timezone": continuation.get("timezone"),
        "clarify_count": continuation.get("clarify_count", 0),
    }
    return f"<!-- {_MARKER_PREFIX} {json.dumps(payload, ensure_ascii=False)} -->"


def strip_clarification_marker(text: str) -> str:
    """从将进入模型提示的文本中移除机器标记（存储保留原文）。"""
    return _MARKER_RE.sub("", text).strip()


def parse_clarification_marker(text: str) -> dict[str, Any] | None:
    """解析正文尾部的机器标记；无标记/版本不识别/JSON 非法返回 None（视为新请求）。"""
    match = _MARKER_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if payload.get("version") != 1:
        return None
    return payload


def build_continuation(
    *, workflow: str, original_request: str, reference_time: str, timezone: str,
    resolved_fields: dict[str, Any], clarify_count: int,
) -> dict[str, Any]:
    """拟定 continuation v1；计数代表已发出的澄清次数，模型不可写入。"""
    return {
        "version": 1,
        "workflow": workflow,
        "original_request": original_request,
        "reference_time": reference_time,
        "timezone": timezone,
        "resolved_fields": resolved_fields,
        "clarify_count": clarify_count,
    }


def merge_continuation(
    continuation: dict[str, Any] | None, supplement: dict[str, Any]
) -> dict[str, Any]:
    """续接合并顺序：原始请求字段 → 已确认字段 → 本次用户补充逐字段覆盖。"""
    merged: dict[str, Any] = dict(continuation or {})
    resolved = dict(merged.get("resolved_fields") or {})
    resolved.update(supplement)
    merged["resolved_fields"] = resolved
    return merged
