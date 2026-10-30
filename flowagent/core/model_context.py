"""模型调用的请求级上下文与实际应用证据，不持有平台快照或凭据。"""
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Mapping

from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
from flowagent.core.stream import close_async_resource


# 原生 SDK 模型也在发起处计数；上下文只保存用量，不包含凭据。
native_request_usage: ContextVar[Any] = ContextVar("flowagent_native_request_usage", default=None)


def record_native_request() -> None:
    usage = native_request_usage.get()
    if usage is not None:
        usage.model_requests += 1


@dataclass
class ModelRequestContext:
    system_prompt: str = ""
    memory_text: str = ""
    memory_ids: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    request_count: int = 0
    applied_parameters: dict[str, Any] = field(default_factory=dict)
    memory_injected: bool = False
    context_bytes: int | None = None
    usage: Any = field(default=None, repr=False)

    def record_request(self, request: dict[str, Any]) -> None:
        # 仅在 SDK 请求实际发起时记账，授权拒绝和零调用不产生应用证据。
        self.request_count += 1
        if self.usage is not None:
            self.usage.model_requests += 1
        self.applied_parameters = {k: request[k] for k in
            ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop")
            if k in request}
        content = str(request["messages"][0].get("content") or "") if request.get("messages") else ""
        self.memory_injected = self.memory_injected or bool(
            self.memory_text.strip() and self.memory_text in content)


class ContextualModel:
    def __init__(self, inner: Any, context: ModelRequestContext) -> None:
        self.inner = inner
        self.context = context

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                     limits: Mapping[str, Any]):
        context = self.context
        prepared = [dict(message) for message in messages]
        additions = [context.system_prompt] if context.system_prompt.strip() else []
        if context.memory_text.strip():
            additions.append("记忆上下文：\n" + context.memory_text)
        if additions:
            if prepared and prepared[0].get("role") == "system":
                prepared[0]["content"] = "\n\n".join(additions + [str(prepared[0].get("content") or "")])
            else:
                prepared.insert(0, {"role": "system", "content": "\n\n".join(additions)})
        # 最终消息而非注入前消息计入预算，所有模型调用路径共享检查。
        payload = json.dumps({"messages": prepared, "tools": tools}, ensure_ascii=False)
        if context.context_bytes is not None and len(payload.encode("utf-8")) > context.context_bytes:
            raise FlowAgentCoreError(ErrorCode.CONTEXT_LIMIT_EXCEEDED,
                                     "model context exceeds request byte budget")
        effective = dict(limits or {})
        # 快照优先，临时 token 上限只允许收紧。
        for name, value in context.parameters.items():
            if name == "max_tokens" and effective.get(name) is not None:
                effective[name] = min(value, effective[name])
            else:
                effective[name] = value
        stream = self.inner.stream(prepared, tools, effective)
        try:
            async for event in stream:
                yield event
        finally:
            await close_async_resource(stream)
