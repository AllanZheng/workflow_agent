"""OpenAI 兼容模型客户端：chunk 聚合、usage 提取与稳定错误映射。

约束（技术方案 §5.1/§6.3）：
- SDK 自动重试关闭（max_retries=0）：恢复只有一个所有者（内核单轮恢复）。
- 不依赖 langchain；消息为 OpenAI 线格式 dict（role/content/tool_calls/tool_call_id）。
- 流式：text_delta / tool_delta 增量 + 一次 completed（完整正文、聚合工具调用、usage）。
- Qwen 模板：默认经 extra_body 关思考；系统消息只有一条（由调用方保证）。
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Mapping

from flowagent.core.contracts import ErrorCode, FlowAgentCoreError, Model, ModelEvent


def _map_openai_error(exc: BaseException) -> FlowAgentCoreError:
    """openai SDK 异常 → 稳定错误码；未识别异常原样抛出（保留堆栈）。"""
    try:
        import openai
    except ImportError:  # pragma: no cover - 环境缺依赖
        return FlowAgentCoreError(ErrorCode.CAPABILITY_UNAVAILABLE, f"openai sdk missing: {exc}")
    if isinstance(exc, (openai.APITimeoutError,)):
        return FlowAgentCoreError(ErrorCode.MODEL_TIMEOUT, str(exc), cause=exc)
    if isinstance(exc, (openai.APIConnectionError,)):
        return FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, str(exc), cause=exc)
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code in (408, 429) or exc.status_code >= 500:
            return FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, str(exc), cause=exc)
        return FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, f"http {exc.status_code}", cause=exc)
    return FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, f"{type(exc).__name__}: {exc}", cause=exc)


class OpenAICompatibleModel(Model):
    """OpenAI 兼容端点的流式客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        extra_body: dict[str, Any] | None = None,
        client: Any | None = None,  # 测试注入 AsyncOpenAI 替身
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key  # 请求级凭据可经 set_api_key 换发；不落日志
        self._extra_body = dict(extra_body or {})
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=self._api_key,
                max_retries=0, timeout=self.timeout_seconds,
            )
        return self._client

    def set_api_key(self, api_key: str) -> None:
        """平台预授权换发后更新凭据；既有 client 随之重建。"""
        self._api_key = api_key
        self._client = None

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        limits: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ModelEvent]:
        """一次模型请求：增量事件 + completed。空 tools 不下发 tools 字段。"""
        limits = dict(limits or {})
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": limits.get("temperature", self.temperature),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = tools
        if "max_tokens" in limits:
            request["max_tokens"] = limits["max_tokens"]
        if self._extra_body:
            request["extra_body"] = self._extra_body

        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, int] | None = None

        try:
            created = self._ensure_client().chat.completions.create(**request)
            stream = await created if hasattr(created, "__await__") else created
            async for chunk in stream:
                usage = getattr(chunk, "usage", None) and _extract_usage(chunk.usage) or usage
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    text_parts.append(piece)
                    yield ModelEvent(kind="text_delta", text=piece)
                for call in getattr(delta, "tool_calls", None) or []:
                    index = call.index
                    slot = tool_calls.setdefault(index, {
                        "id": None, "name": None, "arguments": "",
                    })
                    if call.id:
                        slot["id"] = call.id
                    if call.function and call.function.name:
                        slot["name"] = call.function.name
                    if call.function and call.function.arguments:
                        slot["arguments"] += call.function.arguments
                        yield ModelEvent(kind="tool_delta", tool_calls=[dict(slot)])
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        except FlowAgentCoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一映射为稳定错误码
            raise _map_openai_error(exc) from None

        yield ModelEvent(
            kind="completed",
            text="".join(text_parts),
            tool_calls=[tool_calls[key] for key in sorted(tool_calls)],
            finish_reason=finish_reason,
            usage=usage,
        )


def _extract_usage(usage: Any) -> dict[str, int] | None:
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None and completion is None:
        return None
    return {"input_tokens": prompt, "output_tokens": completion}


def parse_tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """聚合后的 tool_call.arguments（JSON 字符串）→ dict；非法 JSON 抛 MODEL_INVALID_RESPONSE。"""
    raw = call.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FlowAgentCoreError(
                ErrorCode.MODEL_INVALID_RESPONSE, f"tool arguments not json: {call.get('name')}"
            ) from exc
        if not isinstance(parsed, dict):
            raise FlowAgentCoreError(
                ErrorCode.MODEL_INVALID_RESPONSE, "tool arguments must be an object")
        return parsed
    return {}
