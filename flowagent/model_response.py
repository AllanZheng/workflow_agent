"""模型单轮恢复：只重试尚未发布的调用，不重放工具和已经发送的答案。"""
from __future__ import annotations

import logging

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

logger = logging.getLogger(__name__)


class ModelResponseError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def response_text(content) -> str:
    # 文本块只提取可见正文，不能把推理块或 Python 对象表示当成答案。
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item if isinstance(item, str) else item.get("text", "")
                       for item in content if isinstance(item, str) or (
                           isinstance(item, dict) and item.get("type") == "text"
                           and isinstance(item.get("text"), str)))
    return ""


def _validate(response, allowed_tools=None):
    if getattr(response, "invalid_tool_calls", None):
        raise ModelResponseError("MODEL_INVALID_RESPONSE")
    # 先校验整轮工具名再执行，既不放宽授权，也不重放同批次中的有效工具。
    if allowed_tools is not None and any(call.get("name") not in allowed_tools
            for call in (getattr(response, "tool_calls", None) or [])):
        raise ModelResponseError("MODEL_INVALID_RESPONSE")
    if not getattr(response, "tool_calls", None) and not response_text(
            getattr(response, "content", None)).strip():
        raise ModelResponseError("MODEL_EMPTY_RESPONSE")


def _retry_error(exc):
    # 只恢复模型通信错误；鉴权、取消和业务参数错误保持原来的失败语义。
    if isinstance(exc, ModelResponseError):
        return exc
    if isinstance(exc, (TimeoutError, APITimeoutError, httpx.TimeoutException)):
        return ModelResponseError("MODEL_TIMEOUT")
    if isinstance(exc, (APIConnectionError, httpx.TransportError)):
        return ModelResponseError("MODEL_UPSTREAM_ERROR")
    if isinstance(exc, APIStatusError) and (exc.status_code in (408, 429) or exc.status_code >= 500):
        return ModelResponseError("MODEL_UPSTREAM_ERROR")
    return None


def _record(error, response, execution_id, attempt, retry):
    # 日志仅保留分类和长度，避免记录用户正文、工具参数或模型凭据。
    metadata = getattr(response, "response_metadata", {}) or {}
    reason = metadata.get("finish_reason")
    safe_reason = reason if reason in {"stop", "length", "tool_calls", "content_filter"} else "unknown"
    logger.warning("model_response_recovery execution_id=%s attempt=%s code=%s retry=%s "
                   "content_length=%s tool_count=%s finish_reason=%s",
                   execution_id, attempt + 1, error.code, retry,
                   len(response_text(getattr(response, "content", None))),
                   len(getattr(response, "tool_calls", None) or []), safe_reason)


async def invoke_model(model, messages, *, execution_id=None, allowed_tools=None):
    for attempt in range(2):
        response = None
        try:
            response = await model.ainvoke(messages)
            _validate(response, allowed_tools)
            return response
        except Exception as exc:
            error = _retry_error(exc)
            if error is None:
                raise
            _record(error, response, execution_id, attempt, attempt == 0)
            if attempt:
                raise error from exc


async def stream_model(model, messages, *, buffered: bool, execution_id=None, allowed_tools=None):
    for attempt in range(2):
        response, published, pending = None, False, []
        try:
            async for chunk in model.astream(messages):
                response = chunk if response is None else response + chunk
                pending.append(chunk)
                if not buffered and response_text(getattr(response, "content", None)).strip():
                    # 无工具路径继续逐 token 输出；一旦发送即不允许自动重放。
                    for item in pending:
                        yield item
                    pending.clear()
                    published = True
            _validate(response, allowed_tools)
            for item in pending:
                yield item
            return
        except Exception as exc:
            error = _retry_error(exc)
            if error is None:
                raise
            retry = attempt == 0 and not published
            _record(error, response, execution_id, attempt, retry)
            if not retry:
                raise error from exc
