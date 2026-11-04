"""异步流的统一关闭：清理失败不覆盖原始执行错误。"""
import asyncio
import inspect
import logging
import sys
from typing import Any

logger = logging.getLogger("flowagent.stream")


async def close_async_resource(resource: Any) -> None:
    if resource is None:
        return
    closer = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    if not callable(closer):
        return
    active_type = sys.exc_info()[0]
    # GeneratorExit 只是正常关闭信号；关闭期间新到的取消必须继续传播。
    preserve_error = active_type is not None and not issubclass(active_type, GeneratorExit)
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        if not preserve_error:
            raise
        logger.warning("stream cleanup cancelled while handling an existing error")
    except Exception:  # 清理异常写入日志，不能覆盖原始错误或取消。
        logger.exception("stream cleanup failed")
