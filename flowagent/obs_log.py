"""obs_log — 五仓统一可观测日志模块（复制同步，勿在本仓单独修改）.

规范源头：demo_platform/src/agent_platform/obs_log.py
同步工具：demo_platform/scripts/sync_obs_log.py（推送其余仓标准路径）
改动流程：先改源头 -> 升 _OBS_LOG_VERSION -> 跑同步脚本。

能力（设计见 docs/specs/2026-09-27-logging-obs-plan-design.md）：
- setup_logging(service)：双 handler——文本→stdout（人读）、JSON→stderr（机器）
- RequestIdMiddleware：纯 ASGI（SSE 安全）；X-Request-ID 有则透传/无则生成，
  响应回传同头；每请求一条出入日志（2xx/3xx=INFO、4xx=WARNING、5xx=ERROR+堆栈）
- request_id 经 contextvar + Filter 注入所有 LogRecord，业务调用点零改动
- trace_context()/bootstrap_trace_from_env()：后台任务与进程级 trace
- traced_client()/install_trace_hook()：httpx 出站自动附 X-Request-ID
- 环境变量：LOG_LEVEL 覆盖根级别（默认 INFO）；OBS_LOG_JSON=0 关 JSON handler
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager

_OBS_LOG_VERSION = "1.0.0"

# 入站 X-Request-ID 白名单：防日志注入（超长/含空白等一律重新生成）
_SAFE_ID = re.compile(r"^[0-9A-Za-z-]{1,64}$")

# LogRecord 标准属性（JSON 摊平 extra 时排除；request_id 为本模块注入字段）
_RECORD_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    "request_id",
})

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "obs_request_id", default=None)


# ---------------------------------------------------------------- trace 上下文

def get_request_id() -> str | None:
    return _request_id_var.get()


def new_request_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def trace_context(tid: str | None = None):
    """后台任务包裹用：进入设置 trace id，退出恢复先前值。"""
    token = _request_id_var.set(tid if tid else new_request_id())
    try:
        yield _request_id_var.get()
    finally:
        _request_id_var.reset(token)


def bootstrap_trace_from_env() -> str | None:
    """进程级 trace：读取 TRACE_ID 环境变量（由拉起方注入），非法则忽略。"""
    tid = os.environ.get("TRACE_ID", "").strip()
    if tid and _SAFE_ID.match(tid):
        _request_id_var.set(tid)
    return _request_id_var.get()


# ---------------------------------------------------------------- handlers

class _RequestIdFilter(logging.Filter):
    """把当前 contextvar 的 request_id 写进每条记录（handler 级，双 handler 共用）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get() or "-"
        return True


class _DynamicStreamHandler(logging.StreamHandler):
    """每次写入动态解析当前 sys.stdout/sys.stderr。

    StreamHandler 默认在构造时绑定流对象；pytest capsys 等测试设施会在
    运行期替换并关闭旧的 sys.stdout/stderr 对象，绑定死的 handler 会报
    "I/O operation on closed file"。动态解析对生产无影响（进程的
    stdout/stderr 不变），对测试与运行期重定向安全。
    """

    def __init__(self, stream_name: str) -> None:
        logging.Handler.__init__(self)   # 跳过 StreamHandler 的流绑定
        self._stream_name = stream_name  # "stdout" / "stderr"

    @property
    def stream(self):                    # 每次取当前流对象
        return getattr(sys, self._stream_name)

    @stream.setter
    def stream(self, value) -> None:     # StreamHandler.__init__ 会赋值，忽略
        pass


class _TextFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__(
            fmt=f"%(asctime)s %(levelname)s [{service}] "
                f"[req:%(request_id)s] %(name)s - %(message)s",
        )


class _JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "svc": self._service,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():  # extra 摊平（event/job_id/...）
            if key not in _RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(service: str, *, level: str | None = None,
                  force: bool = False) -> None:
    """装配双 handler 并降噪。幂等：已装配则只更新级别。

    只增删带 _obs_log 标记的本模块 handler，绝不动 pytest/uvicorn 等
    他人 handler。force=True（测试用）先拆旧 obs handler 再重装。
    """
    root = logging.getLogger()
    already = any(getattr(h, "_obs_log", False) for h in root.handlers)
    if already and not force:
        root.setLevel(os.environ.get("LOG_LEVEL", level or "INFO").upper())
        return
    for handler in [h for h in root.handlers if getattr(h, "_obs_log", False)]:
        root.removeHandler(handler)
    rid_filter = _RequestIdFilter()
    text_handler = _DynamicStreamHandler("stdout")
    text_handler.setFormatter(_TextFormatter(service))
    text_handler.addFilter(rid_filter)
    text_handler._obs_log = True            # type: ignore[attr-defined]
    root.addHandler(text_handler)
    if os.environ.get("OBS_LOG_JSON", "1") != "0":
        json_handler = _DynamicStreamHandler("stderr")
        json_handler.setFormatter(_JsonFormatter(service))
        json_handler.addFilter(rid_filter)
        json_handler._obs_log = True        # type: ignore[attr-defined]
        root.addHandler(json_handler)
    root.setLevel(os.environ.get("LOG_LEVEL", level or "INFO").upper())
    # 降噪：请求出入已由 RequestIdMiddleware 记录，避免 uvicorn.access 重复
    for noisy in ("uvicorn.access", "httpx", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------- 中间件

class RequestIdMiddleware:
    """纯 ASGI 中间件：透传/生成 X-Request-ID + 每请求一条出入日志。

    纯 ASGI（非 BaseHTTPMiddleware）：对 SSE/流式响应零影响。
    中间件内的日志动作自身容错，日志故障不影响请求。
    """

    def __init__(self, app) -> None:
        self.app = app
        self.logger = logging.getLogger("obs.request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = None
        for key, value in scope.get("headers", []):
            if key == b"x-request-id":
                candidate = value.decode("latin-1")
                rid = candidate if _SAFE_ID.match(candidate) else None
                break
        if rid is None:
            rid = new_request_id()
        token = _request_id_var.set(rid)
        started = time.monotonic()
        fields: dict = {"event": "request", "method": scope.get("method"),
                        "path": scope.get("path")}

        async def _send(message):
            if message["type"] == "http.response.start":
                fields["status"] = message["status"]
                message = {**message, "headers": [
                    *message.get("headers", []),
                    (b"x-request-id", rid.encode("ascii")),
                ]}
            await send(message)

        try:
            try:
                await self.app(scope, receive, _send)
            except Exception:
                fields["status"] = 500
                self.logger.error(
                    "request failed",
                    extra={**fields,
                           "duration_ms": int((time.monotonic() - started) * 1000)},
                    exc_info=True)
                raise
            status = fields.get("status", 0)
            level = (logging.INFO if status < 400
                     else logging.WARNING if status < 500 else logging.ERROR)
            try:
                self.logger.log(
                    level, "request",
                    extra={**fields,
                           "duration_ms": int((time.monotonic() - started) * 1000)})
            except Exception:
                pass                        # 日志故障不影响请求
        finally:
            _request_id_var.reset(token)


# ---------------------------------------------------------------- httpx 传播

async def _attach_trace_header(request) -> None:
    """httpx.AsyncClient 的 request 钩子必须是协程函数（httpx 会 await 它）。"""
    rid = _request_id_var.get()
    if rid:
        request.headers["X-Request-ID"] = rid


def install_trace_hook(client) -> None:
    """给已有 httpx 客户端补 request 钩子（幂等）。"""
    hooks = list(client.event_hooks.get("request", []))
    if _attach_trace_header not in hooks:
        hooks.append(_attach_trace_header)
    client.event_hooks["request"] = hooks


def traced_client(**kwargs):
    """构造自动附 X-Request-ID 的 httpx.AsyncClient。"""
    import httpx                     # 延迟导入：无 httpx 环境不影响模块加载
    client = httpx.AsyncClient(**kwargs)
    install_trace_hook(client)
    return client
