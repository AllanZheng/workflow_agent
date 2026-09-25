"""im-tools-mcp ASGI 应用：Streamable HTTP MCP 子应用 + Bearer 校验 + /health。

组装方式与 demo_platform MCP Edge（mcp 2.2.0 官方 SDK）同款：
  - ``Server(on_list_tools=..., on_call_tool=...)`` 低层回调；
  - ``server.streamable_http_app(streamable_http_path="/", stateless_http=True,
    json_response=True)`` 子应用挂载在 ``/mcp``；
  - 父应用 lifespan 负责运行 ``server.session_manager.run()``（mount 不传播
    子应用 lifespan，父 lifespan 是唯一驱动方）。

鉴权（技术方案 §3）：
  - MCP Edge 按 Admin 凭据注入 ``Authorization: Bearer <im 令牌>``；
  - 本服务最小 bearer 校验：头缺失/非 Bearer 一律 401；设置了
    IM_TOOLS_MCP_TOKEN 时还须精确匹配（常量时间比较）；
  - 令牌经 ContextVar 传给工具实现，用于请求级 ImClient；绝不写日志。

DNS rebinding 防护保持关闭：服务部署在 compose 内网，调用方 Host 是服务名，
开启默认清单（仅 localhost）会整体 403；防线由 bearer + 网络边界承担。
"""

from __future__ import annotations

import hmac
import json
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import mcp.types as mcp_types
from mcp.server import Server
from mcp.server.transport_security import TransportSecuritySettings

from .config import Settings
from .im_client import ImClient, ImError
from .tools import TOOL_HANDLERS, ToolInputError, error_payload
from .window import WindowError

#: 当前请求的 im 令牌（bearer wrapper 写入，工具回调读取）
_current_im_token: ContextVar[str] = ContextVar("im_tools_mcp_token", default="")

#: 工具描述（list_tools 的对外目录文案）
TOOL_DESCRIPTIONS = {
    "find_group": (
        "按名称关键词查找 im 群，返回候选列表（conversation_id/name/member_count，"
        "最多 10 条）。分析前先调用本工具定位群。"
    ),
    "fetch_group_messages": (
        "拉取指定群在分析窗口内某一段（segment_index）的格式化消息语料"
        "（[MM-DD HH:mm] 昵称: 内容），附覆盖率与分段元数据；窗口超过一天时"
        "按 segment_hours 分段、用 segment_index 逐段取。"
    ),
    "fetch_group_analysis_input": (
        "按已授权的 conversation_id 一次性读取固定时间窗，返回去重后的消息与完整性状态。"
    ),
}

ImClientFactory = Callable[[str], ImClient]


def bearer_auth_wrapper(child, expected_token: str = ""):
    """纯 ASGI Bearer 校验包装：失败 401 稳定 JSON，成功写 ContextVar 后放行。"""

    async def wrapped(scope, receive, send) -> None:
        if scope["type"] != "http":
            await child(scope, receive, send)
            return
        token = ""
        for key, value in scope.get("headers") or []:
            if key.decode("latin-1").lower() == "authorization":
                auth = value.decode("latin-1")
                if auth[:7].lower() == "bearer ":
                    token = auth[7:].strip()
                break
        token_ok = bool(token) and (
            not expected_token or hmac.compare_digest(token, expected_token)
        )
        if not token_ok:
            response = JSONResponse(
                status_code=401,
                content={"error": {"code": "unauthorized"}},
            )
            await response(scope, receive, send)
            return
        _current_im_token.set(token)
        await child(scope, receive, send)

    return wrapped


def _error_result(payload: dict[str, Any]) -> mcp_types.CallToolResult:
    """构造 is_error 的 CallToolResult：结构化载荷 + 等义 JSON 文本。"""
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(
            type="text",
            text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )],
        structured_content=payload,
        is_error=True,
    )


def build_app(
    settings: Settings | None = None,
    *,
    im_client_factory: ImClientFactory | None = None,
) -> FastAPI:
    """组装服务应用；im_client_factory 供测试注入（默认按环境配置构造）。"""
    settings = settings or Settings.from_env()
    default_factory: ImClientFactory = lambda token: ImClient(
        settings.base_url, token, timeout_seconds=settings.timeout_seconds
    )
    make_client = im_client_factory or default_factory

    async def list_handler(ctx, params) -> mcp_types.ListToolsResult:
        tools = [
            mcp_types.Tool(
                name=name,
                description=TOOL_DESCRIPTIONS.get(name, ""),
                input_schema=schema,
            )
            for name, (schema, _handler) in sorted(TOOL_HANDLERS.items())
        ]
        return mcp_types.ListToolsResult(tools=tools)

    async def call_handler(ctx, params) -> mcp_types.CallToolResult:
        entry = TOOL_HANDLERS.get(params.name)
        if entry is None:
            return _error_result(error_payload("invalid_arguments"))
        _schema, handler = entry
        arguments = params.arguments if isinstance(params.arguments, dict) else {}
        try:
            im = make_client(_current_im_token.get())
            async with im:
                result = await handler(im, arguments, settings=settings)
        except ToolInputError:
            return _error_result(error_payload("invalid_arguments"))
        except WindowError as exc:
            return _error_result(error_payload(exc.code))
        except ImError as exc:
            return _error_result(error_payload(exc.code))
        text = json.dumps(result, ensure_ascii=False, sort_keys=True)
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=text)],
            structured_content=result,
        )

    server = Server(
        "im-tools-mcp", on_list_tools=list_handler, on_call_tool=call_handler
    )
    child = server.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        # mcp 2.2.0 默认开启 DNS rebinding 防护（仅放行 localhost 形态 Host），
        # compose 内网调用方 Host 是服务名，会被整体 403 —— 显式关闭，
        # 防线由 bearer + 网络边界承担（见模块 docstring）
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 父应用是子应用 MCP 会话管理器的唯一驱动方（mount 不传播 lifespan）
        async with server.session_manager.run():
            yield

    app = FastAPI(title="im-tools-mcp", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    app.mount("/mcp", bearer_auth_wrapper(child, settings.shared_token))
    return app
