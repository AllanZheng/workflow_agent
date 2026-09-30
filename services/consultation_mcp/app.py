from __future__ import annotations

import hmac
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

import httpx
import mcp.types as mcp_types
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server import Server
from mcp.server.transport_security import TransportSecuritySettings

from .config import Settings

logger = logging.getLogger(__name__)

TOOL_NAME = "send_consultation_message"
_CLIENT_MESSAGE_ID = re.compile(r"^[A-Za-z0-9:_-]{1,64}$")
_ARGUMENT_KEYS = frozenset({
    "conversation_id", "content", "client_message_id", "attachment_ids", "reply_to_id",
    "sender_subject_id", "sender_binding_generation", "execution_id", "actual_agent_id",
    "agent_config_id",
})
# IM 受信端点 403 时 message 形如 "AGENT_MUTED: ..."，只放行这四个已知错误码。
_IM_REJECTIONS = frozenset({
    "AGENT_ACCOUNT_INVALID", "AGENT_NOT_CONVERSATION_MEMBER", "AGENT_MUTED",
    "ATTACHMENT_OWNER_MISMATCH",
})


# 解析 IM 403 响应中的拒绝原因码；无法识别时返回通用鉴权失败。
# English: Extract the IM rejection code from a 403 body.
def _im_rejection_code(response: httpx.Response) -> str:
    try:
        message = str(response.json().get("message") or "")
    except (ValueError, AttributeError):
        return "im_unauthorized"
    prefix = message.split(":", 1)[0].strip()
    return prefix if prefix in _IM_REJECTIONS else "im_unauthorized"


# 处理工具错误相关逻辑。
# English: Handle tool error operations.
def _tool_error(code: str) -> mcp_types.CallToolResult:
    payload = {"status": "error", "error_code": code}
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload, is_error=True,
    )


# 构建`app`。
# English: Build app.
def build_app(settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    if not settings.mcp_token:
        logger.warning(
            "consultation-write-mcp 未配置 CONSULTATION_WRITE_MCP_TOKEN，"
            "所有请求将被放行——仅限受信内网/开发环境使用",
        )
    # 列出工具。
    # English: List tools.
    async def list_tools(_ctx, _params):
        return mcp_types.ListToolsResult(tools=[mcp_types.Tool(
            name=TOOL_NAME,
            description="Post an already approved consultation artifact as the bound agent account. The sender identity is verified by IM from the execution context; this tool cannot choose an arbitrary sender.",
            input_schema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "integer", "minimum": 1},
                    "content": {"type": "string", "minLength": 1, "maxLength": 100000},
                    "client_message_id": {"type": "string", "pattern": "^[A-Za-z0-9:_-]{1,64}$"},
                    "attachment_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": 20, "uniqueItems": True},
                    "reply_to_id": {"type": "integer", "minimum": 1},
                    "sender_subject_id": {"type": "string", "format": "uuid"},
                    "sender_binding_generation": {"type": "integer", "minimum": 0},
                    "execution_id": {"type": "string", "minLength": 1},
                    "actual_agent_id": {"type": "string", "minLength": 1},
                    "agent_config_id": {"type": "string", "minLength": 1},
                },
                "required": ["conversation_id", "content", "client_message_id", "sender_subject_id",
                             "sender_binding_generation", "execution_id", "actual_agent_id",
                             "agent_config_id"],
                "additionalProperties": False,
            },
        )])

    # 处理`call`工具相关逻辑。
    # English: Handle call tool operations.
    async def call_tool(_ctx, params):
        args = params.arguments if isinstance(params.arguments, dict) else {}
        if set(args) - _ARGUMENT_KEYS:
            return _tool_error("invalid_arguments")
        conversation_id = args.get("conversation_id")
        content, client_message_id = args.get("content"), args.get("client_message_id")
        attachment_ids = args.get("attachment_ids", [])
        reply_to_id = args.get("reply_to_id")
        # 仅转发已上传附件编号，附件所有权和群访问权限继续由 IM 验证。
        if (not isinstance(attachment_ids, list) or len(attachment_ids) > 20
                or any(type(item) is not int or item <= 0 for item in attachment_ids)
                or len(set(attachment_ids)) != len(attachment_ids)):
            return _tool_error("invalid_arguments")
        if reply_to_id is not None and (type(reply_to_id) is not int or reply_to_id <= 0):
            return _tool_error("invalid_arguments")
        if (type(conversation_id) is not int or conversation_id <= 0
                or not isinstance(content, str) or not content.strip()
                or len(content) > 100000 or not isinstance(client_message_id, str)
                or not _CLIENT_MESSAGE_ID.fullmatch(client_message_id)):
            return _tool_error("invalid_arguments")
        # 发送者身份字段：subject 必须是 UUID 字符串，绑定代数是非负整数（bool 不算整数）。
        try:
            subject = str(uuid.UUID(str(args.get("sender_subject_id"))))
        except ValueError:
            return _tool_error("invalid_arguments")
        generation = args.get("sender_binding_generation")
        if type(generation) is not int or generation < 0:
            return _tool_error("invalid_arguments")
        context = {}
        for key in ("execution_id", "actual_agent_id", "agent_config_id"):
            value = args.get(key)
            if not isinstance(value, str) or not value.strip():
                return _tool_error("invalid_arguments")
            context[key] = value
        if settings.allowed_conversation_ids and conversation_id not in settings.allowed_conversation_ids:
            return _tool_error("conversation_not_allowed")
        payload = {
            "sender_subject_id": subject, "sender_binding_generation": generation,
            "content": content, "message_type": "file" if attachment_ids else "text",
            "client_message_id": client_message_id, **context,
        }
        if attachment_ids:
            payload["attachment_ids"] = attachment_ids
        if reply_to_id is not None:
            payload["reply_to_id"] = reply_to_id
        try:
            async with httpx.AsyncClient(
                base_url=settings.im_base_url,
                headers={"Authorization": f"Bearer {settings.im_service_token}"},
                timeout=settings.timeout_seconds, transport=transport,
            ) as client:
                response = await client.post(
                    f"/api/v1/internal/im/conversations/{conversation_id}/agent-messages",
                    json=payload,
                    headers={"Idempotency-Key": client_message_id,
                             "X-Request-ID": str(uuid.uuid4())},
                )
        except httpx.HTTPError:
            return _tool_error("im_unavailable")
        if response.status_code == 403:
            # IM 的拒绝原因前缀（如 AGENT_MUTED）原样透传，未识别的按鉴权失败处理。
            return _tool_error(_im_rejection_code(response))
        if response.status_code == 401:
            return _tool_error("im_unauthorized")
        if response.status_code >= 500:
            return _tool_error("im_unavailable")
        if response.status_code not in (200, 201):
            return _tool_error("im_send_rejected")
        try:
            body = response.json()
            data = body.get("data", {})
            message_id = data.get("id")
            if type(message_id) is not int:
                raise ValueError("message id missing")
        except (ValueError, AttributeError, TypeError):
            return _tool_error("im_response_invalid")
        result = {"status": "sent", "message_id": message_id,
                  "client_message_id": client_message_id}
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=json.dumps(result))],
            structured_content=result,
        )

    server = Server("consultation-write-mcp", on_list_tools=list_tools,
                    on_call_tool=call_tool)
    child = server.streamable_http_app(
        streamable_http_path="/", stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    # 处理`authenticated``child`相关逻辑。
    # English: Handle authenticated child operations.
    async def authenticated_child(scope, receive, send):
        if scope["type"] != "http":
            await child(scope, receive, send)
            return
        if not settings.mcp_token:
            # 评审 D-11：本服务自身入站鉴权已放开，不影响 IM_INTERNAL_SERVICE_TOKEN 与白名单。
            await child(scope, receive, send)
            return
        supplied = ""
        for key, value in scope.get("headers", ()):
            if key.decode("latin-1").lower() == "authorization":
                header = value.decode("latin-1")
                if header[:7].lower() == "bearer ":
                    supplied = header[7:].strip()
                break
        if not supplied or not hmac.compare_digest(supplied, settings.mcp_token):
            response = JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
            await response(scope, receive, send)
            return
        await child(scope, receive, send)

    # 处理`lifespan`相关逻辑。
    # English: Handle lifespan operations.
    @asynccontextmanager
    async def lifespan(app):
        async with server.session_manager.run():
            yield

    app = FastAPI(title="Risk Consultation Write MCP", lifespan=lifespan)

    # 处理健康状态相关逻辑。
    # English: Handle health operations.
    @app.get("/health")
    async def health():
        return {"ok": True}

    app.mount("/mcp", authenticated_child)
    return app
