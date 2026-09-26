from __future__ import annotations

import hmac
import json
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


TOOL_NAME = "send_consultation_message"
_CLIENT_MESSAGE_ID = re.compile(r"^[A-Za-z0-9:_-]{1,64}$")


def _tool_error(code: str) -> mcp_types.CallToolResult:
    payload = {"status": "error", "error_code": code}
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload, is_error=True,
    )


def build_app(settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    async def list_tools(_ctx, _params):
        return mcp_types.ListToolsResult(tools=[mcp_types.Tool(
            name=TOOL_NAME,
            description="Post an already approved consultation artifact to a configured IM group. This tool cannot choose a group or sender.",
            input_schema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "integer", "minimum": 1},
                    "sender_id": {"type": "integer", "minimum": 1},
                    "content": {"type": "string", "minLength": 1, "maxLength": 100000},
                    "client_message_id": {"type": "string", "pattern": "^[A-Za-z0-9:_-]{1,64}$"},
                    "attachment_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": 20, "uniqueItems": True},
                },
                "required": ["conversation_id", "sender_id", "content", "client_message_id"],
                "additionalProperties": False,
            },
        )])

    async def call_tool(_ctx, params):
        args = params.arguments if isinstance(params.arguments, dict) else {}
        conversation_id, sender_id = args.get("conversation_id"), args.get("sender_id")
        content, client_message_id = args.get("content"), args.get("client_message_id")
        attachment_ids = args.get("attachment_ids", [])
        # 仅转发已上传附件编号，附件所有权和群访问权限继续由 IM 验证。
        if (not isinstance(attachment_ids, list) or len(attachment_ids) > 20
                or any(type(item) is not int or item <= 0 for item in attachment_ids)
                or len(set(attachment_ids)) != len(attachment_ids)):
            return _tool_error("invalid_arguments")
        if (type(conversation_id) is not int or type(sender_id) is not int
                or not isinstance(content, str) or not content.strip()
                or len(content) > 100000 or not isinstance(client_message_id, str)
                or not _CLIENT_MESSAGE_ID.fullmatch(client_message_id)):
            return _tool_error("invalid_arguments")
        if conversation_id not in settings.allowed_conversation_ids:
            return _tool_error("conversation_not_allowed")
        if sender_id not in settings.allowed_sender_ids:
            return _tool_error("sender_not_allowed")
        payload = {
            "sender_id": sender_id, "content": content,
            "message_type": "file" if attachment_ids else "text", "metadata": {},
            "client_message_id": client_message_id,
        }
        if attachment_ids:
            payload["attachment_ids"] = attachment_ids
        try:
            async with httpx.AsyncClient(
                base_url=settings.im_base_url,
                headers={"Authorization": f"Bearer {settings.im_service_token}"},
                timeout=settings.timeout_seconds, transport=transport,
            ) as client:
                response = await client.post(
                    f"/api/v1/internal/im/conversations/{conversation_id}/messages",
                    json=payload,
                    headers={"Idempotency-Key": client_message_id,
                             "X-Request-ID": str(uuid.uuid4())},
                )
        except httpx.HTTPError:
            return _tool_error("im_unavailable")
        if response.status_code in (401, 403):
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

    async def authenticated_child(scope, receive, send):
        if scope["type"] != "http":
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

    @asynccontextmanager
    async def lifespan(app):
        async with server.session_manager.run():
            yield

    app = FastAPI(title="Risk Consultation Write MCP", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ok": True}

    app.mount("/mcp", authenticated_child)
    return app
