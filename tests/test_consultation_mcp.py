from contextlib import asynccontextmanager
import asyncio

import httpx
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from services.consultation_mcp.app import build_app
from services.consultation_mcp.config import Settings


def _settings():
    return Settings(
        im_base_url="http://im.internal", im_service_token="service-held",
        mcp_token="platform-worker", allowed_conversation_ids=frozenset({101}),
        allowed_sender_ids=frozenset({9001}),
    )


@asynccontextmanager
async def _session(app, token="platform-worker"):
    async with app.router.lifespan_context(app):
        http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            async with streamable_http_client("http://testserver/mcp/", http_client=http) as streams:
                read, write = streams[:2]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        finally:
            await http.aclose()


def test_consultation_write_mcp_exposes_only_scoped_send_tool_and_forwards_idempotency():
    seen = []

    async def im_handler(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer service-held"
        assert request.headers["idempotency-key"] == "consultation-1"
        assert request.headers["x-request-id"]
        assert request.url.path == "/api/v1/internal/im/conversations/101/messages"
        payload = __import__("json").loads(request.content)
        assert payload["sender_id"] == 9001
        assert payload["content"] == "approved markdown"
        assert payload["client_message_id"] == "consultation-1"
        assert payload["message_type"] == "text"
        return httpx.Response(200, json={"data": {"id": 88}})

    async def exercise():
        app = build_app(_settings(), transport=httpx.MockTransport(im_handler))
        async with _session(app) as session:
            tools = await session.list_tools()
            assert [tool.name for tool in tools.tools] == ["send_consultation_message"]
            return await session.call_tool("send_consultation_message", {
                "conversation_id": 101, "sender_id": 9001, "content": "approved markdown",
                "client_message_id": "consultation-1",
            })

    result = asyncio.run(exercise())

    assert result.is_error is False
    assert result.structured_content["status"] == "sent"
    assert result.structured_content["message_id"] == 88
    assert len(seen) == 1


def test_consultation_write_mcp_denies_non_allowlisted_group_before_im_call():
    calls = []
    app = build_app(_settings(), transport=httpx.MockTransport(
        lambda request: calls.append(request) or httpx.Response(200, json={"data": {"id": 1}})
    ))
    async def exercise():
        async with _session(app) as session:
            return await session.call_tool("send_consultation_message", {
                "conversation_id": 102, "sender_id": 9001, "content": "do not send",
                "client_message_id": "consultation-2",
            })

    result = asyncio.run(exercise())

    assert result.is_error is True
    assert result.structured_content["error_code"] == "conversation_not_allowed"
    assert calls == []


def test_consultation_write_mcp_requires_platform_worker_bearer():
    app = build_app(_settings())
    with TestClient(app) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "method": "initialize", "id": 1})
    assert response.status_code == 401
    assert response.json() == {"error": {"code": "unauthorized"}}


def test_consultation_file_message_forwards_attachment_and_rejects_invalid_ids():
    import json
    calls = []
    async def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"id": 90}})
    async def exercise():
        app = build_app(_settings(), transport=httpx.MockTransport(handler))
        async with _session(app) as session:
            args = {"conversation_id": 101, "sender_id": 9001, "content": "初稿待审批",
                    "client_message_id": "draft-file-1", "attachment_ids": [7]}
            result = await session.call_tool("send_consultation_message", args)
            assert not result.is_error
            # 非正整数和重复编号不得到达 IM。
            for invalid in ([True], [-1], [7, 7], "7"):
                args["attachment_ids"] = invalid
                rejected = await session.call_tool("send_consultation_message", args)
                assert rejected.is_error
    asyncio.run(exercise())
    assert len(calls) == 1
    assert calls[0]["message_type"] == "file"
    assert calls[0]["attachment_ids"] == [7]
