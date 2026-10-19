from contextlib import asynccontextmanager
import asyncio
import inspect
import json

import httpx
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import services.consultation_mcp.config as config_module
from services.consultation_mcp.app import build_app
from services.consultation_mcp.config import Settings

SUBJECT = "11111111-2222-4333-8444-555555555555"


# 构造配置项；会话白名单默认只放行测试群 101。
# English: Build settings; the conversation allowlist defaults to group 101.
def _settings(conversations=frozenset({101})):
    return Settings(
        im_base_url="http://im.internal", im_service_token="service-held",
        mcp_token="platform-worker", allowed_conversation_ids=conversations,
    )


# 构造一次完整的发送参数（发送者字段来自运行的 sender_snapshot）。
# English: Build full send arguments.
def _args(**overrides):
    args = {
        "conversation_id": 101, "content": "approved markdown", "client_message_id": "consultation-1",
        "sender_subject_id": SUBJECT, "sender_binding_generation": 3,
        "execution_id": "stage-run-1", "actual_agent_id": "agent-1", "agent_config_id": "rev-1",
    }
    args.update(overrides)
    return args


# 处理会话相关逻辑。
# English: Handle session operations.
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


# CMP-01：请求发往 /agent-messages，幂等键等于 client_message_id，字段完整。
# English: CMP-01 forwarding to agent-messages with idempotency.
def test_forwards_to_agent_messages_with_idempotency():
    seen = []

    async def im_handler(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer service-held"
        assert request.headers["idempotency-key"] == "consultation-1"
        assert request.headers["x-request-id"]
        assert request.url.path == "/api/v1/internal/im/conversations/101/agent-messages"
        payload = json.loads(request.content)
        assert payload == {
            "sender_subject_id": SUBJECT, "sender_binding_generation": 3,
            "content": "approved markdown", "message_type": "text",
            "client_message_id": "consultation-1", "execution_id": "stage-run-1",
            "actual_agent_id": "agent-1", "agent_config_id": "rev-1",
        }
        return httpx.Response(200, json={"data": {"id": 88}})

    async def exercise():
        app = build_app(_settings(), transport=httpx.MockTransport(im_handler))
        async with _session(app) as session:
            tools = await session.list_tools()
            assert [tool.name for tool in tools.tools] == ["send_consultation_message"]
            return await session.call_tool("send_consultation_message", _args())

    result = asyncio.run(exercise())
    assert result.is_error is False
    assert result.structured_content["status"] == "sent"
    assert result.structured_content["message_id"] == 88
    assert len(seen) == 1


# CMP-02：不再接受整数 sender_id；配置里没有发送者白名单。
# English: CMP-02 no integer sender or sender allowlist.
def test_integer_sender_is_rejected_and_sender_allowlist_is_gone():
    calls = []
    app = build_app(_settings(), transport=httpx.MockTransport(
        lambda request: calls.append(request) or httpx.Response(200, json={"data": {"id": 1}})
    ))

    async def exercise():
        async with _session(app) as session:
            legacy = _args(sender_id=9001)
            missing = _args()
            del missing["sender_subject_id"]
            bad_subject = _args(sender_subject_id="not-a-uuid")
            bad_generation = _args(sender_binding_generation=True)
            return [await session.call_tool("send_consultation_message", item)
                    for item in (legacy, missing, bad_subject, bad_generation)]

    results = asyncio.run(exercise())
    assert all(item.is_error and item.structured_content["error_code"] == "invalid_arguments"
               for item in results)
    assert calls == []
    assert "allowed_sender_ids" not in Settings.__dataclass_fields__
    assert "CONSULTATION_ALLOWED_SENDER_IDS" not in inspect.getsource(config_module)


# CMP-03：IM 403 的已知拒绝原因原样透传，未知 403 按鉴权失败处理。
# English: CMP-03 pass IM rejection codes through.
def test_im_rejection_codes_are_passed_through():
    messages = iter(["AGENT_MUTED: 群组已开启全员禁言", "AGENT_NOT_CONVERSATION_MEMBER: x",
                     "ATTACHMENT_OWNER_MISMATCH: y", "AGENT_ACCOUNT_INVALID: z", "其他原因"])

    async def handler(request):
        return httpx.Response(403, json={"code": 40301, "message": next(messages), "data": None})

    async def exercise():
        app = build_app(_settings(), transport=httpx.MockTransport(handler))
        async with _session(app) as session:
            return [(await session.call_tool("send_consultation_message", _args(
                client_message_id=f"m-{i}"))).structured_content["error_code"] for i in range(5)]

    assert asyncio.run(exercise()) == [
        "AGENT_MUTED", "AGENT_NOT_CONVERSATION_MEMBER", "ATTACHMENT_OWNER_MISMATCH",
        "AGENT_ACCOUNT_INVALID", "im_unauthorized"]


# 配置了会话白名单时，名单外会话在调用 IM 之前被拒绝；留空则不限制（私聊场景）。
# English: Conversation allowlist is optional.
def test_conversation_allowlist_is_optional():
    calls = []
    transport = httpx.MockTransport(
        lambda request: calls.append(request) or httpx.Response(200, json={"data": {"id": 1}}))

    async def exercise():
        restricted = build_app(_settings(), transport=transport)
        async with _session(restricted) as session:
            denied = await session.call_tool("send_consultation_message", _args(conversation_id=102))
        open_app = build_app(_settings(frozenset()), transport=transport)
        async with _session(open_app) as session:
            allowed = await session.call_tool("send_consultation_message", _args(conversation_id=102))
        return denied, allowed

    denied, allowed = asyncio.run(exercise())
    assert denied.is_error and denied.structured_content["error_code"] == "conversation_not_allowed"
    assert allowed.is_error is False and len(calls) == 1


# 入站必须携带 platform worker 令牌。
# English: Platform worker bearer is required.
def test_consultation_write_mcp_requires_platform_worker_bearer():
    app = build_app(_settings())
    with TestClient(app) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "method": "initialize", "id": 1})
    assert response.status_code == 401
    assert response.json() == {"error": {"code": "unauthorized"}}


# 文件消息转发附件与 reply_to_id；非法附件编号不得到达 IM。
# English: File messages forward attachments and reply_to_id.
def test_file_message_forwards_attachment_and_reply_to():
    calls = []

    async def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"id": 90}})

    async def exercise():
        app = build_app(_settings(), transport=httpx.MockTransport(handler))
        async with _session(app) as session:
            args = _args(content="初稿待审批", client_message_id="draft-file-1",
                         attachment_ids=[7], reply_to_id=55)
            result = await session.call_tool("send_consultation_message", args)
            assert not result.is_error
            for invalid in ([True], [-1], [7, 7], "7"):
                args["attachment_ids"] = invalid
                assert (await session.call_tool("send_consultation_message", args)).is_error
            args["attachment_ids"] = [7]
            args["reply_to_id"] = 0
            assert (await session.call_tool("send_consultation_message", args)).is_error

    asyncio.run(exercise())
    assert len(calls) == 1
    assert calls[0]["message_type"] == "file"
    assert calls[0]["attachment_ids"] == [7] and calls[0]["reply_to_id"] == 55
