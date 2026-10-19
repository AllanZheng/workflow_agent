"""平台 V3 Skill 装载与执行期 MCP 能力测试（V03，M3-1 原生内核适配层）。

覆盖行为编号：
- V03-a PlatformSkills：装载端点回显全字段（五字段 + body_hash）→ load 返回正文
  （CRLF 原样保留）；二次装载走缓存（传输层只被调一次）；catalog 条目与快照一致。
- V03-b 装载绑定校验：回显 body_version_id / body_hash 与绑定不一致 →
  SkillLoadError(skill_load_binding_mismatch)。
- V03-c 装载失败映射：非 2xx → skill_load_denied；传输异常 → skill_load_unavailable；
  授权 URL 不以 /model-authorization 结尾 → skill_loader_endpoint_invalid。
- V03-d 端到端：信封含 skills + 固定 MCP 工具；模型首轮调 load_skill、次轮终文；
  回执 skills_used 携带 skill 三元组（实际装载去重）；正文进入第二次模型请求
  messages；目录枚举只含绑定 skill_id；未知 skill_id → skill_binding_invalid。
- V03-e 执行期 MCP：身份头（Bearer 工作负载令牌 + 执行/智能体/修订/运行时/绑定代次）；
  固定别名调用附 X-Agent-Tool-ID/X-Agent-Tool-Version-ID；未固定/不可见工具 →
  CAPABILITY_UNAVAILABLE；PLATFORM_TOOLS 无 pin 头可调用；isError → error_code 透传。

离线约定：模型经 SimpleNamespace 流替身；模型授权器为进程内桩；Skill 装载经
httpx.MockTransport、MCP 经会话工厂替身，全程不触网。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.platform import context as platform_context
from flowagent.adapters.platform.capabilities_runtime import (
    ModelAuthorizer,
    SkillLoadError,
    tool_alias,
)
from flowagent.adapters.platform.protocol import ExecutionRequest, snapshot_digest
from flowagent.app import build_platform_service
from tests.test_managed_config_v3 import _envelope, _snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """anyio 插件：只需 asyncio 后端（与本仓库其他原生测试一致）。"""
    return "asyncio"


# ---------------------------------------------------------------- 固定标识

EXECUTION_ID = "f04b6209-30da-4e57-bf93-73c6867cc101"
AGENT_ID = uuid.UUID("604b6209-30da-4e57-bf93-73c6867cc101")
REVISION_ID = uuid.UUID("704b6209-30da-4e57-bf93-73c6867cc101")
RUNTIME_ID = uuid.UUID("804b6209-30da-4e57-bf93-73c6867cc101")
MODEL_VERSION_ID = uuid.UUID("a04b6209-30da-4e57-bf93-73c6867cc101")
TOOL_ID = uuid.UUID("11000000-0000-0000-0000-000000000001")
TOOL_VERSION_ID = uuid.UUID("12000000-0000-0000-0000-000000000001")
SERVER_VERSION_ID = uuid.UUID("13000000-0000-0000-0000-000000000001")
SKILL_ID = uuid.UUID("31000000-0000-0000-0000-000000000001")
METADATA_VERSION_ID = uuid.UUID("32000000-0000-0000-0000-000000000001")
BODY_VERSION_ID = uuid.UUID("33000000-0000-0000-0000-000000000001")

# 正文带 CRLF：装载校验按 CRLF 归一后计算 sha256，正文本身原样返回给模型。
BODY_TEXT = "# 巡检制度\r\n每日巡检一次并记录异常。\r\n"


def _normalized_sha256(text: str) -> str:
    """装载协议的正文摘要：CRLF/CR 归一为 LF 后取 sha256。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


BODY_HASH = _normalized_sha256(BODY_TEXT)

EXECUTION_TOKEN = "execution-token-secret"  # MCP 工作负载与 Skill 装载共用的信封令牌


# ---------------------------------------------------------------- 信封构造

def _tool_entry() -> dict[str, Any]:
    """快照固定工具条目（schema_digest 满足 64 位十六进制约束）。"""
    return {
        "tool_id": str(TOOL_ID),
        "tool_version_id": str(TOOL_VERSION_ID),
        "server_version_id": str(SERVER_VERSION_ID),
        "remote_name": "search_orders",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        "description": "检索订单",
        "schema_digest": "a" * 64,
    }


def _skill_entry() -> dict[str, Any]:
    """快照 Skill 条目（与装载回显的绑定字段一一对应）。"""
    return {
        "skill_id": str(SKILL_ID),
        "metadata_version_id": str(METADATA_VERSION_ID),
        "body_version_id": str(BODY_VERSION_ID),
        "name": "巡检制度",
        "summary": "设备巡检制度检索指南",
    }


def _skills_envelope(*, tools: bool = True, skills: bool = True) -> dict[str, Any]:
    """带固定工具与 Skill 的 V3 信封；快照变更后按摘要规则重算 config_digest。"""
    snapshot = _snapshot()
    snapshot["tools"] = [_tool_entry()] if tools else []
    snapshot["skills"] = [_skill_entry()] if skills else []
    snapshot["config"]["tools"] = [
        {"tool_id": t["tool_id"], "tool_version_id": t["tool_version_id"]}
        for t in snapshot["tools"]
    ]
    snapshot["config"]["skills"] = [
        {"skill_id": s["skill_id"], "metadata_version_id": s["metadata_version_id"],
         "body_version_id": s["body_version_id"]}
        for s in snapshot["skills"]
    ]
    # 摘要规则：任何快照变更都必须重算，且快照与信封两处 config_digest 同步。
    snapshot["config_digest"] = snapshot_digest(snapshot)
    envelope = _envelope(snapshot)
    envelope["agent_config"]["mcp_endpoint"] = "http://mcp.test/edge"
    envelope["agent_config"]["tool_access_token"] = "tool-access-secret"
    return envelope


def _caps(envelope: dict[str, Any], **kwargs: Any):
    """解析信封并按注入替身组装能力束。"""
    request = ExecutionRequest.model_validate(envelope)
    return request, platform_context.build_capabilities(request, **kwargs)


# ---------------------------------------------------------------- 替身构造

def _skill_loader_transport(calls: list, *, body_version_id: str | None = None,
                            body_hash: str | None = None, status_code: int = 200):
    """Skill 装载端点替身：回显五字段 + 规范化 sha256；记录请求供契约断言。"""
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status_code, json={
            "skill_id": str(SKILL_ID),
            "metadata_version_id": str(METADATA_VERSION_ID),
            "body_version_id": str(BODY_VERSION_ID if body_version_id is None else body_version_id),
            "name": "巡检制度",
            "summary": "设备巡检制度检索指南",
            "body_text": BODY_TEXT,
            "body_hash": BODY_HASH if body_hash is None else body_hash,
        })

    return httpx.MockTransport(handler)


def _failing_skill_transport() -> httpx.MockTransport:
    """传输层直接抛连接错误的装载替身（skill_load_unavailable 路径）。"""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    return httpx.MockTransport(handler)


def _chunk(delta: Any, *, finish_reason: str | None = None,
           usage: Any = None) -> SimpleNamespace:
    """单条 OpenAI 流式 chunk 替身（choices 单项）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=usage)


def _tool_call_chunk(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    """携带一个工具调用的 chunk（delta.tool_calls 单项聚合完整）。"""
    delta = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            index=0, id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments))],
    )
    return _chunk(delta)


def _model_transport(turns: list[list[SimpleNamespace]]):
    """SimpleNamespace 树模拟 AsyncOpenAI：create(**req) 按轮吐 chunk 并记录请求。"""
    requests: list[dict[str, Any]] = []
    script = list(turns)

    def create(**req: Any):
        requests.append(req)
        chunks = script.pop(0)

        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()  # 客户端容忍可等待或异步生成器，这里直接返回异步生成器

    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return transport, requests


def _mcp_factory(calls: list, *, visible: tuple[str, ...] = (),
                 tool_results: dict[str, Any] | None = None):
    """执行期 MCP 会话替身：记录 endpoint/headers/timeout 与调用参数。

    visible：服务端目录声明可见的工具名（固定别名 + 平台原生工具）。
    tool_results：按名脚本化返回值（异常原样抛出，SimpleNamespace 原样返回）。
    """
    @asynccontextmanager
    async def factory(endpoint: str, headers: dict[str, str], timeout_seconds: float):
        calls.append({"endpoint": endpoint, "headers": dict(headers),
                      "timeout_seconds": timeout_seconds})

        async def list_tools():
            return SimpleNamespace(tools=[
                SimpleNamespace(name=name, description="平台目录项",
                                inputSchema={"type": "object"})
                for name in visible
            ])

        async def call_tool(name: str, arguments: dict[str, Any]):
            calls.append({"tool": name, "arguments": dict(arguments)})
            scripted = (tool_results or {}).get(name)
            if isinstance(scripted, BaseException):
                raise scripted
            if scripted is not None:
                return scripted
            return SimpleNamespace(
                is_error=False,
                structured_content={"ok": True, "tool": name},
                content=[SimpleNamespace(text="raw-text-fallback")],
            )

        yield SimpleNamespace(list_tools=list_tools, call_tool=call_tool)

    return factory


def _install_stub_authorizer(monkeypatch) -> list[dict[str, Any]]:
    """把 DemoModelAuthorizer 换成进程内桩：记录每次预授权，离线不触网。"""
    calls: list[dict[str, Any]] = []

    class _StubAuthorizer(ModelAuthorizer):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__()

        async def authorize(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(dict(kwargs))
            return {"credential_generation": 0, "request_id": None}

    monkeypatch.setattr(platform_context, "DemoModelAuthorizer", _StubAuthorizer)
    return calls


# ---------------------------------------------------------------- V03-a 装载正文/缓存/目录

# 验证“V03-a echo 全字段回显装载正文、二次装载走缓存且目录与快照一致”场景下的预期行为。
async def test_v03_a_skill_load_returns_body_caches_and_catalog_matches_snapshot():
    calls: list = []
    _, caps = _caps(_skills_envelope(tools=False),
                    skill_transport=_skill_loader_transport(calls))
    skills = caps.skills

    # 目录与快照一致（字符串化 UUID）。
    assert skills.catalog() == [{
        "skill_id": str(SKILL_ID),
        "name": "巡检制度",
        "summary": "设备巡检制度检索指南",
        "metadata_version_id": str(METADATA_VERSION_ID),
        "body_version_id": str(BODY_VERSION_ID),
    }]

    body = await skills.load(str(SKILL_ID))
    assert body == BODY_TEXT  # CRLF 原样保留，不做归一改写
    assert await skills.load(str(SKILL_ID)) == BODY_TEXT  # 缓存命中
    assert len(calls) == 1  # 传输层只被调一次

    # 装载请求契约：URL 由授权 URL 派生 + Bearer 信封令牌 + 运行时头 + 固定版本载荷。
    sent = calls[0]
    assert str(sent.url) == (
        f"https://demo.test/internal/executions/{EXECUTION_ID}"
        f"/skills/{SKILL_ID}/load"
    )
    assert sent.headers["Authorization"] == f"Bearer {EXECUTION_TOKEN}"
    assert sent.headers["X-Runtime-ID"] == str(RUNTIME_ID)
    assert json.loads(sent.content) == {
        "generation": 4,
        "metadata_version_id": str(METADATA_VERSION_ID),
        "body_version_id": str(BODY_VERSION_ID),
    }


# ---------------------------------------------------------------- V03-b 绑定校验

# 验证“V03-b 回显 body_version_id/body_hash 与绑定不一致 → skill_load_binding_mismatch”场景下的预期行为。
@pytest.mark.parametrize("field,value", [
    ("body_version_id", "34000000-0000-0000-0000-000000000009"),
    ("body_hash", "f" * 64),
], ids=["body_version_mismatch", "body_hash_mismatch"])
async def test_v03_b_binding_mismatch_rejected(field: str, value: str):
    calls: list = []
    _, caps = _caps(
        _skills_envelope(tools=False),
        skill_transport=_skill_loader_transport(calls, **{field: value}))

    with pytest.raises(SkillLoadError) as excinfo:
        await caps.skills.load(str(SKILL_ID))
    assert excinfo.value.code == "skill_load_binding_mismatch"
    assert len(calls) == 1  # 校验失败发生在装载端点返回之后（一次真实调用）


# ---------------------------------------------------------------- V03-c 装载失败映射

# 验证“V03-c 装载端点非 2xx → skill_load_denied”场景下的预期行为。
async def test_v03_c_non_2xx_maps_to_load_denied():
    _, caps = _caps(_skills_envelope(tools=False),
                    skill_transport=_skill_loader_transport([], status_code=403))
    with pytest.raises(SkillLoadError) as excinfo:
        await caps.skills.load(str(SKILL_ID))
    assert excinfo.value.code == "skill_load_denied"


# 验证“V03-c 装载传输异常 → skill_load_unavailable”场景下的预期行为。
async def test_v03_c_transport_error_maps_to_load_unavailable():
    _, caps = _caps(_skills_envelope(tools=False),
                    skill_transport=_failing_skill_transport())
    with pytest.raises(SkillLoadError) as excinfo:
        await caps.skills.load(str(SKILL_ID))
    assert excinfo.value.code == "skill_load_unavailable"


# 验证“V03-c 授权 URL 不以 /model-authorization 结尾 → skill_loader_endpoint_invalid”场景下的预期行为。
async def test_v03_c_invalid_loader_endpoint_rejected():
    envelope = _skills_envelope(tools=False)
    envelope["agent_config"]["model_authorization_url"] = (
        f"https://demo.test/internal/executions/{EXECUTION_ID}/authorize")
    _, caps = _caps(envelope)
    with pytest.raises(SkillLoadError) as excinfo:
        await caps.skills.load(str(SKILL_ID))
    assert excinfo.value.code == "skill_loader_endpoint_invalid"


# ---------------------------------------------------------------- V03-d 端到端

# 验证“V03-d 信封含 Skill+固定工具：模型先 load_skill 再终文，回执 skills_used 与第二轮 messages 携带正文”场景下的预期行为。
def test_v03_d_end_to_end_skill_load_receipt_and_model_context(monkeypatch):
    envelope = _skills_envelope()
    skill_calls: list = []
    skill_transport = _skill_loader_transport(skill_calls)
    mcp_calls: list = []
    alias = tool_alias(str(TOOL_ID))
    mcp_factory = _mcp_factory(mcp_calls, visible=(alias, "platform_resolve_group"))

    # 两轮脚本：首轮 load_skill，次轮终文；usage 经 chunk 透出。
    turns = [
        [
            _tool_call_chunk("call-1", "load_skill",
                             json.dumps({"skill_id": str(SKILL_ID)})),
            _chunk(SimpleNamespace(content=None, tool_calls=None),
                   finish_reason="tool_calls",
                   usage=SimpleNamespace(prompt_tokens=9, completion_tokens=9)),
        ],
        [
            _chunk(SimpleNamespace(content="制度要点已确认。", tool_calls=None)),
            _chunk(SimpleNamespace(content=None, tool_calls=None),
                   finish_reason="stop",
                   usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5)),
        ],
    ]
    model_transport, model_requests = _model_transport(turns)

    auth_calls = _install_stub_authorizer(monkeypatch)
    original_build = platform_context.build_capabilities

    # routes 经模块属性调用 build_capabilities：替换模块属性注入替身传输层。
    def _inject(request, **kwargs):
        return original_build(
            request,
            model_transport=model_transport,
            skill_transport=skill_transport,
            mcp_session_factory=mcp_factory,
        )

    monkeypatch.setattr(platform_context, "build_capabilities", _inject)

    client = TestClient(build_platform_service(internal_token="internal-secret"))
    response = client.post(
        "/execution-v1/respond", json=envelope,
        headers={"Authorization": "Bearer internal-secret"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["execution_id"] == EXECUTION_ID
    assert body["reply"] == "制度要点已确认。"

    # 回执：skills_used 携带 skill 三元组（实际装载，目录不算使用）。
    applied = body["metadata"]["agent_config_applied"]
    assert applied["schema_version"] == 3
    assert applied["skills_used"] == [{
        "skill_id": str(SKILL_ID),
        "metadata_version_id": str(METADATA_VERSION_ID),
        "body_version_id": str(BODY_VERSION_ID),
    }]
    assert body["metadata"]["usage"]["skill_loads"] == 1

    # 目录枚举只含绑定 skill_id；正文进入第二次模型请求的 tool 消息。
    load_schema = [t for t in model_requests[0]["tools"]
                   if t["function"]["name"] == "load_skill"][0]
    assert (load_schema["function"]["parameters"]["properties"]["skill_id"]["enum"]
            == [str(SKILL_ID)])
    assert any(t["function"]["name"] == alias for t in model_requests[0]["tools"])
    assert any(
        msg.get("role") == "tool" and BODY_TEXT in str(msg.get("content") or "")
        for msg in model_requests[1]["messages"]
    )

    # 预授权：每次模型请求前各一次（两轮），参数来自快照与绑定代次。
    assert auth_calls == [{
        "execution_id": EXECUTION_ID,
        "model_version_id": str(MODEL_VERSION_ID),
        "generation": 4,
    }, {
        "execution_id": EXECUTION_ID,
        "model_version_id": str(MODEL_VERSION_ID),
        "generation": 4,
    }]

    # 执行期身份头：Bearer 工作负载令牌（信封 execution_token）+ 执行/绑定代次。
    headers = mcp_calls[0]["headers"]
    assert headers["Authorization"] == f"Bearer {EXECUTION_TOKEN}"
    assert headers["X-Execution-ID"] == EXECUTION_ID
    assert headers["X-Binding-Generation"] == "4"
    assert skill_calls and str(skill_calls[0].url).endswith(
        f"/skills/{SKILL_ID}/load")


# 验证“V03-d 未知 skill_id 不在目录枚举且直接装载被拒（skill_binding_invalid）”场景下的预期行为。
async def test_v03_d_unknown_skill_id_rejected_as_binding_invalid():
    _, caps = _caps(_skills_envelope(tools=False))
    with pytest.raises(SkillLoadError) as excinfo:
        await caps.skills.load("44000000-0000-0000-0000-000000000009")
    assert excinfo.value.code == "skill_binding_invalid"


# ---------------------------------------------------------------- V03-e 执行期 MCP

# 验证“V03-e list 发送身份头、可见别名与平台原生工具入目录”场景下的预期行为。
async def test_v03_e_list_sends_identity_headers_and_visible_alias():
    calls: list = []
    alias = tool_alias(str(TOOL_ID))
    factory = _mcp_factory(calls, visible=(alias, "platform_resolve_group"))
    _, caps = _caps(_skills_envelope(), mcp_session_factory=factory)

    specs = {spec.name: spec for spec in await caps.tools.list()}

    # 身份头全集：Bearer 工作负载令牌 + 执行/智能体/修订/运行时/绑定代次。
    headers = calls[0]["headers"]
    assert headers["Authorization"] == f"Bearer {EXECUTION_TOKEN}"
    assert headers["X-Execution-ID"] == EXECUTION_ID
    assert headers["X-Agent-ID"] == str(AGENT_ID)
    assert headers["X-Agent-Revision-ID"] == str(REVISION_ID)
    assert headers["X-Runtime-ID"] == str(RUNTIME_ID)
    assert headers["X-Binding-Generation"] == "4"
    assert calls[0]["endpoint"] == "http://mcp.test/edge"

    # 固定别名以服务端目录可见为门；PLATFORM_TOOLS 透传进目录。
    assert alias in specs
    assert specs[alias].description == "检索订单"
    assert specs[alias].input_schema == {"type": "object", "properties": {"q": {"type": "string"}}}
    assert "platform_resolve_group" in specs


# 验证“V03-e 固定别名调用附 X-Agent-Tool-ID/版本头且优先结构化结果”场景下的预期行为。
async def test_v03_e_pinned_call_sends_tool_headers():
    calls: list = []
    alias = tool_alias(str(TOOL_ID))
    factory = _mcp_factory(calls, visible=(alias,))
    _, caps = _caps(_skills_envelope(), mcp_session_factory=factory)
    await caps.tools.list()

    result = await caps.tools.call(alias, {"q": "异常"})

    assert result.status == "success"
    assert result.value == {"ok": True, "tool": alias}  # 结构化结果优先
    # 调用会话头 = 身份头 + 固定版本两件套；调用名仍为别名。
    call_headers = [c["headers"] for c in calls if "headers" in c][-1]
    assert call_headers["X-Agent-Tool-ID"] == str(TOOL_ID)
    assert call_headers["X-Agent-Tool-Version-ID"] == str(TOOL_VERSION_ID)
    assert calls[-1] == {"tool": alias, "arguments": {"q": "异常"}}


# 验证“V03-e 未固定/不可见工具调用 → CAPABILITY_UNAVAILABLE（not pinned）”场景下的预期行为。
async def test_v03_e_unpinned_or_invisible_tool_call_rejected():
    calls: list = []
    alias = tool_alias(str(TOOL_ID))
    # 服务端目录为空：已固定别名不可见；未固定名从未在册。
    factory = _mcp_factory(calls, visible=())
    _, caps = _caps(_skills_envelope(), mcp_session_factory=factory)
    await caps.tools.list()

    invisible = await caps.tools.call(alias, {})
    assert invisible.status == "error"
    assert invisible.error_code == "CAPABILITY_UNAVAILABLE"
    assert "not pinned" in (invisible.message or "")

    unknown = await caps.tools.call("never_heard_of", {})
    assert unknown.status == "error"
    assert unknown.error_code == "CAPABILITY_UNAVAILABLE"


# 验证“V03-e 平台原生工具（PLATFORM_TOOLS）无 pin 头可调用”场景下的预期行为。
async def test_v03_e_platform_tool_callable_without_pin_headers():
    calls: list = []
    factory = _mcp_factory(calls, visible=("platform_resolve_group",))
    _, caps = _caps(_skills_envelope(), mcp_session_factory=factory)

    specs = {spec.name: spec for spec in await caps.tools.list()}
    assert "platform_resolve_group" in specs

    result = await caps.tools.call("platform_resolve_group", {"conversation_id": 6})

    assert result.status == "success"
    call_headers = [c["headers"] for c in calls if "headers" in c][-1]
    assert "X-Agent-Tool-ID" not in call_headers  # 平台工具由服务端目录授权
    assert "X-Agent-Tool-Version-ID" not in call_headers
    assert call_headers["Authorization"] == f"Bearer {EXECUTION_TOKEN}"


# 验证“V03-e 服务端 isError → 结构化 error_code 透传”场景下的预期行为。
async def test_v03_e_is_error_maps_structured_error_code():
    alias = tool_alias(str(TOOL_ID))
    error_result = SimpleNamespace(
        is_error=True,
        structured_content={"error_code": "ORDER_DENIED"},
        content=[SimpleNamespace(text="denied")],
    )
    factory = _mcp_factory([], visible=(alias,), tool_results={alias: error_result})
    _, caps = _caps(_skills_envelope(), mcp_session_factory=factory)
    await caps.tools.list()

    result = await caps.tools.call(alias, {"q": "x"})

    assert result.status == "error"
    assert result.error_code == "ORDER_DENIED"
