"""平台 V3 适配层模型预授权行为（V-02）。

覆盖行为编号（对应测试方案 M3-1 授权篇）：
- V02-A1 AuthorizedModel：每次内层模型流之前都先 authorize（顺序：授权→流→授权→流）。
- V02-A2 授权拒绝时内层模型零调用（fail-closed，先授权后消费）。
- V02-A3 平台 chat 全链路（agent 循环工具轮 + 终轮 = 两次模型请求）：
      authorize 次数 >= usage.model_requests；载荷携带 execution_id/model_version_id/generation。
- V02-B1 DemoModelAuthorizer：200 决策原样回传；请求载荷含三要素、
      未配置 credential_generation 时不下发该键；Bearer + X-Runtime-ID 头齐备。
- V02-B2 409 → AuthorizationError("model_credential_rotated")。
- V02-B3 非 2xx（500）→ AuthorizationError。
- V02-B4 传输不可达（ConnectError）→ AuthorizationError。
- V02-B5 200 但非法 JSON 体 → AuthorizationError。
- V02-B6 credential_generation 与本地不一致（体回 0、本地 7）→ AuthorizationError("model_credential_rotated")。
- V02-B7 200 但体形状非法（非 dict / 缺 credential_generation 键）→ AuthorizationError。
- V02-C 请求流：授权一开始即拒绝 → /execution-v1/respond 映射 403 authorization_denied
      错误信封；模型传输层零调用；recovery 只呈现 primary_error（fallback_attempted=False、
      fallback_error=None、final_source="error"，caller 兜底归属）。
- V02-D 请求流：运行中凭据轮换（第二次 authorize 抛 rotated）→ 403 失败；
      鉴权错误不进入单轮恢复重试，模型传输层至多被调用一次。

全程离线：模型传输层用 SimpleNamespace 假客户端，授权端点用 httpx.MockTransport；
平台服务经 build_platform_service 组装（fallback_owner="caller"）。
"""
from __future__ import annotations  # noqa: I001

import json
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.platform import context as platform_context
from flowagent.adapters.platform.capabilities_runtime import (
    AuthorizationError,
    AuthorizedModel,
    DemoModelAuthorizer,
    ModelAuthorizer,
)
from flowagent.app import build_platform_service
from tests.native_helpers import FakeModel
from tests.test_managed_config_v3 import _envelope

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# 授权端点 URL（形状合法即可，MockTransport 不出网）。
AUTH_URL = "https://demo.test/internal/executions/e1/model-authorization"

# 信封固定身份（见 tests/test_managed_config_v3.py）。
EXECUTION_ID = "f04b6209-30da-4e57-bf93-73c6867cc101"
MODEL_VERSION_ID = "a04b6209-30da-4e57-bf93-73c6867cc101"
BINDING_GENERATION = 4


# ---------------------------------------------------------------- 单元替身


class RecordingAuthorizer(ModelAuthorizer):
    """记录 authorize 调用并按脚本放行/拒绝的桩（单元级，实例可达）。"""

    def __init__(self, decisions: list[Any] | None = None, order: list[str] | None = None):
        self.calls: list[dict[str, Any]] = []
        self.decisions = list(decisions or [])  # 'ok' 或 BaseException
        self.order = order

    async def authorize(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.order is not None:
            self.order.append("authorize")
        action = self.decisions.pop(0) if self.decisions else "ok"
        if isinstance(action, BaseException):
            raise action
        return {"credential_generation": 0, "request_id": None}


class OrderedInnerModel:
    """记录 stream 开始顺序并吐 FakeModel 事件脚本的最小内层模型。"""

    def __init__(self, turns: list[list[Any]], order: list[str]):
        self.turns = list(turns)
        self.order = order
        self.stream_starts = 0

    async def stream(self, messages, tools, limits):
        self.stream_starts += 1
        self.order.append("stream")
        for event in self.turns.pop(0):
            yield event


def _authorized_model(inner, authorizer) -> AuthorizedModel:
    return AuthorizedModel(
        inner, authorizer,
        execution_id=EXECUTION_ID, model_version_id=MODEL_VERSION_ID,
        generation=BINDING_GENERATION,
    )


async def _collect(model: Any, *, times: int = 1) -> list[Any]:
    """完整消费 times 次 stream，返回各次事件（授权发生在首个 __anext__）。"""
    collected: list[Any] = []
    for _ in range(times):
        collected.extend([event async for event in model.stream([], [], {})])
    return collected


# ---------------------------------------------------------------- V02-A1/A2 单元


# 验证“`v02`授权`a1``precedes`每次`stream`”场景下的预期行为。
# English: Verify v02 a1 authorize precedes every inner model stream.
async def test_v02_a1_authorize_precedes_every_inner_stream():
    order: list[str] = []
    inner = OrderedInnerModel(
        [FakeModel.tool_turn({"name": "mystery_tool", "arguments": {}}),
         FakeModel.text_turn("完成")], order)
    authorizer = RecordingAuthorizer(order=order)
    model = _authorized_model(inner, authorizer)

    await _collect(model, times=2)

    # 两次模型流各对应一次前置授权；授权严格先于流开始。
    assert order == ["authorize", "stream", "authorize", "stream"]
    assert authorizer.calls == [
        {"execution_id": EXECUTION_ID, "model_version_id": MODEL_VERSION_ID,
         "generation": BINDING_GENERATION},
    ] * 2


# 验证“`v02`授权`a2``denial`阻止`inner``stream`”场景下的预期行为。
# English: Verify v02 a2 authorization denial blocks the inner model stream.
async def test_v02_a2_authorization_denial_blocks_inner_stream():
    order: list[str] = []
    inner = OrderedInnerModel([FakeModel.text_turn("不应到达")], order)
    authorizer = RecordingAuthorizer(
        decisions=[AuthorizationError("model_credential_rotated")], order=order)
    model = _authorized_model(inner, authorizer)

    with pytest.raises(AuthorizationError, match="model_credential_rotated"):
        await _collect(model)

    # fail-closed：授权失败则内层模型从未启动。
    assert order == ["authorize"]
    assert inner.stream_starts == 0


# ---------------------------------------------------------------- V02-B1..B7 单元


def _demo_authorizer(handler, *, credential_generation: int | None = None) -> DemoModelAuthorizer:
    return DemoModelAuthorizer(
        AUTH_URL, "exec-token", "runtime-9",
        credential_generation=credential_generation,
        transport=httpx.MockTransport(handler),
    )


# 验证“`v02`授权`b1``ok``decision`与`request`契约”场景下的预期行为。
# English: Verify v02 b1 ok decision and request contract of demo authorizer.
async def test_v02_b1_demo_authorizer_ok_decision_and_request_contract():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["runtime_id"] = request.headers.get("x-runtime-id")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, json={"credential_generation": 0, "request_id": "req-1"})

    decision = await _demo_authorizer(handler).authorize(
        execution_id="e1", model_version_id="mv-1", generation=4)

    # 决策原样回传；载荷三要素齐备；未配置 credential_generation 时不下发该键。
    assert decision == {"credential_generation": 0, "request_id": "req-1"}
    assert captured["url"] == AUTH_URL
    assert captured["authorization"] == "Bearer exec-token"
    assert captured["runtime_id"] == "runtime-9"
    assert captured["payload"] == {
        "execution_id": "e1", "model_version_id": "mv-1", "generation": 4}


# 验证“`v02`授权`b2``409`映射`credential``rotated`”场景下的预期行为。
# English: Verify v02 b2 409 maps to model_credential_rotated authorization error.
async def test_v02_b2_demo_authorizer_409_maps_credential_rotated():
    authorizer = _demo_authorizer(lambda request: httpx.Response(409, json={}))
    with pytest.raises(AuthorizationError, match="model_credential_rotated"):
        await authorizer.authorize(
            execution_id="e1", model_version_id="mv-1", generation=4)


# 验证“`v02`授权`b3``non2xx`拒绝”场景下的预期行为。
# English: Verify v02 b3 non-2xx status maps to authorization error.
async def test_v02_b3_demo_authorizer_non_2xx_maps_authorization_error():
    authorizer = _demo_authorizer(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(AuthorizationError, match="500"):
        await authorizer.authorize(
            execution_id="e1", model_version_id="mv-1", generation=4)


# 验证“`v02`授权`b4``transport``error`拒绝”场景下的预期行为。
# English: Verify v02 b4 transport error maps to authorization error.
async def test_v02_b4_demo_authorizer_transport_error_maps_authorization_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    authorizer = _demo_authorizer(handler)
    with pytest.raises(AuthorizationError, match="unavailable"):
        await authorizer.authorize(
            execution_id="e1", model_version_id="mv-1", generation=4)


# 验证“`v02`授权`b5``invalid``json`拒绝”场景下的预期行为。
# English: Verify v02 b5 invalid json body maps to authorization error.
async def test_v02_b5_demo_authorizer_invalid_json_body_maps_authorization_error():
    authorizer = _demo_authorizer(lambda request: httpx.Response(200, content=b"not-json{"))
    with pytest.raises(AuthorizationError, match="invalid body"):
        await authorizer.authorize(
            execution_id="e1", model_version_id="mv-1", generation=4)


# 验证“`v02`授权`b6``credential``generation`不一致`rotated`”场景下的预期行为。
# English: Verify v02 b6 credential generation mismatch maps to rotated.
async def test_v02_b6_demo_authorizer_credential_generation_mismatch_maps_rotated():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"credential_generation": 0})

    authorizer = _demo_authorizer(handler, credential_generation=7)
    with pytest.raises(AuthorizationError, match="model_credential_rotated"):
        await authorizer.authorize(
            execution_id="e1", model_version_id="mv-1", generation=4)

    # 配置了 credential_generation 时必须随载荷下发（服务端据此判定轮换）。
    assert captured["payload"]["credential_generation"] == 7


# 验证“`v02`授权`b7``malformed``body`形状拒绝”场景下的预期行为。
# English: Verify v02 b7 malformed body shapes map to authorization error.
async def test_v02_b7_demo_authorizer_malformed_body_shapes_rejected():
    for body in ([1, 2], {"ok": True}):  # 非 dict 形状 / 缺 credential_generation 键
        authorizer = _demo_authorizer(
            lambda request, body=body: httpx.Response(200, json=body))
        with pytest.raises(AuthorizationError, match="invalid body"):
            await authorizer.authorize(
                execution_id="e1", model_version_id="mv-1", generation=4)


# ---------------------------------------------------------------- 请求流替身


class TransportProbe:
    """OpenAI 兼容假传输层：记录 completions 次数并按脚本吐 chunk 轮次。"""

    def __init__(self, turns: list[list[Any]]):
        self.turns = list(turns)
        self.completions = 0

    def client(self) -> Any:
        probe = self

        def create(**request: Any) -> Any:
            probe.completions += 1
            turn = probe.turns.pop(0) if probe.turns else []

            async def agen():
                for chunk in turn:
                    yield chunk

            return agen()  # 普通函数返回异步生成器；客户端兼容 awaitable-or-agen

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _tool_call_chunks() -> list[Any]:
    """一轮工具调用 chunk（未知工具名，触发 agent 修正轮 → 第二次模型请求）。"""
    call = SimpleNamespace(
        index=0, id="c1", function=SimpleNamespace(name="mystery_tool", arguments="{}"))
    return [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=[call]), finish_reason=None)],
            usage=None),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="tool_calls")],
            usage=None),
    ]


def _text_chunks() -> list[Any]:
    """一轮正文 chunk：增量 + 终止（含 usage）。"""
    return [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="文本", tool_calls=None), finish_reason=None)],
            usage=None),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="", tool_calls=None), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=9)),
    ]


class StubPlatformAuthorizer(ModelAuthorizer):
    """请求流桩：context.build_capabilities 内部实例化，故用类级记录与脚本。"""

    calls: ClassVar[list[dict[str, Any]]] = []
    script: ClassVar[list[Any]] = []  # 'ok' 或 BaseException；按 authorize 次序消费

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass  # 构造参数（url/token/runtime_id）对桩无意义

    async def authorize(self, **kwargs: Any) -> dict[str, Any]:
        type(self).calls.append(kwargs)
        action = type(self).script.pop(0) if type(self).script else "ok"
        if isinstance(action, BaseException):
            raise action
        return {"credential_generation": 0, "request_id": None}


def _install_platform(monkeypatch: pytest.MonkeyPatch, probe: TransportProbe) -> None:
    """把假传输层与桩授权器接进平台能力组装（routes 经模块属性运行期取用）。"""
    real_build = platform_context.build_capabilities

    def _patched(request: Any, **kwargs: Any) -> Any:
        return real_build(request, model_transport=probe.client(), **kwargs)

    monkeypatch.setattr(platform_context, "DemoModelAuthorizer", StubPlatformAuthorizer)
    monkeypatch.setattr(platform_context, "build_capabilities", _patched)


def _respond(monkeypatch: pytest.MonkeyPatch, probe: TransportProbe,
             script: list[Any], turns: list[list[Any]]) -> Any:
    """组装平台服务并 POST /execution-v1/respond；返回 TestClient 响应。"""
    _install_platform(monkeypatch, probe)
    StubPlatformAuthorizer.calls = []
    StubPlatformAuthorizer.script = list(script)
    client = TestClient(build_platform_service())
    return client.post(
        "/execution-v1/respond", json=_envelope(),
        headers={"Authorization": "Bearer internal-token"})


# ---------------------------------------------------------------- V02-A3/C/D 请求流


# 验证“`v02`授权`a3``platform``chat`每轮`model``request`前置`authorize`”场景下的预期行为。
# English: Verify v02 a3 platform chat authorizes before every model request.
def test_v02_a3_platform_chat_authorizes_every_model_request(monkeypatch):
    probe = TransportProbe([_tool_call_chunks(), _text_chunks()])
    response = _respond(monkeypatch, probe, script=[], turns=probe.turns)

    # 工具轮（未知工具回填修正）+ 终轮正文 = 两次模型请求，各自前置授权。
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reply"] == "文本"
    assert body["metadata"]["status"] == "completed"
    assert probe.completions == 2
    assert len(StubPlatformAuthorizer.calls) >= body["metadata"]["usage"]["model_requests"]
    assert len(StubPlatformAuthorizer.calls) == 2
    # 授权载荷携带请求身份与绑定代次。
    assert StubPlatformAuthorizer.calls[0] == {
        "execution_id": EXECUTION_ID,
        "model_version_id": MODEL_VERSION_ID,
        "generation": BINDING_GENERATION,
    }


# 验证“`v02`授权`c`拒绝`403``envelope`且`model`零调用”场景下的预期行为。
# English: Verify v02 c denial maps 403 envelope with zero model transport calls.
def test_v02_c_authorization_denial_maps_403_without_model_call(monkeypatch):
    probe = TransportProbe([_text_chunks()])
    response = _respond(
        monkeypatch, probe,
        script=[AuthorizationError("model_credential_rotated")], turns=probe.turns)

    # fail-closed：403 错误信封；执行号回显；模型传输层零调用。
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["error"]["code"] == "authorization_denied"
    assert "model_credential_rotated" in body["error"]["message"]
    assert body["execution_id"] == EXECUTION_ID
    assert probe.completions == 0
    assert len(StubPlatformAuthorizer.calls) == 1

    # caller 兜底归属：recovery 只呈现 primary_error，不进入兜底。
    recovery = body["metadata"]["recovery"]
    assert recovery["primary_error"] == {
        "code": "AUTHORIZATION_DENIED", "message": "model_credential_rotated"}
    assert recovery["fallback_attempted"] is False
    assert recovery["fallback_error"] is None
    assert recovery["final_source"] == "error"
    assert recovery["requested_workflow"] == "chat"


# 验证“`v02`授权`d`运行中`rotated`失败且`model`至多一次”场景下的预期行为。
# English: Verify v02 d mid-run rotated fails 403 with at most one model call.
def test_v02_d_mid_run_rotation_fails_403_without_model_retry(monkeypatch):
    probe = TransportProbe([_tool_call_chunks(), _text_chunks()])
    response = _respond(
        monkeypatch, probe,
        script=["ok", AuthorizationError("model_credential_rotated")], turns=probe.turns)

    # 第一轮放行（工具轮已消费一次传输层），第二轮授权轮换拒绝。
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["error"]["code"] == "authorization_denied"

    # 鉴权错误不在单轮恢复白名单：不重发模型请求，传输层至多一次。
    assert probe.completions == 1
    assert len(StubPlatformAuthorizer.calls) == 2

    recovery = body["metadata"]["recovery"]
    assert recovery["fallback_attempted"] is False
    assert recovery["final_source"] == "error"
