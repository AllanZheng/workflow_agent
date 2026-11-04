"""验证托管模式不覆盖模型推理配置，独立模式保留显式配置。"""
import json

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from flowagent.app import build_platform_service, build_runner
from flowagent.clients.model import OpenAICompatibleModel
from flowagent.core.contracts import RunInput, Status
from tests.test_managed_config_v3 import _envelope
from tests.test_platform_v3_adapter import _FakeChatTransport, _final_chunk, _install_offline_stack, _reseal, _text_chunk
from tests.test_repair_context import _envelope_with_context


@pytest.mark.parametrize("stream", [False, True])
def test_managed_sdk_requests_do_not_force_disable_thinking(monkeypatch, stream):
    # 通过正式托管入口捕获 SDK 请求，不能以模型替身伪造传参结论。
    transport, _ = _install_offline_stack(monkeypatch, turns=[[_text_chunk("回答"), _final_chunk()]])
    path = "/execution-v1/respond/stream" if stream else "/execution-v1/respond"
    response = TestClient(build_platform_service()).post(path, json=_envelope())
    assert response.status_code == 200
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1]["kind"] == "done" and events[-1]["metadata"]["status"] == "completed"
    else:
        assert response.json()["metadata"]["status"] == "completed"
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert "extra_body" not in request
    assert "chat_template_kwargs" not in request and "enable_thinking" not in request


@pytest.mark.anyio
async def test_standalone_profile_explicit_thinking_configuration_is_forwarded(monkeypatch):
    # 显式本地 profile 属用户配置；托管默认值修复不得改变它。
    transport = _FakeChatTransport([[_text_chunk("回答"), _final_chunk()]])
    monkeypatch.setattr(OpenAICompatibleModel, "_ensure_client", lambda self: transport.sdk)
    profile = {"model": {"base_url": "http://model.test/v1", "name": "qwen",
                        "api_key": "unused", "parameters": {"extra_body": {
                            "chat_template_kwargs": {"enable_thinking": False}}}}}
    result = await build_runner(profile).run(RunInput(workflow="chat", inputs={"message": "你好"}))
    assert result.status is Status.COMPLETED and result.reply == "回答"
    assert transport.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def real_sdk_requests(monkeypatch):
    # 替换网络而不是 SDK：验证真实序列化后的 OpenAI 兼容 HTTP JSON。
    captured = []
    real_sdk = openai.AsyncOpenAI

    def capture(request):
        captured.append(json.loads(request.content))
        body = ('data: {"id":"r1","object":"chat.completion.chunk","created":1,"model":"m",'
                '"choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":"stop"}]}'
                '\n\ndata: [DONE]\n\n')
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    def construct(**kwargs):
        return real_sdk(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(capture)))

    class Allow:
        def __init__(self, *args, **kwargs):
            pass

        async def authorize(self, **kwargs):
            return {"credential_generation": 7}

    from flowagent.adapters.platform import context as platform_context
    monkeypatch.setattr(openai, "AsyncOpenAI", construct)
    monkeypatch.setattr(platform_context, "DemoModelAuthorizer", Allow)
    return captured


@pytest.mark.parametrize("stream", [False, True])
def test_managed_real_sdk_http_json_preserves_frozen_parameters(real_sdk_requests, stream):
    envelope = _envelope_with_context()
    snapshot = envelope["agent_config"]["snapshot"]
    params = {"temperature": 0.0, "max_tokens": 37, "top_p": 0.75,
              "presence_penalty": 0, "frequency_penalty": 0, "stop": []}
    snapshot["model"]["parameters"] = params
    _reseal(envelope)
    path = "/execution-v1/respond/stream" if stream else "/execution-v1/respond"
    response = TestClient(build_platform_service()).post(path, json=envelope)
    assert response.status_code == 200, response.text
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1]["kind"] == "done"
        receipt = events[-1]["metadata"]["agent_config_applied"]
    else:
        receipt = response.json()["metadata"]["agent_config_applied"]
    assert len(real_sdk_requests) == receipt["model_request_count"] == 1
    payload = real_sdk_requests[0]
    assert "chat_template_kwargs" not in payload and "enable_thinking" not in payload
    assert {key: payload[key] for key in params} == params
    assert receipt["applied_model_parameters"] == params


@pytest.mark.anyio
async def test_standalone_real_sdk_http_json_preserves_explicit_thinking(real_sdk_requests):
    profile = {"model": {"base_url": "https://model.test/v1", "name": "m", "api_key": "unused",
                         "parameters": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}}
    result = await build_runner(profile).run(RunInput(workflow="chat", inputs={"message": "你好"}))
    assert result.status is Status.COMPLETED and result.reply == "OK"
    assert len(real_sdk_requests) == 1
    assert real_sdk_requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
