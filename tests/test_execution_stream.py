import asyncio
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

import flowagent.execution_service as execution_service
from flowagent.execution_service import build_execution_app


def _digest(snapshot):
    return hashlib.sha256(json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _envelope(yaml_text, *, history=None):
    snapshot = {
        "schema_version": 1,
        "agent": {"id": "604b6209-30da-4e57-bf93-73c6867cc101", "type": "workflow"},
        "model": {"provider": "openai_compatible", "name": "test-model", "base_url": "http://model.test/v1"},
        "system_prompt": "rules",
        "skills": [{"content": "skill rules"}],
        "workflow": {"yaml": yaml_text},
    }
    return {
        "protocol_version": "execution-v1", "execution_id": "stream-e1",
        "trace": {"trace_id": "not-the-execution-id"}, "binding_generation": 1,
        "continuity_key": "c1", "message": "current", "history": history or [],
        "identity": {"user_id": "u1"}, "scene": {"type": "chat"},
        "agent_config": {
            "agent_id": snapshot["agent"]["id"], "agent_type": "workflow",
            "revision_id": "704b6209-30da-4e57-bf93-73c6867cc101",
            "config_digest": _digest(snapshot), "snapshot": snapshot,
            "model_api_key": "model-secret", "mcp_endpoint": "http://mcp.test/",
        },
    }


_AGENT_YAML = """workflow:
  name: stream-agent
  start: answer
nodes:
  answer:
    type: agent
    output_key: reply
    max_tool_iterations: 2
    next: END
"""


def test_execution_stream_emits_template_delta_matching_done():
    yaml_text = """workflow:
  name: stream-template-only
  start: answer
nodes:
  answer:
    type: template
    template: "Report: {message}\\nCompleted"
    output_key: reply
    next: END
"""

    async def collect():
        return [event async for event in execution_service.stream_managed(
            execution_service.ExecutionRequest.model_validate(_envelope(yaml_text)),
        )]

    events = asyncio.run(collect())

    assert [event["kind"] for event in events] == ["meta", "delta", "done"]
    assert [event["seq"] for event in events] == [0, 1, 2]
    assert events[1]["content"] == events[-1]["content"] == "Report: current\nCompleted"


def test_execution_stream_emits_model_delta_before_terminal_signal(monkeypatch):
    """真实 astream 的首块在模型完成前就必须穿透到服务事件流。"""
    release = asyncio.Event()
    saw_first_chunk = asyncio.Event()
    finished = asyncio.Event()

    class FakeMcp:
        async def list_tools(self): return []

    class FakeBoundModel:
        async def astream(self, messages):
            yield AIMessageChunk(content="early ")
            saw_first_chunk.set()
            await release.wait()
            yield AIMessageChunk(content="reply")
            finished.set()

    class FakeLlm:
        def bind_tools(self, tools): return FakeBoundModel()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    monkeypatch.setattr(execution_service, "McpEdgeClient", lambda endpoint, token: FakeMcp())

    async def exercise():
        events = []

        async def consume():
            async for event in execution_service.stream_managed(
                execution_service.ExecutionRequest.model_validate(_envelope(_AGENT_YAML)), "mcp-token"
            ):
                events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(saw_first_chunk.wait(), timeout=1)
        for _ in range(50):
            if any(event["kind"] == "delta" for event in events):
                break
            await asyncio.sleep(0.01)
        assert [event["content"] for event in events if event["kind"] == "delta"] == ["early "]
        assert not finished.is_set()
        release.set()
        await task
        return events

    events = asyncio.run(exercise())
    assert [event["kind"] for event in events] == ["meta", "delta", "delta", "done"]
    assert [event["seq"] for event in events] == list(range(len(events)))
    assert all(event["trace_id"] == "stream-e1" for event in events)
    assert events[-1]["content"] == "early reply"
    assert events[-1]["metadata"]["agent_config_applied"]["config_digest"] == _envelope(_AGENT_YAML)["agent_config"]["config_digest"]


def test_execution_stream_uses_history_and_only_streams_terminal_reply(monkeypatch):
    yaml_text = """workflow:
  name: stream-final-only
  start: draft
nodes:
  draft:
    type: template
    template: 'internal-draft'
    output_key: draft
    next: answer
  answer:
    type: agent
    output_key: reply
    max_tool_iterations: 2
    next: END
"""
    captured = {}

    class FakeMcp:
        async def list_tools(self): return []

    class FakeBoundModel:
        async def astream(self, messages):
            captured["messages"] = messages
            yield AIMessageChunk(content="final")

    class FakeLlm:
        def bind_tools(self, tools): return FakeBoundModel()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    monkeypatch.setattr(execution_service, "McpEdgeClient", lambda endpoint, token: FakeMcp())

    async def collect():
        return [event async for event in execution_service.stream_managed(
            execution_service.ExecutionRequest.model_validate(_envelope(
                yaml_text,
                history=[
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                ],
            )), "mcp-token"
        )]

    events = asyncio.run(collect())
    assert [event["content"] for event in events if event["kind"] == "delta"] == ["final"]
    assert events[-1]["content"] == "final"
    assert [message.content for message in captured["messages"]] == [
        "rules\n\nskill rules", "old question", "old answer", "current",
    ]


def test_execution_stream_endpoint_auth_and_exception_have_sse_terminal(monkeypatch):
    client = TestClient(build_execution_app("secret-token"))
    assert client.post("/execution-v1/respond/stream", json=_envelope(_AGENT_YAML)).status_code == 401

    def broken_request_llm(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(execution_service, "_request_llm", broken_request_llm)
    response = client.post(
        "/execution-v1/respond/stream", json=_envelope(_AGENT_YAML),
        headers={"Authorization": "Bearer secret-token"},
    )
    events = [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [event["kind"] for event in events] == ["meta", "error"]
    assert [event["seq"] for event in events] == [0, 1]
    assert events[-1]["error_code"] == "execution_failed"


def test_execution_stream_runs_pure_text_agent_without_mcp(monkeypatch):
    """没有 MCP 配置时，纯文本 Agent 绑定空工具列表而不是请求 MCP。"""
    payload = _envelope(_AGENT_YAML)
    payload["agent_config"].pop("mcp_endpoint")
    captured = {}

    class FakeBoundModel:
        async def astream(self, messages):
            yield AIMessageChunk(content="plain reply")

    class FakeLlm:
        def bind_tools(self, tools):
            captured["tools"] = tools
            return FakeBoundModel()

    def unexpected_mcp(*args, **kwargs):
        raise AssertionError("pure text execution must not construct an MCP client")

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    monkeypatch.setattr(execution_service, "McpEdgeClient", unexpected_mcp)

    async def collect():
        request = execution_service.ExecutionRequest.model_validate(payload)
        return [event async for event in execution_service.stream_managed(request)]

    events = asyncio.run(collect())
    assert captured["tools"] == []
    assert [event["kind"] for event in events] == ["meta", "delta", "done"]
    assert events[-1]["content"] == "plain reply"


def test_execution_stream_refuses_partial_mcp_configuration(monkeypatch):
    """声明 MCP endpoint 的配置必须同时提供服务 workload token。"""
    payload = _envelope(_AGENT_YAML)

    class FakeLlm:
        def bind_tools(self, tools): raise AssertionError("must fail before model bind")

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())

    async def consume():
        stream = execution_service.stream_managed(
            execution_service.ExecutionRequest.model_validate(payload)
        )
        first = await anext(stream)
        with pytest.raises(Exception, match="mcp"):
            await anext(stream)
        return first

    assert asyncio.run(consume())["kind"] == "meta"


def test_execution_stream_stops_with_error_after_visible_delta(monkeypatch):
    """首个用户可见 delta 后失败绝不能重试并把部分回复拼进 done。"""
    payload = _envelope(_AGENT_YAML)
    payload["agent_config"]["snapshot"]["workflow"]["yaml"] = _AGENT_YAML.replace(
        "max_tool_iterations: 2", "max_tool_iterations: 2\n    retries: 1"
    )
    payload["agent_config"]["config_digest"] = _digest(payload["agent_config"]["snapshot"])
    calls = []

    class FakeMcp:
        async def list_tools(self): return []

    class FakeBoundModel:
        async def astream(self, messages):
            calls.append("attempt")
            yield AIMessageChunk(content="abandoned")
            raise RuntimeError("upstream dropped")

    class FakeLlm:
        def bind_tools(self, tools): return FakeBoundModel()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    monkeypatch.setattr(execution_service, "McpEdgeClient", lambda endpoint, token: FakeMcp())

    response = TestClient(build_execution_app("secret-token", "mcp-token")).post(
        "/execution-v1/respond/stream", json=payload,
        headers={"Authorization": "Bearer secret-token"},
    )
    events = [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]
    assert calls == ["attempt"]
    assert [event["kind"] for event in events] == ["meta", "delta", "error"]
    assert events[-1]["error_code"] == "execution_failed"
    assert all(event["kind"] != "done" for event in events)


def test_execution_stream_llm_does_not_fallback_after_visible_delta(monkeypatch):
    """终态 LLM 发出文本后失败时，on_error 不能用 fallback 伪造 done。"""
    yaml_text = """workflow:
  name: stream-llm-error
  start: answer
nodes:
  answer:
    type: llm
    prompt: '{message}'
    output_key: reply
    on_error:
      goto: fallback
      set:
        reply: fallback answer
    next: END
  fallback:
    type: template
    template: 'fallback answer'
    output_key: reply
    next: END
"""
    payload = _envelope(yaml_text)

    class FakeLlm:
        async def astream(self, messages):
            yield AIMessageChunk(content="abandoned partial")
            raise RuntimeError("upstream dropped")

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())

    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/respond/stream", json=payload,
        headers={"Authorization": "Bearer secret-token"},
    )
    events = [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]
    assert [event["kind"] for event in events] == ["meta", "delta", "error"]
    assert events[1]["content"] == "abandoned partial"
    assert all(event["kind"] != "done" for event in events)


def test_execution_stream_llm_text_deltas_match_stripped_done_reply(monkeypatch):
    """text parser 的前后空白不得让已发 delta 与 done.reply 不同。"""
    yaml_text = """workflow:
  name: stream-llm-whitespace
  start: answer
nodes:
  answer:
    type: llm
    prompt: '{message}'
    output_key: reply
    next: END
"""

    class FakeLlm:
        async def astream(self, messages):
            yield AIMessageChunk(content="  hello")
            yield AIMessageChunk(content=" world  ")

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/respond/stream", json=_envelope(yaml_text),
        headers={"Authorization": "Bearer secret-token"},
    )
    events = [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]
    delta_reply = "".join(event["content"] for event in events if event["kind"] == "delta")
    assert delta_reply == events[-1]["content"] == "hello world"


def test_execution_stream_llm_retries_before_first_delta(monkeypatch):
    """首个可见文本前的瞬态 LLM 失败仍遵从 node retries。"""
    yaml_text = """workflow:
  name: stream-llm-retry
  start: answer
nodes:
  answer:
    type: llm
    prompt: '{message}'
    output_key: reply
    retries: 1
    retry_delay: 0
    next: END
"""
    attempts = []

    class FakeLlm:
        async def astream(self, messages):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("temporary")
            yield AIMessageChunk(content="recovered")

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/respond/stream", json=_envelope(yaml_text),
        headers={"Authorization": "Bearer secret-token"},
    )
    events = [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]
    assert len(attempts) == 2
    assert [event["kind"] for event in events] == ["meta", "delta", "done"]
    assert events[-1]["content"] == "recovered"


def test_execution_stream_llm_timeout_cancels_model_and_errors(monkeypatch):
    """LLM node.timeout 覆盖整个 async stream，并取消卡住的上游。"""
    yaml_text = """workflow:
  name: stream-llm-timeout
  start: answer
nodes:
  answer:
    type: llm
    prompt: '{message}'
    output_key: reply
    timeout: 0.01
    next: END
"""
    closed = asyncio.Event()

    class FakeLlm:
        async def astream(self, messages):
            try:
                await asyncio.sleep(0.05)
                yield AIMessageChunk(content="late")
            finally:
                closed.set()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/respond/stream", json=_envelope(yaml_text),
        headers={"Authorization": "Bearer secret-token"},
    )
    events = [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]
    assert [event["kind"] for event in events] == ["meta", "error"]
    assert closed.is_set()


def test_execution_stream_agent_timeout_cancels_model_and_errors(monkeypatch):
    """Agent node.timeout 同样覆盖其整个模型 async stream。"""
    yaml_text = _AGENT_YAML.replace("max_tool_iterations: 2", "max_tool_iterations: 2\n    timeout: 0.01")
    closed = asyncio.Event()

    class FakeBoundModel:
        async def astream(self, messages):
            try:
                await asyncio.sleep(0.05)
                yield AIMessageChunk(content="late")
            finally:
                closed.set()

    class FakeLlm:
        def bind_tools(self, tools): return FakeBoundModel()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    payload = _envelope(yaml_text)
    payload["agent_config"].pop("mcp_endpoint")
    response = TestClient(build_execution_app("secret-token")).post(
        "/execution-v1/respond/stream", json=payload,
        headers={"Authorization": "Bearer secret-token"},
    )
    events = [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]
    assert [event["kind"] for event in events] == ["meta", "error"]
    assert closed.is_set()


def test_execution_request_validation_never_reflects_model_api_key():
    payload = _envelope(_AGENT_YAML)
    payload["agent_config"]["model_api_key"] = {"malicious": "do-not-reflect-this-value"}
    client = TestClient(build_execution_app("secret-token"))
    for path in ("/execution-v1/respond", "/execution-v1/respond/stream"):
        response = client.post(path, json=payload, headers={"Authorization": "Bearer secret-token"})
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_request", "message": "invalid request"}}
        assert "do-not-reflect-this-value" not in response.text


def test_execution_stream_cancellation_propagates_and_closes_model_stream(monkeypatch):
    """取消消费者必须取消下游模型 async generator，且不能被转成 SSE error。"""
    payload = _envelope(_AGENT_YAML)
    first_delta = asyncio.Event()
    closed = asyncio.Event()

    class FakeMcp:
        async def list_tools(self): return []

    class FakeBoundModel:
        async def astream(self, messages):
            try:
                yield AIMessageChunk(content="first")
                await asyncio.Event().wait()
            finally:
                closed.set()

    class FakeLlm:
        def bind_tools(self, tools): return FakeBoundModel()

    monkeypatch.setattr(execution_service, "_request_llm", lambda spec, config: FakeLlm())
    monkeypatch.setattr(execution_service, "McpEdgeClient", lambda endpoint, token: FakeMcp())

    async def cancel_consumer():
        async def consume():
            async for event in execution_service.stream_managed(
                execution_service.ExecutionRequest.model_validate(payload), "mcp-token"
            ):
                if event["kind"] == "delta":
                    first_delta.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(first_delta.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closed.wait(), timeout=1)

    asyncio.run(cancel_consumer())
