import asyncio
import hashlib
import json

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
