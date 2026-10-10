import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from flowagent import execution_service as service
from flowagent.mcp_edge_client import McpEdgeError
# 固定快照夹具，避免依赖未入库的旧测试文件。
def _digest(snapshot):
    content = {key: value for key, value in snapshot.items() if key != "config_digest"}
    return hashlib.sha256(json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")).hexdigest()

def _snapshot():
    agent_id = "604b6209-30da-4e57-bf93-73c6867cc101"
    revision_id = "704b6209-30da-4e57-bf93-73c6867cc101"
    runtime_id = "804b6209-30da-4e57-bf93-73c6867cc101"
    model_id = "904b6209-30da-4e57-bf93-73c6867cc101"
    model_version_id = "a04b6209-30da-4e57-bf93-73c6867cc101"
    credential_id = "b04b6209-30da-4e57-bf93-73c6867cc101"
    memory_id = "c04b6209-30da-4e57-bf93-73c6867cc101"
    memory_version_id = "d04b6209-30da-4e57-bf93-73c6867cc101"
    snapshot = {
        "schema_version": 3,
        "agent_id": agent_id,
        "revision_id": revision_id,
        "runtime_id": runtime_id,
        "runtime_capability_digest": "e" * 64,
        "config": {
            "runtime_id": runtime_id,
            "enabled": True,
            "model_version_id": model_version_id,
            "memory_profile_version_id": memory_version_id,
            "memory_read": False,
            "memory_write": False,
            "system_prompt": "Answer concisely.",
            "tools": [],
            "skills": [],
        },
        "system_prompt": "Answer concisely.",
        "model": {
            "model_id": model_id,
            "version_id": model_version_id,
            "provider": "openai_compatible",
            "model_name": "model-test",
            "base_url": "https://model.test/v1",
            "credential_id": credential_id,
            "parameters": {"temperature": 0},
        },
        "tools": [],
        "skills": [],
        "memory": {
            "profile_id": memory_id,
            "profile_version_id": memory_version_id,
            "provider": "disabled",
            "credential_id": None,
            "namespace": "",
            "team_id": "",
            "logical_agent_id": "",
            "service_id": "",
            "read": False,
            "write": False,
        },
    }
    snapshot["config_digest"] = _digest(snapshot)
    return snapshot

def _envelope(snapshot=None):
    snapshot = snapshot or _snapshot()
    return {
        "protocol_version": "execution-v1",
        "execution_id": "f04b6209-30da-4e57-bf93-73c6867cc101",
        "trace": {"trace_id": "f04b6209-30da-4e57-bf93-73c6867cc101"},
        "binding_generation": 4,
        "continuity_key": "conversation-1",
        "message": "hello",
        "identity": {"user_id": "user-1"},
        "scene": {"type": "chat"},
        "agent_config": {
            "agent_id": snapshot["agent_id"],
            "agent_type": "workflow",
            "revision_id": snapshot["revision_id"],
            "config_digest": snapshot["config_digest"],
            "snapshot": snapshot,
            "model_api_key": "model-secret",
            "model_credential_generation": 7,
            "model_authorization_url": "https://demo.test/internal/executions/f04b6209-30da-4e57-bf93-73c6867cc101/model-authorization",
            "execution_token": "execution-token-secret",
        },
    }


@pytest.mark.parametrize("streaming", [False, True])
def test_v3_chat_corrects_tool_arguments_after_error(monkeypatch, streaming):
    # 模拟真实故障：错误文档 ID 应反馈给模型，修正后继续回答。
    request = service.ExecutionRequest.model_validate(_envelope())
    snapshot = request.agent_config.snapshot_v3
    snapshot.tools.append(SimpleNamespace(
        remote_name="list_chunks", model_dump=lambda **kwargs: {}))
    calls = []
    histories = []

    class Mcp:
        async def list_tools(self):
            return [{"name": "list_chunks", "input_schema": {"type": "object"}}]

        async def call_tool(self, name, arguments):
            calls.append(arguments["document_id"])
            if arguments["document_id"] == "attachment-id":
                raise McpEdgeError("MCP tool returned an error: list_chunks")
            return "水位达到 2.20 米进入预警"

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            histories.append(list(messages))
            if len(histories) <= 2:
                return AIMessage(content="", tool_calls=[{
                    "id": str(len(histories)), "name": "list_chunks",
                    "args": {"document_id": "attachment-id" if len(histories) == 1 else "knowledge-id"},
                }])
            return AIMessage(content="2.35 米已超过 2.20 米预警阈值。")

        async def astream(self, messages):
            response = await self.ainvoke(messages)
            yield AIMessageChunk(content=response.content, tool_calls=response.tool_calls)

    model, mcp = Model(), Mcp()
    monkeypatch.setattr(service, "_v3_execution_parts", lambda *args: (snapshot, model, mcp, None))
    # 兼容修复前流式入口的独立构造路径，确保先复现真实工具错误。
    if hasattr(service, "AuthorizedChatModel"):
        monkeypatch.setattr(service, "AuthorizedChatModel", lambda *args, **kwargs: model)
    monkeypatch.setattr(service, "_request_llm", lambda *args: model)
    monkeypatch.setattr(service, "McpEdgeClient", lambda *args, **kwargs: mcp)

    async def execute():
        if streaming:
            events = [event async for event in service.stream_managed(request)]
            return events[-1]["content"]
        return (await service.execute_managed(request))["reply"]

    assert "2.35 米" in asyncio.run(execute())
    assert calls == ["attachment-id", "knowledge-id"]
    assert any(isinstance(message, ToolMessage) and "Tool call failed" in message.content
               for message in histories[1])
