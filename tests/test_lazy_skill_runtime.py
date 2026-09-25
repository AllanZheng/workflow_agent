from langchain_core.messages import AIMessage, ToolMessage
import pytest

from flowagent.agent_runtime import ManagedAgentRuntime
from flowagent.execution_service import MemoryContextV1, _managed_system_prompt


class _Model:
    def __init__(self, script, calls):
        self.script = script
        self.calls = calls

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.script.pop(0)


class _LLM:
    def __init__(self, script):
        self.script = script
        self.calls = []

    def bind_tools(self, schemas):
        self.schemas = schemas
        return _Model(self.script, self.calls)


class _McpClient:
    async def list_tools(self):
        return [{
            "name": "platform_load_skill",
            "description": "Load pinned Skill body",
            "input_schema": {"type": "object"},
        }]

    async def call_tool(self, name, arguments):
        assert name == "platform_load_skill"
        assert arguments == {"skill_key": "order-helper"}
        return {"result": {
            "skill_key": "order-helper",
            "metadata_version_id": "metadata-v1",
            "body_version_id": "body-v1",
            "content_hash": "a" * 64,
            "content": "private instructions: normalize the customer order ID",
            "tool_names": [],
        }}


def test_v2_prompt_has_summary_only_and_model_receives_body_after_tool_message():
    snapshot = {
        "schema_version": 2,
        "system_prompt": "Follow the task.",
        "skills": [{
            "key": "order-helper", "name": "Order helper",
            "summary": "Normalize and inspect order identifiers.",
            "body_version_id": "body-v1",
        }],
    }
    prompt = _managed_system_prompt(snapshot)
    body = "private instructions: normalize the customer order ID"
    assert "Order helper" in prompt
    assert body not in prompt

    llm = _LLM([
        AIMessage(content="", tool_calls=[{
            "name": "platform_load_skill",
            "args": {"skill_key": "order-helper"},
            "id": "load-1",
        }]),
        AIMessage(content="I normalized the order ID."),
    ])
    runtime = ManagedAgentRuntime(llm, _McpClient())

    import asyncio
    reply = asyncio.run(runtime.run("check order", prompt, 4))

    assert reply == "I normalized the order ID."
    assert len(llm.calls) == 2
    first_input = "\n".join(str(item.content) for item in llm.calls[0])
    second_tool_message = next(item for item in llm.calls[1] if isinstance(item, ToolMessage))
    assert body not in first_input
    assert body in str(second_tool_message.content)
    assert runtime.loaded_skills == [{
        "skill_key": "order-helper",
        "metadata_version_id": "metadata-v1",
        "body_version_id": "body-v1",
        "content_hash": "a" * 64,
    }]


def test_v2_prompt_receives_memory_context_as_a_separate_platform_section():
    prompt = _managed_system_prompt(
        {"schema_version": 2, "system_prompt": "Follow policy."},
        MemoryContextV1(
            status="ok", rendered_text="Use the user's preferred date format.",
            included_ids=["m1"],
        ),
    )
    assert "[Platform memory context]" in prompt
    assert "Use the user's preferred date format." in prompt


def test_v2_skill_loader_requires_a_reachable_agent_node():
    from flowagent.execution_service import (
        ExecutionRequest, ManagedAgentConfig, _digest, execute_managed,
    )
    from flowagent.errors import ConfigError
    import asyncio
    import uuid

    snapshot = {
        "schema_version": 2,
        "agent": {"id": str(uuid.uuid4()), "type": "workflow", "name": "A"},
        "adapter": {"endpoint": "http://workflow"},
        "workflow": {
            "yaml": "workflow: {name: no-agent, start: finish}\nnodes:\n"
                    "  finish: {type: template, output_key: reply, template: done, next: END}\n",
        },
        "skills": [{
            "skill_id": 1, "key": "s", "metadata_version_id": str(uuid.uuid4()),
            "body_version_id": str(uuid.uuid4()), "name": "S", "summary": "Summary",
            "metadata_hash": "a" * 64, "content_hash": "b" * 64,
        }],
        "memory": {"profile_id": "disabled"},
    }
    config = ManagedAgentConfig(
        agent_id=snapshot["agent"]["id"], agent_type="workflow",
        revision_id=uuid.uuid4(), config_digest=_digest(snapshot), snapshot=snapshot,
    )
    request = ExecutionRequest(
        protocol_version="execution-v1", execution_id="execution-no-agent",
        trace={"trace_id": "execution-no-agent"}, binding_generation=1,
        continuity_key="desk:c", message="hello", identity={}, scene={}, agent_config=config,
    )
    with pytest.raises(ConfigError, match="skill_loader_requires_agent_node"):
        asyncio.run(execute_managed(request))
