import asyncio
import hashlib

from langchain_core.messages import AIMessage, ToolMessage
import pytest
import httpx

from flowagent.agent_runtime import ManagedAgentRuntime
from flowagent.execution_service import MemoryContextV1, _managed_system_prompt
from flowagent.skill_runtime import DemoSkillBodyLoader, SkillLoadError


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


SKILL = {
    "skill_id": "604b6209-30da-4e57-bf93-73c6867cc101",
    "metadata_version_id": "704b6209-30da-4e57-bf93-73c6867cc101",
    "body_version_id": "804b6209-30da-4e57-bf93-73c6867cc101",
    "name": "Refund policy",
    "summary": "Use the published refund policy when answering refund questions.",
}
BODY = "Refunds are available within 30 days with proof of purchase."
BODY_HASH = hashlib.sha256(BODY.encode("utf-8")).hexdigest()


class FakeMcp:
    async def list_tools(self):
        return []


class FakeModel:
    def __init__(self, responses, captured):
        self.responses = iter(responses)
        self.captured = captured

    async def ainvoke(self, messages):
        self.captured.append(list(messages))
        return next(self.responses)


class FakeLlm:
    def __init__(self, responses, captured):
        self.model = FakeModel(responses, captured)
        self.tools = None

    def bind_tools(self, tools):
        self.tools = tools
        return self.model


def test_skill_body_is_loaded_only_on_model_request_and_uses_pinned_versions():
    captured = []
    loaded = []
    llm = FakeLlm([
        AIMessage(content="", tool_calls=[{
            "id": "load-1", "name": "load_skill", "args": {"skill_id": SKILL["skill_id"]},
        }]),
        AIMessage(content="refund answer"),
    ], captured)

    async def load_skill(*, execution_id, generation, skill):
        loaded.append((execution_id, generation, skill["metadata_version_id"], skill["body_version_id"]))
        return BODY

    runtime = ManagedAgentRuntime(llm, FakeMcp(), skills=[SKILL], skill_loader=load_skill,
                                  execution_id="904b6209-30da-4e57-bf93-73c6867cc101",
                                  generation=4)
    answer = asyncio.run(runtime.run("Can I get a refund?", "General instructions", 3))

    assert answer == "refund answer"
    first_system = captured[0][0].content
    assert SKILL["summary"] in first_system
    assert BODY not in first_system
    assert loaded == [(
        "904b6209-30da-4e57-bf93-73c6867cc101",
        4,
        SKILL["metadata_version_id"],
        SKILL["body_version_id"],
    )]
    assert any(isinstance(message, ToolMessage) and BODY == message.content
               for message in captured[1])
    assert runtime.usage.loaded_skills == [{
        "skill_id": SKILL["skill_id"],
        "metadata_version_id": SKILL["metadata_version_id"],
        "body_version_id": SKILL["body_version_id"],
        "body_hash": BODY_HASH,
        "outcome": "success",
    }]


def test_skill_loader_cannot_resolve_a_skill_outside_the_snapshot():
    llm = FakeLlm([AIMessage(content="", tool_calls=[{
        "id": "load-2", "name": "load_skill", "args": {"skill_id": "unknown"},
    }]), AIMessage(content="I can answer without that Skill.")], [])
    loaded = []

    async def load_skill(**kwargs):
        loaded.append(kwargs)
        return "must not load"

    runtime = ManagedAgentRuntime(llm, FakeMcp(), skills=[SKILL], skill_loader=load_skill,
                                  execution_id="904b6209-30da-4e57-bf93-73c6867cc101",
                                  generation=4)

    answer = asyncio.run(runtime.run("question", "", 2))
    assert answer == "I can answer without that Skill."
    assert loaded == []


def test_pinned_skill_load_failure_is_safe_and_model_can_answer_without_it():
    captured = []
    llm = FakeLlm([
        AIMessage(content="", tool_calls=[{
            "id": "load-3", "name": "load_skill", "args": {"skill_id": SKILL["skill_id"]},
        }]),
        AIMessage(content="I cannot verify the refund policy."),
    ], captured)

    async def broken_loader(**kwargs):
        raise RuntimeError("remote detail contains a secret")

    runtime = ManagedAgentRuntime(llm, FakeMcp(), skills=[SKILL], skill_loader=broken_loader,
                                  execution_id="904b6209-30da-4e57-bf93-73c6867cc101",
                                  generation=4)
    answer = asyncio.run(runtime.run("refund?", "", 2))

    assert answer == "I cannot verify the refund policy."
    assert "secret" not in captured[1][-1].content
    assert runtime.usage.skill_load_failures == [{
        "skill_id": SKILL["skill_id"],
        "metadata_version_id": SKILL["metadata_version_id"],
        "body_version_id": SKILL["body_version_id"],
        "outcome": "failed",
        "error_code": "skill_load_failed",
    }]


def test_demo_skill_loader_requests_the_fixed_versions_with_generation_and_validates_reply():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = request.read().decode("utf-8")
        return httpx.Response(200, json={
            "skill_id": SKILL["skill_id"],
            "metadata_version_id": SKILL["metadata_version_id"],
            "body_version_id": SKILL["body_version_id"],
            "name": SKILL["name"],
            "summary": SKILL["summary"],
            "body_text": BODY,
            "body_hash": BODY_HASH,
        })

    loader = DemoSkillBodyLoader(
        "https://demo.test/internal/executions/904b6209-30da-4e57-bf93-73c6867cc101/model-authorization",
        "execution-secret", "804b6209-30da-4e57-bf93-73c6867cc101",
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(loader(
        execution_id="904b6209-30da-4e57-bf93-73c6867cc101", generation=4, skill=SKILL,
    ))

    assert result == BODY
    assert captured["url"].endswith(f"/skills/{SKILL['skill_id']}/load")
    assert captured["headers"]["authorization"] == "Bearer execution-secret"
    assert captured["headers"]["x-runtime-id"] == "804b6209-30da-4e57-bf93-73c6867cc101"
    assert captured["body"] == (
        '{"generation":4,"metadata_version_id":"704b6209-30da-4e57-bf93-73c6867cc101",'
        '"body_version_id":"804b6209-30da-4e57-bf93-73c6867cc101"}'
    )


def test_demo_skill_loader_rejects_a_body_version_other_than_the_pinned_one():
    loader = DemoSkillBodyLoader(
        "https://demo.test/internal/executions/904b6209-30da-4e57-bf93-73c6867cc101/model-authorization",
        "execution-secret", "804b6209-30da-4e57-bf93-73c6867cc101",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "skill_id": SKILL["skill_id"],
            "metadata_version_id": SKILL["metadata_version_id"],
            "body_version_id": "904b6209-30da-4e57-bf93-73c6867cc101",
            "body_text": "unpublished body",
            "body_hash": "b" * 64,
        })),
    )

    with pytest.raises(SkillLoadError, match="binding_mismatch"):
        asyncio.run(loader(
            execution_id="904b6209-30da-4e57-bf93-73c6867cc101", generation=4, skill=SKILL,
        ))


def test_demo_skill_loader_normalizes_line_endings_before_hash_check():
    canonical_body = BODY + "\nBring your receipt."
    windows_body = canonical_body.replace("\n", "\r\n")
    canonical_hash = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
    loader = DemoSkillBodyLoader(
        "https://demo.test/internal/executions/904b6209-30da-4e57-bf93-73c6867cc101/model-authorization",
        "execution-secret", "804b6209-30da-4e57-bf93-73c6867cc101",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "skill_id": SKILL["skill_id"],
            "metadata_version_id": SKILL["metadata_version_id"],
            "body_version_id": SKILL["body_version_id"],
            "name": SKILL["name"], "summary": SKILL["summary"],
            "body_text": windows_body, "body_hash": canonical_hash,
        })),
    )
    assert asyncio.run(loader(
        execution_id="904b6209-30da-4e57-bf93-73c6867cc101", generation=4, skill=SKILL,
    )) == canonical_body


def test_demo_skill_loader_rejects_body_hash_mismatch():
    loader = DemoSkillBodyLoader(
        "https://demo.test/internal/executions/904b6209-30da-4e57-bf93-73c6867cc101/model-authorization",
        "execution-secret", "804b6209-30da-4e57-bf93-73c6867cc101",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "skill_id": SKILL["skill_id"],
            "metadata_version_id": SKILL["metadata_version_id"],
            "body_version_id": SKILL["body_version_id"],
            "name": SKILL["name"], "summary": SKILL["summary"],
            "body_text": BODY, "body_hash": "0" * 64,
        })),
    )
    with pytest.raises(SkillLoadError, match="binding_mismatch"):
        asyncio.run(loader(
            execution_id="904b6209-30da-4e57-bf93-73c6867cc101", generation=4, skill=SKILL,
        ))
