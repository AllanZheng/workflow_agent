import asyncio
import hashlib
import inspect
import uuid

import httpx
import pytest

from flowagent.skill_runtime import DemoSkillBodyLoader, SkillLoadError


def fixture():
    skill = {
        key: str(uuid.uuid4())
        for key in ("skill_id", "metadata_version_id", "body_version_id")
    }
    skill.update(name="Wiki", summary="Query Wiki")
    tool = {
        key: str(uuid.uuid4())
        for key in ("tool_id", "tool_version_id", "server_version_id")
    }
    tool["remote_name"] = "hybrid_search"
    ref = "{{tool:wiki-mcp/hybrid_search}}"
    body = "Use " + ref + "; then cite the result."
    payload = {
        **skill,
        "body_text": body,
        "body_hash": hashlib.sha256(body.encode()).hexdigest(),
        "tool_bindings": [
            {
                **{
                    k: tool[k]
                    for k in ("tool_id", "tool_version_id", "server_version_id")
                },
                "logical_ref": ref,
            }
        ],
    }
    return skill, tool, payload


def run_loader(skill, tool, payload):
    assert "fixed_tools" in inspect.signature(DemoSkillBodyLoader).parameters, (
        "Pinned Skill reference resolution is missing"
    )
    execution_id = str(uuid.uuid4())
    loader = DemoSkillBodyLoader(
        f"https://demo.test/internal/executions/{execution_id}/model-authorization",
        "test-token",
        str(uuid.uuid4()),
        fixed_tools=[tool],
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    )
    return asyncio.run(loader(execution_id=execution_id, generation=1, skill=skill))


def test_loader_resolves_against_exact_fixed_versions_and_preserves_source_hash():
    skill, tool, payload = fixture()
    result = run_loader(skill, tool, payload)
    assert (
        result["body_text"]
        == "Use tool_" + uuid.UUID(tool["tool_id"]).hex + "; then cite the result."
    )
    assert result["source_body_hash"] == payload["body_hash"]
    assert (
        result["body_hash"] == hashlib.sha256(result["body_text"].encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "tool_id",
        "tool_version_id",
        "server_version_id",
        "unknown_ref",
        "malformed",
    ],
)
def test_loader_rejects_unbound_or_malformed_mapping(mutation):
    skill, tool, payload = fixture()
    if mutation == "missing":
        payload.pop("tool_bindings")
    elif mutation == "duplicate":
        payload["tool_bindings"] *= 2
    elif mutation == "unknown_ref":
        payload["tool_bindings"][0]["logical_ref"] = "{{tool:other/search}}"
    elif mutation == "malformed":
        payload["body_text"] += " {{tool:broken"
        payload["body_hash"] = hashlib.sha256(payload["body_text"].encode()).hexdigest()
    else:
        payload["tool_bindings"][0][mutation] = str(uuid.uuid4())
    with pytest.raises(SkillLoadError, match="skill_tool"):
        run_loader(skill, tool, payload)


def test_runtime_sends_resolved_body_to_model_and_rechecks_revoked_dependencies():
    from flowagent.agent_runtime import ManagedAgentRuntime

    skill, tool, payload = fixture()
    resolved = run_loader(skill, tool, payload)
    calls = []

    async def loader(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            raise SkillLoadError("skill_load_denied")
        return resolved

    runtime = ManagedAgentRuntime(
        None,
        None,
        skills=[skill],
        skill_loader=loader,
        execution_id="isolated",
        generation=1,
    )

    async def check():
        assert await runtime._load_skill(skill["skill_id"]) == resolved["body_text"]
        with pytest.raises(SkillLoadError, match="denied"):
            await runtime._load_skill(skill["skill_id"])

    asyncio.run(check())
    assert runtime.usage.loaded_skills[0]["body_hash"] == payload["body_hash"]
    assert runtime.usage.loaded_skills[0]["resolved_body_hash"] == resolved["body_hash"]
