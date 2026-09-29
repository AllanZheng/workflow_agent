"""Execution-scoped lazy Skill body loading from Demo."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


class SkillLoadError(RuntimeError):
    """Skill content could not be loaded or did not match the pinned binding."""


_TOOL_REF = re.compile(r"\{\{tool:([a-z0-9][a-z0-9-]{0,98})/([A-Za-z0-9][A-Za-z0-9_.-]{0,159})\}\}")


def _resolve_tool_references(body: str, bindings: Any, fixed_tools: list[dict]) -> str:
    """Resolve only server-authorized dependencies matching this execution's snapshot."""
    matches = list(_TOOL_REF.finditer(body))
    references = {match.group(0) for match in matches}
    if len(matches) != body.count("{{tool:") or not isinstance(bindings, list):
        raise SkillLoadError("skill_tool_reference_invalid")
    try:
        pinned = {str(uuid.UUID(tool["tool_id"])): tool for tool in fixed_tools}
        if len(pinned) != len(fixed_tools):
            raise ValueError
        aliases = {}
        seen_ids = set()
        for binding in bindings:
            reference = binding["logical_ref"]
            tool_id = str(uuid.UUID(binding["tool_id"]))
            tool = pinned.get(tool_id)
            if (reference not in references or reference in aliases or tool_id in seen_ids
                    or tool is None or any(str(uuid.UUID(binding[key])) != str(uuid.UUID(tool[key]))
                        for key in ("tool_version_id", "server_version_id"))):
                raise ValueError
            aliases[reference] = "tool_" + uuid.UUID(tool_id).hex
            seen_ids.add(tool_id)
        if set(aliases) != references:
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError):
        raise SkillLoadError("skill_tool_binding_mismatch") from None
    return _TOOL_REF.sub(lambda match: aliases[match.group(0)], body)


class DemoSkillBodyLoader:
    # 初始化对象状态并建立所需依赖。
    # English: Initialize object state and required dependencies.
    def __init__(self, model_authorization_url: str, execution_token: str,
                 runtime_id: str, *, timeout_seconds: float = 3.0,
                 fixed_tools: list[dict[str, Any]] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not model_authorization_url or not execution_token or not runtime_id:
            raise ValueError("Skill loader configuration is incomplete")
        if timeout_seconds <= 0:
            raise ValueError("Skill loader timeout must be positive")
        self._url_template = model_authorization_url
        self._execution_token = execution_token
        self._runtime_id = str(uuid.UUID(runtime_id))
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._fixed_tools = [dict(tool) for tool in (fixed_tools or [])]

    # 生成对象的调试表示。
    # English: Return a developer-oriented representation of the object.
    def __repr__(self) -> str:
        return f"DemoSkillBodyLoader(url={self._url_template!r})"

    # 处理URL 地址相关逻辑。
    # English: Handle URL operations.
    def _url(self, execution_id: str, skill_id: str) -> str:
        parsed = urlsplit(self._url_template)
        suffix = f"/executions/{execution_id}/model-authorization"
        if not parsed.path.endswith(suffix):
            raise SkillLoadError("skill_loader_endpoint_invalid")
        base_path = parsed.path[:-len("/model-authorization")]
        return urlunsplit((parsed.scheme, parsed.netloc,
                           f"{base_path}/skills/{skill_id}/load", "", ""))

    # 执行对象封装的可调用逻辑。
    # English: Invoke the object as a callable.
    async def __call__(self, *, execution_id: str, generation: int,
                       skill: dict[str, Any]) -> str | dict[str, Any]:
        try:
            execution_id = str(uuid.UUID(execution_id))
            skill_id = str(uuid.UUID(skill["skill_id"]))
            metadata_version_id = str(uuid.UUID(skill["metadata_version_id"]))
            body_version_id = str(uuid.UUID(skill["body_version_id"]))
            runtime_id = self._runtime_id
        except (KeyError, ValueError, TypeError):
            raise SkillLoadError("skill_binding_invalid") from None
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise SkillLoadError("skill_generation_invalid")
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds, transport=self._transport,
            ) as client:
                response = await client.post(
                    self._url(execution_id, skill_id),
                    json={
                        "generation": generation,
                        "metadata_version_id": metadata_version_id,
                        "body_version_id": body_version_id,
                    },
                    headers={
                        "Authorization": f"Bearer {self._execution_token}",
                        "X-Runtime-ID": runtime_id,
                    },
                )
            if response.status_code < 200 or response.status_code >= 300:
                raise SkillLoadError("skill_load_denied")
            payload = response.json()
        except SkillLoadError:
            raise
        except Exception:
            raise SkillLoadError("skill_load_unavailable") from None

        body_text = payload.get("body_text") if isinstance(payload, dict) else None
        body_hash = payload.get("body_hash") if isinstance(payload, dict) else None
        canonical_body = body_text.replace("\r\n", "\n").replace("\r", "\n") \
            if isinstance(body_text, str) else None
        actual_hash = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest() \
            if canonical_body is not None else None
        if (not isinstance(payload, dict)
                or payload.get("skill_id") != skill_id
                or payload.get("metadata_version_id") != metadata_version_id
                or payload.get("body_version_id") != body_version_id
                or payload.get("name") != skill.get("name")
                or payload.get("summary") != skill.get("summary")
                or not isinstance(body_hash, str)
                or body_hash != actual_hash):
            raise SkillLoadError("skill_load_binding_mismatch")
        if "{{tool:" in canonical_body or payload.get("tool_bindings"):
            resolved = _resolve_tool_references(canonical_body, payload.get("tool_bindings"), self._fixed_tools)
            return {
                "skill_id": skill_id, "metadata_version_id": metadata_version_id,
                "body_version_id": body_version_id, "body_text": resolved,
                "body_hash": hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
                "source_body_hash": body_hash,
                "requires_dependency_validation": True,
            }
        return canonical_body
