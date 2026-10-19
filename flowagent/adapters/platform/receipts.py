"""回执与结果映射：agent_config_applied、execution-v1 返回体与错误信封。

规则（技术方案 §7.1）：
- 回执记实际发生的使用（成功加载 Skill / 调用工具去重）；配置目录不算使用；
  失败与重试保留在 diagnostics。
- 业务映射：群分析 → analysis_result；会商 → function_result；澄清 → clarification；
  status/usage/execution_path/recovery 作为额外 metadata。
- 执行失败映射非 2xx + 既有 error 信封（error_code/error_message），
  不用 HTTP 200 普通回复伪装失败；CLARIFICATION_EXHAUSTED 携带有限缺失字段。
"""
from __future__ import annotations

from typing import Any

from flowagent.core.contracts import RunResult, Status

# 内部码 → 既有外部码/HTTP 状态（保持平台已消费的错误字段）。
_ERROR_MAP: dict[str, tuple[str, int]] = {
    "INVALID_INPUT": ("invalid_input", 422),
    "CLARIFICATION_EXHAUSTED": ("clarification_exhausted", 422),
    "MODEL_TIMEOUT": ("model_timeout", 502),
    "MODEL_UPSTREAM_ERROR": ("model_upstream_error", 502),
    "MODEL_INVALID_RESPONSE": ("model_invalid_response", 502),
    "MODEL_BUDGET_EXCEEDED": ("model_budget_exceeded", 502),
    "TOOL_BUDGET_EXCEEDED": ("tool_budget_exceeded", 502),
    "TOOL_ARGUMENT_INVALID": ("invalid_arguments", 422),
    "MCP_TOOL_ERROR": ("mcp_tool_error", 502),
    "CAPABILITY_UNAVAILABLE": ("capability_unavailable", 502),
    "CONTEXT_LIMIT_EXCEEDED": ("context_limit_exceeded", 422),
    "DEADLINE_EXCEEDED": ("deadline_exceeded", 504),
    "AUTHORIZATION_DENIED": ("authorization_denied", 403),
    "CANCELLED_BY_USER": ("cancelled", 499),
    "OUTPUT_INVALID": ("invalid_structured_output", 422),
    "DUPLICATE_TOOL_CALL": ("duplicate_tool_call", 502),
    "INTERNAL_ERROR": ("internal_error", 500),
    "FALLBACK_FAILED": ("fallback_failed", 500),
    "SKILL_LOAD_FAILED": ("skill_load_failed", 502),
    "SKILL_BINDING_MISMATCH": ("skill_binding_mismatch", 502),
}


def error_envelope(code: str, message: str) -> tuple[dict[str, Any], int]:
    """内部错误码 → (error 信封, HTTP 状态)；未识别按 internal_error。"""
    external, status = _ERROR_MAP.get(code, ("internal_error", 500))
    return {"error": {"code": external, "message": message}}, status


def build_receipt(
    request: Any, result: RunResult, *, caps: Any,
) -> dict[str, Any]:
    """agent_config_applied：与平台验证器逐字段对齐的 V3 实际使用回执。

    契约（PostgresConfigApplyReceiptStore._validate_actual_receipt）：
    runtime_id/schema_version=3/model_version_id/memory_profile_version_id 与发布快照
    一致；loaded_skill_versions 条目含 body_hash 且 outcome='success'；called_tool_versions
    条目含版本三元组且 outcome='success'；均去重。
    """
    snapshot = request.agent_config.snapshot_v3
    loaded_skills, called_tools, skill_failures = _actual_usage(caps, result)
    memory_status = (
        request.memory_context.status if request.memory_context is not None else "unavailable")
    return {
        "execution_id": request.execution_id,
        "runtime_id": str(snapshot.runtime_id),
        "schema_version": 3,
        "revision_id": str(snapshot.revision_id),
        "config_digest": request.agent_config.config_digest,
        "model_version_id": str(snapshot.model.version_id),
        "memory_profile_version_id": str(snapshot.memory.profile_version_id),
        "memory_context_status": memory_status,
        "loaded_skill_versions": loaded_skills,
        "skill_load_failures": skill_failures,
        "called_tool_versions": called_tools,
        "applied_fields": ["system_prompt", "model", "tools", "skills", "memory"],
    }


def _actual_usage(caps: Any, result: RunResult) -> tuple[list[dict], list[dict], list[dict]]:
    """实际成功使用（去重）：Skill 正文哈希 + 固定工具版本三元组；失败单列。"""
    import hashlib

    loaded: list[dict] = []
    called: list[dict] = []
    failures: list[dict] = []
    seen_tools: set[tuple] = set()
    bindings = getattr(caps.tools, "tool_bindings", {}) if caps.tools else {}
    # 反查：逻辑工具名/远端名 → 固定别名绑定（工作流按逻辑名调用，绑定键是别名）。
    by_remote = {
        str(pinned.get("remote_name")): pinned for pinned in bindings.values()
    }
    for event in (result.recovery.get("_evidence_tool_calls") or []):
        name = str(event.get("name") or "")
        if name == "load_skill":
            if event.get("status") != "success":
                failures.append({"outcome": "failed",
                                 "error_code": event.get("error_code")})
            continue
        if event.get("status") != "success":
            continue
        pinned = (bindings.get(name)
                  or by_remote.get(str(event.get("remote_name") or ""))
                  or by_remote.get(name))
        if not pinned:
            continue
        key = (str(pinned.get("tool_id")), str(pinned.get("tool_version_id")),
               str(pinned.get("server_version_id")))
        if key in seen_tools:
            continue
        seen_tools.add(key)
        called.append({
            "tool_id": key[0], "tool_version_id": key[1], "server_version_id": key[2],
            "remote_name": pinned.get("remote_name"), "outcome": "success",
        })
    skills = caps.skills
    if skills is not None and getattr(skills, "_loaded", None):
        catalog = {str(item["skill_id"]): item for item in skills.catalog()}
        for skill_id, body in skills._loaded.items():
            item = catalog.get(str(skill_id), {"skill_id": skill_id})
            normalized = str(body).replace("\r\n", "\n").replace("\r", "\n")
            loaded.append({
                "skill_id": str(skill_id),
                "metadata_version_id": item.get("metadata_version_id"),
                "body_version_id": item.get("body_version_id"),
                "body_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                "outcome": "success",
            })
    return loaded, called, failures


def build_response(
    request: Any, result: RunResult, *, caps: Any,
) -> dict[str, Any]:
    """RunResult → execution-v1 respond 返回体。"""
    output = result.output or {}
    metadata: dict[str, Any] = {
        "agent_config_applied": build_receipt(request, result, caps=caps),
        "status": result.status.value,
        "usage": {
            "model_requests": result.usage.model_requests,
            "model_retries": result.usage.model_retries,
            "mcp_attempts": result.usage.mcp_attempts,
            "skill_loads": result.usage.skill_loads,
            "tokens_in": result.usage.tokens_in,
            "tokens_out": result.usage.tokens_out,
            "elapsed_ms": result.usage.elapsed_ms,
        },
        "execution_path": result.execution_path,
        "recovery": _public_recovery(result),
    }
    body: dict[str, Any] = {
        "execution_id": request.execution_id,
        "reply": result.reply,
        "used_context_refs": list((result.output or {}).get("memory_included_ids") or []),
        "metadata": metadata,
    }
    if "analysis_result" in output:
        metadata["analysis_result"] = output["analysis_result"]
    if "function_result" in output:
        metadata["function_result"] = output["function_result"]
    if result.clarification is not None:
        metadata["clarification"] = {
            "reason": result.clarification.reason,
            "missing_fields": result.clarification.missing_fields,
            "confirmed_fields": result.clarification.confirmed_fields,
            "hints": result.clarification.hints,
            "continuation": result.clarification.continuation,
        }
    return body


def _public_recovery(result: RunResult) -> dict[str, Any]:
    recovery = dict(result.recovery)
    recovery.pop("_evidence_tool_calls", None)
    return recovery

