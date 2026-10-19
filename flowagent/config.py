"""配置装载：原生 V3 profile / platform_policy / 限制（技术方案 §6.2）。

YAML 工作流引擎配置随 M4 退役；历史配置由旧制品继续绑定。
"""
from __future__ import annotations

from flowagent.errors import ConfigError

# =====================================================================
# 原生 V3 profile 装载（技术方案 §6.2）：配置能力和限制，不描述流程图。
# 与上方 YAML WorkflowSpec 装载互不影响；旧引擎能力在 M4 退出前保持可用。
# =====================================================================

import os as _os
from pathlib import Path as _Path

from flowagent.core.context import Budget

_PROFILE_VERSION = 1
_DEFAULT_LIMITS: dict = {
    "deadline_ms": 90_000, "fallback_timeout_ms": 30_000,
    "main_deadline_ms": 60_000,
    "model_calls": 24, "tool_calls": 32,
    "autonomous": {"model_calls": 12, "tool_calls": 16},
    "context_bytes": 131_072, "classification_concurrency": 1,
}


def load_profile_config(path: str | _Path) -> dict:
    """读取并校验原生 profile 配置（version 必须为 1）。"""
    import json as _json

    data = _json.loads(_Path(path).read_text(encoding="utf-8"))
    if data.get("version") != _PROFILE_VERSION:
        raise ConfigError(f"profile version must be {_PROFILE_VERSION}")
    if not data.get("profiles"):
        # 平台-only 部署可以没有本地 profiles（合法 V3 请求不受影响）。
        data.setdefault("default_profile", None)
    return data


def select_profile(config: dict, name: str | None) -> dict:
    """按名称选择 profile；缺省取 default_profile；未知名称报稳定错误。"""
    profiles = config.get("profiles") or {}
    chosen = name or config.get("default_profile")
    if chosen is None and profiles:
        raise ConfigError("no default_profile configured")
    if chosen is not None and chosen not in profiles:
        raise ConfigError(f"unknown profile: {chosen}")
    return profiles.get(chosen) if chosen else {}


def resolve_model_spec(profile: dict) -> dict:
    """profile.model 的 env 解析：base_url/api_key/模型名/参数。

    必需模型变量启动校验（缺失报 ConfigError）；MCP 惰性发现不在此处。
    """
    model = profile.get("model") or {}
    base_url = _env_or_value(model.get("base_url_env"), model.get("base_url"))
    api_key = _env_or_value(model.get("api_key_env"), model.get("api_key"))
    model_name = _env_or_value(model.get("name_env"), model.get("name"))
    if not model_name:
        raise ConfigError("profile model name unresolved")
    parameters = dict(model.get("parameters") or {})
    return {
        "base_url": base_url, "api_key": api_key or "EMPTY",
        "model_name": model_name, "temperature": float(parameters.get("temperature", 0.0)),
        "extra_body": parameters.get("extra_body") or {},
    }


def _env_or_value(env_key: str | None, direct: str | None) -> str | None:
    if env_key:
        value = _os.environ.get(env_key)
        if not value:
            raise ConfigError(f"required env not set: {env_key}")
        return value
    return direct


def build_budget(profile: dict) -> Budget:
    """limits → Budget（独立模式默认值见 _DEFAULT_LIMITS）。"""
    limits = dict(_DEFAULT_LIMITS)
    limits.update(profile.get("limits") or {})
    autonomous = dict(_DEFAULT_LIMITS["autonomous"])
    autonomous.update((profile.get("limits") or {}).get("autonomous") or {})
    return Budget(
        model_total=int(limits["model_calls"]),
        mcp_total=int(limits["tool_calls"]),
        deadline_ms=float(limits["deadline_ms"]),
        main_deadline_ms=float(limits["main_deadline_ms"]),
        fallback_timeout_ms=float(limits["fallback_timeout_ms"]),
        context_bytes=int(limits["context_bytes"]),
    )


def build_mcp_configs(profile: dict) -> list["McpServerConfig"]:
    """mcp_servers → McpServerConfig 列表（惰性发现：此处不连接）。"""
    from flowagent.clients.mcp import McpServerConfig  # 懒加载：clients 属 M1-2

    configs = []
    for server in profile.get("mcp_servers") or []:
        configs.append(McpServerConfig(
            name=server["name"],
            transport=server.get("transport", "streamable_http"),
            url=server.get("url"),
            url_env=server.get("url_env"),
            headers=server.get("headers"),
            headers_env=server.get("headers_env"),
            allowed_tools=server.get("allowed_tools"),
            timeout_seconds=float(server.get("timeout_seconds", 30.0)),
        ))
    return configs


def workflow_enabled_map(profile: dict) -> dict[str, bool]:
    """workflows.<name>.enabled；省略默认 true。"""
    return {
        name: bool((cfg or {}).get("enabled", True))
        for name, cfg in (profile.get("workflows") or {}).items()
    }


def platform_policy(config: dict) -> dict:
    """platform_policy：仅执行策略，无本地模型/工具；与 default profile 分离。"""
    policy = dict(config.get("platform_policy") or {})
    if not policy.get("enabled"):
        return {"enabled": False}
    return {
        "enabled": True,
        "workflow_enabled_map": {
            name: bool((cfg or {}).get("enabled", True))
            for name, cfg in (policy.get("workflows") or {}).items()
        },
        "limits": dict(policy.get("limits") or {}),
    }
