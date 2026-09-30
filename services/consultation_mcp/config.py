from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(ValueError):
    pass


# 处理正数标识集合相关逻辑；空值表示不限制。
# English: Handle positive IDs; an empty value means unrestricted.
def _positive_ids(value: str, name: str) -> frozenset[int]:
    try:
        ids = frozenset(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise SettingsError(f"{name} must contain comma-separated positive integers") from exc
    if any(item <= 0 for item in ids):
        raise SettingsError(f"{name} must contain comma-separated positive integers")
    return ids


@dataclass(frozen=True)
class Settings:
    im_base_url: str
    im_service_token: str
    mcp_token: str
    # 为空表示不限制会话：私聊会话编号无法预先配置，成员资格由 IM 受信端点校验。
    allowed_conversation_ids: frozenset[int] = frozenset()
    host: str = "0.0.0.0"
    port: int = 8091
    timeout_seconds: float = 10.0

    # 处理来自环境变量相关逻辑。
    # English: Build from env.
    @classmethod
    def from_env(cls, env=None) -> "Settings":
        source = os.environ if env is None else env
        base_url = source.get("IM_BASE_URL", "").strip().rstrip("/")
        service_token = source.get("IM_INTERNAL_SERVICE_TOKEN", "").strip()
        mcp_token = source.get("CONSULTATION_WRITE_MCP_TOKEN", "").strip()
        if not base_url.startswith(("http://", "https://")):
            raise SettingsError("IM_BASE_URL must be an http(s) URL")
        if not service_token:
            raise SettingsError("IM_INTERNAL_SERVICE_TOKEN is required")
        # CONSULTATION_WRITE_MCP_TOKEN 评审 D-11：允许留空以放开本服务自身的
        # 入站鉴权（不影响 IM_INTERNAL_SERVICE_TOKEN 与会话白名单）。
        conversations = _positive_ids(
            source.get("CONSULTATION_ALLOWED_CONVERSATION_IDS", ""),
            "CONSULTATION_ALLOWED_CONVERSATION_IDS",
        )
        try:
            port = int(source.get("CONSULTATION_WRITE_MCP_PORT", "8091"))
            timeout = float(source.get("CONSULTATION_WRITE_TIMEOUT_SECONDS", "10"))
        except ValueError as exc:
            raise SettingsError("CONSULTATION_WRITE_MCP_PORT and timeout must be numeric") from exc
        if not 1 <= port <= 65535 or timeout <= 0:
            raise SettingsError("consultation MCP port and timeout are out of range")
        return cls(base_url, service_token, mcp_token, conversations,
                   source.get("CONSULTATION_WRITE_MCP_HOST", "0.0.0.0"), port, timeout)
