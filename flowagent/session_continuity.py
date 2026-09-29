"""会话连续性规划：同会话新配置可执行，不兼容 checkpoint 明确要求新会话。

P3 契约：一次执行解析当前生效配置并固定快照；同一会话（continuity key）
的下一次执行允许更新配置——同摘要复用 checkpoint，不同摘要不静默续跑旧图，
返回 SESSION_RESTART_REQUIRED 由调用方开启新会话（历史由请求携带）。

本模块为纯决策函数：不读写存储，便于在运行镜像启用连续性复用前先固定语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SESSION_RESTART_REQUIRED = "SESSION_RESTART_REQUIRED"


@dataclass(frozen=True)
class SessionDecision:
    action: Literal["start_new", "reuse_checkpoint", "restart_required"]
    reason: str

    # 处理错误`code`相关逻辑。
    # English: Handle error code operations.
    @property
    def error_code(self) -> str | None:
        return SESSION_RESTART_REQUIRED if self.action == "restart_required" else None


# English: Handle plan session operations.
def plan_session(
    *,
    has_checkpoint: bool,
    checkpoint_config_digest: str | None,
    request_config_digest: str,
    checkpoint_schema_version: int | None = None,
    request_schema_version: int = 1,
) -> SessionDecision:
    """决定会话下一步动作。

    - 无 checkpoint：直接开新会话（首次执行或上次已结束）。
    - 有 checkpoint 且摘要一致：复用，保证继续同一张图。
    - 有 checkpoint 但配置或 schema 已变：不静默续跑，返回 restart_required，
      调用方带历史开启新会话（配置变更从下一次执行生效）。
    """
    if not request_config_digest:
        raise ValueError("request_config_digest is required")
    if not has_checkpoint:
        return SessionDecision("start_new", "no checkpoint for this session")
    if checkpoint_config_digest is None:
        # 旧 checkpoint 未记录摘要：无法证明兼容，不静默续跑
        return SessionDecision("restart_required", "checkpoint config digest unknown")
    if checkpoint_config_digest != request_config_digest:
        return SessionDecision("restart_required",
                               "config changed since the checkpoint was written")
    if (checkpoint_schema_version is not None
            and checkpoint_schema_version != request_schema_version):
        return SessionDecision("restart_required",
                               "checkpoint schema version is incompatible")
    return SessionDecision("reuse_checkpoint", "config and schema unchanged")
