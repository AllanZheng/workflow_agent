"""平台 V3 适配层入口：信封协议、能力运行时、上下文映射、回执与路由。"""
from flowagent.adapters.platform.protocol import (
    ExecutionRequest,
    FunctionTask,
    ManagedAgentConfig,
    ManagedSnapshotV3,
    MemoryContextV1,
    parse_managed_snapshot_v3,
    snapshot_digest,
)
from flowagent.adapters.platform.routes import build_platform_app

__all__ = [
    "ExecutionRequest", "FunctionTask", "ManagedAgentConfig", "ManagedSnapshotV3",
    "MemoryContextV1", "parse_managed_snapshot_v3", "snapshot_digest",
    "build_platform_app",
]
