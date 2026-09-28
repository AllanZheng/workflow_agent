"""D-1 模块划分：`ExecutionRequest` 与 V1/V2/V3 snapshot 模型的规范导入位置。

实际定义仍在 `flowagent/execution_service.py`
（大范围物理搬迁这些相互引用的 Pydantic 模型收益有限、风险不小，本次不做）；
这里做一层具名再导出，新代码可以从 `flowagent.execution.contracts` 导入，
与技术方案 §3.1 的模块划分保持一致，且两个导入路径指向同一个类对象。
"""

from __future__ import annotations

from ..execution_service import (
    ExecutionRequest,
    GroupAnalysisTarget,
    ManagedAgentConfig,
    ManagedSnapshot,
    ManagedSnapshotV1,
    ManagedSnapshotV2,
    MemoryContextItem,
    MemoryContextV1,
    StrictModel,
    Trace,
)

__all__ = [
    "ExecutionRequest",
    "GroupAnalysisTarget",
    "ManagedAgentConfig",
    "ManagedSnapshot",
    "ManagedSnapshotV1",
    "ManagedSnapshotV2",
    "MemoryContextItem",
    "MemoryContextV1",
    "StrictModel",
    "Trace",
]
