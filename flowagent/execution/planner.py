"""D-1：flow 路由——把「请求属于哪条 flow」的判断收敛到一处。

respond()/stream() 原先各自内联同一套 if/elif（V3→qa，deterministic→
group_analysis，其余→managed_yaml），本函数抽出后两条路径共用同一份判断，
避免后续修改路由规则时又要改两处。
"""

from __future__ import annotations

from typing import Any


def select_flow(config: Any, spec: Any = None) -> str:
    """按技术方案 §4.4 的规则选择 flow 名称。

    V3 snapshot 不需要 `spec` 即可判定为 "qa"。V1/V2 必须先
    `load_managed_spec()` 解出 `spec` 才能区分 group_analysis 与
    managed_yaml——调用方在拿到 spec 之前只应依赖本函数对 V3 的判断，spec
    到手后再调用一次确认具体走哪条 V1/V2 子路径。
    """
    if getattr(config, "snapshot_v3", None) is not None:
        return "qa"
    if spec is not None and spec.workflow.group_analysis_mode == "deterministic":
        return "group_analysis"
    return "managed_yaml"
