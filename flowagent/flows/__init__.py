"""D-1：flow 注册表——每个业务能力对应一个具名 flow。

本次未把 qa/group_analysis 的内部实现改写成字面意义的多节点 StateGraph（那
部分逻辑复杂且被大量既有测试锁定行为，改写风险与收益不成比例）；`managed_yaml`
本身就是 `flowagent.engine.build_graph` 产出的图。`FlowDefinition` 在这里先作为
统一的路由/元数据登记点使用：`flowagent.execution.planner.select_flow()` 返回
的名字必须能在 `FLOWS` 里查到，`needs_checkpoint` 供 D-2（checkpoint 默认关闭）
判断依据。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowDefinition:
    name: str
    description: str
    needs_checkpoint: bool = False
    streams_reply: bool = True


FLOWS: dict[str, FlowDefinition] = {
    "qa": FlowDefinition(
        name="qa", description="V3 直接问答：agent 与工具循环（ManagedAgentRuntime）",
    ),
    "group_analysis": FlowDefinition(
        name="group_analysis", description="V1/V2 群消息定制化分析（deterministic 模式）",
    ),
    "managed_yaml": FlowDefinition(
        name="managed_yaml", description="V1/V2 通用受管 YAML（含风险会商），由 build_graph 驱动",
    ),
}


# 获取流程。
def get_flow(name: str) -> FlowDefinition:
    try:
        return FLOWS[name]
    except KeyError as exc:
        raise ValueError(f"unknown flow: {name}") from exc
