"""工作流图的全局状态定义。

所有节点共享同一个状态通道：
- vars: 业务变量，节点写入的输出都合并进这里（自定义 reducer 做字典合并，
        使并行分支各自写不同 key 时互不覆盖）。
- trace: 执行轨迹，只追加（operator.add reducer）。
- loop_counts: 每个节点的执行次数，用于循环护栏（max_loops）。
- error: 最近一个发生错误并走了 on_error 降级分支的节点信息。
"""

import operator
from typing import Annotated, TypedDict


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """LangGraph reducer：右侧更新合并进左侧（None 安全）。"""
    if left is None:
        return dict(right or {})
    if right is None:
        return dict(left)
    return {**left, **right}


class WFState(TypedDict):
    vars: Annotated[dict, merge_dict]
    trace: Annotated[list, operator.add]
    loop_counts: Annotated[dict, merge_dict]
    error: dict
