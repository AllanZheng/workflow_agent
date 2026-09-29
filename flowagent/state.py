"""工作流图的全局状态定义。

所有节点共享同一个状态通道：
- vars: 业务变量，节点写入的输出都合并进这里（自定义 reducer 做字典合并，
        使并行分支各自写不同 key 时互不覆盖）。
- trace: 执行轨迹，只追加，最多保留最近 200 条（evaluation 问题 6：避免长循环撑爆状态/checkpoint）。
- loop_counts: 每个节点的执行次数，用于循环护栏（max_loops）。
- error: 最近一个发生错误并走了 on_error 降级分支的节点信息；该节点自己
  重跑并成功后会被清除，其余节点成功不影响它（保留给下游节点做诊断展示）。
"""

from typing import Annotated, TypedDict

#: trace 只保留最近 N 条，避免长循环把状态和 checkpoint 撑爆（评审问题 6）。
_MAX_TRACE_ENTRIES = 200


# English: Handle merge dict operations.
def merge_dict(left: dict | None, right: dict | None) -> dict:
    """LangGraph reducer：右侧更新合并进左侧（None 安全）。"""
    if left is None:
        return dict(right or {})
    if right is None:
        return dict(left)
    return {**left, **right}


# English: Handle bounded trace operations.
def bounded_trace(left: list | None, right: list | None) -> list:
    """追加式 reducer，但只保留最近 _MAX_TRACE_ENTRIES 条。"""
    combined = list(left or []) + list(right or [])
    if len(combined) > _MAX_TRACE_ENTRIES:
        combined = combined[-_MAX_TRACE_ENTRIES:]
    return combined


# English: Handle last write operations.
def last_write(_left: dict | None, right: dict | None) -> dict | None:
    """error 的 reducer：单写者场景等价于旧的隐式覆盖；同一 super-step 内
    两个并行分支都写入时，避免 LangGraph 因无 reducer 而抛 InvalidUpdateError。"""
    return right


class WFState(TypedDict):
    vars: Annotated[dict, merge_dict]
    trace: Annotated[list, bounded_trace]
    loop_counts: Annotated[dict, merge_dict]
    error: Annotated[dict | None, last_write]
