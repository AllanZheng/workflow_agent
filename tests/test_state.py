"""单元测试：flowagent.state 的 reducer（评审问题 6）。"""

from flowagent.state import bounded_trace, last_write


# 验证“`bounded`追踪`appends``but``caps``at`200`entries`”场景下的预期行为。
def test_bounded_trace_appends_but_caps_at_200_entries():
    left = [{"node": f"n{i}"} for i in range(199)]
    right = [{"node": "new1"}, {"node": "new2"}]
    combined = bounded_trace(left, right)
    assert len(combined) == 200
    # 保留的是最近的 200 条：最旧的一条被挤出
    assert combined[0] == {"node": "n1"}
    assert combined[-1] == {"node": "new2"}


# 验证“`bounded`追踪处理无输入”场景下的预期行为。
def test_bounded_trace_handles_none_inputs():
    assert bounded_trace(None, None) == []
    assert bounded_trace(None, [{"node": "a"}]) == [{"node": "a"}]
    assert bounded_trace([{"node": "a"}], None) == [{"node": "a"}]


def test_last_write_resolves_without_raising_when_two_parallel_branches_both_error():
    """并行分支同一步都失败时，两次写入 error 不应导致 LangGraph 抛
    InvalidUpdateError——reducer 必须能处理"同一 super-step 内两次调用"且
    不崩溃（具体保留哪一个允许是实现细节，这里只保证确定性、不抛异常）。
    """
    branch_a_error = {"node": "a", "type": "ToolError"}
    branch_b_error = {"node": "b", "type": "ToolError"}
    # LangGraph 对 Annotated reducer 是逐步 fold：reduce(reduce(None, a), b)
    merged = last_write(last_write(None, branch_a_error), branch_b_error)
    assert merged == branch_b_error
