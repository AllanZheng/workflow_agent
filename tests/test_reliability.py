"""单元/集成测试：节点可靠性——重试(retries)与超时(timeout)。"""

import pytest

import tests.flaky as flaky
from flowagent import WorkflowEngine, load_spec
from flowagent.errors import ToolError
from flowagent.llm import MockLLM


# 重置`flaky`。
@pytest.fixture(autouse=True)
def _reset_flaky():
    flaky.reset()
    yield


# 处理`engine`相关逻辑。
def _engine(spec_dict) -> WorkflowEngine:
    return WorkflowEngine(load_spec(spec_dict), MockLLM())


FLAKY_NODE = {
    "type": "tool",
    "tool": "tests.flaky:flaky_tool",
    "args": {"key": "k", "fail_times": 2},
    "output_key": "r",
    "retry_delay": 0,
    "next": "END",
}


class TestRetry:
    # 验证“重试`recovers``within``budget`”场景下的预期行为。
    def test_retry_recovers_within_budget(self):
        spec = {"workflow": {"name": "r", "start": "n"},
                "nodes": {"n": {**FLAKY_NODE, "retries": 3}}}
        final = _engine(spec).run({})
        assert final["vars"]["r"] == "ok after 3 calls"
        assert flaky.calls("k") == 3  # 首次 + 2 次失败 + 第3次成功 = 3 次调用

    # 验证“重试`exhausted`抛出不包含启用错误”场景下的预期行为。
    def test_retry_exhausted_raises_without_on_error(self):
        spec = {"workflow": {"name": "r", "start": "n"},
                "nodes": {"n": {**FLAKY_NODE, "retries": 1}}}
        with pytest.raises(ToolError):
            _engine(spec).run({})
        assert flaky.calls("k") == 2  # 首次 + 1 次重试

    # 验证“重试`exhausted``degrades`包含启用错误”场景下的预期行为。
    def test_retry_exhausted_degrades_with_on_error(self):
        node = {
            **FLAKY_NODE,
            "retries": 1,
            "on_error": {"goto": "END", "set": {"r": "降级"}},
        }
        spec = {"workflow": {"name": "r", "start": "n"}, "nodes": {"n": node}}
        final = _engine(spec).run({})
        assert final["error"]["type"] == "ToolError"
        assert final["vars"]["r"] == "降级"

    # 验证“无重试按默认项”场景下的预期行为。
    def test_no_retry_by_default(self):
        spec = {"workflow": {"name": "r", "start": "n"}, "nodes": {"n": dict(FLAKY_NODE)}}
        with pytest.raises(ToolError):
            _engine(spec).run({})
        assert flaky.calls("k") == 1

    def test_retry_delay_is_applied(self, monkeypatch):
        """重试间隔确实生效：拦截 time.sleep 验证调用参数（避免计时抖动）。"""
        sleeps = []
        monkeypatch.setattr("flowagent.nodes.time.sleep", lambda s: sleeps.append(s))
        spec = {
            "workflow": {"name": "r", "start": "n"},
            "nodes": {
                "n": {
                    "type": "tool",
                    "tool": "tests.flaky:flaky_tool",
                    "args": {"key": "d", "fail_times": 1},
                    "retries": 2,
                    "retry_delay": 0.05,
                    "output_key": "r",
                    "next": "END",
                }
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["r"] == "ok after 2 calls"
        assert sleeps == [0.05]  # 仅第一次失败后休眠一次，随后成功

    def test_exponential_backoff(self, monkeypatch):
        """backoff 倍率：第 n 次重试间隔 = retry_delay * backoff^(n-1)。"""
        sleeps = []
        monkeypatch.setattr("flowagent.nodes.time.sleep", lambda s: sleeps.append(s))
        spec = {
            "workflow": {"name": "r", "start": "n"},
            "nodes": {
                "n": {
                    "type": "tool",
                    "tool": "tests.flaky:flaky_tool",
                    "args": {"key": "bk", "fail_times": 2},
                    "retries": 3,
                    "retry_delay": 0.1,
                    "backoff": 2.0,
                    "output_key": "r",
                    "next": "END",
                }
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["r"] == "ok after 3 calls"
        assert sleeps == [0.1, 0.2]  # 两次失败后的间隔按倍率递增


class TestTimeout:
    # 验证“`timeout`抛出不包含启用错误”场景下的预期行为。
    def test_timeout_raises_without_on_error(self):
        spec = {
            "workflow": {"name": "t", "start": "n"},
            "nodes": {
                "n": {
                    "type": "tool",
                    "tool": "flowagent.tools:slow_tool",
                    "args": {"seconds": 2},
                    "timeout": 0.3,
                    "output_key": "r",
                    "next": "END",
                }
            },
        }
        import concurrent.futures

        with pytest.raises((TimeoutError, concurrent.futures.TimeoutError)):
            _engine(spec).run({})

    # 验证“`timeout``degrades`包含启用错误”场景下的预期行为。
    def test_timeout_degrades_with_on_error(self):
        spec = {
            "workflow": {"name": "t", "start": "n"},
            "nodes": {
                "n": {
                    "type": "tool",
                    "tool": "flowagent.tools:slow_tool",
                    "args": {"seconds": 2},
                    "timeout": 0.3,
                    "output_key": "r",
                    "on_error": {"goto": "fb", "set": {"r": "超时降级"}},
                    "next": {"field": "r", "rules": [], "default": "fb", "on_error": "fb"},
                },
                "fb": {
                    "type": "template",
                    "template": "完成:{r}",
                    "output_key": "msg",
                    "next": "END",
                },
            },
        }
        final = _engine(spec).run({})
        assert final["error"]["type"] == "TimeoutError"
        assert final["vars"]["msg"] == "完成:超时降级"

    # 验证“`fast`工具`within``timeout``succeeds`”场景下的预期行为。
    def test_fast_tool_within_timeout_succeeds(self):
        spec = {
            "workflow": {"name": "t", "start": "n"},
            "nodes": {
                "n": {
                    "type": "tool",
                    "tool": "flowagent.tools:slow_tool",
                    "args": {"seconds": 0.05},
                    "timeout": 5,
                    "output_key": "r",
                    "next": "END",
                }
            },
        }
        final = _engine(spec).run({})
        assert final["error"] is None
        assert final["vars"]["r"] == "slept 0.05s"


class TestOnErrorGoto:
    """4.5：node.on_error.goto 是权威错误路由，三种 next 形态都要生效。"""

    FLAKY_GOTO = {
        "type": "tool",
        "tool": "tests.flaky:flaky_tool",
        "args": {"key": "og", "fail_times": 99},
        "output_key": "r",
        "retry_delay": 0,
        "on_error": {"goto": "fallback", "set": {"r": "降级"}},
    }

    # 验证“`str``next`错误`goes``goto`”场景下的预期行为。
    def test_str_next_error_goes_to_goto(self):
        spec = {
            "workflow": {"name": "og1", "start": "n"},
            "nodes": {
                "n": {**self.FLAKY_GOTO, "next": "normal"},
                "normal": {"type": "template", "template": "正常", "output_key": "x", "next": "END"},
                "fallback": {"type": "template", "template": "兜底", "output_key": "y", "next": "END"},
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["r"] == "降级"
        assert final["vars"]["y"] == "兜底"   # 走了 goto 分支
        assert "x" not in final["vars"]       # 正常分支未执行

    # 验证“列出`next`错误`goes``goto`”场景下的预期行为。
    def test_list_next_error_goes_to_goto(self):
        spec = {
            "workflow": {"name": "og2", "start": "n"},
            "nodes": {
                "n": {**self.FLAKY_GOTO, "next": ["a1", "a2"]},
                "a1": {"type": "template", "template": "A", "output_key": "x", "next": "END"},
                "a2": {"type": "template", "template": "B", "output_key": "w", "next": "END"},
                "fallback": {"type": "template", "template": "兜底", "output_key": "y", "next": "END"},
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["y"] == "兜底"
        assert "x" not in final["vars"] and "w" not in final["vars"]

    # 验证“路由`next``goto``takes``precedence``over`路由启用错误”场景下的预期行为。
    def test_route_next_goto_takes_precedence_over_route_on_error(self):
        node = {
            **self.FLAKY_GOTO,
            "next": {
                "field": "r",
                "rules": [{"when": {"op": "exists"}, "goto": "normal"}],
                "default": "normal",
                "on_error": "route_fallback",
            },
        }
        spec = {
            "workflow": {"name": "og3", "start": "n"},
            "nodes": {
                "n": node,
                "normal": {"type": "template", "template": "正常", "output_key": "x", "next": "END"},
                "fallback": {"type": "template", "template": "节点级", "output_key": "y", "next": "END"},
                "route_fallback": {"type": "template", "template": "路由级", "output_key": "z", "next": "END"},
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["y"] == "节点级"  # node.on_error.goto 优先
        assert "z" not in final["vars"]

    # 验证“无效`goto`目标`rejected``at`加载”场景下的预期行为。
    def test_invalid_goto_target_rejected_at_load(self):
        from flowagent.errors import ConfigError

        with pytest.raises(ConfigError):
            load_spec({
                "workflow": {"name": "og4", "start": "n"},
                "nodes": {
                    "n": {**self.FLAKY_GOTO, "next": "END"},  # goto: fallback 不存在
                },
            })

    def test_node_clears_its_own_stale_error_after_a_later_success(self):
        """评审问题 6：节点失败一次后被路由重试并成功，不应再被旧 error 误路由。

        旧实现：成功节点的返回值不含 "error" 键，channel 保留上一次失败时
        写入的值；下一次对同一节点求值 next 时 `err.get("node") == name` 仍为
        真，会被反复错误地送回 on_error 目标，直到 loop_exhaust 兜底才停止
        （本例会变成 3 次调用而不是 2 次）。
        """
        node = {
            "type": "tool",
            "tool": "tests.flaky:flaky_tool",
            "args": {"key": "selfclear", "fail_times": 1},
            "output_key": "r",
            "retries": 0,
            "on_error": {"goto": "route", "set": {}},
            "next": "END",
        }
        spec = {
            "workflow": {"name": "selfclear", "start": "n"},
            "nodes": {
                "n": node,
                "route": {
                    "type": "template", "template": "retry", "output_key": "note",
                    "next": {
                        "field": "note",
                        "rules": [{"when": {"op": "eq", "value": "retry"},
                                   "goto": "n", "max_loops": 3, "loop_exhaust": "END"}],
                        "default": "END",
                    },
                },
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["r"] == "ok after 2 calls"
        assert final["error"] is None

    def test_other_nodes_success_does_not_erase_an_earlier_nodes_error(self):
        """只清自己：下游节点（on_error.goto 目标）成功执行不应抹掉上游节点
        记录的错误——保留给最终状态做诊断展示（对齐既有测试
        test_timeout_degrades_with_on_error 的期望）。
        """
        node = {**self.FLAKY_GOTO, "args": {"key": "keep", "fail_times": 99}, "next": "END"}
        spec = {
            "workflow": {"name": "keeperr", "start": "n"},
            "nodes": {
                "n": node,
                "fallback": {"type": "template", "template": "兜底", "output_key": "y", "next": "END"},
            },
        }
        final = _engine(spec).run({})
        assert final["error"]["node"] == "n"
        assert final["vars"]["y"] == "兜底"


class TestTimeoutSemantics:
    """4.3：超时按时返回；超时不参与重试（副作用至多一次）。"""

    SLOW_NODE = {
        "type": "tool",
        "tool": "tests.flaky:slow_counting_tool",
        "args": {"key": "slow", "seconds": 0.6},
        "output_key": "r",
        "timeout": 0.05,
        "next": "END",
    }

    # 验证“`timeout``returns``promptly`”场景下的预期行为。
    def test_timeout_returns_promptly(self):
        import time as _t
        spec = {"workflow": {"name": "t1", "start": "n"}, "nodes": {"n": dict(self.SLOW_NODE)}}
        start = _t.monotonic()
        with pytest.raises(TimeoutError):
            _engine(spec).run({})
        elapsed = _t.monotonic() - start
        # 此前 shutdown(wait=True) 会让耗时 ≈ 0.6s；现在必须在 timeout 附近返回
        assert elapsed < 0.3, f"超时未按时返回，耗时 {elapsed:.3f}s"

    # 验证“`timeout``does`不重试`side``effect`”场景下的预期行为。
    def test_timeout_does_not_retry_side_effect(self):
        spec = {"workflow": {"name": "t2", "start": "n"},
                "nodes": {"n": {**self.SLOW_NODE, "retries": 3, "retry_delay": 0}}}
        with pytest.raises(TimeoutError):
            _engine(spec).run({})
        # 超时后不重试：副作用工具只执行一次
        assert flaky.calls("slow") == 1

    # 验证“`timeout``degrades``via`启用错误”场景下的预期行为。
    def test_timeout_degrades_via_on_error(self):
        spec = {"workflow": {"name": "t3", "start": "n"},
                "nodes": {"n": {**self.SLOW_NODE,
                                "on_error": {"goto": "END", "set": {"r": "超时降级"}}}}}
        final = _engine(spec).run({})
        assert final["vars"]["r"] == "超时降级"
