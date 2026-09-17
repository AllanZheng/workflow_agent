"""节点执行器：把配置里的五类节点包装成 LangGraph 节点函数。

可靠性机制（对 llm/tool/template 生效；human 节点除外）：
- retries/retry_delay: 失败重试（指数退避的简化版：固定间隔）；
- timeout: 用单线程池实现超时中断（超时抛 TimeoutError，可被 on_error 捕获）。

human 节点必须**绕过**线程池与重试：LangGraph 的 interrupt() 依赖当前线程的
上下文（config/contextvars），放入子线程会丢失；且 GraphInterrupt 是控制流
异常，必须原样向上抛给 LangGraph 运行时，绝不能被重试/on_error 吞掉。
"""

import concurrent.futures
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

try:  # GraphBubbleUp 是 GraphInterrupt 等控制流异常的基类
    from langgraph.errors import GraphBubbleUp
except ImportError:  # pragma: no cover - 兼容旧版本
    from langgraph.errors import GraphInterrupt as GraphBubbleUp
from langgraph.types import interrupt

from .config import NodeConfig
from .errors import FlowAgentError
from .parsers import parse_output, validate_schema
from .state import WFState
from .template import get_path, render_template
from .tools import load_tool

_MAX_TRACE_LEN = 120

# 子图引擎缓存：同一路径只编译一次
_SUB_ENGINES: dict = {}


def _summarize(result: dict) -> dict:
    """轨迹里只留简短摘要，避免大对象撑爆状态。"""
    out = {}
    for key, value in result.items():
        if isinstance(value, (int, float)):
            text = str(value)
        elif isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        if len(text) > _MAX_TRACE_LEN:
            text = text[: _MAX_TRACE_LEN - 3] + "..."
        out[key] = text
    return out


def _with_timeout(func, args: tuple, timeout: float | None):
    """带超时的函数执行；timeout 为 None 时直接调用。

    超时语义 = 停止等待（评审 4.3）：到点即 shutdown(wait=False) 返回
    TimeoutError，不等待工具线程结束。Python 线程无法强制终止，工具线程
    可能仍在后台完成并产生副作用——因此上层约定超时不自动重试
    （见 _run_with_reliability），避免副作用重复执行。
    """
    if timeout is None:
        return func(*args)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowagent-node")
    future = pool.submit(func, *args)
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as e:
        # 3.10 上 concurrent.futures.TimeoutError 与内建 TimeoutError 不同类，
        # 统一规范化为内建 TimeoutError，保证上层 except 语义跨版本一致
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(
            f"执行超过 {timeout}s，已停止等待（工具线程可能仍在后台运行）"
        ) from e
    pool.shutdown(wait=True)
    return result


class NodeHooks:
    """节点级事件钩子（可选）：观测/审计用，钩子异常不影响工作流执行。"""

    def __init__(self, on_node_start=None, on_node_end=None):
        self.on_node_start = on_node_start  # (node_name, state) -> None
        self.on_node_end = on_node_end      # (node_name, result|None, error|None) -> None

    def _safe(self, event, *args):
        handler = getattr(self, event, None)
        if handler is None:
            return
        try:
            handler(*args)
        except Exception as e:  # noqa: BLE001 - 观测组件故障不阻断业务
            print(f"[flowagent] 钩子 {event} 执行失败: {e}", file=sys.stderr)

    def emit_start(self, name, state):
        self._safe("on_node_start", name, state)

    def emit_end(self, name, result, error):
        self._safe("on_node_end", name, result, error)


def _sub_workflow_engine(workflow_file: str, llm=None, checkpointer=None):
    """加载（并缓存）子图引擎；避免 import 环，延迟导入。

    缓存键含 llm/checkpointer 实例标识：子图与父级共用同一 LLM 客户端与
    检查点后端（实例一致性），不同父引擎各自缓存。
    """
    from .config import load_spec_file
    from .engine import WorkflowEngine

    key = (
        str(Path(workflow_file).resolve()) if workflow_file else workflow_file,
        id(llm),
        id(checkpointer),
    )
    if key not in _SUB_ENGINES:
        _SUB_ENGINES[key] = WorkflowEngine(
            load_spec_file(key[0]), llm=llm, checkpointer=checkpointer
        )
    return _SUB_ENGINES[key]


def _current_thread_id() -> str | None:
    """读取当前 LangGraph 执行上下文的 thread_id（用于派生稳定的子图线程标识）。

    在图内节点执行时可用；单元测试直接调用 execute_node 等场景下返回 None。
    """
    try:
        from langgraph.config import get_config
        cfg = get_config() or {}
        return (cfg.get("configurable") or {}).get("thread_id")
    except Exception:  # noqa: BLE001 - 无执行上下文时退回旧行为（随机 id）
        return None


def _enrich_payload(payload, node_name: str, sub_tid: str):
    """把子图 interrupt 载荷提升到父图时补充定位上下文（不覆盖原字段）。"""
    if isinstance(payload, dict):
        enriched = dict(payload)
        enriched.setdefault("workflow_node", node_name)
        enriched.setdefault("sub_thread_id", sub_tid)
        return enriched
    return payload


def execute_node(node: NodeConfig, state: dict, llm, node_name: str = "", checkpointer=None) -> dict:
    """执行单个节点，返回要合并进 vars 的增量字典。"""
    ctx = dict(state.get("vars") or {})

    if node.type == "template":
        return {node.output_key: render_template(node.template, ctx)}

    if node.type == "tool":
        func = load_tool(node.tool)
        kwargs = {}
        for arg_name, arg_value in (node.args or {}).items():
            if isinstance(arg_value, str) and arg_value.startswith("$"):
                kwargs[arg_name] = get_path(ctx, arg_value[1:])
            else:
                kwargs[arg_name] = arg_value
        return {node.output_key: func(**kwargs)}

    if node.type == "human":
        question = render_template(node.message, ctx)
        payload = {"node": node_name, "question": question}
        if node.options:
            payload["options"] = list(node.options)
        # 声明了 options 即强校验（失败关闭）：无效答复绝不进入业务分支，
        # 而是携带错误提示重新 interrupt，让人工重新作答；
        # interrupt() 抛出的 GraphBubbleUp 由外层原样上抛（见 _run_with_reliability）
        while True:
            answer = interrupt(payload)
            if not node.options or answer in node.options:
                return {node.output_key or "human_decision": answer}
            payload = {
                **payload,
                "error": f"输入 {answer!r} 不在可选范围 {list(node.options)}，请重新答复",
            }

    if node.type == "workflow":
        from .engine import WorkflowEngine  # 延迟导入避免环

        engine = _sub_workflow_engine(node.workflow_file, llm=llm, checkpointer=checkpointer)
        # 稳定子线程 ID：由父 thread_id + 节点名派生。此前每次 run 随机生成，
        # 子图中断后检查点根本无法再次定位（评审 4.4 的隐性根因之一）
        parent_tid = _current_thread_id()
        sub_tid = f"{parent_tid}::{node_name}" if parent_tid else uuid.uuid4().hex[:8]
        snapshot = engine.get_state(sub_tid)
        pending = WorkflowEngine.pending_payload(snapshot)
        if pending is not None:
            # 父图恢复后本节点重放：子图仍挂起中断。
            # 先 interrupt() 取回父级 resume 值，再转发给子图
            value = interrupt(_enrich_payload(pending, node_name, sub_tid))
            sub_final = engine.resume(value, thread_id=sub_tid)
        else:
            sub_final = engine.run(dict(ctx), thread_id=sub_tid)
        # 子图可能再次中断（多人工节点/无效答复重新询问）：
        # 持续把中断提升到父图，直到子图真正结束
        while WorkflowEngine.waiting_on(sub_final) is not None:
            payload = WorkflowEngine.waiting_on(sub_final)
            value = interrupt(_enrich_payload(payload, node_name, sub_tid))
            sub_final = engine.resume(value, thread_id=sub_tid)
        sub_vars = dict(sub_final.get("vars") or {})
        if node.output_key:
            return {node.output_key: sub_vars}
        return sub_vars

    if node.type == "llm":
        if llm is None:
            raise FlowAgentError("LLM 未初始化")
        # Mock 模式下，节点若配置了 mock_response 则渲染后当作 LLM 输出，
        # 使离线演示与测试具备确定语义；真实 LLM 忽略 mock_response。
        if getattr(llm, "name", "") == "mock" and node.mock_response:
            content = render_template(node.mock_response, ctx)
        else:
            messages = []
            if node.system:
                messages.append(SystemMessage(render_template(node.system, ctx)))
            messages.append(HumanMessage(render_template(node.prompt, ctx)))
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):  # 某些供应商返回内容块列表
                content = "".join(str(block) for block in content)
        parsed = parse_output(content, node.parser, node.labels)
        # 声明了 output_schema 的输出必须过字段校验：缺字段/类型错/越界
        # 都视为解析失败（失败关闭），交给 retries/on_error 或直接失败
        if node.output_schema:
            validate_schema(parsed, node.output_schema)
        return {node.output_key: parsed}

    raise FlowAgentError(f"未知节点类型: {node.type}")


def _run_with_reliability(
    name: str, node: NodeConfig, state: dict, llm, checkpointer=None
) -> dict:
    """重试 + 超时包装（human 节点直通，见模块 docstring）。

    非 human 节点的 execute_node 不会产生 GraphInterrupt（interrupt 仅存在于
    human 节点），因此这里只需处理普通异常；控制流异常的再抛出统一由外层
    executor 负责。
    """
    if node.type == "human":
        return execute_node(node, state, llm, node_name=name)
    attempts = node.retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return _with_timeout(
                execute_node, (node, state, llm, name, checkpointer), node.timeout
            )
        except TimeoutError:
            # 超时不是瞬态故障：工具可能仍在后台执行并产生副作用，
            # 自动重试会造成副作用重复，一律立即上抛（评审 4.3）
            raise
        except Exception:
            if attempt >= attempts:
                raise
            if node.retry_delay > 0:
                time.sleep(node.retry_delay * (node.backoff ** (attempt - 1)))


def make_node_executor(name: str, node: NodeConfig, llm, checkpointer=None, hooks: "NodeHooks | None" = None):
    """构造 LangGraph 节点函数：执行业务逻辑 + 事件钩子 + 轨迹/计数 + 错误降级。"""

    def executor(state: WFState) -> dict:
        if hooks is not None:
            hooks.emit_start(name, dict(state))
        counts = dict(state.get("loop_counts") or {})
        counts[name] = counts.get(name, 0) + 1
        started = time.monotonic()
        try:
            result = _run_with_reliability(name, node, state, llm, checkpointer)
        except GraphBubbleUp:
            raise  # interrupt 等控制流：LangGraph 运行时负责处理
        except Exception as exc:
            if hooks is not None:
                hooks.emit_end(name, None, exc)
            if node.on_error is None:
                raise
            error = {"node": name, "type": type(exc).__name__, "message": str(exc)}
            return {
                "vars": dict(node.on_error.set or {}),
                "loop_counts": counts,
                "trace": [
                    {
                        "node": name,
                        "status": "error",
                        "error": error,
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    }
                ],
                "error": error,
            }
        if hooks is not None:
            hooks.emit_end(name, result, None)
        return {
            "vars": result,
            "loop_counts": counts,
            "trace": [
                {
                    "node": name,
                    "status": "ok",
                    "outputs": _summarize(result),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
            ],
        }

    return executor
