"""工作流引擎：默认 PostgreSQL 持久化；测试可选择 memory。

FLOWAGENT_DATABASE_URL 为连接串；引擎应使用 with 管理连接。
异步服务使用 open_postgres_checkpointer 管理原生异步连接。
"""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

try:  # langgraph 1.x 起更名，保留旧名回退
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import (  # type: ignore[attr-defined]
        InMemorySaver as MemorySaver,
    )
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .config import RouteConfig, WorkflowSpec
from .errors import ConfigError, FlowAgentError
from .llm import create_llm
from .nodes import make_node_executor
from .router import make_router
from .state import WFState


# 打开PostgreSQL检查点存储。
@asynccontextmanager
async def open_postgres_checkpointer(dsn: str | None = None) -> AsyncIterator[object]:
    """Open and initialize the async PostgreSQL saver for one app lifespan.

    Checkpoints share the configured PostgreSQL service but use LangGraph's own
    tables.  The serializer allowlist is deliberately strict because checkpoint
    rows are deserialized on every resume.
    """
    connection_string = (dsn or os.getenv("FLOWAGENT_DATABASE_URL", "")).strip()
    if not connection_string:
        raise ConfigError("FLOWAGENT_DATABASE_URL is required for PostgreSQL checkpoints")
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError as exc:
        raise ConfigError(
            "PostgreSQL checkpoints require the postgres extra: "
            "pip install '.[postgres]'"
        ) from exc

    from .postgres import SETUP_LOCK_SQL, SETUP_UNLOCK_SQL

    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    async with AsyncPostgresSaver.from_conn_string(
        connection_string, serde=serde,
    ) as saver:
        # 与同步入口共用会话锁，避免多实例首次启动争抢迁移表。
        while True:
            cursor = await saver.conn.execute(SETUP_LOCK_SQL)
            if (await cursor.fetchone())["acquired"]:
                break
            await asyncio.sleep(0.05)
        try:
            await saver.setup()
        finally:
            await saver.conn.execute(SETUP_UNLOCK_SQL)
        yield saver

_RECURSION_HINT = (
    "工作流在 recursion_limit={limit} 步内未结束，可能存在未设护栏的循环。"
    "可调大 workflow.recursion_limit，或为循环边配置 max_loops/loop_exhaust。"
)


def make_checkpointer(setting: str | None = None):
    """创建检查点；调用方负责关闭 PostgreSQL saver 连接。"""
    setting = (setting or os.getenv("FLOWAGENT_CHECKPOINT", "postgres")).strip().lower()
    if setting == "memory":
        return MemorySaver()
    if setting == "postgres":
        dsn = os.getenv("FLOWAGENT_DATABASE_URL", "").strip()
        if not dsn:
            raise ConfigError("FLOWAGENT_DATABASE_URL is required for PostgreSQL checkpoints")
        from .postgres import create_postgres_saver
        return create_postgres_saver(dsn)
    raise ConfigError(f"未知 FLOWAGENT_CHECKPOINT 配置: {setting}（可选 memory/postgres）")


# 执行解析对应的流程。
def _resolve(target: str):
    return END if target == "END" else target


def _route_path_map(route: RouteConfig, error_goto: str | None = None) -> dict:
    """条件边所有可能的返回值 -> 节点名映射（LangGraph 用它做静态校验）。"""
    mapping: dict = {}

    # 执行添加对应的流程。
    def add(target: str | None):
        if target:
            mapping[target] = _resolve(target)

    for rule in route.rules:
        add(rule.goto)
        add(rule.loop_exhaust)
    add(route.default)
    add(route.on_error)
    add(error_goto)  # node.on_error.goto 也可能成为条件边返回值（评审 4.5）
    return mapping


def _fixed_or_error(node_name: str, normal_targets, error_goto: str):
    """固定 next（字符串/并行列表）的错误改道路由：正常走原目标，错误走 goto。

    返回值保持原类型（列表 = 并行扇出），错误时改道单个 goto 目标。
    """
    # 处理`router`相关逻辑。
    def router(state: dict):
        err = state.get("error")
        if err and err.get("node") == node_name:
            return error_goto
        return normal_targets

    return router


def build_graph(spec: WorkflowSpec, llm=None, checkpointer=None, hooks=None, agent_runtime=None,
                stream_final_reply: bool = False):
    """将 WorkflowSpec 编译为可执行的 LangGraph CompiledStateGraph。"""
    if llm is None:
        llm = create_llm()
    if checkpointer is None:
        checkpointer = make_checkpointer()

    graph = StateGraph(WFState)
    for name, node in spec.nodes.items():
        graph.add_node(
            name,
            make_node_executor(name, node, llm, checkpointer, hooks, agent_runtime, stream_final_reply),
        )

    for name, node in spec.nodes.items():
        nxt = node.next
        error_goto = node.on_error.goto if node.on_error else None
        if isinstance(nxt, str):
            if error_goto:
                # 配置了 on_error.goto：固定边升级为条件边（错误时改道，评审 4.5）
                graph.add_conditional_edges(
                    name,
                    _fixed_or_error(name, nxt, error_goto),
                    path_map={nxt: _resolve(nxt), error_goto: _resolve(error_goto)},
                )
            else:
                graph.add_edge(name, _resolve(nxt))
        elif isinstance(nxt, list):
            if error_goto:
                graph.add_conditional_edges(
                    name,
                    _fixed_or_error(name, list(nxt), error_goto),
                    path_map={
                        **{t: _resolve(t) for t in nxt},
                        error_goto: _resolve(error_goto),
                    },
                )
            else:
                for target in nxt:  # 多条边 = 并行扇出，汇合点自动等待全部分支
                    graph.add_edge(name, _resolve(target))
        else:
            graph.add_conditional_edges(
                name, make_router(name, nxt, error_goto),
                path_map=_route_path_map(nxt, error_goto),
            )

    graph.add_edge(START, spec.workflow.start)
    # 受管 execution 服务传入 False，显式禁用跨请求 checkpoint。
    if checkpointer is False:
        return graph.compile()
    return graph.compile(checkpointer=checkpointer)


class WorkflowEngine:
    """工作流引擎门面：编译一次，多次运行/恢复/流式。"""

    # 初始化对象状态并建立所需依赖。
    def __init__(self, spec: WorkflowSpec, llm=None, checkpointer=None, hooks=None):
        self.spec = spec
        self.llm = llm if llm is not None else create_llm()
        self._owns_checkpointer = checkpointer is None
        self.checkpointer = make_checkpointer() if checkpointer is None else checkpointer
        self.hooks = hooks
        self.durability = spec.workflow.durability
        try:
            self.graph = build_graph(spec, self.llm, self.checkpointer, hooks)
        except BaseException:
            self.close()
            raise

    def close(self):
        """只释放引擎创建的连接；注入 saver 由调用方管理。"""
        if self._owns_checkpointer and hasattr(self.checkpointer, "close"):
            self.checkpointer.close()

    # 进入上下文并准备所需资源。
    def __enter__(self):
        return self

    # 退出上下文并释放所持资源。
    def __exit__(self, *exc):
        self.close()

    async def aresume(self, value, thread_id: str) -> dict:
        """异步恢复人工中断。"""
        try:
            return await self.graph.ainvoke(
                Command(resume=value), self._config(thread_id), durability=self.durability,
            )
        except GraphRecursionError as e:
            raise self._wrap_recursion() from e

    async def aget_state(self, thread_id: str):
        """异步读取持久化状态。"""
        return await self.graph.aget_state(self._config(thread_id))

    # 处理配置相关逻辑。
    def _config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.spec.workflow.recursion_limit,
        }

    def _wrap_recursion(self) -> FlowAgentError:
        """构造带修复建议的业务异常；原始因果链由调用处 raise ... from 建立。"""
        return FlowAgentError(
            _RECURSION_HINT.format(limit=self.spec.workflow.recursion_limit)
        )

    def run(self, inputs: dict | None = None, thread_id: str | None = None) -> dict:
        """运行工作流，返回最终状态（含 vars/trace/loop_counts/error）。

        若图中包含 human 节点并触发 interrupt，返回值会带有 "__interrupt__"
        键（暂停挂起信息），可用 waiting_on() 读取并调用 resume() 恢复。
        """
        tid = thread_id or uuid.uuid4().hex
        initial = {"vars": dict(inputs or {}), "trace": [], "loop_counts": {}, "error": None}
        try:
            return self.graph.invoke(
                initial, self._config(tid), durability=self.durability
            )
        except GraphRecursionError as e:
            raise self._wrap_recursion() from e

    def resume(self, value, thread_id: str) -> dict:
        """恢复一次因 human 节点而暂停的运行（LangGraph Command(resume=...)）。"""
        try:
            return self.graph.invoke(
                Command(resume=value),
                self._config(thread_id),
                durability=self.durability,
            )
        except GraphRecursionError as e:
            raise self._wrap_recursion() from e

    def stream(self, inputs: dict | None = None, thread_id: str | None = None):
        """逐步产出执行后的完整状态（stream_mode="values"），供 CLI/前端实时展示。"""
        tid = thread_id or uuid.uuid4().hex
        initial = {"vars": dict(inputs or {}), "trace": [], "loop_counts": {}, "error": None}
        try:
            yield from self.graph.stream(
                initial,
                self._config(tid),
                stream_mode="values",
                durability=self.durability,
            )
        except GraphRecursionError as e:
            raise self._wrap_recursion() from e

    async def arun(self, inputs: dict | None = None, thread_id: str | None = None) -> dict:
        """异步运行（语义同 run），适合 FastAPI 等异步服务。"""
        tid = thread_id or uuid.uuid4().hex
        initial = {"vars": dict(inputs or {}), "trace": [], "loop_counts": {}, "error": None}
        try:
            return await self.graph.ainvoke(
                initial, self._config(tid), durability=self.durability
            )
        except GraphRecursionError as e:
            raise self._wrap_recursion() from e

    async def astream(self, inputs: dict | None = None, thread_id: str | None = None):
        """异步流式（语义同 stream）。"""
        tid = thread_id or uuid.uuid4().hex
        initial = {"vars": dict(inputs or {}), "trace": [], "loop_counts": {}, "error": None}
        try:
            async for chunk in self.graph.astream(
                initial,
                self._config(tid),
                stream_mode="values",
                durability=self.durability,
            ):
                yield chunk
        except GraphRecursionError as e:
            raise self._wrap_recursion() from e

    @staticmethod
    def waiting_on(final: dict):
        """运行处于暂停（等待人工输入）时返回 interrupt 载荷，否则 None。"""
        if isinstance(final, dict):
            items = final.get("__interrupt__")
            if items:
                first = items[0]
                return getattr(first, "value", first)
        return None

    @staticmethod
    def pending_payload(snapshot):
        """从检查点快照提取待恢复的 interrupt 载荷（无则 None）。"""
        for task in getattr(snapshot, "tasks", None) or ():
            for intr in getattr(task, "interrupts", None) or ():
                return getattr(intr, "value", None)
        return None

    def get_state(self, thread_id: str):
        """读取某次运行的检查点状态（演示 checkpoint 持久化）。"""
        return self.graph.get_state({"configurable": {"thread_id": thread_id}})

    def mermaid(self) -> str:
        """导出 Mermaid 图源码，便于文档化与可视化。"""
        return self.graph.get_graph().draw_mermaid()
