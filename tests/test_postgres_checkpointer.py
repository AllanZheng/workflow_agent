"""PostgreSQL checkpointer integration; isolated by a per-test schema."""

import asyncio
import os
import uuid

import pytest

from flowagent.config import load_spec
from flowagent.engine import open_postgres_checkpointer
from flowagent.llm import MockLLM

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

SPEC = {
    "workflow": {"name": "postgres-checkpoint", "start": "reply"},
    "nodes": {
        "reply": {
            "type": "template", "template": "saved {value}",
            "output_key": "reply", "next": "END",
        }
    },
}


@pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not configured")
def test_async_postgres_saver_persists_graph_state_across_connections():
    async def run():
        import psycopg
        from psycopg.conninfo import make_conninfo
        from psycopg.rows import dict_row

        from flowagent.engine import build_graph

        schema = f"flowagent_test_{uuid.uuid4().hex}"
        dsn = make_conninfo(DATABASE_URL, options=f"-c search_path={schema}")
        async with await psycopg.AsyncConnection.connect(
            DATABASE_URL, autocommit=True, row_factory=dict_row,
        ) as admin_conn:
            await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
            try:
                thread_id = uuid.uuid4().hex
                async with open_postgres_checkpointer(dsn) as saver:
                    graph = build_graph(load_spec(SPEC), MockLLM(), checkpointer=saver)
                    final = await graph.ainvoke(
                        {"vars": {"value": "round-trip"}, "trace": [],
                         "loop_counts": {}, "error": None},
                        {"configurable": {"thread_id": thread_id}},
                    )
                    assert final["vars"]["reply"] == "saved round-trip"

                async with open_postgres_checkpointer(dsn) as saver:
                    graph = build_graph(load_spec(SPEC), MockLLM(), checkpointer=saver)
                    snapshot = await graph.aget_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    assert snapshot.values["vars"]["reply"] == "saved round-trip"
            finally:
                await admin_conn.execute(f'DROP SCHEMA "{schema}" CASCADE')

    asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop)


@pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not configured")
def test_postgres_checkpoint_resumes_pending_human_node_after_reconnect():
    async def run():
        import psycopg
        from langgraph.types import Command
        from psycopg.conninfo import make_conninfo
        from psycopg.rows import dict_row

        from flowagent.engine import build_graph
        from flowagent.engine import WorkflowEngine

        human_spec = {
            "workflow": {"name": "postgres-hitl", "start": "approve"},
            "nodes": {
                "approve": {
                    "type": "human", "message": "Approve {ticket}?",
                    "output_key": "decision", "next": "finish",
                },
                "finish": {
                    "type": "template", "template": "Decision: {decision}",
                    "output_key": "reply", "next": "END",
                },
            },
        }
        schema = f"flowagent_test_{uuid.uuid4().hex}"
        dsn = make_conninfo(DATABASE_URL, options=f"-c search_path={schema}")
        async with await psycopg.AsyncConnection.connect(
            DATABASE_URL, autocommit=True, row_factory=dict_row,
        ) as admin_conn:
            await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
            try:
                thread_id = uuid.uuid4().hex
                config = {"configurable": {"thread_id": thread_id}}
                async with open_postgres_checkpointer(dsn) as saver:
                    graph = build_graph(load_spec(human_spec), MockLLM(), checkpointer=saver)
                    paused = await graph.ainvoke(
                        {"vars": {"ticket": "T-42"}, "trace": [],
                         "loop_counts": {}, "error": None}, config,
                    )
                    approval = WorkflowEngine.waiting_on(paused)
                    assert approval["node"] == "approve"
                    assert approval["question"] == "Approve T-42?"

                async with open_postgres_checkpointer(dsn) as saver:
                    graph = build_graph(load_spec(human_spec), MockLLM(), checkpointer=saver)
                    final = await graph.ainvoke(Command(resume="approved"), config)
                    assert final["vars"]["reply"] == "Decision: approved"
            finally:
                await admin_conn.execute(f'DROP SCHEMA "{schema}" CASCADE')

    asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop)


def test_postgres_checkpointer_requires_connection_string(monkeypatch):
    from flowagent.errors import ConfigError

    monkeypatch.delenv("FLOWAGENT_DATABASE_URL", raising=False)

    async def run():
        with pytest.raises(ConfigError, match="FLOWAGENT_DATABASE_URL"):
            async with open_postgres_checkpointer():
                pass

    asyncio.run(run())
