"""检查点工厂与同步/异步 PostgreSQL 持久化测试。"""

import asyncio
import os
import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver

from flowagent import WorkflowEngine, load_spec
from flowagent.engine import make_checkpointer
from flowagent.errors import ConfigError
from flowagent.llm import MockLLM

SPEC = {
    "workflow": {"name": "checkpoint", "start": "ask"},
    "nodes": {
        "ask": {"type": "human", "message": "确认?", "output_key": "answer", "next": "done"},
        "done": {"type": "template", "template": "结果 {answer}", "output_key": "reply", "next": "END"},
    },
}


def test_default_requires_postgres_dsn(monkeypatch):
    monkeypatch.delenv("FLOWAGENT_CHECKPOINT", raising=False)
    monkeypatch.delenv("FLOWAGENT_DATABASE_URL", raising=False)
    with pytest.raises(ConfigError, match="FLOWAGENT_DATABASE_URL"):
        make_checkpointer()
    assert isinstance(make_checkpointer("memory"), MemorySaver)


def test_invalid_setting():
    with pytest.raises(ConfigError):
        make_checkpointer("removed-backend")


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
@pytest.mark.parametrize("asynchronous", [False, True])
def test_persistent_human_resume_and_connection_close(monkeypatch, asynchronous):
    monkeypatch.setenv("FLOWAGENT_CHECKPOINT", "postgres")
    tid = uuid.uuid4().hex
    with WorkflowEngine(load_spec(SPEC), MockLLM()) as first:
        if asynchronous:
            paused = asyncio.run(first.arun({}, tid))
        else:
            paused = first.run({}, tid)
        assert first.waiting_on(paused)
        connection = first.checkpointer.conn
    assert connection.closed
    with WorkflowEngine(load_spec(SPEC), MockLLM()) as second:
        if asynchronous:
            async def resume():
                assert second.pending_payload(await second.aget_state(tid))
                final = await second.aresume("通过", tid)
                chunks = [item async for item in second.astream({}, uuid.uuid4().hex)]
                assert chunks
                return final
            final = asyncio.run(resume())
        else:
            assert second.pending_payload(second.get_state(tid))
            final = second.resume("通过", tid)
            assert list(second.stream({}, uuid.uuid4().hex))
        assert final["vars"]["reply"] == "结果 通过"
    with make_checkpointer("postgres") as saver:
        with WorkflowEngine(load_spec(SPEC), MockLLM(), checkpointer=saver):
            pass
        assert not saver.conn.closed
    assert saver.conn.closed


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
def test_setup_failure_closes_connection(monkeypatch):
    from flowagent.postgres import ManagedPostgresSaver

    connections = []

    def fail_setup(saver):
        connections.append(saver.conn)
        raise RuntimeError("setup failed")

    monkeypatch.setattr(ManagedPostgresSaver, "setup", fail_setup)
    with pytest.raises(RuntimeError, match="setup failed"):
        make_checkpointer("postgres")
    assert connections[0].closed


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
def test_native_async_engine_resume_and_close():
    from flowagent.engine import open_postgres_checkpointer

    async def roundtrip():
        tid = uuid.uuid4().hex
        async with open_postgres_checkpointer() as saver:
            first = WorkflowEngine(load_spec(SPEC), MockLLM(), checkpointer=saver)
            assert first.waiting_on(await first.arun({}, tid))
            connection = saver.conn
        assert connection.closed
        async with open_postgres_checkpointer() as saver:
            second = WorkflowEngine(load_spec(SPEC), MockLLM(), checkpointer=saver)
            assert second.pending_payload(await second.aget_state(tid))
            final = await second.aresume("通过", tid)
            assert final["vars"]["reply"] == "结果 通过"
    # Windows 的 psycopg 异步连接需要 selector 事件循环。
    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(roundtrip())
    finally:
        loop.close()


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
def test_legacy_api_resume_after_lifespan_restart(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from examples import api_server

    monkeypatch.setenv("FLOWAGENT_CHECKPOINT", "postgres")
    tid = uuid.uuid4().hex
    with TestClient(api_server.app) as client:
        response = client.post("/runs", json={"thread_id": tid, "inputs": {"ticket": "PG", "amount": 1}})
        assert response.status_code == 202
        connection = api_server.get_engine().checkpointer.conn
    assert connection.closed
    with TestClient(api_server.app) as client:
        assert client.get(f"/runs/{tid}").json()["status"] == "waiting_human"
        response = client.post(f"/runs/{tid}/resume", json={"value": "批准"})
        assert response.status_code == 200
        assert response.json()["vars"]["decision"] == "批准"


def test_execution_service_default_disables_checkpoint_without_env(monkeypatch):
    """评审问题 2：受管 execution 服务默认不再依赖 Postgres（3 条现有 flow
    都不需要 HITL）；未设置 FLOWAGENT_CHECKPOINT 时不应报错，也不建库连接。
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from flowagent.execution_service import build_execution_app

    monkeypatch.delenv("FLOWAGENT_CHECKPOINT", raising=False)
    monkeypatch.delenv("FLOWAGENT_DATABASE_URL", raising=False)
    app = build_execution_app("test-token")
    with TestClient(app):
        assert app.state.checkpointer is None


def test_execution_service_multiple_requests_never_touch_checkpoint_when_disabled(monkeypatch):
    """WF-CKPT-02：checkpoint 默认关闭时，连续多次真实请求不应触发任何
    checkpoint 读写——build_graph 收到 checkpointer=False，图内部根本不持有
    saver 实例，因此不是"写入被跳过"，而是"从未触碰"。
    """
    import hashlib
    import json

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from flowagent.execution_service import build_execution_app

    monkeypatch.delenv("FLOWAGENT_CHECKPOINT", raising=False)
    monkeypatch.delenv("FLOWAGENT_DATABASE_URL", raising=False)
    app = build_execution_app("test-token")

    payload = {
        "protocol_version": "execution-v1", "execution_id": "e1",
        "trace": {"trace_id": "e1"}, "binding_generation": 1,
        "continuity_key": "c1", "message": "hello",
        "identity": {"user_id": "u1"}, "scene": {"type": "chat"},
        "agent_config": {
            "agent_id": "604b6209-30da-4e57-bf93-73c6867cc101", "agent_type": "workflow",
            "revision_id": "704b6209-30da-4e57-bf93-73c6867cc101",
        },
    }
    snapshot = {
        "schema_version": 1,
        "agent": {"id": payload["agent_config"]["agent_id"], "type": "workflow", "name": "Workflow"},
        "adapter": {"endpoint": "http://workflow:8080"},
        "model": None,
        "system_prompt": "answer",
        "workflow": {"yaml": "workflow:\n  name: t\n  start: answer\nnodes:\n  answer:\n    type: template\n    template: '{message}'\n    output_key: reply\n    next: END\n"},
    }
    payload["agent_config"]["config_digest"] = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["agent_config"]["snapshot"] = snapshot

    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/execution-v1/respond", json=payload,
                headers={"Authorization": "Bearer test-token"},
            )
            assert response.status_code == 200
            assert response.json()["reply"] == "hello"
        assert app.state.checkpointer is None


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
def test_concurrent_sync_async_setup_on_fresh_schema_then_repeat():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import psycopg

    from flowagent.engine import open_postgres_checkpointer
    from flowagent.postgres import SETUP_LOCK_SQL, SETUP_UNLOCK_SQL

    dsn = os.environ["FLOWAGENT_DATABASE_URL"]

    def initialize(index, barrier):
        barrier.wait(timeout=10)
        if index % 2:
            async def open_async():
                async with open_postgres_checkpointer(dsn) as saver:
                    assert not saver.conn.closed
            loop = asyncio.SelectorEventLoop()
            try:
                loop.run_until_complete(open_async())
            finally:
                loop.close()
        else:
            with make_checkpointer("postgres") as saver:
                assert not saver.conn.closed

    # 首次并发建表与已完成迁移后的重复启动都必须成功。
    for _ in range(2):
        barrier = Barrier(6)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(initialize, index, barrier) for index in range(6)]
            for future in futures:
                future.result(timeout=30)
    with psycopg.connect(dsn, autocommit=True) as conn:
        versions = conn.execute("SELECT v FROM checkpoint_migrations ORDER BY v").fetchall()
        assert len(versions) > 1
        assert len(versions) == len(set(versions))
        assert conn.execute(SETUP_LOCK_SQL).fetchone()[0]
        conn.execute(SETUP_UNLOCK_SQL)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
def test_async_setup_failure_releases_connection_and_lock(monkeypatch):
    import psycopg
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from flowagent.engine import open_postgres_checkpointer
    from flowagent.postgres import SETUP_LOCK_SQL, SETUP_UNLOCK_SQL

    connections = []

    async def fail_setup(saver):
        connections.append(saver.conn)
        raise RuntimeError("async setup failed")

    monkeypatch.setattr(AsyncPostgresSaver, "setup", fail_setup)

    async def attempt():
        with pytest.raises(RuntimeError, match="async setup failed"):
            async with open_postgres_checkpointer():
                pass
    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(attempt())
    finally:
        loop.close()
    assert connections[0].closed
    with psycopg.connect(os.environ["FLOWAGENT_DATABASE_URL"], autocommit=True) as conn:
        assert conn.execute(SETUP_LOCK_SQL).fetchone()[0]
        conn.execute(SETUP_UNLOCK_SQL)
