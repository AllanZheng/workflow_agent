"""测试：checkpointer 工厂与 sqlite 持久化。"""

import sqlite3

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from flowagent import WorkflowEngine, load_spec
from flowagent.engine import make_checkpointer
from flowagent.errors import ConfigError
from flowagent.llm import MockLLM

SPEC = {
    "workflow": {"name": "ck", "start": "a"},
    "nodes": {
        "a": {"type": "template", "template": "hi {x}", "output_key": "msg", "next": "END"}
    },
}


class TestMakeCheckpointer:
    def test_default_memory(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_CHECKPOINT", raising=False)
        assert isinstance(make_checkpointer(), MemorySaver)
        assert isinstance(make_checkpointer("memory"), MemorySaver)

    def test_sqlite_by_argument(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_CHECKPOINT_PATH", str(tmp_path / "a.sqlite"))
        assert isinstance(make_checkpointer("sqlite"), SqliteSaver)

    def test_sqlite_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_CHECKPOINT", "sqlite")
        monkeypatch.setenv("FLOWAGENT_CHECKPOINT_PATH", str(tmp_path / "b.sqlite"))
        assert isinstance(make_checkpointer(), SqliteSaver)

    def test_sqlite_missing_dependency_message(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langgraph.checkpoint.sqlite":
                raise ImportError("boom")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ConfigError):
            make_checkpointer("sqlite")

    def test_invalid_setting(self):
        with pytest.raises(ConfigError):
            make_checkpointer("redis")


class TestSqlitePersistence:
    def _engine(self, db):
        saver = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
        return WorkflowEngine(load_spec(SPEC), MockLLM(), checkpointer=saver)

    def test_state_survives_new_engine_instance(self, tmp_path):
        db = str(tmp_path / "ck.sqlite")
        engine1 = self._engine(db)
        final = engine1.run({"x": "持久"}, thread_id="p1")
        engine2 = self._engine(db)  # 全新实例（模拟进程重启）
        snap = engine2.get_state("p1").values
        assert snap["vars"] == final["vars"]
        assert snap["vars"]["msg"] == "hi 持久"

    def test_injected_checkpointer_used_by_graph(self, tmp_path):
        db = str(tmp_path / "ck2.sqlite")
        engine = self._engine(db)
        engine.run({"x": "y"}, thread_id="p2")
        # 换一个 saver 连接同一文件验证落盘
        saver2 = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
        engine3 = WorkflowEngine(load_spec(SPEC), MockLLM(), checkpointer=saver2)
        assert engine3.get_state("p2").values["vars"]["x"] == "y"
