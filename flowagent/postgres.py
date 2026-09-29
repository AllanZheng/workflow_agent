"""同步 PostgreSQL saver，同时支持异步图调用。"""

import asyncio
import time

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import Connection
from psycopg.rows import dict_row

# 按数据库与 schema 共用会话锁；迁移含并发建索引，不能放进事务锁。
_SETUP_LOCK_KEY = "hashtextextended(current_database() || ':' || current_schema() || ':flowagent-checkpoint-setup', 0)"
SETUP_LOCK_SQL = f"SELECT pg_try_advisory_lock({_SETUP_LOCK_KEY}) AS acquired"
SETUP_UNLOCK_SQL = f"SELECT pg_advisory_unlock({_SETUP_LOCK_KEY})"


class ManagedPostgresSaver(PostgresSaver):
    """异步方法在线程中执行；父类锁保护同一连接。"""

    # 处理`aget``tuple`相关逻辑。
    # English: Handle aget tuple operations.
    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    # 处理`aput`相关逻辑。
    # English: Handle aput operations.
    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    # 处理`aput`写入相关逻辑。
    # English: Handle aput writes operations.
    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    # 处理`alist`相关逻辑。
    # English: Handle alist operations.
    async def alist(self, config, *, filter=None, before=None, limit=None):
        # 在线程内消费游标，避免阻塞事件循环。
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    # 关闭当前连接并释放相关资源。
    # English: Close the current operation.
    def close(self):
        self.conn.close()

    # 进入上下文并准备所需资源。
    # English: Enter the context and return its resource.
    def __enter__(self):
        return self

    # 退出上下文并释放所持资源。
    # English: Exit the context and release its resource.
    def __exit__(self, *exc):
        self.close()


# English: Create postgres saver.
def create_postgres_saver(dsn):
    """初始化失败时也释放连接。"""
    conn = Connection.connect(dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    try:
        saver = ManagedPostgresSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
        # 阻塞式锁等待会保留查询快照，与并发建索引形成死锁；轮询间释放快照。
        while not conn.execute(SETUP_LOCK_SQL).fetchone()["acquired"]:
            time.sleep(0.05)
        try:
            saver.setup()
        finally:
            conn.execute(SETUP_UNLOCK_SQL)
        return saver
    except BaseException:
        conn.close()
        raise
