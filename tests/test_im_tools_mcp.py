"""L1 单元测试 — im-tools-mcp（全部离线：Mock im + ASGI 进程内 MCP）。

覆盖测试计划的 L1 用例组：
  - find_group：模糊匹配、候选 ≤10、无匹配空、im 5xx -> im_unreachable；
  - fetch_group_messages：游标翻页拉全量、时间窗过滤、segment_hours 分段、
    max_chars 截断、coverage 统计；
  - 脱敏：语料只含 昵称/时间/文本，不含 sender_id/subject_id/uuid；
  - 鉴权：令牌缺失/无效 -> 401 明确错误码；令牌不落日志；
  - 契约：input_schema 与快照对拍（含 MCP list_tools 线上对拍）；
  - 协议：真实 mcp 2.2.0 客户端（含 flowagent McpEdgeClient）打到 ASGI 应用。
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx2
import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from flowagent.mcp_edge_client import McpEdgeClient, McpEdgeError
from services.im_tools_mcp import (
    FETCH_GROUP_MESSAGES_INPUT_SCHEMA,
    FIND_GROUP_INPUT_SCHEMA,
    build_app,
)
from services.im_tools_mcp.window import ANALYSIS_TIMEZONE
from services.im_tools_mcp.config import Settings
from services.im_tools_mcp.im_client import ImClient
from services.im_tools_mcp.tools import (
    fetch_group_messages,
    find_group,
)

pytestmark = pytest.mark.anyio

UTC = timezone.utc


# 处理`anyio``backend`相关逻辑。
# English: Handle anyio backend operations.
@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------- 假 im 服务

@dataclass
class FakeIm:
    """进程内假 im：按 im 公网接口语义分页/游标，可注入强制状态码。"""

    conversations: list[dict] = field(default_factory=list)
    details: dict[int, dict] = field(default_factory=dict)
    messages: dict[int, list[dict]] = field(default_factory=dict)
    forced_status: dict[str, int] = field(default_factory=dict)  # 路径前缀 -> 状态码
    message_page_calls: int = 0  # 9a 回归测试：记录 /messages 分页请求的实际次数

    # 处理处理器相关逻辑。
    # English: Handle handler operations.
    def handler(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        for prefix, status in self.forced_status.items():
            if path.startswith(prefix):
                return httpx2.Response(status, json={"detail": "forced"})
        if path == "/conversations/" and request.method == "GET":
            return self._list_conversations(request)
        match = re.fullmatch(r"/conversations/(\d+)", path)
        if match and request.method == "GET":
            return self._detail(int(match.group(1)))
        match = re.fullmatch(r"/conversations/(\d+)/messages", path)
        if match and request.method == "GET":
            return self._messages(int(match.group(1)), request)
        return httpx2.Response(404, json={"detail": "not found"})

    # 列出会话。
    # English: List conversations.
    def _list_conversations(self, request: httpx2.Request) -> httpx2.Response:
        page = int(request.url.params.get("page", "1"))
        page_size = int(request.url.params.get("page_size", "20"))
        start = (page - 1) * page_size
        items = self.conversations[start:start + page_size]
        total = len(self.conversations)
        return httpx2.Response(200, json={
            "items": items, "total": total, "page": page,
            "page_size": page_size, "has_more": page * page_size < total,
        })

    # 处理详情相关逻辑。
    # English: Handle detail operations.
    def _detail(self, conversation_id: int) -> httpx2.Response:
        detail = self.details.get(conversation_id)
        if detail is None:
            return httpx2.Response(404, json={"detail": "会话不存在"})
        return httpx2.Response(200, json=detail)

    # 处理消息相关逻辑。
    # English: Handle messages operations.
    def _messages(self, conversation_id: int, request: httpx2.Request) -> httpx2.Response:
        if conversation_id not in self.details:
            return httpx2.Response(404, json={"detail": "会话不存在"})
        self.message_page_calls += 1
        page_size = int(request.url.params.get("page_size", "50"))
        before_raw = request.url.params.get("before_id")
        after_raw = request.url.params.get("after_id")
        rows = sorted(self.messages.get(conversation_id, []), key=lambda m: m["id"])
        if before_raw:
            rows = [m for m in rows if m["id"] < int(before_raw)]
            rows_desc = list(reversed(rows))[: page_size + 1]
            has_more = len(rows_desc) > page_size
            items = list(reversed(rows_desc[:page_size]))
        elif after_raw:
            rows = [m for m in rows if m["id"] > int(after_raw)]
            items = rows[:page_size]
            has_more = len(rows) > page_size
        else:
            rows_desc = list(reversed(rows))[: page_size + 1]
            has_more = len(rows_desc) > page_size
            items = list(reversed(rows_desc[:page_size]))
        return httpx2.Response(200, json={
            "items": items, "total": len(items), "page": 1,
            "page_size": page_size, "has_more": has_more,
            "oldest_message_id": items[0]["id"] if items else None,
            "newest_message_id": items[-1]["id"] if items else None,
        })


# English: Handle msg operations.
def _msg(mid: int, sender: int, ts: str, content, mtype: str = "text", **extra) -> dict:
    """构造一条 im 消息（形状对齐 MessageViewV1.to_rest 的关键子集）。"""
    return {
        "id": mid, "conversation_id": 1, "sender_id": sender,
        "message_type": mtype, "content": content, "created_at": ts,
        "is_deleted": False, "file_name": extra.get("file_name"),
    }


# 执行群组对应的流程。
# English: Handle group operations.
def _group(cid: int, name: str, member_count: int = 5, ctype: str = "group") -> dict:
    return {"id": cid, "name": name, "type": ctype, "member_count": member_count}


# 处理详情相关逻辑。
# English: Handle detail operations.
def _detail(members: list[dict], name: str = "测试群") -> dict:
    return {"id": 1, "name": name, "type": "group", "members": members,
            "member_count": len(members)}


SETTINGS = Settings(base_url="http://im.test")


# 构建`im`。
# English: Create IM.
def _make_im(fake: FakeIm) -> ImClient:
    transport = httpx2.MockTransport(fake.handler)
    return ImClient(
        "http://im.test", "im-token", timeout_seconds=5,
        client=httpx2.AsyncClient(transport=transport, base_url="http://im.test"),
    )


# ---------------------------------------------------------------- find_group

# 验证“查找群组`fuzzy`匹配`case``insensitive`”场景下的预期行为。
# English: Verify find group fuzzy match case insensitive.
async def test_find_group_fuzzy_match_case_insensitive():
    fake = FakeIm(conversations=[
        _group(11, "项目协作群"), _group(12, "Project-X"),
        _group(13, "私密小群", ctype="private"),
    ])
    async with _make_im(fake) as im:
        result = await find_group(im, {"name_query": "project"})
    assert [g["conversation_id"] for g in result["groups"]] == [12]
    assert result["count"] == 1


# 验证“查找群组`caps`候选项`at``ten`”场景下的预期行为。
# English: Verify find group caps candidates at ten.
async def test_find_group_caps_candidates_at_ten():
    fake = FakeIm(conversations=[
        _group(i, f"售前群-{i}") for i in range(1, 26)
    ])
    async with _make_im(fake) as im:
        result = await find_group(im, {"name_query": "售前群"})
    assert result["count"] == 10
    assert len(result["groups"]) == 10


# 验证“查找群组无匹配`returns`空值不错误”场景下的预期行为。
# English: Verify find group no match returns empty not error.
async def test_find_group_no_match_returns_empty_not_error():
    fake = FakeIm(conversations=[_group(11, "项目协作群")])
    async with _make_im(fake) as im:
        result = await find_group(im, {"name_query": "不存在"})
    assert result == {"groups": [], "count": 0}


# 验证“查找群组`im``5xx`映射`unreachable`”场景下的预期行为。
# English: Verify find group IM 5xx maps to unreachable.
async def test_find_group_im_5xx_maps_to_unreachable():
    fake = FakeIm(forced_status={"/conversations": 503})
    async with _make_im(fake) as im:
        with pytest.raises(Exception) as exc_info:
            await find_group(im, {"name_query": "任意"})
    assert getattr(exc_info.value, "code", "") == "im_unreachable"


# ------------------------------------------------------ fetch_group_messages

# English: Handle august fixture operations.
def _august_fixture() -> FakeIm:
    """8/1–8/3（UTC）两天消息 + 7/31 与 8/4 的窗口外消息；id=6 已删除。"""
    deleted = _msg(6, 1003, "2026-08-02T13:00:00+00:00", "已删除消息")
    deleted["is_deleted"] = True
    return FakeIm(
        conversations=[_group(1, "项目协作群", 3)],
        details={1: _detail([
            {"user_id": 1001, "nickname": "阿明", "username": "aming", "full_name": "王小明"},
            {"user_id": 1002, "nickname": None, "username": "lihua", "full_name": "李华"},
            {"user_id": 1003, "nickname": "知言"},
        ])},
        messages={1: [
            _msg(1, 1001, "2026-07-31T23:00:00+00:00", "窗口前"),
            _msg(2, 1001, "2026-08-01T08:00:00+00:00", "8月1日 上午的任务安排"),
            _msg(3, 1002, "2026-08-01T09:30:00+00:00", "收到，我跟进一下"),
            _msg(4, 1001, "2026-08-02T10:00:00+00:00", "8月2日的评审结论"),
            _msg(5, 1002, "2026-08-02T12:00:00+00:00", "明天发布", mtype="image"),
            deleted,
            _msg(7, 1001, "2026-08-04T00:30:00+00:00", "窗口后"),
        ]},
    )


# 验证“获取筛选`window`与格式化`corpus`”场景下的预期行为。
# English: Verify fetch filters window and formats corpus.
async def test_fetch_filters_window_and_formats_corpus():
    fake = _august_fixture()
    async with _make_im(fake) as im:
        result = await fetch_group_messages(im, {
            "conversation_id": 1,
            "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-03T00:00:00+00:00",
            "segment_hours": 48,  # 整个 48h 窗口并为一段
        }, settings=SETTINGS)
    lines = result["corpus"].splitlines()
    assert lines == [
        "[08-01 08:00] 阿明: 8月1日 上午的任务安排",
        "[08-01 09:30] 李华: 收到，我跟进一下",
        "[08-02 10:00] 阿明: 8月2日的评审结论",
        "[08-02 12:00] 李华: [图片] 明天发布",
    ]
    assert result["truncated"] is False
    assert result["coverage"]["messages"] == 4
    assert result["coverage"]["first_ts"] == "2026-08-01T08:00:00+00:00"
    assert result["coverage"]["last_ts"] == "2026-08-02T12:00:00+00:00"
    assert result["segment"]["total"] == 1


# 验证“获取`segments`按`segment``hours`”场景下的预期行为。
# English: Verify fetch segments by segment hours.
async def test_fetch_segments_by_segment_hours():
    fake = _august_fixture()
    async with _make_im(fake) as im:
        args = {
            "conversation_id": 1,
            "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-03T00:00:00+00:00",
            "segment_hours": 24,
        }
        seg0 = await fetch_group_messages(im, {**args, "segment_index": 0}, settings=SETTINGS)
        seg1 = await fetch_group_messages(im, {**args, "segment_index": 1}, settings=SETTINGS)
    assert seg0["segment"]["total"] == 2
    assert seg0["segment"]["start"] == "2026-08-01T00:00:00+00:00"
    assert seg0["segment"]["end"] == "2026-08-02T00:00:00+00:00"
    assert seg0["coverage"]["messages"] == 2
    assert seg1["segment"]["start"] == "2026-08-02T00:00:00+00:00"
    assert seg1["segment"]["end"] == "2026-08-03T00:00:00+00:00"
    assert seg1["coverage"]["messages"] == 2  # 图片占位也算一条语料


# 验证“`end`日期纳入`day`与`boundary`消息判断是否不`duplicated`”场景下的预期行为。
# English: Verify end date includes day and boundary message is not duplicated.
async def test_end_date_includes_day_and_boundary_message_is_not_duplicated():
    from services.im_tools_mcp.tools import fetch_group_analysis_input
    fake = FakeIm(
        conversations=[_group(1, "项目协作群")],
        details={1: _detail([{"user_id": 1001, "nickname": "阿明"}])},
        messages={1: [
            _msg(1, 1001, "2026-08-02T00:00:00+00:00", "边界"),
            _msg(2, 1001, "2026-08-02T23:59:59.999999+00:00", "末尾"),
            _msg(3, 1001, "2026-08-03T00:00:00+00:00", "次日"),
        ]},
    )
    async with _make_im(fake) as im:
        direct = await fetch_group_analysis_input(im, {
            "conversation_id": 1, "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-03T00:00:00+00:00",
        }, settings=SETTINGS)
        seg0 = await fetch_group_messages(im, {
            "conversation_id": 1, "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-03T00:00:00+00:00", "segment_index": 0,
        }, settings=SETTINGS)
        seg1 = await fetch_group_messages(im, {
            "conversation_id": 1, "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-03T00:00:00+00:00", "segment_index": 1,
        }, settings=SETTINGS)
    assert [m["message_ref"] for m in direct["messages"]] == ["m1", "m2"]
    assert "边界" not in seg0["corpus"]
    assert "边界" in seg1["corpus"]


# English: Verify fetch group analysis input stops paging once past window start.
async def test_fetch_group_analysis_input_stops_paging_once_past_window_start():
    """评审性能问题 9a：翻到早于 window.start 的一页后应立即停止，不再翻更老的页。

    构造 3 页量级的消息（页大小 100）：第 1 页全部落在窗口内；第 2 页全部
    早于窗口起点（触发提前停止）；第 3 页更老。断言只翻了 2 页，且不会被
    误标为 page_limit（旧实现会翻满 max_pages 并误标）。
    """
    from services.im_tools_mcp.tools import fetch_group_analysis_input

    old_ts = "2026-01-01T00:00:00+00:00"        # 远早于窗口
    ancient_ts = "2025-01-01T00:00:00+00:00"    # 更早（不应被翻到）
    in_window = [
        _msg(i, 1001, (datetime(2026, 8, 1, tzinfo=UTC) + timedelta(minutes=i)).isoformat(),
             f"新消息{i}")
        for i in range(121, 221)   # 100 条，全部落在窗口内（最新一页）
    ]
    before_window = [_msg(i, 1001, old_ts, f"旧消息{i}") for i in range(21, 121)]      # 100 条
    ancient = [_msg(i, 1001, ancient_ts, f"远古消息{i}") for i in range(1, 21)]        # 20 条
    fake = FakeIm(
        conversations=[_group(1, "项目协作群")],
        details={1: _detail([{"user_id": 1001, "nickname": "阿明"}])},
        messages={1: ancient + before_window + in_window},
    )
    async with _make_im(fake) as im:
        result = await fetch_group_analysis_input(im, {
            "conversation_id": 1,
            "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-02T00:00:00+00:00",
        }, settings=SETTINGS)
    assert result["status"] == "complete"
    assert result["partial_reasons"] == []
    assert result["counts"]["fetched_unique"] == 100
    assert all("新消息" in msg["text"] for msg in result["messages"])
    # 核心断言：翻到第 2 页（全部早于窗口起点）后立即停止，不再翻第 3 页。
    assert fake.message_page_calls == 2


# 验证“获取`segment``index``out``range`”场景下的预期行为。
# English: Verify fetch segment index out of range.
async def test_fetch_segment_index_out_of_range():
    fake = _august_fixture()
    async with _make_im(fake) as im:
        with pytest.raises(Exception) as exc_info:
            await fetch_group_messages(im, {
                "conversation_id": 1,
                "start": "2026-08-01T00:00:00+00:00",
                "end": "2026-08-03T00:00:00+00:00",
                "segment_hours": 24, "segment_index": 2,
            }, settings=SETTINGS)
    assert getattr(exc_info.value, "code", "") == "segment_out_of_range"


# 验证“获取游标`pagination``pulls``everything`”场景下的预期行为。
# English: Verify fetch cursor pagination pulls everything.
async def test_fetch_cursor_pagination_pulls_everything():
    messages = [
        _msg(i, 1001, (datetime(2026, 8, 1, tzinfo=UTC) + timedelta(minutes=5 * i)).isoformat(),
             f"消息{i}")
        for i in range(1, 151)  # 150 条（5 分钟一条 ≈ 12.5h）-> 2 页（页大小 100）
    ]
    fake = FakeIm(
        conversations=[_group(1, "项目协作群")],
        details={1: _detail([{"user_id": 1001, "nickname": "阿明"}])},
        messages={1: messages},
    )
    async with _make_im(fake) as im:
        result = await fetch_group_messages(im, {
            "conversation_id": 1,
            "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-02T00:00:00+00:00",
        }, settings=SETTINGS)
    assert result["coverage"]["messages"] == 150
    assert result["corpus"].splitlines()[0].endswith("消息1")
    assert result["corpus"].splitlines()[-1].endswith("消息150")


# 验证“获取`truncates``corpus``at``max``chars`”场景下的预期行为。
# English: Verify fetch truncates corpus at max chars.
async def test_fetch_truncates_corpus_at_max_chars():
    messages = [
        _msg(i, 1001, (datetime(2026, 8, 1, tzinfo=UTC) + timedelta(minutes=i)).isoformat(),
             f"很长的消息内容{'x' * 200}")
        for i in range(1, 101)
    ]
    fake = FakeIm(
        conversations=[_group(1, "项目协作群")],
        details={1: _detail([{"user_id": 1001, "nickname": "阿明"}])},
        messages={1: messages},
    )
    async with _make_im(fake) as im:
        result = await fetch_group_messages(im, {
            "conversation_id": 1,
            "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-02T00:00:00+00:00",
            "max_chars": 3000,
        }, settings=SETTINGS)
    assert result["truncated"] is True
    assert len(result["corpus"]) == 3000


# 验证“获取默认项`window`判断是否`seven``days``relative`”场景下的预期行为。
# English: Verify fetch default window is seven days relative.
async def test_fetch_default_window_is_seven_days_relative():
    fake = FakeIm(conversations=[_group(1, "项目协作群")],
                  details={1: _detail([{"user_id": 1001, "nickname": "阿明"}])},
                  messages={1: []})
    async with _make_im(fake) as im:
        result = await fetch_group_messages(im, {"conversation_id": 1}, settings=SETTINGS)
    start = datetime.fromisoformat(result["window"]["start"])
    end = datetime.fromisoformat(result["window"]["end"])
    span_days = (end - start).total_seconds() / 86400
    assert 6.99 <= span_days <= 7.01  # 两次取 now 的微秒级漂移容差


# 验证“获取`relative`令牌解析”场景下的预期行为。
# English: Verify fetch relative tokens resolve.
async def test_fetch_relative_tokens_resolve():
    fake = FakeIm(conversations=[_group(1, "项目协作群")],
                  details={1: _detail([{"user_id": 1001, "nickname": "阿明"}])},
                  messages={1: []})
    async with _make_im(fake) as im:
        result = await fetch_group_messages(im, {
            "conversation_id": 1, "start": "now-30d", "end": "now",
        }, settings=SETTINGS)
    start = datetime.fromisoformat(result["window"]["start"])
    end = datetime.fromisoformat(result["window"]["end"])
    assert 29.9 <= (end - start).total_seconds() / 86400 <= 30.1


# 验证“获取`window``too``large``guard`”场景下的预期行为。
# English: Verify fetch window too large guard.
async def test_fetch_window_too_large_guard():
    fake = FakeIm(conversations=[_group(1, "项目协作群")],
                  details={1: _detail([{"user_id": 1001, "nickname": "阿明"}])},
                  messages={1: []})
    async with _make_im(fake) as im:
        with pytest.raises(Exception) as exc_info:
            await fetch_group_messages(im, {
                "conversation_id": 1,
                "start": "2026-08-01T00:00:00+00:00",
                "end": "2026-10-15T00:00:00+00:00",  # 跨度 75 天
            }, settings=SETTINGS)
    assert getattr(exc_info.value, "code", "") == "window_too_large"


# 验证“获取`cn`日期不包含`year`推断`recent``past`”场景下的预期行为。
# English: Verify fetch cn date without year infers recent past.
async def test_fetch_cn_date_without_year_infers_recent_past():
    fake = FakeIm(conversations=[_group(1, "项目协作群")],
                  details={1: _detail([{"user_id": 1001, "nickname": "阿明"}])},
                  messages={1: []})
    async with _make_im(fake) as im:
        result = await fetch_group_messages(im, {
            "conversation_id": 1, "start": "9月1日", "end": "9月15日",
        }, settings=SETTINGS)
    start = datetime.fromisoformat(result["window"]["start"])
    now = datetime.now(ANALYSIS_TIMEZONE)
    expected = datetime(now.year, 9, 1)
    if expected > now.replace(tzinfo=None):
        expected = datetime(now.year - 1, 9, 1)
    # 窗口记号按服务器本地时区解析，断言须换回本地时区再比对年月日
    local_start = start.astimezone(ANALYSIS_TIMEZONE)
    assert (local_start.year, local_start.month, local_start.day) == (
        expected.year, 9, 1
    )


# -------------------------------------------------------------------- 脱敏

# English: Verify corpus never contains internal identifiers.
async def test_corpus_never_contains_internal_identifiers():
    fake = _august_fixture()
    fake.details[1] = _detail([
        {"user_id": 1001, "nickname": "阿明", "subject_id": "uuid-should-not-leak"},
    ])
    fake.messages[1].append(_msg(99, 1001, "2026-08-01T18:00:00+00:00",
                                 "内容带标识 1001 与 550e8400-e29b-41d4-a716-446655440000"))
    async with _make_im(fake) as im:
        result = await fetch_group_messages(im, {
            "conversation_id": 1,
            "start": "2026-08-01T00:00:00+00:00",
            "end": "2026-08-03T00:00:00+00:00",
        }, settings=SETTINGS)
    corpus = result["corpus"]
    # 语料行格式只有 [MM-DD HH:mm] 昵称: 文本
    for line in corpus.splitlines():
        assert re.fullmatch(r"\[\d{2}-\d{2} \d{2}:\d{2}\] [^:]+: .*", line)
    # 用户消息正文里的内容原样保留（文本本身允许出现数字），
    # 但昵称映射与结构化字段绝不把 sender_id 当昵称、不出现 subject_id 字段名
    assert "subject_id" not in corpus
    assert "阿明" in corpus


# ---------------------------------------------------------------- MCP 协议层

# English: Handle app with fake operations.
def _app_with_fake(fake: FakeIm, *, shared_token: str = "") -> tuple:
    settings = Settings(base_url="http://im.test", shared_token=shared_token)

    # 处理工厂方法相关逻辑。
    # English: Handle factory operations.
    def factory(token: str) -> ImClient:
        transport = httpx2.MockTransport(fake.handler)
        return ImClient(
            "http://im.test", token, timeout_seconds=5,
            client=httpx2.AsyncClient(transport=transport, base_url="http://im.test"),
        )

    return build_app(settings, im_client_factory=factory), settings


# English: Handle asgi MCP session operations.
@asynccontextmanager
async def asgi_mcp_session(app, token: str = "im-token"):
    """官方 mcp 2.2.0 客户端打 ASGI 应用（显式进入父 lifespan）。

    端点必须带尾斜杠（/mcp/）：不带斜杠会命中子应用 307 重定向，
    而 MCP 客户端不跟随重定向。"""
    async with app.router.lifespan_context(app):
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx2.Timeout(30),
        )
        try:
            async with streamable_http_client(
                "http://testserver/mcp/", http_client=http
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        finally:
            await http.aclose()


# 验证“MCP列出群组`analysis`工具包含契约结构”场景下的预期行为。
# English: Verify MCP lists group analysis tools with contract schemas.
async def test_mcp_lists_group_analysis_tools_with_contract_schemas():
    app, _ = _app_with_fake(FakeIm())
    async with asgi_mcp_session(app) as session:
        result = await session.list_tools()
    by_name = {tool.name: tool for tool in result.tools}
    assert set(by_name) == {"find_group", "fetch_group_messages", "fetch_group_analysis_input",
                            "read_group_attachment"}
    assert by_name["find_group"].input_schema == FIND_GROUP_INPUT_SCHEMA
    assert by_name["fetch_group_messages"].input_schema == FETCH_GROUP_MESSAGES_INPUT_SCHEMA


# 验证“MCP`call`查找群组`returns``structured`内容”场景下的预期行为。
# English: Verify MCP call find group returns structured content.
async def test_mcp_call_find_group_returns_structured_content():
    fake = FakeIm(conversations=[_group(11, "项目协作群", 7)])
    app, _ = _app_with_fake(fake)
    async with asgi_mcp_session(app) as session:
        result = await session.call_tool("find_group", {"name_query": "项目"})
    assert result.is_error is False
    assert result.structured_content == {"groups": [
        {"conversation_id": 11, "name": "项目协作群", "member_count": 7},
    ], "count": 1}


# 验证“MCP工具错误判断是否错误包含`code`”场景下的预期行为。
# English: Verify MCP tool error is error with code.
async def test_mcp_tool_error_is_error_with_code():
    fake = FakeIm(forced_status={"/conversations": 503})
    app, _ = _app_with_fake(fake)
    async with asgi_mcp_session(app) as session:
        result = await session.call_tool("find_group", {"name_query": "x"})
    assert result.is_error is True
    assert result.structured_content["error_code"] == "im_unreachable"


# English: Verify flowagent MCP edge client round trip.
async def test_flowagent_mcp_edge_client_round_trip():
    """flowagent 真实 McpEdgeClient（阶段 2 的接线件）打本服务可用。"""
    fake = FakeIm(conversations=[_group(11, "项目协作群", 7)])
    app, _ = _app_with_fake(fake)

    # lifespan 只进一次：会话管理器 run() 每实例只允许一次（部署面同理，
    # 由进程生命周期承担；这里模拟"应用已就绪，客户端按请求开会话"）
    async with app.router.lifespan_context(app):

        # 处理会话工厂方法相关逻辑。
        # English: Handle session factory operations.
        def session_factory(endpoint, headers, timeout_seconds):
            # 处理会话相关逻辑。
            # English: Handle session operations.
            @asynccontextmanager
            async def _session():
                http = httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app),
                    headers=dict(headers), timeout=timeout_seconds,
                )
                async with streamable_http_client(endpoint, http_client=http) as streams:
                    read, write = streams
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
            return _session()

        client = McpEdgeClient(
            "http://testserver/mcp/", "im-token", session_factory=session_factory
        )
        tools = await client.list_tools()
        assert {t["name"] for t in tools} == {
            "find_group", "fetch_group_messages", "fetch_group_analysis_input",
            "read_group_attachment"}
        payload = await client.call_tool("find_group", {"name_query": "项目"})
        assert payload["count"] == 1
        with pytest.raises(McpEdgeError):
            await client.call_tool("fetch_group_messages", {"conversation_id": 999})


# ---------------------------------------------------------------------- 鉴权

# English: Verify bearer missing is allowed when token not configured.
def test_bearer_missing_is_allowed_when_token_not_configured():
    """评审 D-11：IM_TOOLS_MCP_TOKEN 未配置时放开鉴权，不再要求 Bearer 头。"""
    app, _ = _app_with_fake(FakeIm())  # shared_token 默认空串
    with TestClient(app) as client:
        response = client.post(
            "/mcp/", json={"jsonrpc": "2.0", "method": "initialize", "id": 1},
        )
    assert response.status_code != 401


# 验证“Bearer 令牌缺失`returns`401当令牌已配置”场景下的预期行为。
# English: Verify bearer missing returns 401 when token configured.
def test_bearer_missing_returns_401_when_token_configured():
    fake = FakeIm()
    app, _ = _app_with_fake(fake, shared_token="expected-shared")
    client = TestClient(app)
    response = client.post("/mcp", json={"jsonrpc": "2.0", "method": "initialize", "id": 1})
    assert response.status_code == 401
    assert response.json() == {"error": {"code": "unauthorized"}}


# 验证“Bearer 令牌`wrong``shared`令牌`returns`401”场景下的预期行为。
# English: Verify bearer wrong shared token returns 401.
def test_bearer_wrong_shared_token_returns_401():
    fake = FakeIm()
    app, _ = _app_with_fake(fake, shared_token="expected-shared")
    client = TestClient(app)
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "initialize", "id": 1},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


# 验证“Bearer 令牌有效`reaches`MCP`app`不401”场景下的预期行为。
# English: Verify bearer valid reaches MCP app not 401.
def test_bearer_valid_reaches_mcp_app_not_401():
    fake = FakeIm()
    app, _ = _app_with_fake(fake, shared_token="expected-shared")
    with TestClient(app) as client:  # with 语句驱动 lifespan（会话管理器就绪）
        response = client.post(
            "/mcp/", json={"jsonrpc": "2.0", "method": "initialize", "id": 1},
            headers={"Authorization": "Bearer expected-shared"},
        )
    assert response.status_code != 401


# 验证“令牌`never``logged`”场景下的预期行为。
# English: Verify token never logged.
def test_token_never_logged(caplog):
    import logging

    fake = FakeIm(conversations=[_group(11, "项目协作群")])
    app, _ = _app_with_fake(fake)
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            client.post(
                "/mcp/", json={"jsonrpc": "2.0", "method": "initialize", "id": 1},
                headers={"Authorization": "Bearer im-token"},
            )
    assert "im-token" not in caplog.text


# ---------------------------------------------------------------------- 契约

# English: Verify input schemas match admin registration snapshot.
def test_input_schemas_match_admin_registration_snapshot():
    """Admin mcp_tools 注册快照对拍：schema 变化必须显式更新此快照。"""
    assert FIND_GROUP_INPUT_SCHEMA == {
        "type": "object",
        "properties": {
            "name_query": {
                "type": "string",
                "description": "群名称关键词，服务端做大小写不敏感的模糊匹配",
            },
        },
        "required": ["name_query"],
        "additionalProperties": False,
    }
    assert FETCH_GROUP_MESSAGES_INPUT_SCHEMA == {
        "type": "object",
        "properties": {
            "conversation_id": {"type": "integer", "description": "群会话 ID"},
            "start": {
                "type": "string",
                "description": (
                    "分析窗口起点记号：ISO 日期(时间)、缺年日期(MM-DD / M月D日)、"
                    "或相对记号 now / now-Nd / now-Nh；与 end 同时缺省时按 days 取相对窗口"
                ),
            },
            "end": {
                "type": "string",
                "description": "分析窗口终点记号，格式同 start；缺省取 now",
            },
            "days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 31,
                "description": "相对窗口天数（start/end 均未提供时生效，默认 7）",
            },
            "segment_hours": {
                "type": "integer",
                "minimum": 1,
                "maximum": 720,
                "default": 24,
                "description": "分段时长（小时）；窗口按此切分为 segments_total 段",
            },
            "max_chars": {
                "type": "integer",
                "minimum": 1000,
                "maximum": 60000,
                "default": 30000,
                "description": "单段语料字符上限，超出截断并置 truncated=true",
            },
            "segment_index": {
                "type": "integer",
                "minimum": 0,
                "description": "取第几段（0 起计，上限见响应 segments_total-1）",
            },
        },
        "required": ["conversation_id"],
        "additionalProperties": False,
    }
