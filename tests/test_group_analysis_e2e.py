"""L3 离线端到端 — 群分析受管工作流全链路（进程内 ASGI，零外部依赖）。

链路（技术方案 §1 的 workflow_agent 侧全段）：
  execution-v1 HTTP 端点（真实 FastAPI app）
    -> 受管引擎（真实 build_graph/agent_runtime/McpEdgeClient，真 MCP 协议）
    -> im-tools-mcp（真实服务 ASGI）
    -> 假 im（httpx2 MockTransport）

  LLM 用注入式替身（monkeypatch _request_llm，与既有执行服务测试同款）：
  parse/stats/report 按节点语义出活；fetch 的工具循环按
  find_group -> seg0 -> 并行取余段 -> digest 状态机驱动，真实穿过 MCP 协议。

覆盖测试计划 L3 六用例：
  1 正常 7 天分析；2 缺群名反问+补充后成功（history 生效）；3 30 天分段多轮；
  4 分段超护栏采样兜底+覆盖率声明；5 im 不可达用户可读降级；6 SSE 事件序合法。
"""

from __future__ import annotations

import ast
import hashlib
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx2
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import flowagent.execution_service as execution_service
from flowagent.execution_service import build_execution_app
from flowagent.mcp_edge_client import McpEdgeClient
from services.im_tools_mcp import build_app as build_im_tools_app
from services.im_tools_mcp.config import Settings
from services.im_tools_mcp.im_client import ImClient

# 复用 L1 的假 im（分页/游标/详情语义与 im 公网接口一致）
from tests.test_im_tools_mcp import FakeIm, _detail, _group, _msg

pytestmark = pytest.mark.anyio


def _parse_tool_payload(text: str) -> dict:
    """解析 ToolMessage 内容：引擎把 dict 结果 str() 成 Python repr（单引号），
    真实模型两种文本都能读；替身同样两者兼容。"""
    try:
        payload = json.loads(text)
    except ValueError:
        payload = ast.literal_eval(text)
    assert isinstance(payload, dict)
    return payload


UTC = timezone.utc
AGENT_ID = "604b6209-30da-4e57-bf93-73c6867cc101"
REVISION_ID = "704b6209-30da-4e57-bf93-73c6867cc101"
YAML_TEXT = (Path(__file__).resolve().parents[1]
             / "configs" / "managed" / "group_analysis.yaml").read_text(encoding="utf-8")


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ------------------------------------------------------------ LLM 替身

def _is_tool_message(message) -> bool:
    return type(message).__name__ == "ToolMessage"


def _segment_payloads(messages) -> list[dict]:
    payloads = []
    for message in messages:
        if not _is_tool_message(message):
            continue
        try:
            payload = _parse_tool_payload(str(message.content))
        except (ValueError, SyntaxError, AssertionError):
            continue
        if "segment" in payload:
            payloads.append(payload)
    return payloads


class AgentLoopModel:
    """受管 agent 工具循环的状态机替身（bind_tools 的返回值）。

    行为对齐 group_analysis.yaml 的 fetch_sys 指令：
    find_group -> fetch 第 0 段 -> 读 segment.total 并行取余段 -> 输出分段纪要。
    sample_cap 模拟"段数过多时采样"的护栏行为（用例 4）。
    完成判定：ToolMessage 数 >= 已发调用数（runtime 每轮把全部结果 append 后
    才进入下一次 ainvoke）。
    """

    def __init__(self, days: int, sample_cap: int | None = None):
        self.days = days
        self.sample_cap = sample_cap
        self.issued = 0            # 已发出的工具调用总数
        self.conversation_id: int | None = None
        self.parallel_issued = False

    async def ainvoke(self, messages):
        tool_messages = [m for m in messages if _is_tool_message(m)]
        if not tool_messages:
            self.issued = 1
            return self._call("fetch_group_messages", {"conversation_id": 11, "days": self.days}, "seg0")
        assert len(tool_messages) >= self.issued, "工具结果未到齐不应再次进入循环"

        payload = _parse_tool_payload(str(tool_messages[-1].content))
        self.conversation_id = 11

        segments = _segment_payloads(messages)
        assert segments, "fetch 结果应携带分段元数据"
        total = segments[0]["segment"]["total"]
        if not self.parallel_issued:
            self.parallel_issued = True
            remaining = list(range(1, total))
            if self.sample_cap is not None and total > self.sample_cap:
                # 采样兜底：含第 0 段在内最多 sample_cap 段（隔段均匀取样）
                step = max(1, total // self.sample_cap)
                remaining = sorted(set(range(1, total, step)))[: self.sample_cap - 1]
            if remaining:
                self.issued += len(remaining)
                return AIMessage(content="", tool_calls=[
                    self._spec("fetch_group_messages", {
                        "conversation_id": self.conversation_id,
                        "days": self.days, "segment_index": index,
                    }, f"seg{index}")
                    for index in remaining
                ])
        return AIMessage(content=self._digest(messages, total))

    def _digest(self, messages, total: int) -> str:
        segments = _segment_payloads(messages)
        covered = sum(seg.get("coverage", {}).get("messages", 0) for seg in segments)
        got = len({seg["segment"]["index"] for seg in segments})
        window = segments[0].get("window", {})
        coverage_note = (
            f"采样覆盖 {got}/{total} 段" if got < total else f"全量覆盖 {total}/{total} 段"
        )
        return (
            "【分段纪要】\n"
            f"分析窗口：{window.get('start', '?')} ~ {window.get('end', '?')}\n"
            f"分段覆盖：{coverage_note}\n"
            "分类计数：问题 12 条；通知 8 条；讨论 20 条；闲聊 5 条；其他 3 条\n"
            "关键事项：完成了发布准备与回归验证。\n"
            f"覆盖率：纳入约 {covered} 条消息"
        )

    @staticmethod
    def _spec(name: str, args: dict, call_id: str) -> dict:
        return {"name": name, "args": args, "id": call_id, "type": "tool_call"}

    @classmethod
    def _call(cls, name: str, args: dict, call_id: str) -> AIMessage:
        return AIMessage(content="", tool_calls=[cls._spec(name, args, call_id)])

    async def astream(self, messages):
        yield await self.ainvoke(messages)


class E2ELLM:
    """按节点语义出活的注入式 LLM：parse/stats/report + agent bind_tools。"""

    name = "e2e-fake"

    def __init__(self, *, parse_output: str | None = None, parse_bad: bool = False,
                 report_note: str = "", sample_cap: int | None = None):
        self.parse_output = parse_output or (
            '{"group_query": "项目协作群", "start": "", "end": "", "days": 7, '
            '"taxonomy_override": ""}'
        )
        self.parse_bad = parse_bad
        self.report_note = report_note
        self.sample_cap = sample_cap
        self.last_days = 7
        self.parse_prompts: list[str] = []

    def bind_tools(self, tools) -> AgentLoopModel:
        # agent_runtime 传入 OpenAI 形状 schema：{"type":"function","function":{...}}
        assert {tool["function"]["name"] for tool in tools} == {"fetch_group_messages"}
        return AgentLoopModel(self.last_days, sample_cap=self.sample_cap)

    def invoke(self, messages, config=None, **kwargs):
        prompt = str(messages[-1].content)
        if "只输出 JSON" in prompt and "消息：" in prompt:
            batch = json.loads(prompt.split("消息：", 1)[1])
            category = "技术类" if "技术类" in prompt else "问题"
            return AIMessage(content=json.dumps({"assignments": [
                {"message_ref": item["message_ref"], "category": category}
                for item in batch
            ]}, ensure_ascii=False))
        if "请输出解析结果 JSON" in prompt:
            self.parse_prompts.append(prompt)
            if self.parse_bad:
                return AIMessage(content="抱歉，我没听懂")  # 非 JSON -> 失败关闭
            self.last_days = json.loads(self.parse_output).get("days") or 7
            return AIMessage(content=self.parse_output)
        if "请输出分类统计 JSON" in prompt:
            return AIMessage(content=(
                '{"total_messages": 48, "categories": [{"name": "问题", "count": 12, '
                '"ratio": "25%", "quotes": ["[08-01 08:00] 阿明: 帮我看下配置"]}], '
                '"override_applied": false}'
            ))
        if "请输出群分析报告" in prompt:
            return AIMessage(content=self._report_text())
        return AIMessage(content="{}")

    def _report_text(self) -> str:
        return (
            "## 总结\n本周期完成发布准备与回归验证。\n"
            f"{self.report_note}"
            "## 分类统计\n- 问题 25%（代表：[08-01 08:00] 阿明: 帮我看下配置）\n"
            "- 通知 17%；讨论 41%；闲聊 11%；其他 6%\n"
            "## 汇报要点\n- 待办：发布前附件权限确认\n- 风险：无"
        )

    async def astream(self, messages, config=None, **kwargs):
        prompt = str(messages[-1].content)
        assert "请输出群分析报告" in prompt
        text = self._report_text()
        for i in range(0, len(text), 12):
            yield AIMessageChunk(content=text[i:i + 12])


# ------------------------------------------------------------ 环境装配

def _fake_im(days: int, per_day: int = 3) -> FakeIm:
    """最近 N 天、每天 per_day 条消息的项目协作群（含一个干扰群）。"""
    now = datetime.now(UTC)
    messages = []
    mid = 1
    for day in range(days):
        base = (now - timedelta(days=days - 1 - day)).replace(
            hour=9, minute=0, second=0, microsecond=0)
        for n in range(per_day):
            sender = 1001 if n % 2 == 0 else 1002
            messages.append(_msg(
                mid, sender, (base + timedelta(hours=3 * n)).isoformat(),
                f"{days - day}天前的第{n + 1}条：讨论发布进度",
            ))
            mid += 1
    return FakeIm(
        conversations=[_group(11, "项目协作群", 3), _group(12, "别的群", 2)],
        details={11: _detail([
            {"user_id": 1001, "nickname": "阿明"},
            {"user_id": 1002, "nickname": "李华"},
        ], name="项目协作群")},
        messages={11: messages},
    )


def _envelope(message: str, history: list[dict] | None = None,
              execution_id: str = "e2e-e1", conversation_id: int | None = 11) -> dict:
    snapshot = {
        "schema_version": 1,
        "agent": {"id": AGENT_ID, "type": "workflow", "name": "群分析助手"},
        "adapter": {"endpoint": "http://workflow:8080"},
        "model": {"provider": "openai_compatible", "name": "qwen3.5-35b",
                  "base_url": "http://vllm.test/v1"},
        "system_prompt": "你是群分析助手。",
        "skills": [],
        "workflow": {"yaml": YAML_TEXT},
    }
    digest = hashlib.sha256(json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    envelope = {
        "protocol_version": "execution-v1", "execution_id": execution_id,
        "trace": {"trace_id": execution_id}, "binding_generation": 1,
        "continuity_key": "c-e2e", "message": message, "history": history or [],
        "identity": {"user_id": "u1"}, "scene": {"type": "chat"},
        "agent_config": {
            "agent_id": AGENT_ID, "agent_type": "workflow",
            "revision_id": REVISION_ID, "config_digest": digest,
            "snapshot": snapshot, "model_api_key": "model-secret",
            "mcp_endpoint": "http://edge.test/mcp/runtime/e2e-e1/",
        },
    }
    if conversation_id is not None:
        envelope["analysis_target"] = {"conversation_id": conversation_id}
    return envelope


class E2EStack:
    """离线全链路栈：execution app（真实）+ MCP 会话工厂 + im-tools-mcp（真实）。"""

    def __init__(self, fake_im: FakeIm, llm: E2ELLM):
        self.fake_im = fake_im
        self.llm = llm
        self.im_tools_app = build_im_tools_app(
            Settings(base_url="http://im.test"), im_client_factory=self._client
        )
        self.execution_app = build_execution_app("internal-token", "mcp-workload-token")
        self._lifespan_depth = 0
        self._lifespan_cm = None

    @asynccontextmanager
    async def lifespan(self):
        """im-tools-mcp 会话管理器生命周期：每实例只允许 run 一次，
        因此用引用计数让同一栈内的多次请求共享一次进入。"""
        if self._lifespan_depth == 0:
            self._lifespan_cm = self.im_tools_app.router.lifespan_context(
                self.im_tools_app)
            await self._lifespan_cm.__aenter__()
        self._lifespan_depth += 1
        try:
            yield
        finally:
            self._lifespan_depth -= 1
            if self._lifespan_depth == 0:
                await self._lifespan_cm.__aexit__(None, None, None)
                self._lifespan_cm = None

    def _client(self, token: str) -> ImClient:
        return ImClient(
            "http://im.test", token, timeout_seconds=5,
            client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(self.fake_im.handler),
                base_url="http://im.test",
            ),
        )

    def mcp_session_factory(self):
        """McpEdgeClient 会话工厂：ASGI 直连 im-tools-mcp 应用（真 MCP 协议）。

        生产链路中 endpoint 是平台 MCP Edge 的地址（由 Edge 代理到远端工具）；
        离线测试没有 Edge，工厂把任意 endpoint 重定向到 im-tools-mcp 应用，
        等价于"Edge 透传"（鉴权头与超时语义保持不变）。
        """

        def factory(endpoint, headers, timeout_seconds):
            @asynccontextmanager
            async def _session():
                http = httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=self.im_tools_app),
                    headers=dict(headers), timeout=timeout_seconds,
                )
                async with streamable_http_client(
                    "http://testserver/mcp/", http_client=http,
                ) as streams:
                    read, write = streams
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
            return _session()

        return factory

    def install(self, monkeypatch) -> None:
        """把替身 LLM 与 MCP 会话工厂接入 execution_service（唯一两处注入点）。"""
        monkeypatch.setattr(
            execution_service, "_request_llm", lambda spec, config: self.llm
        )
        factory = self.mcp_session_factory()
        monkeypatch.setattr(
            execution_service, "McpEdgeClient",
            lambda endpoint, token: McpEdgeClient(
                endpoint, token, session_factory=factory),
        )

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=self.execution_app),
            base_url="http://workflow.test",
        )


async def _respond(stack: E2EStack, message: str, history: list[dict] | None = None,
                   conversation_id: int | None = 11):
    """在 im-tools-mcp lifespan 就绪的前提下打 respond 端点。"""
    async with stack.lifespan():
        async with stack.client() as client:
            return await client.post(
                "/execution-v1/respond", json=_envelope(message, history, conversation_id=conversation_id),
                headers={"Authorization": "Bearer internal-token"},
            )


# ------------------------------------------------------------ L3 用例

async def test_normal_seven_day_analysis_full_chain(monkeypatch):
    """用例 1：「分析群 X 最近 7 天」→ 结构完整报告（三段标题齐全）。"""
    stack = E2EStack(_fake_im(days=7), E2ELLM())
    stack.install(monkeypatch)
    response = await _respond(stack, "分析项目协作群最近7天")
    assert response.status_code == 200, response.text
    body = response.json()
    reply = body["reply"]
    assert "## 总结" in reply and "## 分类统计" in reply and "## 汇报要点" in reply
    receipt = body["metadata"]["agent_config_applied"]
    assert receipt["revision_id"] == REVISION_ID
    assert "workflow.yaml" in receipt["applied_fields"]


async def test_custom_taxonomy_is_applied_to_program_counts(monkeypatch):
    llm = E2ELLM(parse_output=(
        '{"start":"","end":"","days":7,'
        '"taxonomy_override":"按消息性质分技术类/管理类"}'
    ))
    stack = E2EStack(_fake_im(days=7), llm)
    stack.install(monkeypatch)
    response = await _respond(stack, "按消息性质分技术类/管理类分析最近7天")
    assert response.status_code == 200
    result = response.json()["metadata"]["analysis_result"]
    assert result["status"] == "complete"
    assert result["statistics"]["total_messages"] > 0
    assert result["statistics"]["categories"][0]["name"] == "技术类"
    assert "问题：" not in response.json()["reply"]


async def test_missing_group_name_clarifies_then_converges(monkeypatch):
    """缺目标 ID 先反问；给出 ID 后继续分析。"""
    stack = E2EStack(_fake_im(days=7), E2ELLM(parse_output=(
        '{"group_query":"","group_id":null,"start":"","end":"",'
        '"days":7,"taxonomy_override":""}'
    )))
    stack.install(monkeypatch)
    # 会话管理器每实例只能 run 一次：两轮请求共享同一次生命周期（嵌套进入）
    async with stack.lifespan():
        first = await _respond(stack, "帮我分析一下群", conversation_id=None)
        assert first.status_code == 200
        assert "请选择要分析的群" in first.json()["reply"]

        history = [
            {"role": "user", "content": "帮我分析一下群"},
            {"role": "assistant", "content": first.json()["reply"]},
        ]
        second = await _respond(stack, "分析最近7天", history=history, conversation_id=11)
        assert second.status_code == 200
        assert "## 汇报要点" in second.json()["reply"]
    # history 确实进入了 parse_args 提示词（多轮收敛的管线证明）
    assert any("帮我分析一下群" in prompt for prompt in stack.llm.parse_prompts)


async def test_thirty_day_window_segments_across_tool_rounds(monkeypatch):
    """用例 3：30 天大窗口 → 分段纪要多轮调用 → 汇总成功，报告含覆盖率声明。"""
    llm = E2ELLM(
        parse_output='{"group_query": "项目协作群", "start": "", "end": "", '
                     '"days": 30, "taxonomy_override": ""}',
        report_note="⚠ 覆盖率：已纳入全部 30 段。\n",
    )
    stack = E2EStack(_fake_im(days=30, per_day=2), llm)
    stack.install(monkeypatch)
    response = await _respond(stack, "分析项目协作群最近30天")
    assert response.status_code == 200, response.text
    reply = response.json()["reply"]
    assert "## 总结" in reply
    assert "覆盖率" in reply


async def test_segment_overflow_falls_back_to_sampling(monkeypatch):
    """用例 4：分段数超护栏 → 采样兜底路径 + 报告声明覆盖率。"""
    llm = E2ELLM(
        parse_output='{"group_query": "项目协作群", "start": "", "end": "", '
                     '"days": 30, "taxonomy_override": ""}',
        sample_cap=8,  # 30 段 > 护栏，只采样 8 段
        report_note="⚠ 覆盖率：段数超护栏，采样 8/30 段。\n",
    )
    stack = E2EStack(_fake_im(days=30, per_day=2), llm)
    stack.install(monkeypatch)
    response = await _respond(stack, "分析项目协作群最近30天")
    assert response.status_code == 200
    reply = response.json()["reply"]
    assert "采样" in reply and "覆盖率" in reply


async def test_im_unreachable_degrades_gracefully(monkeypatch):
    """用例 5：im 不可达 → 用户可读错误提示，请求仍 200（会话不崩）。"""
    fake_im = _fake_im(days=7)
    fake_im.forced_status["/conversations/11"] = 503
    stack = E2EStack(fake_im, E2ELLM())
    stack.install(monkeypatch)
    response = await _respond(stack, "分析项目协作群最近7天")
    assert response.status_code == 200
    reply = response.json()["reply"]
    assert "无法获取群消息" in reply
    assert "im-tools-mcp" in reply  # 运维指引而非堆栈


async def test_sse_stream_event_sequence_is_valid(monkeypatch):
    """用例 6：流式 SSE meta/delta/done 事件序合法（meta 首、seq 单调、单一终态）。"""
    stack = E2EStack(_fake_im(days=7), E2ELLM())
    stack.install(monkeypatch)
    events: list[dict] = []
    async with stack.lifespan():
        async with stack.client() as client:
            async with client.stream(
                "POST", "/execution-v1/respond/stream",
                json=_envelope("分析项目协作群最近7天"),
                headers={"Authorization": "Bearer internal-token"},
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert "error" not in kinds
    assert sum(1 for kind in kinds if kind == "delta") >= 1
    assert [event["seq"] for event in events] == list(range(len(events)))
    assert "## 汇报要点" in events[-1]["content"]
    assert all(event["trace_id"] == "e2e-e1" for event in events)
