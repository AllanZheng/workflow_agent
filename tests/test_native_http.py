"""原生 HTTP 适配测试（行为 H01–H08）。

覆盖行为编号：
- H01 POST /runs 合法信封 → 200，status=completed，reply 来自模型，execution_path=workflow。
- H02 信封非法（inputs/history/continuation 类型错）→ 422，
  且校验先于执行：不创建任何模型实例。
- H02a workflow 字段缺失/为空 → 422（内核 2026-10-20 修复 KeyError→500，保留为回归）。
- H03 POST /runs/stream → 200 + text/event-stream，seq 从 0 连续递增，唯一 run.finished。
- H04 SSE 建立后执行失败仍返回 200，run.finished 携带 failed 结果（HTTP 状态不再变化）。
- H05 GET /health 报告 ok/workflows/fallback_owner，且不实例化模型（零付费请求）。
- H06 POST /runs/cancel/{未知 run_id} → 200 {cancelled: false}。
- H07 准入：max_active_runs=1 时首个运行未结束，第二个请求 429；释放后首个正常完成。
- H08 未知 profile：装配期抛 ConfigError（启动即失败，而不是运行期 500）。

模型替换策略：adapters/http 复用 flowagent.app.build_runner，而 build_runner 的
context_factory 在调用时按模块全局名 OpenAICompatibleModel 实例化；因此只要在
build_http_app 之前 monkeypatch flowagent.app.OpenAICompatibleModel 为返回 FakeModel
的替身，就能保持配置加载/装配/执行全链路真实且完全离线。
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.http import build_http_app
from flowagent.core.contracts import ErrorCode, FlowAgentCoreError, ModelEvent
from flowagent.errors import ConfigError
from tests.native_helpers import FakeModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """anyio 插件：只需 asyncio 后端（与本仓库其他原生测试一致）。"""
    return "asyncio"


VALID_RUN = {"workflow": "chat", "inputs": {"message": "hi"}, "history": []}


# ---------------------------------------------------------------- 测试工具


def write_profile(tmp_path, *, max_active_runs: int = 2) -> str:
    """写入最小原生 profile（默认模型地址不可达——模型客户端由替身接管）。

    注意 max_active_runs 放在配置顶层：build_http_app 先取 profile 值，
    再回落到配置顶层值，本测试用它验证并发准入。
    """
    config = {
        "version": 1,
        "default_profile": "default",
        "max_active_runs": max_active_runs,
        "profiles": {
            "default": {
                "model": {"base_url": "http://x/v1", "api_key": "k", "name": "m"},
                "workflows": {"chat": {"enabled": True}},
                "limits": {"model_calls": 24},
            }
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return str(path)


def install_model(monkeypatch, factory):
    """把 flowagent.app 的模型类换成替身工厂。

    替身对象本身暴露 stream 并转发给脚本化 FakeModel（Runner 只持有这个对象）。
    返回 (created, construction_calls)：每次执行新建的 FakeModel 实例列表，
    以及装配层实际传入的构造参数（用于验证 profile 解析确实生效）。
    """
    created: list[FakeModel] = []
    construction_calls: list[dict] = []

    class FakeModelShim:
        def __init__(self, **kwargs):
            construction_calls.append(dict(kwargs))
            self.fake = factory()
            created.append(self.fake)

        def stream(self, messages, tools, limits):
            return self.fake.stream(messages, tools, limits)

    monkeypatch.setattr("flowagent.app.OpenAICompatibleModel", FakeModelShim)
    return created, construction_calls


def make_app(
    tmp_path,
    monkeypatch,
    factory,
    *,
    max_active_runs: int = 2,
    profile_name: str | None = None,
    fallback_owner: str = "local",
):
    """装配 FastAPI 应用并返回 (app, created, construction_calls)。"""
    created, construction_calls = install_model(monkeypatch, factory)
    app = build_http_app({
        "config_path": write_profile(tmp_path, max_active_runs=max_active_runs),
        "profile_name": profile_name,
        "fallback_owner": fallback_owner,
    })
    return app, created, construction_calls


def parse_sse(body: str) -> list[dict]:
    """解析 SSE 正文：每帧形如 'data: {json}'，帧之间以空行分隔。"""
    events: list[dict] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        # 严格的 SSE 帧格式断言：任何非 data 行都说明适配层输出被污染
        assert line.startswith("data: "), f"unexpected SSE line: {line!r}"
        events.append(json.loads(line[len("data: "):]))
    return events


def _finished_events(events: list[dict]) -> list[dict]:
    """取出全部 run.finished（按契约必须恰好一个）。"""
    return [event for event in events if event["kind"] == "run.finished"]


# ---------------------------------------------------------------- H01


def test_run_completed_envelope(tmp_path, monkeypatch):
    """H01：合法信封经 /runs 同步执行 → 200 completed（业务执行走真实装配链路）。"""
    app, created, construction_calls = make_app(
        tmp_path, monkeypatch, lambda: FakeModel(default_text="hi"))
    client = TestClient(app)

    response = client.post("/runs", json=VALID_RUN)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"] == "hi"
    assert body["output"] is None
    assert body["error"] is None
    assert body["execution_path"] == "workflow"
    # 路由元数据：请求的 chat 已注册且 preflight 通过（固定路径）
    assert body["recovery"]["requested_workflow"] == "chat"
    assert body["recovery"]["route_reason"] == "registered"
    assert body["usage"]["model_requests"] == 1
    # 装配层把 profile 中的模型参数真实传给了（被替换的）客户端
    assert construction_calls[0]["base_url"] == "http://x/v1"
    assert construction_calls[0]["model_name"] == "m"
    assert construction_calls[0]["api_key"] == "k"
    assert len(created) == 1


# ---------------------------------------------------------------- H02


@pytest.mark.parametrize(
    "payload",
    [
        {"workflow": "chat", "inputs": "not-an-object"},     # inputs 非对象
        {"workflow": "chat", "inputs": ["x"]},               # inputs 非对象（数组）
        {"workflow": "chat", "history": "not-a-list"},       # history 非数组
        {"workflow": "chat", "history": {"role": "user"}},   # history 非空映射同样非法
        {"workflow": "chat", "continuation": "not-object"},  # continuation 非对象
        {"workflow": "chat", "continuation": [1]},           # continuation 非对象（数组）
    ],
)
def test_invalid_envelope_returns_422(tmp_path, monkeypatch, payload):
    """H02：信封非法 → 422，且在校验阶段就拒绝（不触发任何模型调用）。"""
    app, created, _ = make_app(
        tmp_path, monkeypatch, lambda: FakeModel(default_text="should-not-run"))
    client = TestClient(app)

    response = client.post("/runs", json=payload)

    assert response.status_code == 422
    # 校验先于执行：没有为本次请求创建模型实例
    assert created == []


def test_missing_workflow_returns_422(tmp_path, monkeypatch):
    """H02a：workflow 缺失 → 422（曾因 payload['workflow'] KeyError 变 500，回归）。"""
    app, created, _ = make_app(
        tmp_path, monkeypatch, lambda: FakeModel(default_text="should-not-run"))
    client = TestClient(app)

    response = client.post("/runs", json={"inputs": {"message": "hi"}})

    assert response.status_code == 422
    assert created == []


# ---------------------------------------------------------------- H03


def test_stream_success_sse_contract(tmp_path, monkeypatch):
    """H03：SSE 成功流 → 200 text/event-stream、seq 连续、唯一 run.finished。"""
    app, created, _ = make_app(
        tmp_path, monkeypatch, lambda: FakeModel(default_text="hello-stream"))
    client = TestClient(app)

    response = client.post("/runs/stream", json=VALID_RUN)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events, "SSE 流不应为空"
    # seq 从 0 开始逐条递增；同一执行的 run_id 不变
    assert [event["seq"] for event in events] == list(range(len(events)))
    assert len({event["run_id"] for event in events}) == 1
    # 首帧是 run.started，证明事件来自同一次执行
    assert events[0]["kind"] == "run.started"
    finished = _finished_events(events)
    assert len(finished) == 1
    result = finished[0]["data"]["result"]
    assert result["status"] == "completed"
    assert result["reply"] == "hello-stream"
    assert result["execution_path"] == "workflow"
    assert len(created[0].requests) == 1


# ---------------------------------------------------------------- H04


def test_stream_failure_still_returns_200(tmp_path, monkeypatch):
    """H04：SSE 建立后执行失败——HTTP 仍 200，run.finished 携带 failed 终态。"""
    # 第一轮 MODEL_UPSTREAM_ERROR 触发单轮恢复；脚本耗尽后恢复与本地兜底均失败
    app, created, _ = make_app(
        tmp_path, monkeypatch,
        lambda: FakeModel(turns=[FlowAgentCoreError(
            ErrorCode.MODEL_UPSTREAM_ERROR, "boom")]))
    client = TestClient(app)

    response = client.post("/runs/stream", json=VALID_RUN)

    # 关键断言：SSE 已建立，业务失败不改变 HTTP 状态码
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [event["seq"] for event in events] == list(range(len(events)))
    finished = _finished_events(events)
    assert len(finished) == 1
    result = finished[0]["data"]["result"]
    assert result["status"] == "failed"
    # 主阶段与本地兜底都失败 → 稳定错误码 FALLBACK_FAILED，且保留双错误链
    assert result["error"]["code"] == "FALLBACK_FAILED"
    assert result["recovery"]["fallback_attempted"] is True
    assert result["recovery"]["primary_error"]["code"] == "MODEL_INVALID_RESPONSE"
    assert result["recovery"]["fallback_error"]["code"] == "MODEL_INVALID_RESPONSE"


# ---------------------------------------------------------------- H05


def test_health_reports_config_without_model_calls(tmp_path, monkeypatch):
    """H05：/health 返回配置健康信息，且不创建模型（不触发付费请求）。"""
    app, created, construction_calls = make_app(
        tmp_path, monkeypatch, lambda: FakeModel(default_text="never-called"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["workflows"] == ["chat"]
    assert body["fallback_owner"] == "local"
    assert body["max_active_runs"] == 2
    # 健康检查是纯配置检查：构造期与请求期都没有模型实例
    assert created == []
    assert construction_calls == []


# ---------------------------------------------------------------- H06


def test_cancel_unknown_run_returns_not_cancelled(tmp_path, monkeypatch):
    """H06：取消未知 run → 200 且明确 cancelled=false（不虚构撤销）。"""
    app, created, _ = make_app(tmp_path, monkeypatch, lambda: FakeModel(default_text="hi"))
    client = TestClient(app)

    response = client.post("/runs/cancel/does-not-exist")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "does-not-exist",
        "cancelled": False,
        "reason": "not_found_or_finished",
    }
    assert created == []


# ---------------------------------------------------------------- H07


class _StallModel(FakeModel):
    """首个模型流阻塞在 release 事件上：让准入信号量保持占用。"""

    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__()
        self._started = started
        self._release = release

    async def stream(self, messages, tools, limits):
        self.requests.append({
            "messages": messages, "tools": list(tools), "limits": dict(limits or {}),
        })
        self._started.set()          # 通知测试：首个运行已持有准入
        await self._release.wait()   # 不使用 sleep：由测试显式释放
        yield ModelEvent(kind="completed", text="released",
                         tool_calls=[], finish_reason="stop")


async def test_admission_rejects_concurrent_run_with_429(tmp_path, monkeypatch):
    """H07：max_active_runs=1 时并发第二个请求 429；首个运行释放后正常完成。"""
    started = asyncio.Event()
    release = asyncio.Event()
    app, created, _ = make_app(
        tmp_path, monkeypatch,
        lambda: _StallModel(started, release), max_active_runs=1)

    # 并发场景使用 httpx.ASGITransport（TestClient 同步阻塞，无法制造重叠）
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://flowagent.test") as client:
        first = asyncio.create_task(client.post("/runs", json=VALID_RUN))
        # 等待首个运行真正进入模型调用（此时它已持有准入信号量）
        await asyncio.wait_for(started.wait(), timeout=5)

        second = await client.post("/runs", json=VALID_RUN)

        assert second.status_code == 429
        assert second.json()["detail"] == "max active runs reached"
        # 被拒请求不得创建模型（准入先于执行装配）
        assert len(created) == 1

        release.set()
        first_response = await asyncio.wait_for(first, timeout=5)

    assert first_response.status_code == 200
    first_body = first_response.json()
    assert first_body["status"] == "completed"
    assert first_body["reply"] == "released"


# ---------------------------------------------------------------- H08


def test_unknown_profile_fails_at_build_time(tmp_path):
    """H08：未知 profile 在装配期抛 ConfigError（启动即失败，避免运行期 500）。"""
    config_path = write_profile(tmp_path)

    with pytest.raises(ConfigError) as exc_info:
        build_http_app({"config_path": config_path, "profile_name": "nope"})

    assert "unknown profile: nope" in str(exc_info.value)
