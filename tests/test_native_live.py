"""真实调用链（L01/L02）：显式 live 标记，默认不参与离线回归。

运行：
    BENCH_MODEL_BASE_URL=... BENCH_MODEL_NAME=... python -m pytest tests/test_native_live.py -m live -ra

覆盖：
- L01a 独立模式真实模型问答（无工具）。
- L01b 独立模式真实模型 + 真实 MCP（streamable-http）工具调用，结果可核对。
- L01c MCP 连接失败：非必需服务离线不阻止执行，记录能力不可用诊断。
- L02 平台 V3 真实链路：需要完整平台栈/Edge/Skill 环境；缺条件时 skip 并如实标注未验证。
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

pytestmark = [pytest.mark.live, pytest.mark.anyio]


@pytest.fixture(autouse=True)
def require_explicit_model_environment(request):
    if request.node.name.startswith("test_l01"):
        required = ("BENCH_MODEL_BASE_URL", "BENCH_MODEL_NAME", "BENCH_MODEL_API_KEY")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            pytest.skip("L01 未运行：缺少显式配置 " + ", ".join(missing))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_server() -> tuple[str, list[dict]]:
    """进程内 MCP streamable-http 服务器（mcp 2.x MCPServer；1.x FastMCP 回退）。"""
    port = _free_port()
    try:
        from mcp.server.mcpserver import MCPServer
        server = MCPServer("live-tools")
        run_kwargs = {"transport": "streamable-http", "port": port}
    except ImportError:  # pragma: no cover - mcp 1.x 回退
        from mcp.server.fastmcp import FastMCP
        server = FastMCP("live-tools")
        os.environ["FASTMCP_PORT"] = str(port)
        run_kwargs = {"transport": "streamable-http"}

    calls: list[dict] = []

    @server.tool()
    def hybrid_search(query: str) -> dict:
        """知识库检索（live 测试桩）。"""
        result = {"data": [{
            "knowledge_id": "kb-live-001",
            "knowledge_title": "设备巡检制度",
            "id": "kb-live-001-c1",
            "text": "核心机房每日巡检一次，普通机房每周一次。",
        }], "query": query}
        # 服务端记录实际执行及完整返回，不能用模型的常识答案代替工具证据。
        calls.append({"query": query, "result": result})
        return result

    import uvicorn  # noqa: F401 - 确保 SDK 服务器依赖可用
    thread = threading.Thread(target=lambda: server.run(**run_kwargs), daemon=True)
    thread.start()
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    else:  # pragma: no cover - 环境缺依赖
        pytest.skip("MCP server failed to start")
    return f"http://127.0.0.1:{port}/mcp", calls


def _profile(tmp_path: Path, *, mcp_url: str | None) -> Path:
    model = {"base_url": os.environ["BENCH_MODEL_BASE_URL"],
             "api_key_env": "BENCH_MODEL_API_KEY", "name": os.environ["BENCH_MODEL_NAME"]}
    if os.environ.get("BENCH_MODEL_DISABLE_THINKING") == "1":
        # 仅显式选择时为兼容模型发送模板参数，其他模型保持原行为。
        model["parameters"] = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    profile = {
        "version": 1, "default_profile": "default", "max_active_runs": 2,
        "profiles": {"default": {
            "model": model,
            "mcp_servers": ([{
                "name": "wiki", "transport": "streamable_http",
                "url": mcp_url, "allowed_tools": ["hybrid_search"],
            }] if mcp_url else []),
            "workflows": {"chat": {"enabled": True}},
            "limits": {"model_calls": 6, "tool_calls": 6, "deadline_ms": 60_000,
                       "main_deadline_ms": 55_000, "fallback_timeout_ms": 5_000},
            "max_active_runs": 2,
        }},
    }
    path = tmp_path / "live-config.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


async def test_l01a_live_model_chat(tmp_path) -> None:
    """L01a：真实模型问答可完成且回复非空。"""
    from flowagent.app import build_runner
    from flowagent.config import load_profile_config, select_profile
    from flowagent.core.contracts import RunInput

    config = load_profile_config(_profile(tmp_path, mcp_url=None))
    runner = build_runner(select_profile(config, None))
    result = await runner.run(RunInput(
        workflow="chat", inputs={"message": "用一句话说明你是谁。"}))
    assert result.status.value in ("completed", "degraded"), result
    assert result.reply.strip()
    assert result.usage.model_requests >= 1
    print(f"L01a status={result.status.value} model_requests={result.usage.model_requests}")


async def test_l01b_live_model_with_real_mcp_tool(tmp_path, mcp_server) -> None:
    """L01b：真实模型经真实 MCP 调用工具，结果进入回复（可核对）。"""
    from flowagent.app import build_runner
    from flowagent.config import load_profile_config, select_profile
    from flowagent.core.contracts import RunInput

    mcp_url, calls = mcp_server
    config = load_profile_config(_profile(tmp_path, mcp_url=mcp_url))
    runner = build_runner(select_profile(config, None))
    result = await runner.run(RunInput(workflow="chat", inputs={
        "message": "请用知识库工具检索：核心机房的巡检频率是多少？回答时引用检索到的原文。"}))
    _assert_l01b_evidence(result, calls)
    print("L01b MCP evidence: " + json.dumps(calls, ensure_ascii=False))


def _assert_l01b_evidence(result, calls: list[dict]) -> None:
    assert result.status.value == "completed", f"主任务未完成：{result.status.value}"
    assert result.usage.mcp_attempts >= 1, "没有 MCP 调用尝试"
    assert any("机房" in call.get("query", "") and "巡检" in call.get("query", "") and
               any("核心机房每日巡检一次" in item.get("text", "")
                   for item in call.get("result", {}).get("data", [])) for call in calls), \
        "没有服务端成功返回预期机房巡检证据的调用"
    assert "每日" in result.reply or "每天" in result.reply, "回复未体现真实工具返回的巡检频率"


async def test_l01c_unreachable_mcp_does_not_block_execution(tmp_path) -> None:
    """L01c：MCP 连接失败不阻止执行；记录能力不可用诊断，模型仍可作答。"""
    from flowagent.app import build_runner
    from flowagent.config import load_profile_config, select_profile
    from flowagent.core.contracts import RunInput

    dead = f"http://127.0.0.1:{_free_port()}/mcp"
    config = load_profile_config(_profile(tmp_path, mcp_url=dead))
    runner = build_runner(select_profile(config, None))
    result = await runner.run(RunInput(
        workflow="chat", inputs={"message": "简单回答：1+1 等于几？"}))
    assert result.status.value in ("completed", "degraded"), result
    assert result.reply.strip()
    assert result.usage.mcp_attempts == 0
    print(f"L01c status={result.status.value} model_requests={result.usage.model_requests} "
          f"mcp_attempts={result.usage.mcp_attempts}")


_PLATFORM_ENV = {
    "platform_url": "IT_PLATFORM_URL", "agent_id": "IT_AGENT_ID", "desk_token": "IT_DESK_TOKEN",
    "management_token": "IT_MANAGEMENT_TOKEN", "runtime_id": "IT_RUNTIME_ID",
    "revision_id": "IT_REVISION_ID", "config_digest": "IT_CONFIG_DIGEST",
    "skill_id": "IT_SKILL_ID", "skill_body_version_id": "IT_SKILL_BODY_VERSION_ID",
    "tool_id": "IT_TOOL_ID", "tool_version_id": "IT_TOOL_VERSION_ID",
    "server_version_id": "IT_SERVER_VERSION_ID", "prompt": "IT_L02_PROMPT",
    "expected_text": "IT_L02_EXPECTED_TEXT",
}


async def _run_platform_v3_check(client, config: dict[str, str]) -> dict[str, str]:
    """使用专用只读会话关联实际响应与平台保存的冻结版本回执。"""
    base = config["platform_url"].rstrip("/")
    key = "sample-live-" + uuid.uuid4().hex
    body = {"identity": {"user_id": "ignored-by-trusted-principal", "role_keys": []},
            "scene": {"type": "chat", "ref_id": key}, "session": {"continuity_key": key},
            "agent_id": config["agent_id"], "message": config["prompt"],
            "idempotency_key": key, "deadline_ms": 90000}
    response = await client.post(base + "/v1/agent/respond", json=body,
                                 headers={"Authorization": "Bearer " + config["desk_token"]})
    assert response.status_code == 200, f"平台请求失败：HTTP {response.status_code}"
    value = response.json()
    assert value.get("metadata", {}).get("response_source") == "agent", "主路径未成功"
    assert value.get("metadata", {}).get("status") == "completed", "Sample 未完成主任务"
    assert config["expected_text"] in value.get("reply", ""), "回复未包含预设检索证据"
    execution_id = value.get("trace_id")
    assert execution_id, "平台响应缺少执行号"
    receipt_response = await client.get(base + f"/internal/management/v1/agents/{config['agent_id']}/receipts",
                                        headers={"Authorization": "Bearer " + config["management_token"]})
    assert receipt_response.status_code == 200, f"回执查询失败：HTTP {receipt_response.status_code}"
    rows = receipt_response.json().get("data", [])
    matching = [row for row in rows if str(row.get("execution_id")) == str(execution_id)]
    assert len(matching) == 1, "未找到唯一执行回执"
    receipt = matching[0]
    assert receipt.get("status") == "loaded"
    for field in ("runtime_id", "revision_id", "config_digest"):
        assert str(receipt.get(field)) == config[field], f"回执 {field} 与预期不符"
    details = receipt.get("details_json") or {}
    assert details.get("schema_version") == 3 and details.get("runtime_id") == config["runtime_id"]
    assert details.get("model_request_count", 0) >= 1, "模型请求未计入实际回执"
    assert any(item.get("skill_id") == config["skill_id"] and
               item.get("body_version_id") == config["skill_body_version_id"] and item.get("outcome") == "success"
               for item in details.get("loaded_skill_versions", [])), "未加载预期 Skill 版本"
    assert any(all(item.get(field) == config[field] for field in
                   ("tool_id", "tool_version_id", "server_version_id")) and item.get("outcome") == "success"
               for item in details.get("called_tool_versions", [])), "未调用预期 Edge 工具版本"
    return {"execution_id": str(execution_id), "runtime_id": config["runtime_id"],
            "revision_id": config["revision_id"], "config_digest": config["config_digest"],
            "skill_id": config["skill_id"], "tool_id": config["tool_id"]}


async def test_l02_platform_v3_live() -> None:
    """L02 主链：真实平台、模型、Skill、Edge 和已持久化回执；故障切换另行验收。"""
    if os.environ.get("LIVE_PLATFORM_STACK") != "1":
        pytest.skip("L02 未运行：设置 LIVE_PLATFORM_STACK=1 并提供专用联调环境")
    missing = [name for name in _PLATFORM_ENV.values() if not os.environ.get(name)]
    if missing:
        pytest.skip("L02 未运行：缺少显式配置 " + ", ".join(missing))
    import httpx
    config = {key: os.environ[name] for key, name in _PLATFORM_ENV.items()}
    async with httpx.AsyncClient(timeout=100.0, trust_env=False) as client:
        evidence = await _run_platform_v3_check(client, config)
    print("L02 evidence: " + json.dumps(evidence, ensure_ascii=False))
