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
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

BASE_URL = os.environ.get("BENCH_MODEL_BASE_URL", "http://172.18.127.124:18007/v1")
MODEL_NAME = os.environ.get("BENCH_MODEL_NAME", "Qwen3.6-35B")
API_KEY = os.environ.get("BENCH_MODEL_API_KEY", "token-abc123")

pytestmark = [pytest.mark.live, pytest.mark.anyio]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_server() -> str:
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

    @server.tool()
    def hybrid_search(query: str) -> dict:
        """知识库检索（live 测试桩）。"""
        return {"data": [{
            "knowledge_id": "kb-live-001",
            "knowledge_title": "设备巡检制度",
            "id": "kb-live-001-c1",
            "text": "核心机房每日巡检一次，普通机房每周一次。",
        }], "query": query}

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
    return f"http://127.0.0.1:{port}/mcp"


def _profile(tmp_path: Path, *, mcp_url: str | None) -> Path:
    profile = {
        "version": 1, "default_profile": "default", "max_active_runs": 2,
        "profiles": {"default": {
            "model": {"base_url": BASE_URL, "api_key": API_KEY, "name": MODEL_NAME},
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


async def test_l01b_live_model_with_real_mcp_tool(tmp_path, mcp_server) -> None:
    """L01b：真实模型经真实 MCP 调用工具，结果进入回复（可核对）。"""
    from flowagent.app import build_runner
    from flowagent.config import load_profile_config, select_profile
    from flowagent.core.contracts import RunInput

    config = load_profile_config(_profile(tmp_path, mcp_url=mcp_server))
    runner = build_runner(select_profile(config, None))
    result = await runner.run(RunInput(workflow="chat", inputs={
        "message": "请用知识库工具检索：核心机房的巡检频率是多少？回答时引用检索到的原文。"}))
    assert result.status.value in ("completed", "degraded"), result
    assert result.usage.mcp_attempts >= 1, "模型未调用真实 MCP 工具"
    assert "每日" in result.reply or "每周" in result.reply


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


@pytest.mark.skipif(
    not os.environ.get("LIVE_PLATFORM_STACK"),
    reason="L02 需要完整平台栈（runtime/revision/Edge/Skill）；缺条件=未验证，不得算通过",
)
async def test_l02_platform_v3_live() -> None:
    """L02：真实平台 V3 + 预授权 + Skill + Edge（需 LIVE_PLATFORM_STACK 环境）。

    由验收环境执行：注册候选 runtime → 新会话 → chat/群分析/会商/知识；
    记录 runtime/revision、调用轨迹与回执。本机缺失网络与平台栈时占位。
    """
    pytest.skip("L02 占位：在验收环境按测试方案 §5.11 执行并记录证据")
