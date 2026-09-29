"""L4 真实链路冒烟脚本（阶段 4，运维在验收环境执行）。

覆盖测试计划 L4 的数据面（真实 im）：
  登录 auth → 本进程起 im-tools-mcp → 真实 MCP 协议调用
  find_group / fetch_group_messages（真实群、真实消息、真实脱敏语料输出）。

用法（在 workflow_agent 仓库根，venv 已装 .[api]）：
  export SMOKE_AUTH_URL=http://<网关>/admin-api/v1/auth/login
  export SMOKE_IM_BASE_URL=http://<im 地址>          # 如 http://localhost:10501
  export SMOKE_USERNAME=admin
  export SMOKE_PASSWORD=<密码>
  export SMOKE_GROUP_QUERY=<群名关键词>
  python deploy/group-analysis/smoke_real_im.py

输出：候选群、7 天窗口分段语料头几行、覆盖率与窗口，供人工核对时间窗与脱敏。
LLM 报告面（L4 #1/#3）待 vLLM 环境可达后走 Demo 前端全链路（见 README 第 4 节）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx2
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from services.im_tools_mcp import build_app  # noqa: E402
from services.im_tools_mcp.config import Settings  # noqa: E402


class SmokeConfigurationError(ValueError):
    """冒烟环境变量缺失。"""


# 读取必需配置值，并在缺失时停止后续处理。
def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SmokeConfigurationError(f"{name} is required")
    return value


async def _login_token() -> str:
    """经 auth 登录换取 im 可用的 access token（凭据只从环境变量读）。"""
    async with httpx2.AsyncClient(timeout=15) as client:
        response = await client.post(
            _required("SMOKE_AUTH_URL"),
            json={"username": _required("SMOKE_USERNAME"),
                  "password": _required("SMOKE_PASSWORD")},
        )
    if response.status_code != 200:
        raise SmokeConfigurationError(
            f"login failed with status {response.status_code}")
    token = response.json().get("access_token")
    if not token:
        raise SmokeConfigurationError("login response has no access_token")
    return token


# 解析入口参数并启动主流程。
async def main() -> int:
    token = await _login_token()
    group_query = _required("SMOKE_GROUP_QUERY")
    im_base_url = _required("SMOKE_IM_BASE_URL")

    settings = Settings(base_url=im_base_url)
    app = build_app(settings)

    # 环回口随机端口起服务，真实走一遍 Streamable HTTP MCP 协议
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):  # 等待端口就绪
            if server.started:
                break
            await asyncio.sleep(0.1)
        # uvicorn 0.5x：监听 socket 挂在 server.servers[*].sockets
        port = None
        for listener in getattr(server, "servers", None) or []:
            for sock in listener.sockets:
                port = sock.getsockname()[1]
        if not port:
            print("[smoke] im-tools-mcp failed to start", file=sys.stderr)
            return 1

        from flowagent.mcp_edge_client import McpEdgeClient
        client = McpEdgeClient(f"http://127.0.0.1:{port}/mcp/", token,
                               timeout_seconds=60)
        tools = await client.list_tools()
        print(f"[smoke] tools: {[t['name'] for t in tools]}")

        found = await client.call_tool("find_group", {"name_query": group_query})
        print(f"[smoke] find_group({group_query!r}) -> count={found.get('count')}")
        groups = found.get("groups") or []
        if not groups:
            print("[smoke] no group matched; adjust SMOKE_GROUP_QUERY")
            return 1
        for group in groups[:10]:
            print(f"  - {group['conversation_id']} {group['name']}"
                  f" (members={group['member_count']})")
        conversation_id = groups[0]["conversation_id"]

        result = await client.call_tool("fetch_group_messages", {
            "conversation_id": conversation_id, "days": 7,
        })
        coverage = result.get("coverage", {})
        print(f"[smoke] window: {result['window']['start'][:19]} ~ "
              f"{result['window']['end'][:19]}")
        print(f"[smoke] segment 0/{result['segment']['total']}, "
              f"messages={coverage.get('messages')}, "
              f"truncated={result.get('truncated')}")
        corpus_lines = (result.get("corpus") or "").splitlines()
        for line in corpus_lines[:8]:  # 语料样例（脱敏格式：仅昵称/时间/文本）
            print(f"  | {line[:80]}")
        if not corpus_lines:
            print("  | （本段无消息）")
        return 0
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve_task, timeout=10)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except SmokeConfigurationError as exc:
        print(f"[smoke] configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
