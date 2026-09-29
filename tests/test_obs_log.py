"""obs_log 同步副本冒烟测试：格式、trace、中间件回显（不依赖异步插件）。"""
import asyncio
import json
import logging

httpx = __import__("pytest").importorskip("httpx")

from flowagent.obs_log import (
    RequestIdMiddleware,
    get_request_id,
    setup_logging,
    trace_context,
)


# 处理最后一个JSON`line`相关逻辑。
def _last_json_line(stderr_text: str) -> dict:
    for line in reversed(stderr_text.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError("stderr 中没有 JSON 日志行")


# 验证“`dual`流式处理与追踪”场景下的预期行为。
def test_dual_stream_and_trace(capsys):
    setup_logging("workflow_agent", force=True)
    try:
        with trace_context("smoke-rid-1"):
            logging.getLogger("smoke").info("hello",
                                            extra={"event": "smoke.ok"})
        captured = capsys.readouterr()
        line = _last_json_line(captured.err)
        assert line["request_id"] == "smoke-rid-1"
        assert line["event"] == "smoke.ok"
        assert "[workflow_agent] [req:smoke-rid-1]" in captured.out
    finally:
        root = logging.getLogger()
        for handler in [h for h in root.handlers
                        if getattr(h, "_obs_log", False)]:
            root.removeHandler(handler)


# 验证“`middleware``echo`”场景下的预期行为。
def test_middleware_echo():
    # 处理`echo``app`相关逻辑。
    async def echo_app(scope, receive, send):
        body = (get_request_id() or "-").encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": []})
        await send({"type": "http.response.body", "body": body})

    # 处理场景相关逻辑。
    async def scenario():
        transport = httpx.ASGITransport(app=RequestIdMiddleware(echo_app))
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t") as client:
            return await client.get("/ping",
                                    headers={"X-Request-ID": "smoke-2"})

    response = asyncio.run(scenario())
    assert response.headers["x-request-id"] == "smoke-2"
    assert response.text == "smoke-2"
