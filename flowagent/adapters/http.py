"""原生 HTTP 适配：/runs、/runs/stream、/health（技术方案 §6.1）。

契约：
- POST /runs 等待 RunResult；已接受业务失败仍 200，保留 usage 与错误链。
- POST /runs/stream 相同输入 → SSE RunEvent（data: json），唯一 run.finished；
  SSE 建立后不改变 HTTP 状态。
- GET /health 进程/配置健康；不触发付费模型请求。
- 信封非法 422；未知 profile 404；并发准入过载 429。
"""
from __future__ import annotations

import asyncio
from typing import Any

from flowagent.app import build_runner
from flowagent.config import load_profile_config, select_profile
from flowagent.core.contracts import RunInput


def build_http_app(settings: dict[str, Any]) -> Any:
    """settings: {config_path, profile_name?, fallback_owner?}；返回 FastAPI 实例。"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - api extra 未安装
        raise RuntimeError("fastapi not installed; install flowagent[api]") from exc

    config = load_profile_config(settings["config_path"])
    profile_name = settings.get("profile_name")
    try:
        profile = select_profile(config, profile_name)
    except Exception:  # noqa: BLE001 - 未知 profile：启动即报错（404 语义）
        raise
    runner = build_runner(profile, fallback_owner=settings.get("fallback_owner", "local"))
    max_active = int(profile.get("max_active_runs", config.get("max_active_runs", 8)))
    admission = asyncio.Semaphore(max_active)
    app = FastAPI(title="flowagent native", version="3")

    def _parse_run_input(payload: dict[str, Any]) -> RunInput:
        workflow = payload.get("workflow") if isinstance(payload, dict) else None
        if not isinstance(workflow, str) or not workflow.strip():
            raise HTTPException(status_code=422, detail="invalid envelope: workflow required")
        inputs = payload.get("inputs")
        if inputs is not None and not isinstance(inputs, dict):
            raise HTTPException(status_code=422, detail="invalid envelope: inputs must be object")
        history = payload.get("history") or []
        if not isinstance(history, list):
            raise HTTPException(status_code=422, detail="invalid envelope: history must be list")
        continuation = payload.get("continuation")
        if continuation is not None and not isinstance(continuation, dict):
            raise HTTPException(status_code=422, detail="invalid envelope: continuation must be object")
        return RunInput(
            workflow=payload["workflow"],
            inputs=dict(inputs or {}),
            history=[dict(item) for item in history if isinstance(item, dict)],
            correlation_id=payload.get("correlation_id"),
            continuation=continuation,
        )

    @app.post("/runs")
    async def runs(payload: dict[str, Any]) -> Any:
        request = _parse_run_input(payload)
        if admission.locked():
            raise HTTPException(status_code=429, detail="max active runs reached")
        async with admission:
            result = await runner.run(request)
        return result.to_public_dict()

    @app.post("/runs/stream")
    async def runs_stream(payload: dict[str, Any]) -> Any:
        request = _parse_run_input(payload)
        if admission.locked():
            raise HTTPException(status_code=429, detail="max active runs reached")

        async def event_source() -> Any:
            # SSE 建立后不再改变 HTTP 状态；准入在流内以非阻塞方式等待。
            async with admission:
                import json as _json

                async for event in runner.stream(request):
                    yield f"data: {_json.dumps({
                        'run_id': event.run_id, 'seq': event.seq, 'kind': event.kind.value,
                        'step': event.step, 'data': event.data,
                    }, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.get("/health")
    async def health() -> Any:
        # 配置健康检查：不触发付费模型请求（模型连通性不在 health 内验证）。
        return JSONResponse({
            "ok": True,
            "version": "3",
            "workflows": sorted(runner.workflows),
            "fallback_owner": runner.fallback_owner,
            "max_active_runs": max_active,
        })

    @app.post("/runs/cancel/{run_id}")
    async def cancel(run_id: str) -> Any:
        outcome = runner.cancel(run_id)
        return JSONResponse(outcome)

    app.state.runner = runner
    return app
