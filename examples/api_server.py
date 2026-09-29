"""flowagent API 服务示例（FastAPI，可选依赖）。

    pip install fastapi uvicorn httpx
    uvicorn examples.api_server:app --port 8080
    # 或: python examples/api_server.py

端点：
- POST /runs               {"inputs": {...}, "thread_id": "..."} → 200 完成 / 202 等待人工
- GET  /runs/{thread_id}   查看运行状态（等待人工 / 已结束）与输出变量
- POST /runs/{thread_id}/resume {"value": ...} → 200 完成 / 202 仍在等待

服务默认加载 configs/approval.yaml（可用环境变量 FLOWAGENT_CONFIG 覆盖）。
检查点后端遵循 FLOWAGENT_CHECKPOINT 环境变量（跨进程/重启请用 PostgreSQL）。
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 允许直接脚本启动

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import flowagent
from flowagent import WorkflowEngine, load_spec_file
from flowagent.errors import ConfigError, FlowAgentError
from flowagent.obs_log import (
    RequestIdMiddleware,
    bootstrap_trace_from_env,
    setup_logging,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "approval.yaml"
CONFIG_PATH = os.getenv("FLOWAGENT_CONFIG", str(_DEFAULT_CONFIG))

_ENGINE = None


# 处理`lifespan`相关逻辑。
@asynccontextmanager
async def lifespan(app):
    global _ENGINE
    with WorkflowEngine(load_spec_file(CONFIG_PATH)) as engine:
        _ENGINE = engine
        try:
            yield
        finally:
            _ENGINE = None


# 获取`engine`。
def get_engine() -> WorkflowEngine:
    if _ENGINE is None:
        raise RuntimeError("API lifespan 尚未启动")
    return _ENGINE


class RunIn(BaseModel):
    inputs: dict = {}
    thread_id: str | None = None


class ResumeIn(BaseModel):
    value: object


# OpenAPI 显式声明响应码：客户端据 HTTP 状态区分“同步完成/异步等待”
_RESPONSES = {
    200: {"description": "运行完成"},
    202: {"description": "等待人工输入（用 /runs/{thread_id}/resume 恢复）"},
    400: {"description": "工作流执行错误"},
    404: {"description": "运行不存在"},
    409: {"description": "没有待恢复的人工节点"},
}


setup_logging("workflow_agent")
bootstrap_trace_from_env()   # 平台拉起时可经 TRACE_ID 环境变量注入进程级 trace

app = FastAPI(
    lifespan=lifespan,
    title="flowagent API",
    description="配置驱动的 LangGraph 工作流智能体服务示例",
    version=flowagent.__version__,  # 版本单一来源：flowagent.__version__（评审 4.8）
)

app.add_middleware(RequestIdMiddleware)


def _result(final: dict, thread_id: str) -> JSONResponse:
    """构造响应：真实 HTTP 语义——等待人工=202（异步未完成），完成=200。"""
    payload = WorkflowEngine.waiting_on(final)
    if payload is not None:
        body = {"thread_id": thread_id, "status": "waiting_human", "pending": payload}
        return JSONResponse(status_code=202, content=body)
    body = {
        "thread_id": thread_id,
        "status": "finished",
        "vars": final.get("vars") or {},
        "error": final.get("error"),
    }
    return JSONResponse(status_code=200, content=body)


@app.post("/runs", responses=_RESPONSES)
def create_run(body: RunIn):
    """启动一次运行；若流程包含 human 节点且命中，则返回 202 与待办问题。"""
    engine = get_engine()
    thread_id = body.thread_id or uuid.uuid4().hex
    try:
        final = engine.run(body.inputs, thread_id=thread_id)
    except FlowAgentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _result(final, thread_id)


# 获取运行。
@app.get("/runs/{thread_id}", responses=_RESPONSES)
def get_run(thread_id: str):
    snapshot = get_engine().get_state(thread_id)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"未找到运行: {thread_id}")
    pending = WorkflowEngine.pending_payload(snapshot)
    if pending is not None:
        return {
            "thread_id": thread_id,
            "status": "waiting_human",
            "pending": pending,
        }
    if snapshot.next:
        return {"thread_id": thread_id, "status": "running", "vars": snapshot.values.get("vars", {})}
    return {
        "thread_id": thread_id,
        "status": "finished",
        "vars": snapshot.values.get("vars", {}),
        "error": snapshot.values.get("error"),
    }


# 处理`resume`运行相关逻辑。
@app.post("/runs/{thread_id}/resume", responses=_RESPONSES)
def resume_run(thread_id: str, body: ResumeIn):
    engine = get_engine()
    snapshot = engine.get_state(thread_id)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"未找到运行: {thread_id}")
    if WorkflowEngine.pending_payload(snapshot) is None:
        raise HTTPException(status_code=409, detail="该运行没有待恢复的人工节点")
    try:
        final = engine.resume(body.value, thread_id=thread_id)
    except FlowAgentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _result(final, thread_id)


# 处理健康状态相关逻辑。
@app.get("/health")
def health():
    return {"ok": True, "config": CONFIG_PATH}


if __name__ == "__main__":
    import uvicorn

    try:
        load_spec_file(CONFIG_PATH)  # 启动前先校验配置
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        raise SystemExit(2) from e
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("FLOWAGENT_PORT", "8080")))
