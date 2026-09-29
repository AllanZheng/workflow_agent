"""FastAPI 服务示例测试（examples/api_server.py）。

fastapi/httpx 未安装时自动跳过：pip install fastapi httpx
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("api_server", ROOT / "examples" / "api_server.py")
api_server = importlib.util.module_from_spec(_spec)
sys.modules["api_server"] = api_server
_spec.loader.exec_module(api_server)

client = None


# 处理API`lifespan`相关逻辑。
@pytest.fixture(autouse=True)
def api_lifespan(_suite_memory_checkpoint):
    global client
    with TestClient(api_server.app) as running_client:
        client = running_client
        yield


# 验证“健康状态”场景下的预期行为。
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# 验证“`full``hitl``roundtrip``via`API”场景下的预期行为。
def test_full_hitl_roundtrip_via_api():
    # 1) 启动运行 → 命中 human 节点 → 真实 HTTP 202 等待人工（评审 4.6）
    r1 = client.post("/runs", json={"inputs": {"ticket": "API工单", "amount": 1}, "thread_id": "api-1"})
    assert r1.status_code == 202
    body = r1.json()
    assert body["status"] == "waiting_human"
    assert "请审批" in body["pending"]["question"]

    # 2) 查询状态
    r2 = client.get("/runs/api-1")
    assert r2.status_code == 200
    assert r2.json()["status"] == "waiting_human"

    # 3) 恢复 → 完成（200）
    r3 = client.post("/runs/api-1/resume", json={"value": "批准"})
    assert r3.status_code == 200
    assert r3.json()["status"] == "finished"
    assert r3.json()["vars"]["decision"] == "批准"

    # 4) 重复恢复 → 409
    r4 = client.post("/runs/api-1/resume", json={"value": "批准"})
    assert r4.status_code == 409


# 验证“获取未知运行`returns`404”场景下的预期行为。
def test_get_unknown_run_returns_404():
    assert client.get("/runs/nope").status_code == 404


# 验证“`finished`状态查询”场景下的预期行为。
def test_finished_state_query():
    client.post("/runs", json={"inputs": {"ticket": "B", "amount": 2}, "thread_id": "api-2"})
    client.post("/runs/api-2/resume", json={"value": "批准"})
    r = client.get("/runs/api-2")
    assert r.json()["status"] == "finished"
    assert "final" in r.json()["vars"]


class TestHttpStatusCodes:
    """4.6：等待人工时真实 HTTP 202，OpenAPI 同步声明。"""

    # 验证“创建运行`waiting``human`判断是否202”场景下的预期行为。
    def test_create_run_waiting_human_is_202(self):
        resp = client.post("/runs", json={"inputs": {"ticket": "T", "amount": 1},
                                          "thread_id": "api202"})
        assert resp.status_code == 202
        assert resp.json()["status"] == "waiting_human"
        assert "status_code" not in resp.json()  # 伪字段已删除

    # 验证“`resume`无效值`still``waiting`202”场景下的预期行为。
    def test_resume_invalid_value_still_waiting_202(self):
        client.post("/runs", json={"inputs": {"ticket": "t", "amount": 1},
                                   "thread_id": "api203"})
        # 恢复无效值（options 强校验）→ 仍等待人工 → 202
        resp = client.post("/runs/api203/resume", json={"value": "不批准"})
        assert resp.status_code == 202
        assert resp.json()["status"] == "waiting_human"

    # 验证“`openapi``declares`202”场景下的预期行为。
    def test_openapi_declares_202(self):
        spec = client.get("/openapi.json").json()
        assert "202" in spec["paths"]["/runs"]["post"]["responses"]
        assert "202" in spec["paths"]["/runs/{thread_id}/resume"]["post"]["responses"]

    # 验证“`openapi`版本`matches``package`”场景下的预期行为。
    def test_openapi_version_matches_package(self):
        import flowagent
        spec = client.get("/openapi.json").json()
        assert spec["info"]["version"] == flowagent.__version__
