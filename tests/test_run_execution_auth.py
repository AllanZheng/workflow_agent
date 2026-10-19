"""原生启动入口默认关闭校验，显式开启后仍验证原有令牌。"""

import pytest
from fastapi.testclient import TestClient

import run_execution


@pytest.mark.parametrize("mode", [None, "0", "1"])
def test_native_launcher_auth_mode_controls_http_requests(monkeypatch, mode):
    monkeypatch.setenv("FRAMEWORK_INTERNAL_TOKEN", "retained-framework-token")
    monkeypatch.setenv("AGENT_PLATFORM_MCP_WORKFLOW_TOKEN", "independent-mcp-token")
    monkeypatch.setenv("FLOWAGENT_CHECKPOINT", "disabled")
    monkeypatch.delenv("FRAMEWORK_AUTH_ENABLED", raising=False)
    if mode is not None:
        monkeypatch.setenv("FRAMEWORK_AUTH_ENABLED", mode)
    apps = []
    monkeypatch.setattr(run_execution.uvicorn, "run", lambda app, **kwargs: apps.append(app))
    run_execution.main()
    with TestClient(apps[0]) as client:
        for headers in ({}, {"Authorization": "Bearer wrong-token"}):
            assert client.get("/capabilities", headers=headers).status_code == (401 if mode == "1" else 200)
        response = client.get("/capabilities", headers={"Authorization": "Bearer retained-framework-token"})
        assert response.status_code == 200
        assert response.json()["tools"]["managed_mcp"] is True


@pytest.mark.parametrize("mode, token", [("invalid", "secret"), ("1", "")])
def test_native_launcher_reports_invalid_auth_config(monkeypatch, capsys, mode, token):
    monkeypatch.setenv("FRAMEWORK_AUTH_ENABLED", mode)
    monkeypatch.setenv("FRAMEWORK_INTERNAL_TOKEN", token)
    # 防止旧实现忽略开关时真的启动监听进程。
    monkeypatch.setattr(run_execution.uvicorn, "run", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit) as error:
        run_execution.main()
    assert error.value.code == 1
    assert "FRAMEWORK_" in capsys.readouterr().err
