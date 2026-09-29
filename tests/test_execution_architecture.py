"""flowagent/execution/ 新架构模块的直接测试：errors 分类、logs 有界缓冲、
planner 的 flow 选择（技术方案 §7.1、§7.3、§4.4；测试方案 WF-ERR/WF-LOG/WF-PLAN）。
"""

import asyncio

import pytest

from flowagent.errors import ConfigError
from flowagent.execution.deps import model_credential_bindings
from flowagent.execution.errors import classify_exception, error_response_body
from flowagent.execution.logs import ExecutionLog
from flowagent.execution.planner import select_flow
from flowagent.execution_authorization import AuthorizationError


# 验证“分类`timeout`错误映射执行`timeout`504”场景下的预期行为。
# English: Verify classify timeout error maps to execution timeout 504.
def test_classify_timeout_error_maps_to_execution_timeout_504():
    error = classify_exception(TimeoutError("execution exceeded 90.0s deadline"))
    assert error.code == "execution_timeout"
    assert error.http_status == 504
    assert error.stage == "execution"
    assert error.retryable is True
    assert error.fallback_eligible is True


# 验证“分类`asyncio``timeout`错误`also`映射执行`timeout`”场景下的预期行为。
# English: Verify classify asyncio timeout error also maps to execution timeout.
def test_classify_asyncio_timeout_error_also_maps_to_execution_timeout():
    error = classify_exception(asyncio.TimeoutError())
    assert error.code == "execution_timeout"
    assert error.http_status == 504


# 验证“分类授权错误`denied`判断是否403与不可重试”场景下的预期行为。
# English: Verify classify authorization error denied is 403 and not retryable.
def test_classify_authorization_error_denied_is_403_and_not_retryable():
    error = classify_exception(AuthorizationError("model_authorization_denied"))
    assert error.code == "model_authorization_denied"
    assert error.http_status == 403
    assert error.stage == "model_auth"
    assert error.retryable is False
    assert error.fallback_eligible is False


# 验证“分类授权错误`unavailable`判断是否503与可重试”场景下的预期行为。
# English: Verify classify authorization error unavailable is 503 and retryable.
def test_classify_authorization_error_unavailable_is_503_and_retryable():
    error = classify_exception(AuthorizationError("model_authorization_unavailable"))
    assert error.http_status == 503
    assert error.retryable is True
    assert error.fallback_eligible is True


# 验证“分类值错误`digest``mismatch`判断是否409”场景下的预期行为。
# English: Verify classify value error digest mismatch is 409.
def test_classify_value_error_digest_mismatch_is_409():
    error = classify_exception(ValueError("config_digest_mismatch"))
    assert error.code == "config_digest_mismatch"
    assert error.http_status == 409
    assert error.retryable is False


# 验证“分类值错误身份`mismatch`判断是否409”场景下的预期行为。
# English: Verify classify value error identity mismatch is 409.
def test_classify_value_error_identity_mismatch_is_409():
    error = classify_exception(ValueError("config_identity_mismatch"))
    assert error.http_status == 409


# English: Verify classify other value error uses message as code and 422.
def test_classify_other_value_error_uses_message_as_code_and_422():
    """现状行为：respond() 把非 digest/identity 的 ValueError 消息原样当 code。"""
    error = classify_exception(ValueError("memory_context_not_enabled_in_snapshot"))
    assert error.code == "memory_context_not_enabled_in_snapshot"
    assert error.http_status == 422


# 验证“分类配置错误判断是否`managed`配置无效422”场景下的预期行为。
# English: Verify classify config error is managed config invalid 422.
def test_classify_config_error_is_managed_config_invalid_422():
    error = classify_exception(ConfigError("managed_yaml_missing"))
    assert error.code == "managed_config_invalid"
    assert error.message == "managed_yaml_missing"
    assert error.http_status == 422


# English: Verify classify unrecognized exception falls back to execution failed without leaking message.
def test_classify_unrecognized_exception_falls_back_to_execution_failed_without_leaking_message():
    """兜底分类不能把原始异常文本（可能含配置/密钥线索）透传给调用方。"""
    error = classify_exception(RuntimeError("dsn=postgresql://user:secret@host/db"))
    assert error.code == "execution_failed"
    assert error.http_status == 500
    assert "secret" not in error.message
    assert error.message == "execution failed"


# 验证“错误响应内容体判断是否`additive`仅”场景下的预期行为。
# English: Verify error response body is additive only.
def test_error_response_body_is_additive_only():
    error = classify_exception(ValueError("config_digest_mismatch"))
    body = error_response_body(error)
    assert body["error"]["code"] == "config_digest_mismatch"
    assert {"stage", "retryable", "fallback_eligible"} <= body["error"].keys()


# 验证“执行日志`truncates`之后100`entries`”场景下的预期行为。
# English: Verify execution log truncates after 100 entries.
def test_execution_log_truncates_after_100_entries():
    log = ExecutionLog()
    for i in range(150):
        log.add("stage", "info", f"entry-{i}")
    assert len(log.entries) == 100
    assert log.truncated is True
    assert log.to_metadata()["logs_truncated"] is True


# 验证“执行日志`truncates`之后`64kib`”场景下的预期行为。
# English: Verify execution log truncates after 64kib.
def test_execution_log_truncates_after_64kib():
    log = ExecutionLog()
    big_message = "x" * 2000
    for _ in range(40):
        log.add("stage", "info", big_message)
    assert log.truncated is True
    assert len(log.entries) < 40


# 验证“执行日志不`truncated``under`数量上限”场景下的预期行为。
# English: Verify execution log not truncated under limits.
def test_execution_log_not_truncated_under_limits():
    log = ExecutionLog()
    log.add("stage", "info", "hello", duration_ms=12)
    metadata = log.to_metadata()
    assert metadata["logs"] == [{"stage": "stage", "level": "info", "message": "hello", "duration_ms": 12}]
    assert "logs_truncated" not in metadata


class _FakeConfig:
    # 初始化对象状态并建立所需依赖。
    # English: Initialize object state and required dependencies.
    def __init__(self, snapshot_v3=None):
        self.snapshot_v3 = snapshot_v3


class _FakeWorkflow:
    # 初始化对象状态并建立所需依赖。
    # English: Initialize object state and required dependencies.
    def __init__(self, group_analysis_mode):
        self.group_analysis_mode = group_analysis_mode


class _FakeSpec:
    # 初始化对象状态并建立所需依赖。
    # English: Initialize object state and required dependencies.
    def __init__(self, group_analysis_mode="legacy"):
        self.workflow = _FakeWorkflow(group_analysis_mode)


# 验证“`select`流程`v3`生成快照路由`qa`”场景下的预期行为。
# English: Verify select flow v3 snapshot routes to qa.
def test_select_flow_v3_snapshot_routes_to_qa():
    assert select_flow(_FakeConfig(snapshot_v3=object())) == "qa"


# 验证“`select`流程`deterministic``v1v2`路由群组`analysis`”场景下的预期行为。
# English: Verify select flow deterministic v1v2 routes to group analysis.
def test_select_flow_deterministic_v1v2_routes_to_group_analysis():
    assert select_flow(_FakeConfig(), _FakeSpec("deterministic")) == "group_analysis"


# 验证“`select`流程其他`v1v2`路由`managed``yaml`”场景下的预期行为。
# English: Verify select flow other v1v2 routes to managed yaml.
def test_select_flow_other_v1v2_routes_to_managed_yaml():
    assert select_flow(_FakeConfig(), _FakeSpec("legacy")) == "managed_yaml"


# 验证“`select`流程不包含`spec`默认项`managed``yaml`”场景下的预期行为。
# English: Verify select flow without spec defaults to managed yaml.
def test_select_flow_without_spec_defaults_to_managed_yaml():
    assert select_flow(_FakeConfig()) == "managed_yaml"


# 验证“模型凭据绑定关系默认项不包含环境变量”场景下的预期行为。
# English: Verify model credential bindings default without env.
def test_model_credential_bindings_default_without_env(monkeypatch):
    monkeypatch.delenv("FLOWAGENT_MODEL_CREDENTIALS", raising=False)
    bindings = model_credential_bindings()
    assert bindings[0]["workflow_name"] == "risk-consultation"
    assert bindings[0]["env_var"] == "FLOWAGENT_RISK_CONSULTATION_MODEL_API_KEY"


# 验证“模型凭据绑定关系拒绝无效JSON”场景下的预期行为。
# English: Verify model credential bindings rejects invalid JSON.
def test_model_credential_bindings_rejects_invalid_json(monkeypatch):
    monkeypatch.setenv("FLOWAGENT_MODEL_CREDENTIALS", "{not json")
    with pytest.raises(ConfigError, match="不是合法 JSON"):
        model_credential_bindings()


# 验证“模型凭据绑定关系拒绝缺失必需密钥”场景下的预期行为。
# English: Verify model credential bindings rejects missing required keys.
def test_model_credential_bindings_rejects_missing_required_keys(monkeypatch):
    monkeypatch.setenv("FLOWAGENT_MODEL_CREDENTIALS", '[{"credential_id": "x"}]')
    with pytest.raises(ConfigError, match="credential_id/workflow_name/env_var"):
        model_credential_bindings()


# 验证“流程`registry``exposes``three`当前项流程”场景下的预期行为。
# English: Verify flows registry exposes the three current flows.
def test_flows_registry_exposes_the_three_current_flows():
    from flowagent.flows import FLOWS, FlowDefinition, get_flow

    assert set(FLOWS) == {"qa", "group_analysis", "managed_yaml"}
    assert all(isinstance(flow, FlowDefinition) for flow in FLOWS.values())
    assert all(flow.needs_checkpoint is False for flow in FLOWS.values())  # D-2
    assert get_flow("qa").name == "qa"
    with pytest.raises(ValueError, match="unknown flow"):
        get_flow("does-not-exist")


# 验证“执行契约`reexports`相同`classes``as`执行服务”场景下的预期行为。
# English: Verify execution contracts reexports the same classes as execution service.
def test_execution_contracts_reexports_the_same_classes_as_execution_service():
    from flowagent import execution_service
    from flowagent.execution import contracts

    assert contracts.ExecutionRequest is execution_service.ExecutionRequest
    assert contracts.ManagedAgentConfig is execution_service.ManagedAgentConfig
