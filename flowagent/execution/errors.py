"""D-12：统一错误分类。

respond()（非流式）和 respond_stream()（流式）此前各自维护一套异常到错误码
的映射，流式路径的兜底 catch 比非流式粗（一律 execution_failed），导致同一
个异常在两条路径下返回不同错误码（评审 §7.2）。本模块提供唯一的
`classify_exception()`，两条路径共用，保证一致。

错误码表里的 `stage`/`retryable`/`fallback_eligible` 是本次新增字段，供兜底
方案判断是否可以安全重试或降级；只新增字段，不删除/更名既有的 `code`。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..errors import ConfigError
from ..execution_authorization import AuthorizationError


@dataclass(frozen=True)
class ExecutionError:
    """分类后的执行错误：code 是外部可见的稳定标识，其余字段供兜底/重试判断。"""

    code: str
    http_status: int
    stage: str
    retryable: bool
    fallback_eligible: bool
    message: str = ""


#: 错误码表（技术方案 §7.1）。未在表中出现的 code（例如 ConfigError/ValueError
#: 直接把异常消息当 code 使用的历史行为）复用 execution_failed 的元数据兜底。
ERROR_TABLE: dict[str, dict[str, object]] = {
    "unauthorized": {"http_status": 401, "stage": "auth", "retryable": False, "fallback_eligible": False},
    "invalid_request": {"http_status": 422, "stage": "validate", "retryable": False, "fallback_eligible": False},
    "config_digest_mismatch": {"http_status": 409, "stage": "validate", "retryable": False, "fallback_eligible": False},
    "config_identity_mismatch": {"http_status": 409, "stage": "validate", "retryable": False, "fallback_eligible": False},
    "managed_config_invalid": {"http_status": 422, "stage": "validate", "retryable": False, "fallback_eligible": False},
    "managed_yaml_missing": {"http_status": 422, "stage": "validate", "retryable": False, "fallback_eligible": False},
    "snapshot_version_not_supported": {"http_status": 422, "stage": "validate", "retryable": False, "fallback_eligible": False},
    "managed_model_config_missing": {"http_status": 422, "stage": "validate", "retryable": False, "fallback_eligible": False},
    "model_authorization_denied": {"http_status": 403, "stage": "model_auth", "retryable": False, "fallback_eligible": False},
    "model_authorization_unavailable": {"http_status": 503, "stage": "model_auth", "retryable": True, "fallback_eligible": True},
    "model_credential_rotated": {"http_status": 503, "stage": "model_auth", "retryable": True, "fallback_eligible": True},
    "MODEL_TIMEOUT": {"http_status": 504, "stage": "model_call", "retryable": True, "fallback_eligible": True},
    "MODEL_UPSTREAM_ERROR": {"http_status": 502, "stage": "model_call", "retryable": True, "fallback_eligible": True},
    "MCP_TOOL_NOT_ALLOWED": {"http_status": 403, "stage": "mcp_call", "retryable": False, "fallback_eligible": False},
    "MCP_TOOL_ERROR": {"http_status": 502, "stage": "mcp_call", "retryable": True, "fallback_eligible": True},
    "TOOL_ITERATION_LIMIT": {"http_status": 422, "stage": "agent_loop", "retryable": False, "fallback_eligible": True},
    "execution_timeout": {"http_status": 504, "stage": "execution", "retryable": True, "fallback_eligible": True},
    "execution_failed": {"http_status": 500, "stage": "execution", "retryable": False, "fallback_eligible": False},
}

#: 现状行为：这些 ValueError 消息直接被当作 code 使用，且映射到 409（digest/
#: 身份不符），其余 ValueError 一律 422，code 就是异常消息本身（保持兼容，不
#: 改造成统一的 invalid_request，避免影响 demo 现有的 error.code 判断）。
_VALUE_ERROR_409_CODES = frozenset({"config_digest_mismatch", "config_identity_mismatch"})


# 处理`lookup`相关逻辑。
# English: Look up the current operation.
def _lookup(code: str) -> dict[str, object]:
    return ERROR_TABLE.get(code, ERROR_TABLE["execution_failed"])


# English: Handle classify exception operations.
def classify_exception(exc: BaseException) -> ExecutionError:
    """把已知异常类型映射为统一错误码；未识别的异常归入 execution_failed。

    与现状（respond() 的 except 链）保持逐条一致，只是把散落在多处的映射
    收敛到一个函数，供 respond 和 stream 共用。
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        meta = _lookup("execution_timeout")
        return ExecutionError(code="execution_timeout", message=str(exc) or "execution timed out", **meta)

    if isinstance(exc, AuthorizationError):
        # 现状：respond() 把异常消息原样当作 code（不是固定的
        # model_authorization_denied/unavailable），这里保持一致，只是同时
        # 补上 stage/retryable/fallback_eligible 分类。
        text = str(exc)
        denied = "denied" in text
        return ExecutionError(
            code=text, http_status=403 if denied else 503, stage="model_auth",
            retryable=not denied, fallback_eligible=not denied, message=text,
        )

    if isinstance(exc, ValueError):
        code = str(exc)
        status = 409 if code in _VALUE_ERROR_409_CODES else 422
        stage = "validate"
        return ExecutionError(
            code=code, http_status=status, stage=stage,
            retryable=False, fallback_eligible=False, message=code,
        )

    if isinstance(exc, ConfigError):
        meta = _lookup("managed_config_invalid")
        return ExecutionError(code="managed_config_invalid", message=str(exc), **meta)

    # 未识别的异常：对外只给稳定错误码和固定文案，绝不把原始异常文本（可能
    # 包含配置、工具参数或内部路径）返回给调用方（respond_stream 现状已如
    # 此处理，这里保持一致，避免信息泄露——OWASP A05/A09）。
    meta = _lookup("execution_failed")
    return ExecutionError(code="execution_failed", message="execution failed", **meta)


# English: Handle error response body operations.
def error_response_body(error: ExecutionError, *, request_id: str | None = None) -> dict:
    """respond() 的 JSON 错误体：只新增 stage/retryable/fallback_eligible/request_id 字段。"""
    body: dict[str, object] = {
        "code": error.code, "message": error.message,
        "stage": error.stage, "retryable": error.retryable,
        "fallback_eligible": error.fallback_eligible,
    }
    if request_id:
        body["request_id"] = request_id
    return {"error": body}
