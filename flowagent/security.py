"""受管服务的入站 Bearer 鉴权：token 为空时按评审 D-11 显式放开，仅打印
一次性 WARNING；token 非空时保持原有的常数时间精确比较语义不变。

范围仅限 workflow_agent 自身对外暴露的三个入口令牌（execution 服务、
im_tools_mcp、consultation_mcp）；模型授权、config_digest/skill sha256
校验、MCP Edge 执行头、会商群白名单等安全机制不受本模块影响。
"""

from __future__ import annotations

import hmac
import logging

logger = logging.getLogger(__name__)


# English: Handle bearer ok operations.
def bearer_ok(authorization: str, expected_token: str) -> bool:
    """expected_token 为空表示该入口鉴权已放开，一律放行；否则要求
    Authorization 头精确等于 "Bearer <expected_token>"（常数时间比较）。
    """
    if not expected_token:
        return True
    return hmac.compare_digest(authorization, f"Bearer {expected_token}")


# English: Handle warn if disabled operations.
def warn_if_disabled(service: str, expected_token: str) -> None:
    """token 为空时打印一次性告警；调用方在启动阶段调用一次即可。"""
    if not expected_token:
        logger.warning(
            "%s 未配置鉴权 token，所有请求将被放行——仅限受信内网/开发环境使用",
            service,
        )
