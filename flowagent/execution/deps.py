"""D-7e：模型凭据映射——V1/V2 受管 YAML 的模型密钥绑定表。

`credential_id` 由 `uuid5` 确定性派生、并非秘密——仅凭它匹配不构成安全边界，
因此每条绑定必须同时核对 `workflow_name`
（见 test_non_consultation_workflow_cannot_use_consultation_service_credential）。
"""

from __future__ import annotations

import json
import os
import uuid

from ..errors import ConfigError

#: 内置凭据映射：风险会商的模型密钥由 FlowAgent 服务持有，平台快照只带
#: 这个非秘密 credential_id 引用。
DEFAULT_MODEL_CREDENTIAL_BINDINGS: tuple[dict[str, str], ...] = (
    {
        "credential_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "risk-consultation:model")),
        "workflow_name": "risk-consultation",
        "env_var": "FLOWAGENT_RISK_CONSULTATION_MODEL_API_KEY",
    },
)


def model_credential_bindings() -> tuple[dict[str, str], ...]:
    """追加 `FLOWAGENT_MODEL_CREDENTIALS`（JSON 数组）声明的额外凭据映射。

    每项须提供 credential_id / workflow_name / env_var 三个字段；未设置该
    环境变量时只使用内置的风险会商默认项，行为与改造前完全一致。
    """
    raw = os.environ.get("FLOWAGENT_MODEL_CREDENTIALS", "").strip()
    if not raw:
        return DEFAULT_MODEL_CREDENTIAL_BINDINGS
    try:
        extra = json.loads(raw)
    except ValueError as exc:
        raise ConfigError(f"FLOWAGENT_MODEL_CREDENTIALS 不是合法 JSON: {exc}") from exc
    if not isinstance(extra, list) or not all(
        isinstance(item, dict) and {"credential_id", "workflow_name", "env_var"} <= item.keys()
        for item in extra
    ):
        raise ConfigError(
            "FLOWAGENT_MODEL_CREDENTIALS 必须是数组，每项含 "
            "credential_id/workflow_name/env_var"
        )
    return DEFAULT_MODEL_CREDENTIAL_BINDINGS + tuple(extra)
