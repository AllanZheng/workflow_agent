"""Demo 对外开放的受管 Workflow 配置子集。"""

from __future__ import annotations

import yaml

from .config import WorkflowSpec, load_spec
from .errors import ConfigError


_ALLOWED_MANAGED_NODE_TYPES = frozenset({"agent", "llm", "template"})


def load_managed_spec(text: str) -> WorkflowSpec:
    """解析并限制受管 YAML，防止绕过 MCP 授权执行本地代码或文件。"""

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"managed_yaml_invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("managed_yaml_invalid: 顶层必须是对象")

    # 类型门禁先于通用节点字段校验，确保禁止类型返回稳定、可识别的错误码。
    nodes = data.get("nodes")
    if not isinstance(nodes, dict):
        raise ConfigError("managed_yaml_invalid: nodes 必须是对象")
    for name, raw_node in nodes.items():
        node_type = raw_node.get("type") if isinstance(raw_node, dict) else None
        if node_type not in _ALLOWED_MANAGED_NODE_TYPES:
            raise ConfigError(
                f"managed_node_type_not_allowed: 节点 {name} type={node_type}"
            )
    return load_spec(data)
