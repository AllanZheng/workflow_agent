"""内置工具与按 "module:function" 引用动态加载外部工具。"""

import importlib
import os
import time

from .errors import ConfigError, ToolError


_DEFAULT_ALLOWED = ("flowagent.tools",)  # 默认白名单：仅内置工具模块


# English: Check tool allowlist.
def _check_tool_allowlist(module_name: str) -> None:
    """工具模块白名单：FLOWAGENT_TOOL_MODULES=前缀1,前缀2。

    默认失败关闭（评审 4.9）：未配置时只放行内置工具模块 flowagent.tools，
    外部模块（如 shutil）一律拒绝——防止不可信 YAML 直接调用任意 Python callable。
    显式设为 "*" 才全放行（仅限可信本地环境手动开启）。
    """
    allowed = os.getenv("FLOWAGENT_TOOL_MODULES")
    if allowed is not None and allowed.strip() == "*":
        return
    if not allowed:
        prefixes = list(_DEFAULT_ALLOWED)
    else:
        prefixes = [p.strip() for p in allowed.split(",") if p.strip()]
    if prefixes and not any(
        module_name == p or module_name.startswith(p + ".") for p in prefixes
    ):
        raise ConfigError(
            f"工具模块 {module_name} 不在白名单内。默认只放行内置工具；"
            f"如需外部工具请设置 FLOWAGENT_TOOL_MODULES=<模块前缀,逗号分隔>"
            f"（可信环境可设 * 全放行）"
        )

# ---- 内置工具（配置里通过 flowagent.tools:函数名 引用） ----


# English: Handle knowledge base operations.
def knowledge_base(query: str) -> str:
    """Mock 知识库检索：按关键词命中返回固定知识条目。"""
    kb = {
        "退款": "退款政策：签收后7天内可无理由退款，款项3-5个工作日原路退回。",
        "发票": "发票支持电子普票与增值税专票，可在订单详情页自助开具。",
        "登录": "无法登录请依次尝试：清除缓存、重置密码、检查网络；仍失败请转技术支持。",
        "物流": "同城48小时内送达，跨城3-5天，可在订单页查看实时轨迹。",
    }
    q = str(query)
    for keyword, answer in kb.items():
        if keyword in q:
            return answer
    return "知识库未命中，建议转人工处理。"


# English: Handle bump counter operations.
def bump_counter(count=0) -> int:
    """计数器工具：用于演示/测试循环护栏。"""
    try:
        return int(count or 0) + 1
    except (TypeError, ValueError):
        return 1


# English: Handle slow tool operations.
def slow_tool(seconds: float = 5) -> str:
    """慢工具：休眠指定秒数，用于验证节点超时控制。"""
    time.sleep(float(seconds))
    return f"slept {seconds}s"


# English: Handle unreliable tool operations.
def unreliable_tool(text: str) -> str:
    """演示用不稳定工具：输入含“触发故障”时抛异常，用于验证 on_error 降级。"""
    if "触发故障" in str(text):
        raise ToolError("模拟工具故障")
    return f"processed:{text}"


# ---- 工具加载 ----


# English: Load tool.
def load_tool(ref: str):
    """按 "module:function" 点分引用加载可调用工具（受白名单约束）。"""
    if ":" not in ref:
        raise ConfigError(f"tool 引用格式应为 module:function，实际为: {ref}")
    module_name, func_name = ref.split(":", 1)
    _check_tool_allowlist(module_name)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ConfigError(f"无法导入工具模块 {module_name}: {e}") from e
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ConfigError(f"工具 {ref} 不是可调用对象")
    return func
