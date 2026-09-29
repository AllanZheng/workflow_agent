"""im-tools-mcp 环境变量配置。

约定：
  - IM_BASE_URL：im 服务 REST 根地址（如 http://im:8000），必填；
  - IM_TOOLS_MCP_TOKEN：服务自身共享令牌（可选）。设置后调用方 Bearer 必须精确
    匹配；未设置时仅要求 Bearer 存在，真实授权由 im 校验（401 -> im_unauthorized）；
  - IM_TOOLS_MCP_HOST / IM_TOOLS_MCP_PORT：监听地址，默认 0.0.0.0:8090；
  - IM_TOOLS_TIMEOUT_SECONDS：调 im 的单请求超时，默认 10 秒；
  - IM_TOOLS_MAX_WINDOW_DAYS：分析窗口护栏，默认 31 天；
  - IM_TOOLS_MAX_PAGES_PER_SEGMENT：单段消息翻页护栏，默认 100 页（约 1 万条）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(ValueError):
    """配置非法；消息只含变量名与期望，不含任何令牌值。"""


@dataclass(frozen=True)
class Settings:
    base_url: str
    shared_token: str = ""
    host: str = "0.0.0.0"
    port: int = 8090
    timeout_seconds: float = 10.0
    max_window_days: int = 31
    max_pages_per_segment: int = 100

    # 处理来自环境变量相关逻辑。
    @staticmethod
    def from_env(environ: dict[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        base_url = env.get("IM_BASE_URL", "").strip().rstrip("/")
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise SettingsError("IM_BASE_URL 必须设置为 im 服务的 http(s) 根地址")
        try:
            port = int(env.get("IM_TOOLS_MCP_PORT", "8090"))
            timeout_seconds = float(env.get("IM_TOOLS_TIMEOUT_SECONDS", "10"))
            max_window_days = int(env.get("IM_TOOLS_MAX_WINDOW_DAYS", "31"))
            max_pages_per_segment = int(env.get("IM_TOOLS_MAX_PAGES_PER_SEGMENT", "100"))
        except ValueError as exc:
            raise SettingsError("IM_TOOLS_* 数值型变量必须为合法数字") from exc
        if not 1 <= port <= 65535:
            raise SettingsError("IM_TOOLS_MCP_PORT 必须在 1..65535")
        if timeout_seconds <= 0:
            raise SettingsError("IM_TOOLS_TIMEOUT_SECONDS 必须为正数")
        if max_window_days < 1:
            raise SettingsError("IM_TOOLS_MAX_WINDOW_DAYS 至少为 1")
        if max_pages_per_segment < 1:
            raise SettingsError("IM_TOOLS_MAX_PAGES_PER_SEGMENT 至少为 1")
        return Settings(
            base_url=base_url,
            shared_token=env.get("IM_TOOLS_MCP_TOKEN", "").strip(),
            host=env.get("IM_TOOLS_MCP_HOST", "0.0.0.0"),
            port=port,
            timeout_seconds=timeout_seconds,
            max_window_days=max_window_days,
            max_pages_per_segment=max_pages_per_segment,
        )
