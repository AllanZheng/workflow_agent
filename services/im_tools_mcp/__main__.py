"""im-tools-mcp 启动入口：python -m services.im_tools_mcp。

环境变量见 config.py（IM_BASE_URL 必填，其余可选）。
"""

from __future__ import annotations

import sys

import uvicorn

from .app import build_app
from .config import Settings


def main() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as exc:  # SettingsError：只含变量名与期望，不含令牌
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    app = build_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
