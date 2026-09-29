from __future__ import annotations

import sys

import uvicorn

from .app import build_app
from .config import Settings


# 解析入口参数并启动主流程。
# English: Run the main entry point.
def main() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"[consultation-write-mcp] configuration error: {exc}", file=sys.stderr)
        return 2
    uvicorn.run(build_app(settings), host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
