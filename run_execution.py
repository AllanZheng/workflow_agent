"""启动 execution-v1 平台服务（V3 原生内核）。"""

import os
import sys

import uvicorn

from flowagent.app import build_platform_service


def _load_internal_token() -> str:
    auth_mode = os.environ.get("FRAMEWORK_AUTH_ENABLED", "0").strip()
    if auth_mode not in {"0", "1"}:
        raise ValueError("FRAMEWORK_AUTH_ENABLED 必须为 0（关闭）或 1（开启）")
    token = os.environ.get("FRAMEWORK_INTERNAL_TOKEN", "")
    if auth_mode == "1" and not token:
        raise ValueError("FRAMEWORK_AUTH_ENABLED=1 时必须配置 FRAMEWORK_INTERNAL_TOKEN")
    return token if auth_mode == "1" else ""


def main() -> None:
    try:
        token = _load_internal_token()
    except ValueError as exc:
        # 只打印变量名与用途，不把令牌值写入日志。
        print(f"[run_execution] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    app = build_platform_service(internal_token=token)
    uvicorn.run(
        app,
        host=os.environ.get("WORKFLOW_HOST", "0.0.0.0"),
        port=int(os.environ.get("WORKFLOW_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
