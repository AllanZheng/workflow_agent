"""启动受管 Workflow execution-v1 服务。"""

import os
import sys

import uvicorn

from flowagent.execution_service import build_execution_app


def _load_internal_token() -> str:
    # 与平台兼容框架使用相同开关；默认放开，显式开启后保留原有 Bearer 比较。
    auth_mode = os.environ.get("FRAMEWORK_AUTH_ENABLED", "0").strip()
    if auth_mode not in {"0", "1"}:
        raise ValueError("FRAMEWORK_AUTH_ENABLED 必须为 0（关闭）或 1（开启）")
    token = os.environ.get("FRAMEWORK_INTERNAL_TOKEN", "")
    if auth_mode == "1" and not token:
        raise ValueError("FRAMEWORK_AUTH_ENABLED=1 时必须配置 FRAMEWORK_INTERNAL_TOKEN")
    return token if auth_mode == "1" else ""


# 解析入口参数并启动主流程。
# English: Run the main entry point.
def main() -> None:
    try:
        token = _load_internal_token()
    except ValueError as exc:
        # 只打印变量名与用途，不把令牌值写入日志。
        print(f"[run_execution] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    mcp_token = os.environ.get("AGENT_PLATFORM_MCP_WORKFLOW_TOKEN", "")
    app = build_execution_app(token, mcp_token)
    uvicorn.run(
        app,
        host=os.environ.get("WORKFLOW_HOST", "0.0.0.0"),
        port=int(os.environ.get("WORKFLOW_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
