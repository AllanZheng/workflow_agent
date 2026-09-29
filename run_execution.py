"""启动受管 Workflow execution-v1 服务。"""

import os

import uvicorn

from flowagent.execution_service import build_execution_app


# 解析入口参数并启动主流程。
def main() -> None:
    token = os.environ.get("FRAMEWORK_INTERNAL_TOKEN", "")
    mcp_token = os.environ.get("AGENT_PLATFORM_MCP_WORKFLOW_TOKEN", "")
    app = build_execution_app(token, mcp_token)
    uvicorn.run(
        app,
        host=os.environ.get("WORKFLOW_HOST", "0.0.0.0"),
        port=int(os.environ.get("WORKFLOW_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
