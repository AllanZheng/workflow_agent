"""启动受管 Workflow execution-v1 服务。"""

import os

import uvicorn

from flowagent.execution_service import build_execution_app


def main() -> None:
    token = os.environ.get("FRAMEWORK_INTERNAL_TOKEN", "")
    app = build_execution_app(token)
    uvicorn.run(
        app,
        host=os.environ.get("WORKFLOW_HOST", "0.0.0.0"),
        port=int(os.environ.get("WORKFLOW_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
