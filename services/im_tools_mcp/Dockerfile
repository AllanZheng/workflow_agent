# im-tools-mcp 服务镜像：im 群消息只读 MCP 适配（群分析助手 MVP 阶段 3）。
#
# 与 Dockerfile.execution 同款基座；额外拷贝 services/（服务不属于
# flowagent 引擎包，按仓库根目录布局运行：python -m services.im_tools_mcp）。
# 环境变量见 services/im_tools_mcp/config.py（IM_BASE_URL 必填）。

FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY flowagent ./flowagent
COPY services ./services
RUN pip install --no-cache-dir ".[api]"

EXPOSE 8090
CMD ["python", "-m", "services.im_tools_mcp"]
