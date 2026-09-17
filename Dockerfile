FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

COPY . .

ENV PYTHONIOENCODING=utf-8 \
    FLOWAGENT_LLM=auto \
    FLOWAGENT_CHECKPOINT=sqlite \
    FLOWAGENT_CHECKPOINT_PATH=/data/flowagent.sqlite

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8080

# 默认以 Mock 模式启动 API 服务（挂载真实 OPENAI_API_KEY 环境变量即可切换真实 LLM）
CMD ["python", "-m", "uvicorn", "examples.api_server:app", "--host", "0.0.0.0", "--port", "8080"]
