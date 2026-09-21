FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
# 版本钉死为本地 venv 已验证组合（裸装最新 fastapi/uvicorn 会与 langgraph 依赖
# 冲突，pip 回溯到远古 uvicorn 后 sdist 构建失败）；清华源加速国内构建。
# 升级依赖时先在本地 venv 验证再同步这两处 pin。
RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt fastapi==0.141.1 uvicorn==0.53.0

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
