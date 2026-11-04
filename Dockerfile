FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml requirements.txt MANIFEST.in README.md LICENSE ./
COPY flowagent ./flowagent
COPY services ./services
RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    ".[api]"

ENV PYTHONIOENCODING=utf-8
EXPOSE 8080
# 原生 HTTP 入口（config.json 由挂载提供）：flowagent serve --config /app/config.json
CMD ["flowagent", "serve", "--config", "/app/config.json", "--host", "0.0.0.0", "--port", "8080"]
