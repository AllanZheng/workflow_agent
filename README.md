# flowagent（sample_agent V3 原生内核）

业务工作流智能体：普通 Python async 工作流为主、少量内核负责模型—工具循环、预算、终态与错误；平台兼容集中在适配层。

## 结构

```text
flowagent/
├─ core/        契约、预算上下文、Agent 循环、Runner（路由/兜底/取消）
├─ clients/     OpenAI 兼容模型客户端（流式聚合）、MCP 客户端（mcp 2.x/1.x）
├─ workflows/   chat / group_analysis（group_rules）/ consultation_stage / knowledge_query
├─ adapters/    http（原生 /runs、/runs/stream、/health）；platform（execution-v1 V3）
├─ app.py       build_runner(profile) / build_app / build_platform_service
├─ config.py    原生 profile 装载（config.json）
└─ cli.py       flowagent run / serve
```

## 独立运行

```bash
pip install .[api]
cp configs/native.example.json config.json   # 按需修改
export MODEL_BASE_URL=... MODEL_API_KEY=... MODEL_NAME=...
flowagent run --config config.json --input '{"workflow":"chat","inputs":{"message":"你好"}}'
flowagent serve --config config.json --port 8080
```

HTTP：`POST /runs`（等待结果）、`POST /runs/stream`（SSE：run.started → … → run.finished）、`GET /health`。

## 平台模式（execution-v1 + V3 快照）

```bash
python run_execution.py   # FRAMEWORK_AUTH_ENABLED=0/1；WORKFLOW_PORT 默认 8080
```

路由：`/execution-v1/respond`、`/execution-v1/respond/stream`、`/execution-v1/reset`、`/execution-v1/cancel`、`/capabilities`、`/capabilities/v2`、`/health`。仅接受 V3 快照；模型每次请求预授权、MCP 执行期固定工具、Skill 正文按固定版本装载。

## 退出码（CLI run）

completed / needs_input / partial = 0；degraded = 2；failed = 1；取消 = 130。调用方应读取结果 `status` 字段。

## 测试

```bash
python -m pytest tests -m "not installation and not live" -q   # 离线回归
python -m pytest tests/test_native_installation.py -m installation -ra  # 制品安装验收
python -m pytest tests/test_native_live.py -m live -ra          # 真实模型/MCP（显式）
```

基准：`benchmarks/run.py --engine baseline-v3|native --suite business|financebench|tau2`（详见 benchmarks/manifests）。

## 说明

- 旧 YAML 引擎（LangGraph/managed_yaml/checkpoint）已在 V3 重构 M4 退役；依赖旧协议的环境继续绑定旧制品。
- 详细设计与测试方案见仓库外 docs/features/agent平台演进。
