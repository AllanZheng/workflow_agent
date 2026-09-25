# flowagent — 配置驱动的 LangGraph 工作流智能体

用 **YAML 自由编排节点**，引擎把配置编译成 **LangGraph 状态图**执行：
条件路由、并行扇出/汇合、LLM 质检修订循环（带循环护栏）、工具节点、错误降级、
**人工介入（HITL，多级审批链、选项强校验与交互式恢复）**、**子图嵌套编排（支持子图人工中断传播）**、节点**重试/超时/退避（超时不重试，防副作用重复）**、
**LLM 输出 schema 强校验（失败关闭）**、**工具白名单（默认仅内置）**、**事件钩子与轨迹导出（含节点耗时）**、**postgres 检查点持久化**、
**流式与异步 API**、**静态引用检查（check --strict）**、**FastAPI 服务化示例（真实 HTTP 202）**，
开箱即用。无 API key 时自动降级 Mock，**离线可跑、测试全绿**。

> 完整设计文档见 [docs/TECHNICAL_DOC.md](docs/TECHNICAL_DOC.md)
>（框架调研对比、架构设计、五类节点配置规范、HITL/持久化机制、测试体系）。
> 更新记录见 [CHANGELOG.md](CHANGELOG.md)。

## 快速开始

命令行示例统一使用 PowerShell / Bash 双兼容写法：JSON 用单引号包双引号
（`'{"key": "value"}'`）；环境变量两种写法都给出。Windows 下 venv 解释器为
`.venv/Scripts/python`，macOS/Linux 为 `.venv/bin/python`。

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
# 开发安装（注册 flowagent 命令与包元数据）：
.venv/Scripts/python -m pip install -e . --no-deps

# 两种调用方式等价：console script 或 python -m
.venv/Scripts/flowagent check configs/customer_service.yaml
.venv/Scripts/python -m flowagent show configs/customer_service.yaml

# 从零创建一个新工作流（脚手架，生成即可 check/run）
.venv/Scripts/flowagent new my_flow.yaml --name my_flow

# 运行前配置 PostgreSQL 连接串（替换账号、密码和库名）：
export FLOWAGENT_DATABASE_URL='postgresql://user:password@localhost:5432/workflow'  # Bash
$env:FLOWAGENT_DATABASE_URL = 'postgresql://user:password@localhost:5432/workflow'  # PowerShell

# 离线运行客服工单流水线（无 key 自动 Mock；--stream 实时输出）
.venv/Scripts/flowagent run configs/customer_service.yaml --input '{"ticket": "商品有质量问题，要求退款"}' --stream

# 人工审批工作流：运行即暂停（退出码3）→ 恢复继续（跨进程需 postgres 检查点）
export FLOWAGENT_CHECKPOINT=postgres          # Bash
$env:FLOWAGENT_CHECKPOINT = "postgres"        # PowerShell（二选一）
.venv/Scripts/flowagent run configs/approval.yaml --thread-id t1 --input '{"ticket": "大额退款", "amount": 8800}'
.venv/Scripts/flowagent resume configs/approval.yaml --thread-id t1 --value 批准
# 省略 --value 可进入交互式问答；轨迹可导出：--trace-out trace.jsonl

# FastAPI 服务化（可选依赖：pip install fastapi uvicorn httpx）
uvicorn examples.api_server:app --port 8080   # POST /runs → HTTP 202 等待人工 → POST /runs/{id}/resume

# 循环护栏演示（评审固定 6 分 → polish 两次后被强制放行）
.venv/Scripts/flowagent run configs/report_writer.yaml --input '{"topic": "智能体框架选型"}'

# 全量测试 + 覆盖率
.venv/Scripts/python -m pytest --cov=flowagent --cov-report=term-missing
```

## 受管 execution-v1 服务的 PostgreSQL 检查点

`Dockerfile.execution` 默认启用 PostgreSQL 检查点。运行 `run_execution.py` 时需注入
`FLOWAGENT_DATABASE_URL`；连接串应指向部署 PostgreSQL 中由 Workflow saver 管理的隔离 schema。
服务在 lifespan 中打开异步 saver 并执行幂等 `setup()`，缺少连接串或 PostgreSQL extra 时会拒绝启动。
手动安装可执行 `pip install -e ".[api,postgres]"`。CLI、旧 API 与异步执行服务均使用 PostgreSQL；测试可显式选择 memory。

## 接入真实 LLM（OpenAI 兼容，任选一家）

```bash
export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # Bash（智谱GLM；DeepSeek/Ollama/OpenAI 同理）
export OPENAI_API_KEY=你的key
export LLM_MODEL=glm-4.6
# PowerShell 写法：
#   $env:OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
#   $env:OPENAI_API_KEY = "你的key"; $env:LLM_MODEL = "glm-4.6"
# 重新运行即可，代码与配置零改动
```

## 安全边界：可信配置 vs 不可信输入

- 工具白名单默认**失败关闭**：未设置 `FLOWAGENT_TOOL_MODULES` 时只允许内置工具
  （`flowagent.tools:*`），YAML 中的外部 `module:function` 引用会被拒绝。
- 只在配置来源可信（你自己维护的 YAML）且确需外部工具时，显式设置：
  `FLOWAGENT_TOOL_MODULES=mytools,company.utils`；本地可信环境可设 `*` 全放行。
- 永远不要把 LLM 生成的模块/函数名直接写进 YAML 再交给引擎加载。
- 提供服务时，不要让最终用户上传 YAML（当前 API 示例只读取服务端本地配置）。

## 一个工作流长什么样

```yaml
nodes:
  classify:
    type: llm                      # LLM 意图分类
    prompt: "把工单分类（退款/技术/投诉/咨询）：{ticket}"
    parser: classifier
    labels: [退款, 技术, 投诉, 咨询]
    output_key: category
    next:
      field: category              # 条件路由
      rules:
        - {when: {op: eq, value: 退款}, goto: refund}
        - {when: {op: eq, value: 技术}, goto: diagnose}
      default: faq

  ask:                             # 人工介入：运行到此暂停，恢复值写入 decision
    type: human
    message: "请审批：{summary}（批准/拒绝）"
    output_key: decision

  risk_check:                      # 子图：把另一份 YAML 作为节点嵌入
    type: workflow
    workflow_file: subflows/risk.yaml
    output_key: risk

  extract:                         # 可靠性：重试 + 超时
    type: llm
    prompt: "提取JSON：{text}"
    parser: json
    output_key: data
    retries: 2
    timeout: 30

  review:                          # LLM 质检打分 + 循环护栏
    type: llm
    parser: json
    output_key: review
    next:
      field: review.score
      rules:
        - when: {op: lt, value: 7}
          goto: revise
          max_loops: 2             # revise 最多 2 次，防死循环
          loop_exhaust: finalize
      default: finalize
```

## 项目结构

```
flowagent/            引擎（config/state/engine/nodes/router/template/parsers/llm/tools/cli）
configs/              示例：customer_service、report_writer、approval（HITL）、refund_approval（综合）
tests/                pytest 全量测试（单元/集成/端到端/CLI/API，覆盖率 96%+）
examples/             api_server.py（FastAPI 服务）、basic_usage.py（Python API 入门）
docs/                 TECHNICAL_DOC.md（技术文档）、API.md（公开 API 参考）
CHANGELOG.md          更新记录（当前 0.9.0）
.github/workflows/    CI（Ubuntu/Windows × Python 3.10/3.12，ruff + pytest）
scripts/              冒烟脚本
LICENSE               MIT
```

### 引擎连接生命周期

```python
with WorkflowEngine(spec) as engine:
    result = engine.run(inputs, thread_id=thread_id)
```

同步 saver 同时支持 `arun`、`astream`、`aresume`、`aget_state`，数据库操作在线程中执行。
原生异步服务应在 lifespan 使用 `open_postgres_checkpointer()`，将 saver 注入引擎，
并调用异步方法。注入的 saver 由调用方负责释放。缺少连接串会直接报配置错误。
单元测试显式使用 `FLOWAGENT_CHECKPOINT=memory`；真实 PostgreSQL 测试设置
`TEST_DATABASE_URL`，测试自行创建和清理隔离 schema。无需迁移旧数据。
