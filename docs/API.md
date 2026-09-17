# flowagent 公开 API 参考

> 面向编程接口使用者。快速上手见 [README](../README.md)，配置规范见
> [TECHNICAL_DOC](TECHNICAL_DOC.md) 第 4 节。

## 顶层导入

```python
from flowagent import (
    WorkflowEngine,      # 引擎门面
    build_graph,         # 配置 → LangGraph 编译图
    load_spec,           # dict → WorkflowSpec（校验）
    load_spec_file,      # YAML 文件 → WorkflowSpec（含子图校验/路径改写）
    MockLLM,             # 离线 LLM 替身
    create_llm,          # LLM 工厂（auto/mock/real）
    WFState,             # 状态 TypedDict
    ConfigError, OutputParseError, ToolError, FlowAgentError, TemplateError,
)
```

## WorkflowSpec

pydantic 模型（`extra="forbid"`，全部字段带校验）：

| 属性 | 说明 |
| --- | --- |
| `spec.workflow.name / description / start / recursion_limit / inputs` | 元信息；`inputs` 为静态引用分析声明的运行时输入 |
| `spec.nodes: dict[str, NodeConfig]` | 节点定义 |

`NodeConfig` 常用字段：`type`（llm/template/tool/human/workflow）、`prompt/system/parser/labels/mock_response`、
`template`、`tool/args`、`message/options`、`workflow_file`、`retries/retry_delay/backoff/timeout`、
`output_key/on_error/next`。

加载器行为：

- `load_spec_file(path)` 会把子图 `workflow_file` 改写为绝对路径、递归校验子图
  （循环引用 → `ConfigError`，深度上限 8）；
- `analyze_references(spec)` 返回模板/提示词/路由中引用但无产出来源的变量名列表；
- `check_spec(spec)` = 引用分析 + 子图输入交叉检查（CLI `check --strict` 的内核）。

## WorkflowEngine

```python
engine = WorkflowEngine(spec, llm=None, checkpointer=None, hooks=None)
```

构造即编译（`build_graph`：StateGraph + checkpointer + MemorySaver/SqliteSaver 工厂）。
`llm=None` 时按环境变量自动选择（无 `OPENAI_API_KEY` → MockLLM）。

| 方法 | 说明 |
| --- | --- |
| `run(inputs: dict, thread_id=None) -> dict` | 同步运行；返回最终状态（vars/trace/loop_counts/error；命中 human 时含 `__interrupt__`）。递归超限抛带修复建议的 `FlowAgentError` |
| `resume(value, thread_id) -> dict` | 恢复暂停的运行（`Command(resume=value)`） |
| `stream(inputs, thread_id=None)` | 生成器：按超步产出完整状态（`stream_mode="values"`） |
| `arun(inputs, thread_id=None) -> dict` | 异步运行（`ainvoke`） |
| `astream(inputs, thread_id=None)` | 异步流式生成器 |
| `get_state(thread_id)` | 读取检查点快照（`.values` / `.next` / `.tasks`） |
| `mermaid() -> str` | Mermaid 图源码 |
| `waiting_on(final) -> dict \| None`（静态） | 从 run/resume 返回值提取待办 interrupt 载荷 |
| `pending_payload(snapshot) -> dict \| None`（静态） | 从检查点快照提取待恢复载荷（含 options） |

trace 条目结构：`{"node", "status": "ok"|"error", "outputs"?, "error"?, "duration_ms"}`。

## hooks（NodeHooks）

```python
from flowagent.nodes import NodeHooks

hooks = NodeHooks(
    on_node_start=lambda name, state: ...,
    on_node_end=lambda name, result, error: ...,   # error 为 None 表示成功
)
engine = WorkflowEngine(spec, hooks=hooks)
```

钩子异常仅打印告警，不阻断工作流。

## create_llm(provider=None) / MockLLM

- `provider`：`"mock"` / `"auto"`（默认，无 key 降级）/ `"real"`（缺 key 报错）；
- 环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_MODEL`、`LLM_TEMPERATURE`；
- Mock 模式下节点 `mock_response` 会被渲染为该节点的 LLM 输出（仍走 parser）。

## make_checkpointer(setting=None)

`"memory"`（默认）/ `"sqlite"`（`FLOWAGENT_CHECKPOINT_PATH` 指定库文件）。
返回对象可直接传入 `WorkflowEngine(checkpointer=...)`（生产可注入 PostgresSaver 等）。

## CLI

```
flowagent new <path> [--name N]        生成配置脚手架（已存在拒绝）
flowagent check <cfg> [--strict]       校验 + 静态引用分析（strict 时未解析引用退出 1）
flowagent show <cfg>                   结构 + Mermaid
flowagent run <cfg> [--input JSON | --input-file F] [--thread-id T]
              [--stream] [--trace-out F.jsonl] [--output json]
flowagent resume <cfg> --thread-id T [--value V]     # 省略 --value 进入交互式
```

退出码：`0` 正常｜`1` 运行/恢复失败｜`2` 用法/配置错误｜`3` 等待人工输入
（`--output json` 同样遵守）。

环境变量汇总：

| 变量 | 说明 |
| --- | --- |
| `FLOWAGENT_LLM` / `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` / `LLM_TEMPERATURE` | LLM 接入 |
| `FLOWAGENT_CHECKPOINT` / `FLOWAGENT_CHECKPOINT_PATH` | memory / sqlite |
| `FLOWAGENT_TOOL_MODULES` | 工具模块白名单（逗号分隔前缀） |
| `FLOWAGENT_CONFIG` / `FLOWAGENT_PORT` | api_server 示例的配置与端口 |
