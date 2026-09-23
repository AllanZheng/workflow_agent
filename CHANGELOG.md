# 更新日志

## 0.10.0（2026-09-23）

群分析助手 MVP（引擎零改动）：新增 im 群消息只读 MCP 适配服务与群分析受管工作流。

### 新增（feature/group-analysis-mvp）

- **im-tools-mcp 服务**（`services/im_tools_mcp/`）：Streamable HTTP MCP 端点
  （mcp 2.2.0 官方 SDK）+ 最小 bearer 校验；`find_group`（模糊匹配 ≤10 条）与
  `fetch_group_messages`（before_id 游标翻页 + 时间窗过滤 + segment_hours 分段
  [segment_index 寻址] + max_chars 截断 + coverage 统计）；时间窗记号服务端解析
  （ISO/缺年日期/now-Nd，引擎无时钟）；脱敏语料仅含昵称/时间/文本；
  稳定错误码 im_unreachable/im_unauthorized/group_not_found/window_*。
- **群分析受管工作流**（`configs/managed/group_analysis.yaml`）：parse_args →
  fetch（agent 工具循环，每轮并行多段）→ stats → report（流式三段式报告）；
  解析失败反问澄清（history 多轮收敛）；「无法完成分析」contains 路由；
  im 故障用户可读降级；stats 失败不阻断报告；采样兜底 + 覆盖率声明；
  分类口径内嵌 YAML，taxonomy_override 对话级覆盖。
- **部署与发布**（`deploy/group-analysis/`）：compose 服务项（叠加 agent-mvp 栈）、
  Admin 发布脚本（幂等；模型 key + im 令牌 mcp_headers 双凭据；schema 与服务实现
  同源；--dry-run 离线预演）、L4 真实链路冒烟脚本（运维执行）、发布与联调手册。
- **测试**：L1 服务单测 23 项（含 McpEdgeClient 真 MCP 对拍）；L2 工作流配置
  12 项（双门禁固化 + 发布契约对拍）；L3 离线端到端 6 项（真实 execution 端点 +
  真 MCP 协议 + 假 im + 注入式 LLM）。全量回归 382 passed，引擎零改动未被破坏。

## 0.9.0（2026-09-15）

评审修复：`docs/PROJECT_REVIEW_2026-09-15.md` 确认的全部 10 项问题。

### 修复（P1：错误放行与可靠性语义）

- **审批否定语义（4.1）**：示例配置 `contains` → `eq`（"不批准"不再误入批准分支）；
  human 节点声明 `options` 即强校验——无效答复携带错误提示重新 `interrupt` 询问，
  绝不进入业务分支（失败关闭）；CLI 恢复预检改用 `pending_payload` 判定，
  与 API 一致（修复二次询问后 `snapshot.next` 为空导致的误报"已结束"）。
- **LLM 输出信任边界（4.2）**：新增 `output_schema` 字段校验
  （`type/min/max/enum/optional`，仅 `parser: json` 可用）；畸形输出抛
  `OutputParseError` 走失败/降级路径；数值路由操作符（gt/ge/lt/le）遇字段缺失
  或不可数值化时规则不命中，不再退化字符串比较；三份示例配置补齐质检 schema。
- **timeout 按时返回（4.3）**：超时即停止等待（`shutdown(wait=False)`），
  耗时不再被工具线程拖长；超时不参与自动重试（避免副作用重复执行）；
  `concurrent.futures.TimeoutError` 规范化为内建 `TimeoutError`（3.10 兼容）。
- **子图人工中断传播（4.4）**：workflow 节点把子图 `__interrupt__` 提升到父图；
  子线程 ID 由父 `thread_id` 派生（`{parent}::{node}`，此前随机生成导致无法恢复）；
  同一父 thread_id 即可完成"父 → 子 human → resume → 父继续"闭环。
- **on_error.goto 生效（4.5）**：字符串/并行列表/条件路由三种 `next` 形态都编译
  错误改道边；`node.on_error.goto` 优先于 `route.on_error`（唯一权威，只需配一处）；
  goto 目标参与引用校验，非法目标加载期报错。

### 修复（P2/P3）

- **API 真实 HTTP 202（4.6）**：等待人工返回 `JSONResponse(202)`，删除 body 伪
  `status_code` 字段；OpenAPI 显式声明 200/202/400/404/409。
- **配置校验补强（4.7）**：`recursion_limit ≥ 1`、`max_loops ≥ 1` 且必须有
  `loop_exhaust` 或 `default` 出口、`regex` 模式编译期校验、并行 `next` 列表非空。
- **版本与发布一致（4.8）**：单一版本源 `flowagent.__version__`（pyproject 动态
  读取）；注册 console script `flowagent`；`MANIFEST.in` 收录 configs/docs/examples；
  CI 新增 build-and-install 验证 job；新增 `sqlite` 可选依赖组；OpenAPI 版本读
  `__version__`。
- **工具白名单默认拒绝（4.9）**：未配置时仅放行内置模块 `flowagent.tools`，
  外部模块一律拒绝；显式 `*` 才全放行；README 新增安全边界说明。
- **文档漂移（4.10）**：导出 `TemplateError`；README 命令 PowerShell/Bash 双兼容
  （去反斜杠续行、环境变量双写法、console script 用法）；TECHNICAL_DOC 同步
  options 强校验语义并修正 `contains` 反模式示例。

### 测试

- 从 259 → 308 个（+49）：审批否定反例、options 重新询问、畸形 LLM 输出、
  路由数值硬化、超时耗时与副作用次数、子图恢复闭环、错误路由三形态、
  HTTP 状态码与 OpenAPI、校验拒绝、白名单默认拒绝、版本一致性。
- 套件环境隔离：autouse fixture 默认钉死 Mock LLM（屏蔽外部
  `OPENAI_API_KEY`/`FLOWAGENT_LLM` 干扰，杜绝配置真实端点后套件偶发变红），
  真实 LLM 验证集中于 `test_real_llm.py`（豁免该 fixture）。
- `examples/basic_usage.py` 输出重配 UTF-8（与 CLI 同款），修复 Windows
  GBK 管道环境下子进程输出解码失败。

## 0.8.0（2026-09-15）

第九轮优化（夜间定时任务）：工具参数静态分析与运行输入提示。

### 新增

- **`tool.args` 的 `$var` 引用纳入静态分析**：`check --strict` 此前不校验工具参数
  引用（`args: {query: "$tickit"}` 拼写错误只有运行期才暴露），现在 `$` 前缀引用
  的根段与模板引用同等参与来源校验（点路径取根段）。
- **CLI run 输入键提示**：`workflow.inputs` 已声明时，`run --input` 中未声明的键
  （疑似拼写错误）与已声明但未提供的键均给出 `[输入提示]` 告警（stderr，不阻断、
  不改变退出码）。未声明 `inputs` 的工作流不提示。

### 变更

- 测试从 252 → 259 个。

## 0.7.0（2026-09-15）

第八轮优化（夜间定时任务）：子图输入链递归检查与可观测文档。

### 新增

- **子图输入交叉检查递归化**：`check_spec` 现在递归遍历所有嵌套层级——深层子图
  声明的 inputs / 未解析引用若在整条供给链上无来源，会被报告并前缀标注完整链路
  （如 `[链路 w1<mid> [链路 w2<leaf>]]`）。此前只检查一层，多层嵌套的输入链断裂
  会漏报。每层子图仍遵循"显式声明 inputs"契约。
- **文档**：TECHNICAL_DOC 新增 5.3 节"LangSmith 追踪零代码接入"
  （`LANGSMITH_TRACING/LANGSMITH_API_KEY` 环境变量，未配置时零开销）。

### 变更

- 测试从 249 → 252 个。

## 0.6.0（2026-09-15）

第七轮优化（夜间定时任务）：检查点写入策略、并发验证与故障排查指南。

### 新增

- **检查点写入策略 `workflow.durability`**（LangGraph 1.x durability 参数，已确认
  1.2.11 支持）：`exit`（默认，超级步结束落盘）/ `async`（异步落盘，吞吐优先）/
  `all`（每次变更落盘，最稳）。引擎在 run/resume/stream/arun/astream 全路径透传。
  高吞吐场景可将工作流设为 `async` 降低检查点开销。
- **并发安全验证**：同一引擎实例 8 线程 × 16 次并发运行（不同 thread_id）互不串扰、
  检查点相互独立的回归测试。
- **文档**：TECHNICAL_DOC 新增"故障排查 FAQ"（递归超限、分类未命中、跨进程恢复、
  Mock 模式、中文乱码、子图路径等高频问题）。

### 变更

- 测试从 240 → 249 个。

## 0.5.1（2026-09-15）

第六轮优化（夜间定时任务）：递归超限友好报错与版本查询。

### 新增

- **递归超限友好报错**：`recursion_limit` 耗尽时，引擎（run/resume/stream/arun/astream
  全路径）把 LangGraph 的 `GraphRecursionError` 转换为带修复建议的 `FlowAgentError`
  （提示调大上限或为循环边配置 max_loops/loop_exhaust），原始异常保留在 `__cause__`；
  CLI 呈现为 `[运行失败]` 并以退出码 1 结束，不再打印裸 traceback。
- **CLI `--version`**：打印引擎版本号。

### 变更

- 测试从 234 → 240 个。

## 0.5.0（2026-09-15）

第五轮优化：静态检查器、机器可读输出、可观测与异步。

### 新增

- **静态引用分析**：`flowagent.config.analyze_references/check_spec`——扫描全部
  模板/提示词/路由字段中引用的变量根段，对比产出来源（`workflow.inputs` 声明、
  各节点 `output_key`、`on_error.set`、human 默认键），报告无来源引用。
- **子图输入交叉检查**：`check_spec` 校验子图声明的 inputs 及内部未解析引用能否被
  父级上下文供给。
- **CLI `check --strict`**：存在未解析引用时退出码 1，可纳入 CI 防止配置引用错别字；
  非严格模式仅告警。全部示例配置已声明 `workflow.inputs` 并通过 strict 校验。
- **CLI `run --output json`**：机器可读输出（status/pending/vars/error/trace），
  等待人工时退出码 3，便于上游系统编排。
- **CLI `new` 脚手架**：生成可直接 check/run 的 starter 配置。
- **节点耗时观测**：trace 条目新增 `duration_ms`（成功/失败路径均记录）。
- **异步 API**：`WorkflowEngine.arun/astream`（FastAPI 等异步服务可用）。
- **多级人工审批**：验证并文档化连续 human 节点链（逐级暂停/恢复）。
- **示例与资产**：`examples/basic_usage.py`（Python API 入门）、`LICENSE`（MIT）、
  `docs/API.md`（公开 API 参考）。
- **CI** 增加 ruff lint 步骤。

### 变更

- 测试从 209 → 234 个；覆盖率 96.7%。
- trace 条目新增 `duration_ms` 字段（向后兼容，纯新增）。

## 0.4.0（2026-09-15）

第四轮优化：打包、退避策略、综合示例与部署产物。

### 新增

- **包可安装**：补全 `[build-system]`，支持 `pip install -e .`（包名 `flowagent`）。
- **指数退避**：节点重试新增 `backoff` 倍率字段，第 n 次重试间隔 =
  `retry_delay * backoff^(n-1)`（默认 1.0 即固定间隔；<1 校验拒绝）。
- **综合示例** `configs/refund_approval.yaml`：子图风险评级 → 条件路由 →
  人工审批（options）→ 带重试的处理 → 质检修订循环 → 通知/驳回，
  一个配置串起全部引擎特性；配套端到端测试（自动/批准/驳回/质检回炉 4 条路径）。
- **Docker**：`Dockerfile`（slim 镜像，sqlite 检查点落盘 /data，默认启动 API 服务）+
  `.dockerignore`。

### 变更

- 测试从 204 → 209 个；ruff（E/F/W/B/UP）全量通过；压测：同一引擎 50 次运行 0.34s、
  500 层受控循环 0.25s。

## 0.3.0（2026-09-15）

第三轮优化：HITL 体验、观测能力、安全加固与服务化示例。

### 新增

- **human 节点 `options` 选项**：interrupt 载荷携带提示选项；CLI `resume` 省略
  `--value` 时进入交互式问答（展示问题与选项）；选项外输入仅告警不阻断。
- **轨迹导出**：`run --trace-out trace.jsonl` 将执行轨迹导出为 JSONL。
- **事件钩子**：`NodeHooks(on_node_start, on_node_end)` 传入引擎，观测/审计用，
  钩子异常只告警不阻断。
- **工具白名单**：`FLOWAGENT_TOOL_MODULES` 限制可加载的工具模块前缀，防止配置
  引入任意代码执行。
- **子图实例透传**：workflow 子图复用父级 LLM 客户端与 checkpointer（按实例缓存）。
- **FastAPI 服务示例**：`examples/api_server.py`（POST /runs、GET /runs/{id}、
  POST /runs/{id}/resume），HITL 以 202+pending 表达；配套 TestClient 测试
  （fastapi/httpx 未安装时自动跳过）。
- **调研**：技术文档新增"代码式引擎 vs 低代码平台"（n8n/Dify/Coze）对比章节。

### 变更

- 测试从 188 → 204 个，覆盖率保持 97%。

## 0.2.0（2026-09-15）

第二轮优化：人工介入、子图编排、可靠性、持久化与流式。

### 新增

- **human 节点类型**：基于 LangGraph `interrupt()` 的人工介入。运行到该节点时图暂停
  （CLI 退出码 3 并打印问题），通过 `WorkflowEngine.resume(value, thread_id)` 或
  `python -m flowagent resume` 恢复；human 节点自动绕过重试/超时（interrupt 依赖
  当前线程上下文，且控制流异常不可被吞掉）。
- **workflow 子图节点**：把另一份 YAML 工作流作为节点嵌入，支持相对/绝对路径与
  `output_key` 包装或变量直并；加载期检测循环引用与嵌套深度（上限 8 层），并把
  子配置路径改写为绝对路径。
- **节点重试与超时**：`retries` / `retry_delay` / `timeout` 三个配置字段，适用于
  llm/tool/template 节点；超时经单线程池实现，触发 `TimeoutError` 可被 `on_error`
  捕获降级。
- **可配置 checkpointer**：`FLOWAGENT_CHECKPOINT=memory|sqlite`（配合
  `FLOWAGENT_CHECKPOINT_PATH`），sqlite 后端跨进程/重启持久化，是跨进程 HITL 的
  前提；也可直接向 `WorkflowEngine(checkpointer=...)` 注入任意 LangGraph checkpointer。
- **CLI**：新增 `check`（配置校验，不运行）与 `resume` 子命令；`run` 新增 `--stream`
  实时流式输出；退出码约定 0/1/2/3（正常/运行失败/用法或配置错误/等待人工输入）；
  resume 前置校验检查点存在性与待恢复状态，拦截“落到未知线程静默重跑”的坑。
- **示例**：`configs/approval.yaml` 人工审批工作流（human → 条件路由）。
- **CI**：GitHub Actions 工作流（多 OS 矩阵，pytest + 覆盖率）。
- 真实 LLM 可选集成测试（默认跳过，配置 `OPENAI_API_KEY` 后启用）。

### 变更

- `NodeConfig` 支持新类型与字段（`human`/`workflow`、`message`、`workflow_file`、
  `retries`、`retry_delay`、`timeout`），并新增相应校验。
- 测试从 141 → 188 个，覆盖率 99% → 98.1%（代码量增大后仍保持高位）。

## 0.1.0（2026-09-15）

首版：配置驱动 LangGraph 工作流引擎。

- 三类节点（llm/template/tool）、三种流转（固定边/并行扇出/条件路由 + 11 种操作符）
- 循环护栏（max_loops/loop_exhaust）、错误降级（on_error）
- 模板引擎（JSON 花括号安全）、4 种输出解析器、OpenAI 兼容 LLM 工厂 + Mock 降级
- MemorySaver 检查点、Mermaid 导出、CLI（run/show）
- 141 个测试，覆盖率 99%
