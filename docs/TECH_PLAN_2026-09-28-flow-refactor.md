# workflow_agent「功能即图」重构技术方案

- 日期：2026-09-28；状态：技术设计，尚未实施。
- 输入：[ENG_REVIEW_2026-09-28.md](ENG_REVIEW_2026-09-28.md) 的评审决策，以及 2026-09-28 本轮补充确认的 4 项（见 §1.2）。
- 配套测试方案：[TEST_PLAN_2026-09-28-flow-refactor.md](TEST_PLAN_2026-09-28-flow-refactor.md)。
- 必须兼容的在研方案，出现冲突时以它们为准：
  - `../../docs/tech/2026-09-27-智能体管理与统一账户技术方案.md`（下称「账户方案」）
  - `../../docs/tech/2026-09-28-全局LLM兜底增量技术方案.md`（下称「兜底方案」）
  - `../../docs/plan/2026-09-27-admin-agent-simplification-execution-plan.md`（下称「执行计划」；workflow_agent 已有 P3 提交 c799fa5 `session_continuity.py`）
- 仓库基线：master@c799fa5；431 个测试通过，22 个跳过，覆盖率 82.6%。

## 1. 目标、边界与决策

### 1.1 目标

workflow_agent 定位为基于 LangGraph 的专用工作流智能体，不做通用智能体。现有能力是直接问答、群消息定制分析和风险研判；后续扩展知识库查询和流程图管理。本次重构要做到：

1. 每个业务能力是一张代码定义的 StateGraph，放在 `flowagent/flows/`，全部经同一个执行入口运行。
2. **execution-v1 对外契约保持不变**：demo 不需要改代码，账户方案和兜底方案正在开发的调用路径不受影响。
3. 代码更少、更易读。消除重复的执行路径和重复的鉴权代码，修复评审发现的 bug 和性能问题。

### 1.2 决策汇总

| 编号 | 决策 | 来源 |
|---|---|---|
| D-1 | 一个功能一张图，统一入口 | 评审问题 1 |
| D-2 | checkpoint 默认关闭；只有需要 HITL 的 flow 才启用，并配 TTL | 评审问题 2 |
| D-3 | 合并成一个 Dockerfile，靠不同的 command 启动不同服务 | 评审问题 3 |
| D-4 | 知识库和流程图先只定义 MCP 接口 | 评审问题 4 |
| D-5 | 只保留一条流式执行路径；鉴权收敛成一个 Depends | 评审问题 5 |
| D-6 | 节点成功时清除 error；trace 设上限 | 评审问题 6 |
| D-7 | 7a、7c–7h 清理（7b 作废，见 D-10） | 评审问题 7 |
| D-8 | 路径级测试，覆盖率 ≥ 85% | 评审问题 8 |
| D-9 | 9a–9d 性能修复 | 评审问题 9 |
| D-10 | **V1、V2、V3 三个 snapshot 版本全部保留**，在内部统一转换成 flow 路由；demo 不改 | 本轮 Q2 |
| D-11 | **安全只放开 workflow_agent 自身的三处**：execution token、im_tools_mcp token、consultation_mcp token。模型授权、config_digest、skill sha256、MCP Edge 头、会商群白名单、sender 注入全部保持原样 | 本轮 Q1，覆盖评审 §3.3 |
| D-12 | 使用分类错误码，SSE 输出 log/warning 阶段信息，响应带 `logs` 字段 | 本轮 Q3 |
| D-13 | 文档放在 workflow_agent/docs | 本轮 Q4 |

D-10 和 D-11 修正了评审报告中的 7b 和 §3.3。评审报告保留原文作为历史记录，后续以本文为准。

### 1.3 不做的事

- 不改 execution-v1 的路径、请求体、snapshot 字段，以及 demo 端的 `StreamValidator` 状态机。
- 不实现兜底逻辑。兜底由 demo 负责（兜底方案 §1）；workflow 只负责提供可分类的错误和“空回复”信号。
- 不实现知识库和流程图的服务本体，只冻结 MCP 接口草案。
- 不实现 HITL，也不启用 checkpoint 续跑。
- 不新增表、队列或日志存储（与账户方案 §1.1、兜底方案 §9 一致）。

## 2. 兼容性约束（与在研方案对齐）

| 在研约束 | 来源 | 本方案如何遵守 |
|---|---|---|
| demo 会商通过 `Bearer FRAMEWORK_INTERNAL_TOKEN` 调用非流式 `/execution-v1/respond`，发送 schema_version=2，credential_id=uuid5("risk-consultation:model") | demo `consultation/integrations.py:193,229` | 请求和响应形状不变；密钥仍从 `FLOWAGENT_RISK_CONSULTATION_MODEL_API_KEY` 取，作为内置映射保留（§6.4） |
| 管理面发布 V3，带模型授权 URL、execution_token 和 MCP Edge 执行头 | demo `control_plane/publication.py:486`；账户方案 §7.3、§12 | 授权、固定工具、`X-Execution-ID` 等头部保持不变 |
| demo `StreamValidator` 只接受 meta/delta/done/error 四种 kind，seq 单调递增，出现未知 kind 即判 `protocol_violation` | demo `platform/streaming.py`、`contracts/events.py` | 默认不发送新的 kind，日志放进终态的 metadata；独立的 log/warning 事件需要 opt-in（§7.3） |
| 同会话不兼容时返回 `SESSION_RESTART_REQUIRED` | 执行计划 P3；workflow c799fa5 | checkpoint 默认关闭时，`plan_session` 总是返回 start_new；启用 HITL 的 flow 时再接入 |
| 兜底要求能区分“可兜底的上游故障”和“安全/业务拒绝”，并识别空回复 | 兜底方案 §3 | 统一错误码表，标注 `fallback_eligible`；空回复单独给出码（§7） |
| sender、requester、actual_agent_id 由 demo 注入，工具层不改 | 账户方案 §13、§16.3 | workflow 不读写 sender，会商 MCP 保留群白名单与批准条件 |
| 支持 HTTP/HTTPS，不关闭证书校验 | 账户方案 §7.0 | 模型、MCP、授权客户端保持现有 http/https 校验，不加 `verify=False` |
| 当次日志通过响应和流返回，不新增日志存储 | 账户方案 §12 | 使用请求级有界缓冲 `ExecutionLog`，最多 100 条或 64 KiB，超出时标记 `logs_truncated` |

## 3. 目标架构

```
POST /execution-v1/respond          POST /execution-v1/respond/stream
          │                                    │
          └──── Depends(require_internal_token)┘   ← 配置了 token 就校验，为空则放行并打 WARNING
                          │
                 ExecutionRequest.model_validate        （模型不变）
                          │
                 planner.plan(request) -> ExecutionPlan
                   ├─ 校验 digest / 身份（V1/V2：_digest；V3：snapshot_digest）
                   ├─ 选择 flow：
                   │    V3                                   → "qa"
                   │    V1/V2 + group_analysis_mode=deterministic → "group_analysis"
                   │    V1/V2 其他 YAML                        → "managed_yaml"
                   ├─ 构造 RequestDeps（llm、mcp、skill_loader、authorizer）
                   └─ 截断 history 到 runtime.history_limit，并计算 deadline
                          │
                 runner.run_stream(plan) ── 唯一执行路径
                   meta → delta* → done|error    （metadata 中带 logs / warnings / error 详情）
                          │
     respond：collect(run_stream) → JSON         stream：逐事件写出 SSE
```

### 3.1 模块划分

| 模块 | 职责 | 来源 |
|---|---|---|
| `flowagent/execution/contracts.py` | `ExecutionRequest` 与 V1/V2/V3 snapshot 模型 | 从 execution_service.py 原样拆出，字段不改 |
| `flowagent/execution/planner.py` | 请求 → `ExecutionPlan`：校验、选择 flow、组装依赖 | 合并 execute_managed/stream_managed 前半段的四处重复 |
| `flowagent/execution/runner.py` | `run_stream(plan)` 与 `collect(stream)` | 唯一执行路径 |
| `flowagent/execution/errors.py` | `ExecutionError(code, stage, retryable, fallback_eligible)`，以及异常到错误码的映射 | 新增 |
| `flowagent/execution/logs.py` | `ExecutionLog` 有界缓冲 | 新增 |
| `flowagent/execution/app.py` | FastAPI 路由与鉴权 Depends | 替代 `build_execution_app` |
| `flowagent/execution/deps.py` | 模型客户端、共享 httpx、凭据映射 | 合并三处 ChatOpenAI 构造 |
| `flowagent/flows/__init__.py` | `FLOWS: dict[str, FlowDefinition]` 注册表 | 新增 |
| `flowagent/flows/qa.py` | V3 问答：`agent ⇄ tools` 图 | 包装 `ManagedAgentRuntime` |
| `flowagent/flows/group_analysis.py` | 群分析图，把 `analyze_group` 的步骤拆成节点 | 由 `group_analysis.py` 改写 |
| `flowagent/flows/managed_yaml.py` | V1/V2 通用 YAML（风险会商等），内部调用 `build_graph` | 复用现有引擎 |
| `flowagent/execution_service.py` | 保留为兼容 shim，re-export `build_execution_app`、`execute_managed`、`stream_managed` | 测试和 run_execution.py 的 import 不受影响 |

YAML 引擎（`config.py`、`nodes.py`、`router.py`、`engine.py`）继续保留，因为 V1/V2 需要它。它的用法从“产品主线”改为“managed_yaml flow 的实现细节”，外加 CLI/demo。

### 3.2 FlowDefinition 接口

```python
# flowagent/flows/__init__.py（设计草案）
@dataclass(frozen=True)
class FlowDefinition:
    name: str                                    # "qa" | "group_analysis" | "managed_yaml"
    build: Callable[[], CompiledStateGraph] | None  # 静态图：启动时编译一次；None 表示按请求编译（managed_yaml）
    needs_checkpoint: bool = False               # D-2：默认 False
    streams_reply: bool = True                   # 是否流式输出最终回复的 delta

FLOWS: dict[str, FlowDefinition] = {...}         # 新增功能只需注册一行
```

- 静态 flow（qa、group_analysis）在 app lifespan 中编译一次（D-9b）。每个请求的依赖通过 `config["configurable"]["deps"]` 注入，这是 LangGraph 官方推荐的运行时注入方式。
- managed_yaml 的图由 YAML 决定，必须按请求编译。缓存键为 `config_digest`，使用有上限的 LRU（128），取代无上限的 `_SUB_ENGINES`（D-9d）。受管 YAML 本身不允许 workflow 子图节点，所以不需要子图缓存。

### 3.3 统一 State

```python
class FlowState(TypedDict):
    vars: Annotated[dict, merge_dict]           # 业务变量（原样）
    trace: Annotated[list, bounded_add(200)]    # D-6：只保留最近 200 条
    loop_counts: Annotated[dict, merge_dict]
    error: Annotated[dict | None, last_write]   # D-6：成功节点写 None；last_write 容忍并行写入
```

WFState 改为 FlowState 的别名，保证 YAML 引擎和 demo 配置兼容。`bounded_add` 和 `last_write` 在 `state.py` 中实现，各约 5 行。

## 4. 三条 flow 的设计

### 4.1 qa（V3 直接问答）

```
START → agent ──有 tool_calls──→ tools ──→ agent（最多 max_tool_iterations 轮）
          │                        └─ 群选择需要澄清（not_found / ambiguous）→ END（回复澄清文本）
          └─无 tool_calls──→ END（reply）
超出轮数上限 → error: TOOL_ITERATION_LIMIT
```

- 复用 `ManagedAgentRuntime` 的授权、固定工具、Skill 懒加载和 `_GroupSelectionRequired` 语义。本次只把循环从 Python while 改成图边，行为不变。
- `max_tool_iterations` 读取 V3 snapshot 中的对应字段；没有这个字段时仍用 8（D-7f，替换 execution_service.py:526 和 654 两处硬编码）。取值范围 1..8，与 `config.py` 的 agent 节点限制一致。
- 流式行为不变：只输出最终那一轮（没有工具调用的一轮）的文本。

### 4.2 group_analysis（V1/V2 deterministic）

```
START → parse_args ─失败→ END（clarify 文本）
          ↓
     resolve_group ─not_found / ambiguous→ END（统一的澄清文案，D-7d）
          ↓
     fetch_input ─McpEdgeError→ END（status=failed，im_tool_unavailable）
          ↓
     classify_batches（逐批处理，每批最多重试 2 次；失败计入 failed）
          ↓
     report → END（reply + analysis_result）
```

- 各节点保留 `group_analysis.py` 现有的实现，只拆成节点函数；prompt 仍从 YAML 中的 `parse_args` 和 `report` 节点读取，保证已发布配置可以继续使用。
- `analysis_result` 的结构不变（status、partial_reasons 等），放在 `done.metadata`。
- 删除 group_analysis.yaml 中的死节点（fetch_msg、fetch_sys、fetch、stats、agent_question、im_unavailable）（D-7a）。先确认 demo 发布脚本不引用这些节点：**假设**只引用 parse_args 和 report，实施时需要用 grep 核实 `publish_group_analysis.py`。
- 单条消息超长（`single_message_too_large`）时改为截断并在 partial_reasons 里记录，不再抛异常中断整个分析。
- **不改变 LLM 调用次数和 prompt**，避免已有输出质量回退。

### 4.3 managed_yaml（V1/V2 其他 YAML，包括风险会商）

- 内部就是现有的 `build_graph(spec, stream_final_reply=True)`，由 runner 统一驱动 `astream(stream_mode=["custom", "values"])`。
- `load_managed_spec` 的节点白名单（agent、llm、template）保持不变。
- `workflow.timeout_ms` 生效：作为整次执行的 deadline（D-7f）。

### 4.4 flow 选择规则（planner）

```python
def select_flow(config) -> str:
    if config.snapshot_v3 is not None:
        return "qa"
    spec = load_managed_spec(yaml)                 # V1/V2
    if spec.workflow.group_analysis_mode == "deterministic":
        return "group_analysis"
    return "managed_yaml"
```

schema_version 不再出现在执行代码的 if 分支里，只在 planner 中使用一次。V1/V2 的差异（Skill 是全文还是摘要、MCP token 是进程级还是请求级、是否带 memory_context）收敛为 `RequestDeps` 的构造细节，由 `planner._deps_v1/_deps_v2/_deps_v3` 三个小函数分别处理。

## 5. 执行路径（D-5）

```python
async def run_stream(plan) -> AsyncIterator[dict]:
    yield meta
    try:
        async with asyncio.timeout(plan.deadline_s):        # D-7f：deadline 覆盖整次执行
            async for event in plan.flow.astream(plan.inputs, plan.config):
                yield delta 或收集最终状态
        reply 为空且没有结构化结果 → 抛出 ExecutionError(AGENT_EMPTY_REPLY)   # 兜底方案 §3
        yield done(content=reply, metadata={receipt, logs, warnings, ...})
    except ExecutionError as e: yield error(e)              # 分类错误码（§7）
    except Exception as e:      yield error(classify(e))

async def collect(stream) -> dict:  # respond 使用；遇到 error 事件时映射为 HTTP 状态码
```

- 非流式 respond 的返回 JSON 字段不变：`execution_id`、`reply`、`used_context_refs`、`metadata`，另外在 metadata 中新增 `logs`。错误响应仍是 `{"error": {"code": ...}}`，并新增 `stage`、`retryable`、`fallback_eligible`、`request_id` 字段。只增加字段，不删除原有字段，旧 demo 解析不受影响。
- HTTP 状态码映射：保留现有的 401、409（digest/identity）、422（配置和校验错误）、403/503（授权），并新增 502（上游模型或 MCP 失败）、504（超时）。
  - 现在 `McpEdgeError` 和 `ToolIterationLimit` 会返回 500，改为 502 和 422。
  - 这项改动会影响 demo 会商 `integrations.py` 对 `response.status_code != 200` 的判断。它读取的是 `error.code`，所以只要非 200 都会进入错误分支，行为兼容，只是错误码变得更精确。
- reset：请求体不是 JSON，或者 JSON 不是对象时，返回 422 `invalid_reset`。目前这两种情况会返回 500。

## 6. 安全与凭据（D-11）

### 6.1 放开的三处

| 位置 | 现状 | 改后 |
|---|---|---|
| execution 服务 `FRAMEWORK_INTERNAL_TOKEN` | 为空时拒绝启动；6 个 handler 各自校验 | 为空时正常启动并打一条 WARNING，所有请求放行；非空时由 Depends 统一校验，逻辑不变 |
| im_tools_mcp `IM_TOOLS_MCP_TOKEN` | 为空时只要求请求带 Bearer 头 | 为空时放行并打 WARNING，不再要求 Bearer 头；非空时校验（逻辑不变） |
| consultation_mcp `CONSULTATION_WRITE_MCP_TOKEN` | 必填 | 为空时放行并打 WARNING；非空时校验 |

- 这三处统一使用一个工具函数 `flowagent/security.py::bearer_guard(expected: str, service: str)`，避免三份实现各自漂移。
- `/health` 依旧不需要鉴权。

### 6.2 保持不变

以下内容一律不动：
- 模型授权（`DemoModelAuthorizer`，V3 每次调用前授权）
- `config_digest` / `snapshot_digest` 校验
- skill sha256 与身份核对
- `McpEdgeClient` 的固定工具和执行头
- 会商 `allowed_conversation_ids` 与业务批准条件
- sender 由 demo 注入（账户方案 §16.3，A3 会替换 allowed_sender_ids，本方案不提前动）

### 6.3 风险记录

放开的三处都只在内网使用。部署时建议 compose 不对宿主机映射 MCP 端口（`compose.group-analysis.yml` 当前映射了 `${IM_TOOLS_PORT:-8091}:8090`）；隔离测试栈和 0927 部署建议配置 token（测试方案 SEC 组会同时覆盖配置和不配置两种状态）。

### 6.4 模型凭据映射（D-7e）

```
FLOWAGENT_MODEL_CREDENTIALS = '{"<credential_id>": "<ENV_VAR_NAME>", ...}'   # 可选
内置默认：{uuid5(NAMESPACE_URL, "risk-consultation:model"): "FLOWAGENT_RISK_CONSULTATION_MODEL_API_KEY"}
```

- 判断依据只有 credential_id，不再比对 `spec.workflow.name == "risk-consultation"`。
- 请求中直接带了 `model_api_key` 时，优先使用请求里的值（行为不变）。
- 环境变量里只写变量名，不写密钥。已有部署不需要改任何 env。
- `needs_model` 只扫描可达节点，修复“YAML 里存在不可达的 llm 节点也要求模型凭据”的问题。

## 7. 错误分类与日志（D-12）

### 7.1 错误码表

| 错误码 | 触发条件 | HTTP | stage | retryable | fallback_eligible |
|---|---|---|---|---|---|
| `unauthorized` | Bearer 校验失败（token 非空时） | 401 | auth | false | false |
| `invalid_request` | 请求体不满足 `ExecutionRequest` | 422 | validate | false | false |
| `config_digest_mismatch` / `config_identity_mismatch` | V1/V2 digest 或身份不符 | 409 | validate | false | false |
| `managed_config_invalid` / `managed_yaml_missing` / `snapshot_version_not_supported` | 配置类 ConfigError | 422 | validate | false | false |
| `managed_model_config_missing` | 找不到模型凭据 | 422 | validate | false | false |
| `model_authorization_denied` | `AuthorizationError`，消息含 denied | 403 | model_auth | false | false |
| `model_authorization_unavailable` / `model_credential_rotated` | 其余 `AuthorizationError` | 503 | model_auth | true | true |
| `MODEL_TIMEOUT` | LLM 调用超过 deadline | 504 | model_call | true | true |
| `MODEL_UPSTREAM_ERROR` | LLM 客户端抛出的其余异常 | 502 | model_call | true | true |
| `MCP_TOOL_NOT_ALLOWED` | 工具不在白名单/未授权 | 403 | mcp_call | false | false |
| `MCP_TOOL_ERROR` | `McpEdgeError` | 502 | mcp_call | true | true |
| `TOOL_ITERATION_LIMIT` | `ToolIterationLimit` | 422 | agent_loop | false | true（业务未完成，非安全拒绝） |
| `AGENT_EMPTY_REPLY` | 成功执行但 reply 为空且无结构化结果 | 200（respond）/ done 事件里带 `degraded=true` | agent_loop | false | true |
| `execution_timeout` | 整体 deadline 到期 | 504 | execution | true | true |
| `execution_failed` | 未归类的其余异常（兜底分类，仅在补充分类失败时出现） | 500 | execution | false | false |

- `fallback_eligible` 字段是本次新增，专为兜底方案 §3 的判定服务；demo 目前不读取也不受影响（新增字段，向后兼容）。
- `AGENT_EMPTY_REPLY` 不是错误，而是通过 `done` 事件的 `metadata.degraded=true` 传递（不返回 error 事件），因为“执行成功但回复为空”本身不是失败——具体走 done 还是 error 由兜底方案的触发表（§3）决定；这里预留字段，不预判兜底方案的最终形态。

### 7.2 与现状的差异

- 目前 `respond` 只捕获 `AuthorizationError` / `ValueError` / `ConfigError`；`McpEdgeError`、`ToolIterationLimit`、`SkillLoadError` 会变成裸 500。改造后所有已知异常都要过一次 `classify_exception()`，未识别的异常才落到 `execution_failed`（500）。
- 目前流式路径的兜底 catch 一律吐 `execution_failed`（execution_service.py:880 附近）；改造后流式路径与非流式路径共用同一个 `classify_exception()`，保证两条路径错误码一致（这也是 D-5 单路径执行的必然结果，不需要额外测试两套分类逻辑）。

### 7.3 日志与 SSE 事件

- `ExecutionLog`：请求级、进程内、有界（100 条 / 64 KiB），字段为 `{stage, level, message, duration_ms}`（不含 execution_id/service，因为单进程单请求场景下这两个字段冗余；如账户方案要求补充，等 A2 落地时再加，不预先建表）。
- 默认路径：`ExecutionLog` 的内容放进 `done.metadata.logs` 和 `error.metadata.logs`（如果 error 事件也允许 metadata，需要在 demo `StreamEvent` 模型上确认；若不支持，则只在 respond 的 JSON 错误体里带 `logs`，流式 error 事件不带，避免破坏 demo 的校验器）。
- **opt-in 独立事件**：仅当 `ExecutionRequest.request_metadata.get("emit_log_events") is True` 时，才额外发送 `{"kind": "log", ...}` / `{"kind": "warning", ...}` 事件。这两种 kind 在 capabilities 中声明（`/capabilities/v2` 的 `context` 或新增 `logging` 字段），demo 的 `StreamValidator` 需要相应升级 `ALL_EVENTS`（跨仓库改动，不在本方案范围内，属于 demo 侧的后续工作，本方案只保证不发送未声明的 kind）。
- 密钥、Authorization、完整请求体不进入 `ExecutionLog`（与账户方案 §12.1 一致）。

## 8. Checkpoint 与会话连续性（D-2）

- `FlowDefinition.needs_checkpoint` 默认 `False`。三条现有 flow（qa、group_analysis、managed_yaml）都不需要 checkpoint：qa 和 group_analysis 本身无状态延续需求；managed_yaml 覆盖风险会商，风险会商也没有 HITL 节点。
- `build_execution_app` 的 lifespan 只有存在 `needs_checkpoint=True` 的 flow 时才建立 Postgres 连接；否则 `FLOWAGENT_CHECKPOINT` 环境变量被忽略，`app.state.checkpointer = None`。这修复了“每次请求写 checkpoint、从不读、从不清理”的问题（评审问题 2）。
- 已有 `session_continuity.plan_session`（P3，c799fa5）不用改：`has_checkpoint=False` 时总是返回 `start_new`，与“默认关闭”天然兼容。等以后某个 flow 需要 HITL 时，在该 flow 里设置 `needs_checkpoint=True` 并调用 `open_postgres_checkpointer`，同时接入 TTL 清理（复用 `langgraph-checkpoint-postgres` 的现有能力，不新增自定义清理任务，避免评审 D-9 之外再造一套后台任务）。
- `/execution-v1/reset` 保留现状：仍是空操作（无 checkpoint 需要清理），但修复输入校验（§5 已述）。

## 9. 部署整合（D-3）

### 9.1 单一 Dockerfile

```dockerfile
# workflow_agent/Dockerfile（替换现有 4 个 Dockerfile）
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY flowagent ./flowagent
COPY services ./services
COPY configs ./configs
COPY run_execution.py ./run_execution.py
RUN pip install --no-cache-dir ".[api,postgres]"
# 三个服务用不同 command 区分，见 9.2
```

- `Dockerfile.execution`、`Dockerfile.risk-consultation`、`services/im_tools_mcp/Dockerfile` 全部删除。
- 根目录 `Dockerfile`（跑 `examples/api_server.py` demo）和 `examples/` 一起删除或移到独立目录，取决于 D-7g 的清理范围（§11）。

### 9.2 compose 层的 command 区分

```yaml
# 片段示例，实际接入到 deploy/ 下现有 compose 文件（跨仓库，本方案只给 workflow_agent 侧的镜像）
services:
  workflow-execution:
    image: workflow_agent:latest
    command: ["python", "run_execution.py"]
    environment: {FRAMEWORK_INTERNAL_TOKEN: "${FRAMEWORK_INTERNAL_TOKEN:-}", ...}
  im-tools-mcp:
    image: workflow_agent:latest
    command: ["python", "-m", "services.im_tools_mcp"]
    environment: {IM_TOOLS_MCP_TOKEN: "${IM_TOOLS_MCP_TOKEN:-}", ...}
  consultation-mcp:
    image: workflow_agent:latest
    command: ["python", "-m", "services.consultation_mcp"]
    environment: {CONSULTATION_WRITE_MCP_TOKEN: "${CONSULTATION_WRITE_MCP_TOKEN:-}", ...}
```

- 三个服务共用一个镜像，体积比现状的 3 份分散镜像更大（多了不必要的依赖），但换来构建一次、版本一致，评审已判断这个取舍成立。
- `deploy/group-analysis/compose.group-analysis.yml` 和 `deploy/0927/...` 下引用 workflow_agent 镜像的 compose 文件需要同步改 `dockerfile:` 和 `command:` 字段——这是跨仓库改动，本方案在 workflow_agent 侧提供新 Dockerfile 和一份 compose 片段样例，实际改 `deploy/0927` 由部署侧配合。
- `Dockerfile.risk-consultation` 依赖的本地日期标签 `flowagent:group-name-20260926` 问题随之消失（评审问题 3 的核心诉求）。

## 10. 知识库 / 流程图 MCP 接口（D-4，仅接口冻结）

```
kb_mcp
  kb_search(query: str, top_k: int = 5, filters: dict | None = None)
    -> {"hits": [{"doc_id": str, "chunk": str, "score": float, "source": str}], "status": "ok"|"empty"}

flowchart_mcp
  flowchart_get(id: str)                    -> {"id", "name", "mermaid", "version"}
  flowchart_list(query: str | None = None)  -> {"items": [{"id", "name", "version"}]}
  flowchart_upsert(id: str | None, name: str, mermaid: str, expected_version: int | None = None)
    -> {"id", "version"}          # expected_version 做乐观锁；版本不符时返回 error_code=version_conflict
  flowchart_delete(id: str, expected_version: int)
    -> {"ok": true}
```

- 本次不实现这两个 MCP 服务，也不新建 `flowcharts` 表。接口形状参考 `services/im_tools_mcp` 现有结构（工具函数 + schema + `TOOL_HANDLERS` 字典）。
- 对应的 flow（`kb_qa.py`、`flowchart.py`）留到工具接口跑通后再单独评审，届时套用本方案 §3 的 `FlowDefinition` 模式：新增文件 + 注册一行 + 一组路径测试，不需要再碰执行入口。
- 是否需要“无依据拒答”和“保存前 HITL 确认”，留给下一轮评审；HITL 需要用到的 checkpoint 开关已经在 §8 里预留好。

## 11. 清理清单（D-7 其余项）

| 编号 | 内容 | 校验方式 |
|---|---|---|
| 7a | 删除 group_analysis.yaml 死节点 + `find_group`/`fetch_group_messages` 工具 | grep `publish_group_analysis.py`，确认发布脚本只引用 parse_args/report |
| 7c | 合并 `nodes.py` 三套执行器（agent/streamed-llm/通用）的重试、超时、trace、on_error 逻辑 | 抽出 `_run_node_with_reliability(func, node, ...)`，三种类型复用同一个包装器 |
| 7d | 群解析 not_found/ambiguous 文案抽成 `flowagent/group_analysis.py::describe_group_selection()`，`agent_runtime.py` 引用它 | 两处不再各自拼字符串 |
| 7g | 删除根 `Dockerfile`、`examples/api_server.py` 默认指向的 demo 配置链路 | 保留 `examples/basic_usage.py` 和 CLI 相关 demo 配置（approval/customer_service/refund_approval/report_writer 仍作为 CLI/测试样例保留，不在 7g 删除范围——评审原文只提到删根 Dockerfile 和 api_server.py，未决定删全部 demo 配置，此处保持原判断范围，避免超出评审授权） |
| 7h | 依赖清理：删除未使用的 `httpx2`，`requirements.txt` 补 `httpx`，postgres 依赖只在 `pyproject.toml` 声明一处 | `pip check` + `pytest` 全绿 |

## 12. 实施顺序

延续评审报告 §6 的顺序，结合本轮决策调整：

1. **准备（不改行为）**：写 §4 每条 flow 的路径级特征测试（对着现有 execution_service.py 跑一遍，固定当前行为），然后做 7h、7g、7a、7d、9a（IM 分页提前停止）。
2. **引擎修复**：D-6（error 清除 + trace 上限）、7c（执行器合并）、9d（删 `_SUB_ENGINES`）。
3. **架构主线**：在 `flows/` 下建 qa / group_analysis / managed_yaml 三张图 + planner + runner + errors + logs，接入统一执行路径（D-1、D-5、D-12），9b（启动时编译）、9c（MCP 会话复用）、7e（凭据映射）、7f（history_limit/timeout_ms/max_tool_iterations 生效）随这一步一起做，因为它们都发生在 planner/runner 内部。
4. **基础设施**：D-2（checkpoint 默认关闭）、D-3（单 Dockerfile）、D-11（安全放开三处）。
5. **扩展预留**：D-4 的 MCP 接口保持冻结状态，不在本轮实施。

第 1、2 步可以并行；第 3 步依赖第 1 步产出的特征测试（先有测试兜底，再大改执行路径）。

## 13. 风险与未决问题

- **7a 的假设需要实施前用 grep 核实**：`publish_group_analysis.py` 是否真的只注册 `fetch_group_analysis_input`。如果发布脚本还引用了待删节点，则先改发布脚本，再删 YAML。
- **demo 侧不动，但要在实施前对齐**：§7.3 的 opt-in log/warning 事件、§5 的新增错误码字段（`stage`/`retryable`/`fallback_eligible`）需要账户方案和兜底方案的开发者知晓字段已存在，避免各自重新发明。
- **9.2 的 compose 改动是跨仓库工作**：workflow_agent 只能提供新 Dockerfile，`deploy/0927` 和 `deploy/group-analysis` 下的 compose 文件需要配合修改，这部分不在本方案的验收范围内，只在文档里给出片段样例。
