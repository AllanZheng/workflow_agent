# workflow_agent「功能即图」重构测试方案

- 日期：2026-09-28；状态：测试设计，尚未实施。
- 配套技术方案：[TECH_PLAN_2026-09-28-flow-refactor.md](TECH_PLAN_2026-09-28-flow-refactor.md)。
- 基线：431 个测试通过，22 个跳过，行覆盖率 82.6%（`postgres.py` 0%，两个 MCP 服务的 `config.py` ~33%）。
- 门槛（评审 D-8）：整体行覆盖率 ≥ 85%，且本次修改涉及的模块（`flowagent/execution/`、`flowagent/flows/`、`flowagent/security.py`）单独达到 ≥ 90%。
- 与在研方案的用例对应关系见 §8；本方案编号使用 `WF-<组>-<序号>`，避免和账户方案的 `ACC/MCP/MEM/DESK` 等编号混淆。

## 1. 测试分层

| 层 | 范围 | 工具 |
|---|---|---|
| 单元 | planner 的 flow 选择、errors 的分类函数、logs 的有界缓冲、state 的 reducer、security 的 bearer_guard | pytest，纯函数，不起进程 |
| 图级 | 每条 flow 单独用 `MockLLM` + 假 MCP 跑通所有分支 | pytest + `flowagent.llm.MockLLM` |
| 契约回归 | 用当前 demo 实际发送的 V1/V2/V3 请求体录制成夹具，跑 respond 和 stream，断言响应字段是新增而非删除/改名 | pytest + `httpx.ASGITransport` |
| 集成 | 容器化 im_tools_mcp / consultation_mcp + workflow_agent，走真实 HTTP | 隔离 docker compose（§7），仅限本次新增的 SEC/DEPLOY 组 |
| 真实 LLM（可选） | 复用统一账户测试方案 §4 已定义的 Qwen3.6-35B-A3B-FP8 参数，验证 qa/group_analysis/managed_yaml 三条 flow 各跑一次真实请求 | 复用统一账户测试计划的隔离栈，不重新定义模型参数 |

真实 LLM 用例不是必需通过条件（本次重构不改变模型调用语义），只作为可选的最终确认，缺失时标记 BLOCKED 并说明原因，不算整体失败。

## 2. 测试框架检测

按 CLAUDE.md 规范（无 `## Testing` 小节，走自动检测）：

```
RUNTIME:python（pyproject.toml + requirements.txt）
测试命令：.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --ignore=.worktrees
覆盖率：pytest-cov，addopts 已在 pyproject.toml 声明
```

新增测试文件放在 `tests/`，与现有命名一致（`test_<module>.py`）。

## 3. 覆盖率坐标图（重构后新增代码）

```
CODE PATHS                                          USER FLOWS（对 demo 而言的行为）
[+] flowagent/execution/planner.py                  [+] respond 契约不变
  ├── select_flow()                                   ├── [GAP] V1 请求体 → 200，字段与今天一致
  │   ├── [GAP] V3 → qa                                ├── [GAP] V2 请求体 → 200
  │   ├── [GAP] V1/V2 deterministic → group_analysis   └── [GAP] V3 请求体 → 200
  │   └── [GAP] V1/V2 其他 → managed_yaml             [+] stream 契约不变
  ├── _digest 校验失败 → 409                            ├── [GAP] meta→delta*→done，seq 单调
  ├── identity 不符 → 409                               └── [GAP] error 事件后无后续事件
  └── [GAP] 不可达 llm 节点不再误报 missing_model      [+] 安全放开
[+] flowagent/execution/runner.py                       ├── [GAP] token 为空 → 放行 + WARNING 日志
  ├── run_stream() 正常完成                              ├── [GAP] token 非空且正确 → 200
  ├── [GAP] deadline 到期 → execution_timeout            └── [GAP] token 非空且错误 → 401
  ├── [GAP] 空回复 → degraded=true                     [+] checkpoint
  └── [GAP] respond 与 stream 最终内容逐字一致           ├── [GAP] 默认关闭：不写库
[+] flowagent/execution/errors.py                       └── [GAP] 显式启用的 flow：仍可写（非本次新增 flow，暂无测试对象）
  ├── [GAP] McpEdgeError → 502, retryable=true         [+] group_analysis
  ├── [GAP] ToolIterationLimit → 422                     ├── [GAP] 参数解析失败 → clarify
  ├── [GAP] AuthorizationError(denied) → 403              ├── [GAP] 群歧义/找不到 → 统一文案
  └── [GAP] 未知异常 → 500 execution_failed               ├── [GAP] IM 不可用 → status=failed
[+] flowagent/execution/logs.py                          ├── [GAP] 分批分类重试后失败 → partial
  ├── [GAP] 超过 100 条截断，logs_truncated=true          ├── [GAP] 单条消息超长 → 截断而非中断
  └── [GAP] 超过 64 KiB 截断                              └── [GAP] IM 分页遇到窗口外消息提前停止
[+] flowagent/nodes.py（合并后的执行器）                [+] qa
  ├── [GAP] 统一包装器：重试次数一致                       ├── [GAP] 工具循环达到上限 → TOOL_ITERATION_LIMIT
  ├── [GAP] 统一包装器：超时行为一致                       ├── [GAP] 工具不在白名单 → MCP_TOOL_NOT_ALLOWED
  └── [GAP] on_error 分支行为一致                          └── [GAP] 正常问答 → reply 流式输出
[+] flowagent/state.py
  ├── [GAP] 成功节点清除 error（回归测试，评审问题 6）
  ├── [GAP] trace 超过 200 条后只保留最近 200 条
  └── [GAP] 并行分支同时写 error 不抛 InvalidUpdateError（suppressed finding 复测）

COVERAGE：0/34 新路径已测（本次全部是 GAP，因为是新代码/新拆分）
```

Legend：`[GAP]` 表示需要新增测试；本次没有历史 `[★]` 标注，因为绝大多数目标模块是从旧文件拆分或新建。

## 4. 回归测试组（对着旧代码固定行为，先写后重构）

**REG 组必须在动执行路径之前全部跑通并锁定基线**（技术方案 §12 第 1 步）。

> **Stage 1 回填说明（2026-09-28）**：Stage 1 采用的是“增量修复 + 针对性新增测试”，没有另建一套独立编号的 REG 夹具。逐条核对后，REG-01~06 的断言已被下列既有测试实际覆盖并持续跑绿，为避免和计划 §8 一样重复维护两套夹具，这里只做编号到既有测试名的映射（而不是新增同名重复测试）；REG-07 已有独立测试锁定，作为唯一需要新增证据的一项，`flowagent/nodes.py::_SUB_ENGINES` 予以保留（9d 已加有界 LRU）。

| ID | 描述 | 断言 | 覆盖测试（状态） |
|---|---|---|---|
| REG-01 | 用当前 demo 会商请求体（V2, group_analysis_mode 默认 legacy）调 `execute_managed` | 响应字段与今天完全一致（execution_id/reply/used_context_refs/metadata） | ✅ `tests/test_execution_service.py::test_execution_service_executes_managed_yaml_and_returns_receipt` |
| REG-02 | 用 V1 请求体（`ManagedSnapshotV1`，`workflow_config_from_env` 产出的形状）调 `execute_managed` | 响应字段不变；确认 V1 路径未被误删 | ✅ `tests/test_managed_config_v3.py` + `test_execution_service.py` 中沿用 `agent_config.snapshot` V1 形状的用例（D-10：V1 未下线） |
| REG-03 | 用 V3 请求体调 `execute_managed` / `stream_managed` | reply、used_context_refs、metadata.agent_config_applied 字段不变 | ✅ `tests/test_execution_service.py::test_v3_model_authorization_is_checked_before_every_model_call_and_receipt_is_versioned`、`test_v3_injects_only_platform_memory_context_when_snapshot_enables_read` |
| REG-04 | 群分析 deterministic 模式，正常输入 | reply 文本、`analysis_result` 结构（status/partial_reasons/counts）不变 | ✅ `tests/test_group_analysis_processing.py::test_invalid_model_classification_cannot_be_counted`、`test_group_name_is_resolved_before_messages_are_read` |
| REG-05 | `/execution-v1/reset` 正常请求 | 响应体不变（`{"ok": true, "continuity_key", "generation"}`） | ✅ `tests/test_execution_service.py::test_execution_service_accepts_authenticated_reset_without_checkpoint_state` |
| REG-06 | `/capabilities`、`/capabilities/v2` | 响应体不变（新增字段除外，§5 的 `logging` 字段除外） | ✅ `tests/test_execution_service.py::test_execution_service_reports_conservative_workflow_capabilities` |
| REG-07 | 当前 `_SUB_ENGINES` 存在时的子图缓存行为 | 删除前先确认没有生产代码路径依赖它（受管 YAML 禁止 workflow 节点，预期无引用） | ✅ `tests/test_managed_config.py`（`type: workflow` 触发 `ConfigError: managed_node_type_not_allowed`，证实 execution-v1 路径永远到不了 `_sub_workflow_engine`） |

## 5. 契约回归组（demo 兼容性，WF-CONTRACT）

对应技术方案 §2 的兼容性约束表。使用 demo 真实构造请求体的函数作为夹具来源，而不是手写 mock，降低“测试对了但和 demo 实际形状不一致”的风险。

| ID | 描述 | 断言 |
|---|---|---|
| WF-CONTRACT-01 | 用 demo `consultation/integrations.py::generate()` 组装的请求体（V2, schema_version=2, credential_id=uuid5(...)）跑 `/execution-v1/respond` | 200；reply 非空；`FLOWAGENT_RISK_CONSULTATION_MODEL_API_KEY` 映射仍生效 |
| WF-CONTRACT-02 | 用 demo `platform/workflow_config.py::workflow_config_from_env()` 产出的 V1 快照跑 respond | 200；确认 V1 未下线（技术方案 D-10） |
| WF-CONTRACT-03 | demo `orchestrator.py` 的 `StreamValidator` 直接消费 `/execution-v1/respond/stream` 的真实输出（不 mock，跑一次真实 SSE 解析） | 不抛 `ProtocolViolation`；meta 首、done/error 尾、seq 单调 |
| WF-CONTRACT-04 | 请求 `request_metadata` 不携带 `emit_log_events` | 流中不出现 kind=log/warning 事件（避免破坏未升级的 demo 校验器） |
| WF-CONTRACT-05 | 请求 `request_metadata.emit_log_events=true`，但假设 demo 校验器尚未升级 | 明确标记为“需要 demo 同步升级后才能打开”，本次只测 workflow 侧发送的事件结构正确，不断言 demo 一定接受 |
| WF-CONTRACT-06 | 错误响应体新增字段 `stage`/`retryable`/`fallback_eligible`/`request_id` | 用 demo 现有的错误解析代码（`payload.get("error", {}).get("code")`）验证仍能正常取到 `code` |
| WF-CONTRACT-07 | McpEdgeError 场景下，HTTP 状态码从 500 变为 502 | demo `integrations.py` 的 `response.status_code != 200` 分支仍能正确捕获（断言其判断逻辑不依赖具体状态码数值） |

## 6. 单元与图级测试组

### 6.1 planner（WF-PLAN）

| ID | 场景 | 预期 |
|---|---|---|
| WF-PLAN-01 | V3 snapshot | 选中 `qa` |
| WF-PLAN-02 | V1/V2 + `group_analysis_mode: deterministic` | 选中 `group_analysis` |
| WF-PLAN-03 | V1/V2 + 其他 YAML | 选中 `managed_yaml` |
| WF-PLAN-04 | YAML 中存在不可达 llm 节点、可达节点无需模型 | 不再误报 `managed_model_config_missing`（修复项） |
| WF-PLAN-05 | history 超过 `runtime.history_limit` | 截断到限制长度（D-7f 回归） |
| WF-PLAN-06 | `workflow.timeout_ms` / snapshot 中 `adapter.timeout_ms` 存在 | `plan.deadline_s` 按其生效（D-7f 回归） |

### 6.2 errors（WF-ERR）

逐条覆盖 §7.1 错误码表的 15 行，每行至少一个用例，验证 `code`、`stage`、`retryable`、`fallback_eligible`、HTTP 状态码。

### 6.3 logs（WF-LOG）

| ID | 场景 | 预期 |
|---|---|---|
| WF-LOG-01 | 写入 101 条阶段日志 | 只保留前 100 条（或按实现保留最近 100 条，以技术方案最终实现为准），`logs_truncated=true` |
| WF-LOG-02 | 单条日志超大导致累计超过 64 KiB | 截断，`logs_truncated=true` |
| WF-LOG-03 | 日志中人为塞入形如 `sk-xxx` 的字符串（模拟意外写入密钥） | 本组不做脱敏（脱敏是调用方职责），但验证 `ExecutionLog` 本身不会记录框架不该传入的字段（如 headers），用来防止未来误用 |

### 6.4 state（WF-STATE，评审问题 6 回归）

| ID | 场景 | 预期 |
|---|---|---|
| WF-STATE-01 | 节点 A 失败 → on_error 跳到 B → 循环回 A → A 这次成功 | 路由不再被旧 error 误导向 on_error（核心 bug 修复） |
| WF-STATE-02 | 循环节点执行 250 次 | trace 只保留最近 200 条 |
| WF-STATE-03 | 两个并行分支在同一步都写 error | 不抛 `InvalidUpdateError`（suppressed finding 复测，confidence 从 5/10 提升为已验证） |

### 6.5 qa flow（WF-QA）

| ID | 场景 | 预期 |
|---|---|---|
| WF-QA-01 | 正常问答，无工具调用 | 直接流式输出 reply |
| WF-QA-02 | 触发一次工具调用后回答 | 只输出最后一轮文本 |
| WF-QA-03 | 工具调用达到 `max_tool_iterations` 上限 | `TOOL_ITERATION_LIMIT` |
| WF-QA-04 | 调用不在白名单的工具 | `MCP_TOOL_NOT_ALLOWED` |
| WF-QA-05 | `platform_resolve_group` 返回 not_found/ambiguous | 走统一澄清文案（D-7d），不进入错误分支 |
| WF-QA-06 | Skill 加载失败 | 降级继续，不中断（现状行为回归） |

### 6.6 group_analysis flow（WF-GA）

| ID | 场景 | 预期 |
|---|---|---|
| WF-GA-01 | 正常完整分析 | status=complete |
| WF-GA-02 | 参数解析失败（days 超范围等） | clarify 文本 |
| WF-GA-03 | 群名重复（ambiguous） | 统一文案，与 qa 的 WF-QA-05 复用同一份措辞 |
| WF-GA-04 | IM 工具不可用 | status=failed, im_tool_unavailable |
| WF-GA-05 | 单批分类连续 2 次失败 | 计入 failed，不中断其余批次 |
| WF-GA-06 | 单条消息超过 12000 字符 | 截断并记录，不再抛 `single_message_too_large` 中断整个分析（本次行为变更点，需要明确标注） |
| WF-GA-07 | IM 分页遇到早于 `window.start` 的消息 | 提前 break，不翻满 100 页（9a 回归） |

### 6.7 managed_yaml flow（WF-YAML）

| ID | 场景 | 预期 |
|---|---|---|
| WF-YAML-01 | 风险会商单节点 YAML，正常生成 | reply 非空，流式输出 |
| WF-YAML-02 | `workflow.timeout_ms` 到期 | `execution_timeout`，504 |
| WF-YAML-03 | 图启动时编译一次（非每请求） | 用 mock 计数验证 `build_graph` 调用次数与请求次数无关（9b 回归） |

### 6.8 安全放开（WF-SEC）

| ID | 场景 | 预期 |
|---|---|---|
| WF-SEC-01 | execution 服务 `FRAMEWORK_INTERNAL_TOKEN` 为空 | 启动成功，日志含 WARNING，所有请求放行 |
| WF-SEC-02 | 同上，非空且请求携带正确 token | 200 |
| WF-SEC-03 | 同上，非空且请求携带错误 token | 401 |
| WF-SEC-04 | im_tools_mcp `IM_TOOLS_MCP_TOKEN` 为空 | 放行，WARNING，不再要求 Bearer 头 |
| WF-SEC-05 | consultation_mcp `CONSULTATION_WRITE_MCP_TOKEN` 为空 | 放行，WARNING |
| WF-SEC-06 | 模型授权 `DemoModelAuthorizer` 拒绝 | 不受安全开关影响，仍返回 403（确认“保持不变”的部分没有被误放开） |
| WF-SEC-07 | `config_digest` 不匹配 | 不受安全开关影响，仍返回 409 |
| WF-SEC-08 | 会商 `allowed_conversation_ids` 白名单外的会话 | 不受安全开关影响，仍拒绝（该逻辑在 consultation_mcp，本次不改） |

### 6.9 checkpoint（WF-CKPT）

| ID | 场景 | 预期 | 覆盖测试（状态） |
|---|---|---|---|
| WF-CKPT-01 | 所有当前 flow（`needs_checkpoint=False`） | 启动不建立 Postgres 连接，即使 `FLOWAGENT_CHECKPOINT=postgres` | ✅ `tests/test_checkpointer.py::test_execution_service_default_disables_checkpoint_without_env` |
| WF-CKPT-02 | 同上，运行多次请求 | 不产生任何 checkpoint 表写入（用假 saver 断言 `put`/`aput` 未被调用） | ✅ `tests/test_checkpointer.py::test_execution_service_multiple_requests_never_touch_checkpoint_when_disabled`（2026-09-28 新增：`checkpointer=False` 时图内部不持有 saver 实例，两次真实请求后 `app.state.checkpointer` 仍为 `None`） |
| WF-CKPT-03 | `session_continuity.plan_session` 在无 checkpoint 时 | 始终返回 `start_new`（P3 现有测试的延伸） | ✅ `tests/test_managed_agent_runtime.py` / `flowagent/session_continuity.py` 既有测试（P3 遗留） |

### 6.10 部署（WF-DEPLOY，可选，容器级）

| ID | 场景 | 预期 |
|---|---|---|
| WF-DEPLOY-01 | 单 Dockerfile 构建后，分别用三个 command 启动 | 三个进程都能正常 `/health` |
| WF-DEPLOY-02 | im_tools_mcp 容器不再依赖 flowagent 包 | 镜像内可以不装 langgraph 也能跑（验证“合并镜像但功能独立”） |

## 7. 隔离测试栈（仅 SEC/DEPLOY 组需要容器）

复用统一账户测试方案 §3 的隔离 Docker 原则（独立 project 名、独立卷、不碰 0927 实际部署）。本次新增制品：

```
workflow_agent/deploy/compose.flow-refactor-test.yml   # 新增：workflow-execution + im-tools-mcp + consultation-mcp，用单一新 Dockerfile
```

不复用 0927 的固定 container_name 和网络；执行完 `down`（不加 `-v`，除非确认是本次新建的卷）。

## 8. 与在研测试计划的映射（避免重复编号）

| 本方案编号 | 对应统一账户测试计划编号 | 关系 |
|---|---|---|
| WF-CONTRACT-01/02 | 统一账户测试计划 LLM-05（Agent 正式执行新会话） | 本方案验证 workflow 侧契约不变；账户测试计划验证端到端真实模型调用。两者互补，不重复执行 |
| WF-SEC-06/07/08 | 账户方案 ACC-03、MCP-02、MCP-04（P0 阻断用例） | 本方案确认这些安全项没有被本次“放开三处”误伤；执行时机应晚于账户方案对应工作包（A0/P4）落地 |
| WF-DEPLOY-01/02 | 统一账户测试计划 §2 已核查的本地容器清单 | 本方案的隔离栈是全新项目名，不复用清单中列出的 0927 容器 |

如果实施时账户方案的 P4（MCP 授权）已经落地，WF-SEC-06/07/08 应该在账户方案的隔离栈里跑，而不是重新搭一套；本方案对应用例保留作为“变更前后对比”的记录，不重复维护两套夹具。

## 9. 覆盖率与门槛判定

- 命令：`.\.venv\Scripts\python.exe -m pytest --ignore=.worktrees --cov=flowagent --cov=services --cov-report=term`。
- 通过条件：总覆盖率 ≥ 85%；`flowagent/execution/`、`flowagent/flows/`、`flowagent/security.py` 三个新增模块 ≥ 90%。
- `postgres.py` 覆盖率 0% 的问题不在本次范围内解决（checkpoint 默认关闭后，覆盖率意义降低，留待真正启用 HITL 的 flow 时补测）。
- 两个 MCP 服务的 `config.py`（现状 ~33%）随 WF-SEC-04/05 补充到位后应升到 ≥ 70%。

## 10. 执行前置条件

1. §4 的 REG 组必须全部通过，作为重构的行为基线（技术方案 §12 第 1 步的强制前置）。
2. §5 的 WF-CONTRACT 组必须使用 demo 仓库中实际存在的请求构造函数作为夹具来源（而不是手写字面量），执行前确认 `demo_platform` 对应函数签名未变。
3. WF-SEC / WF-DEPLOY 两组涉及容器和环境变量组合，建议放在 CI 的独立 job，避免拖慢主测试套件。
4. 真实 LLM 可选用例（§1 表格最后一行）如果执行，直接复用统一账户测试方案 §4 的 `LLM_API_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`，不重新申请或记录一份模型参数。
