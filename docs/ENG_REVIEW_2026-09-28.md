# FlowAgent 工程评审（2026-09-28）

- 评审范围：整体代码库，包括 `flowagent/`、`services/`、`configs/`、`deploy/` 和各个 Dockerfile。分支为 master@31e5ee7。
- 产品定位：基于 LangGraph 的工作流智能体，不做通用智能体。现有功能有直接问答、群消息定制分析、风险研判；计划扩展知识库查询和流程图管理。
- 约束：越快越好，代码简单可读。
- 基线：431 个测试通过，22 个跳过，行覆盖率 82.6%。`postgres.py` 覆盖率为 0%，两个 MCP 服务的 `config.py` 覆盖率约 33%。

## 1. 现状

```
Platform ──execution-v1──> run_execution.py (execution_service.py, 902 行)
                             │
     ┌───────────────────────┼─────────────────────────────┐
     │ snapshot V3           │ V1/V2 + deterministic        │ V1/V2 + legacy
     ▼                       ▼                              ▼
 ManagedAgentRuntime     analyze_group() 纯 Python      build_graph(YAML) 每请求编译
 (直接问答，手写工具循环)  (群分析，YAML 只借 prompt)     (风险研判：单个 llm 节点)
     │                       │                              │
     └──── McpEdgeClient（每次调用都重新握手）──> im_tools_mcp / consultation_mcp
                                                             │
                                  Postgres：只存 LangGraph checkpoint（只写不读）
```

核心问题：三个功能各走一条路径，LangGraph 引擎在产品里几乎没有用上。子图、HITL、并行、循环护栏这些能力只有 demo 和测试在用。

## 2. 目标架构（D6 已定：一个功能对应一张图）

```
POST /execution-v1/respond[/stream]
   └─ auth Depends（有 token 就校验，没有就放行，见 §3.3）
   └─ run_flow(flow_name, request) ── 唯一的执行路径（astream）
         ├─ FLOWS = {"qa": qa_graph, "group_analysis": ga_graph, "risk": risk_graph, ...}
         │    每张图启动时编译一次；LLM/MCP 客户端通过 config["configurable"] 按请求注入
         ├─ checkpointer：默认 None；只有需要 HITL 的 flow 才启用（Postgres + TTL）
         └─ respond = 消费 stream 并拼装结果（和流式输出逐字一致）

flowagent/flows/
   qa.py              agent ⇄ tools（带 max_tool_iterations 上限）→ reply
   group_analysis.py  parse_args → resolve_group → fetch → classify_batches → report
   risk.py            generate
   （以后）kb_qa.py / flowchart.py，数据接入统一走 MCP
```

新增一个功能 = 新建 `flows/xxx.py`，在 `FLOWS` 里注册一行，再配一组路径测试。通用 YAML 引擎（`config.py`、`nodes.py`、`router.py`）只保留 demo 和 CLI 用得到的最小集合。

## 3. 决策与改动清单

### 3.1 架构

| # | 问题 | 决策 |
|---|---|---|
| 1 | 三个功能三条路径，LangGraph 基本没用上 | 一个功能一张代码定义的 StateGraph，统一入口 |
| 2 | `thread_id=execution_id`，checkpoint 每次请求都写，从不读，也不清理（execution_service.py:461；reset 的注释写的是“显式关闭 checkpoint”） | 默认关闭 checkpoint；只有需要 HITL 的 flow 才启用，并配置 TTL 清理。execution 服务不再强依赖 Postgres |
| 3 | 4 个 Dockerfile；`Dockerfile.risk-consultation` 依赖本地 tag `flowagent:group-name-20260926` 才能构建 | 合并成一个 Dockerfile，compose 里用不同的 `command` 区分服务：`python run_execution.py`、`python -m services.im_tools_mcp`、`python -m services.consultation_mcp` |
| 4 | 知识库和流程图的接入方式 | 先都做成 MCP 服务，接口草案见 §4；flow 图和存储细节尚未决定 |

### 3.2 代码质量

| # | 问题 | 决策 |
|---|---|---|
| 5 | 执行逻辑重复 4 份（execute/stream × V2/V3），鉴权代码复制了 6 次 | 只保留一条流式路径，respond 消费流；鉴权收敛成一个 FastAPI Depends |
| 6 | 节点成功时不清除 `state.error`，循环重试会被旧错误再次路由到 on_error；`trace` 没有长度上限 | 成功时返回 `error=None`；trace 用带上限的 reducer（最多 200 条）；error 使用“后写覆盖”的 reducer；补回归测试 |
| 7a | group_analysis.yaml 里的死节点，以及 `find_group`、`fetch_group_messages` 工具 | 删除 |
| 7b | V1 snapshot | 下线。假设：平台已经不再下发 V1，上线前需要确认 |
| 7c | nodes.py 里三套执行器各自实现重试、超时、trace、on_error | 合并成一个包装器 |
| 7d | 群解析 not_found / ambiguous 的文案写了两遍 | 抽成一个共用函数 |
| 7e | `_request_llm` 里有按 `"risk-consultation"` 名字硬编码的分支 | 改成通用的 credential_id 到环境变量的映射 |
| 7f | `history_limit`、`timeout_ms` 声明了但没有执行；`max_tool_iterations` 写死为 8 | 让这三项真正生效 |
| 7g | demo 配置、`examples/api_server.py`、根 Dockerfile | 删除 |
| 7h | `httpx2` 没有被 import 却列在 requirements.txt 里；requirements.txt 缺少 `httpx`；postgres 依赖在两处重复声明 | 去掉 httpx2，补上 httpx，依赖统一以 pyproject 为准 |

### 3.3 安全（用户决定：C，全部放开）

用户的原话：“安全相关的保留只是放开，其他删除”。实现方式如下：

- 安全相关代码全部保留，由一个开关控制，比如 `FLOWAGENT_SECURITY_ENFORCE`，默认值为 `false`。开关覆盖以下几项：
  - execution 服务、im_tools_mcp、consultation_mcp 的 Bearer token 校验
  - 模型授权（`DemoModelAuthorizer`）
  - `config_digest` 校验、skill sha256 校验
  - 会话白名单、发送者白名单
- token 为空时仍可正常启动，但打印一条 WARNING 日志。

风险记录：放开期间，任何能访问到端口的人都可以读任意群的消息、以平台身份发消息；平台下发的配置被篡改或截断也不会被发现。这个开关可以随时关回去，但放开期间泄露的数据收不回来。**建议 compose 去掉 MCP 端口的宿主机映射，只走容器内网。** 这一条只是建议，没有作为决策。

### 3.4 性能

| # | 问题 | 决策 |
|---|---|---|
| 9a | `fetch_group_analysis_input` 遇到窗口外的消息只 `continue`，不停止翻页，所以每次都翻满 100 页，还会误标 `page_limit` | 当前页最老的消息早于 `window.start` 时直接 break |
| 9b | 每次请求都解析 YAML 并调用 build_graph | 每个 flow 启动时编译一次 |
| 9c | 每次 MCP 调用都新建 httpx 客户端并做一次 initialize 握手 | 每次执行复用同一个 MCP 会话；整个 app 共享一个 AsyncClient |
| 9d | `_SUB_ENGINES` 没有容量上限，缓存键里还带着 `id(llm)` | 删除 |

## 4. 知识库 / 流程图 MCP 接口草案（D10：先做成 MCP）

```
kb_mcp
  kb_search(query: str, top_k: int = 5, filters: dict | None)
    -> {"hits": [{"doc_id", "chunk", "score", "source"}], "status": "ok|empty"}

flowchart_mcp
  flowchart_get(id)                         -> {"id", "name", "mermaid", "version"}
  flowchart_list(query?: str)               -> {"items": [{"id", "name", "version"}]}
  flowchart_upsert(id?, name, mermaid, expected_version?)  -> {"id", "version"}
       # expected_version 用于乐观锁，防止并发编辑互相覆盖
  flowchart_delete(id, expected_version)    -> {"ok": true}
```

- 这两个服务和 im_tools_mcp 采用同样的结构，放在 `services/`，打进同一个镜像。
- 存储方案未定。如果流程图需要落库，只需要一张表：`flowcharts(id, name, mermaid, version, updated_by, updated_at)`。
- 以后 flow 图是否要做“无依据就拒答”和“保存前人工确认（HITL）”，等 MCP 接口跑通后再评审。

## 5. 测试计划（D14：路径级测试 + 覆盖率 ≥ 85% 门槛）

```
请求 → [鉴权：有 token / 无 token / token 错误 × 开关开 / 关]
     → [snapshot v2 | v3 | v1（拒绝）]
     → flow 分发（qa | group | risk | 未知 flow → 422）
        group：参数解析失败 / 找不到群 / 群有歧义 / IM 不可用 / 分批分类重试 / partial / 单条消息超长
        qa：   工具循环达到上限 / 工具不在白名单 / MCP 错误
        risk： 正常 / 模型凭据缺失
     → stream(meta → delta* → done | error)，并断言 respond 结果与流式输出一致
```

需要补的测试：
1. error 清除后循环重试的回归测试
2. checkpoint 关闭时不写库
3. 安全开关开、关两种状态
4. 流式与非流式输出一致
5. history_limit 截断；timeout_ms 到期后返回稳定的错误码
6. reset 接口收到非 JSON 或非对象的 body 时返回 422（目前遇到数组 body 会 500）
7. IM 分页提前停止
8. 两个 MCP 服务的 config.py 在有、无环境变量时的行为
9. 每个新 flow 都要有路径测试

## 6. 实施顺序

1. **准备（不改行为）**：先写 §5 的特征测试，把现有行为固定下来，再做 7h、7g、7a、7d、9a。
2. **引擎修复**：6、7c、9d。
3. **架构主线**：1 + 5 + 9b + 9c + 7e + 7f。在 `flows/` 下建三张图，统一走一条流式入口，然后删掉 V1（7b）。
4. **基础设施**：2、3、§3.3 安全开关。
5. **扩展**：kb_mcp、flowchart_mcp（§4）。

第 1、2 步可以并行；第 3 步依赖第 1 步里的特征测试。

## 不在本次范围

- 向量库选型、embedding 和切块策略（放在 kb_mcp 内部实现）
- 平台侧的 compose 和协议调整（仓库外）
- 压测和长稳测试

## Suppressed findings

- (5/10) 并行扇出时两个分支同时写 `error` 可能触发 `InvalidUpdateError`：这是根据 LangGraph 语义推断的，没有实测。问题 6 里给 error 加 reducer 可以顺带覆盖这个问题。
- (4/10) 模型授权失败时，用 `"denied" in str(exc)` 这种字符串匹配来决定返回 403 还是 503，比较脆弱。安全开关关闭后这条路径不会触发。

## GSTACK REVIEW REPORT

| Run | Status | Findings |
|---|---|---|
| Step 0 范围确认 | DONE | 整体代码库，6 个评审维度 |
| 1 架构 | DONE | 4 项（1、2、3、4） |
| 2 代码质量 | DONE | 9 项（5、6、7a–7h） |
| 3 测试 | DONE | 9 个缺口，覆盖率门槛 85% |
| 4 性能 | DONE | 4 项（9a–9d） |
| Outside voice | SKIPPED | 本机没有可用的 codex / 第二模型 |

VERDICT：通过，按 §6 的顺序实施。安全全部放开是用户明确选择的，风险已在 §3.3 记录。

**UNRESOLVED DECISIONS:**
- 7b：平台是否还在下发 V1 snapshot，需要平台侧确认（假设已停止）。
- 知识库和流程图的 flow 图设计、流程图存储位置：等 MCP 接口跑通后再评审。
