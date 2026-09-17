# 问题复盘 / Issue Postmortem: flowagent 评审修复（10 项语义缺陷）

## 1. 问题现象 / Symptoms

- **错误信息 / Error Message:**
  - [中文] 无显式报错——全部是语义缺陷：`STATUS finished APPROVED_BRANCH True`（"不批准"被判批准）、`MISSING_SCORE_RESULT approved`（缺 score 放行）、`TIMEOUT_ELAPSED 0.613`（0.05s 超时耗时 0.6s）、`PARENT_PENDING None`（子图人工中断被吞）、`ON_ERROR_TRACE ['bad'] FALLBACK_VAR None`（on_error.goto 失效）、`API_HTTP_STATUS 200 BODY_STATUS_CODE 202`（伪 202）。
  - [English] No runtime errors — all silent semantic defects: negative approval matched as approved; missing LLM field defaulted to pass; 0.05s timeout took 0.6s; subgraph human interrupt swallowed; on_error.goto ignored; fake HTTP 202 in body.

- **触发条件 / Trigger Condition:**
  - [中文] 运行探针对特定输入/配置组合：否定式审批答复、缺字段 LLM JSON、慢工具+超时、子图含 human 节点、配置 on_error.goto、API 等待人工。
  - [English] Probes with: negative approval replies, LLM JSON missing fields, slow tool + timeout, subgraph containing human node, on_error.goto config, API waiting-human.

- **影响范围 / Impact Scope:**
  - [中文] 审批/质检语义（错误授权风险）、副作用重复执行、HITL 可恢复性、错误降级、API 客户端状态判断、发布产物一致性、工具加载安全边界。
  - [English] Approval/QA semantics (wrong-authorization risk), duplicate side effects, HITL resumability, error degradation, API client signaling, release artifact consistency, tool-loading security boundary.

## 2. 根因分析 / Root Cause Analysis

- **根因 / Root Cause:**
  - [中文] 高语句覆盖率的测试只验证"代码路径是否执行"，未验证业务语义的失败关闭方向；多处实现选择"宽容默认"（子串匹配、字段缺失继续、默认放行、白名单缺省不限）。
  - [English] Coverage-focused tests verified code paths, not fail-closed business semantics; multiple implementations chose permissive defaults (substring match, proceed on missing field, allow-by-default).

- **直接原因 / Direct Cause:**
  - [中文] 4.1 `contains` 子串匹配 + options 不校验；4.2 解析器只验 JSON 合法、数值比较退化字符串；4.3 `with ThreadPoolExecutor` 退出即 `shutdown(wait=True)`；4.4 只读子图 vars 且子线程 ID 随机；4.5 goto 未编译进图；4.6 202 只写 body；4.7 校验缺下界；4.8 三处版本各自维护；4.9 白名单缺省不限；4.10 导出遗漏。
  - [English] See per-issue causes above: substring match, parse-only validation, blocking shutdown, discarded `__interrupt__` + random sub-thread ids, unwired goto edge, body-only 202, missing bounds, three version sources, allow-by-default, missing export.

- **间接原因 / Contributing Factors:**
  - [中文] on_error 测试让 goto 与 next 指向同一节点；超时测试不断言耗时；API 测试断言与注释相反；子图测试不含 human。
  - [English] Error-path tests pointed goto and next at the same node; timeout tests never asserted elapsed time; API test asserted the opposite of its comment; subgraph tests had no human node.

## 3. 解决方案 / Solution

- **修复思路 / Fix Strategy:**
  - [中文] 统一"失败关闭"原则：未知审批值重新询问不放行；LLM 输出过 `output_schema` 强校验；数值路由遇缺失字段不命中；超时停止等待且不重试；子图中断提升到父图并以稳定 ID 恢复；on_error.goto 编译为条件边并做引用校验；真实 HTTP 202；加载期拒绝必败值；版本单一来源；工具默认仅内置白名单。
  - [English] Uniform fail-closed principle across approval, LLM output, routing, timeout, subgraph interrupts, error routing, HTTP semantics, config validation, versioning, and tool loading.

- **具体改动 / Changes Applied:**
  - [中文] `flowagent/{nodes,router,config,engine,parsers,tools,cli,__init__}.py`、`examples/api_server.py`、`pyproject.toml`、`MANIFEST.in`（新建）、`.github/workflows/ci.yml`、`configs/{approval,refund_approval,customer_service,report_writer}.yaml`、`README.md`、`docs/TECHNICAL_DOC.md`、`CHANGELOG.md`；新增 49 个行为测试。
  - [English] Engine modules, API server, packaging, CI, four configs, docs; +49 behavior tests.

- **为什么这样改 / Why This Fix:**
  - [中文] 修改集中在语义边界（引擎层强校验 + 配置层精确匹配双层防护），保持既有模块边界与公开 API 兼容；超时/线程限制如实文档化而非过度承诺。
  - [English] Fixes concentrate at semantic boundaries (engine-level enforcement + exact-match configs), preserving module boundaries and public API; thread-limitation honestly documented.

## 4. 验证结果 / Validation

- **验证步骤 / Validation Steps:**
  - [中文] 全量 pytest（308 passed, 3 skipped，分支覆盖 95.8%）；Ruff（E,F,W,B,UP）通过；`python -m build` 成功；wheel 元数据 `Version: 0.9.0` + console_scripts + extras（dev/api/sqlite）；sdist 含 configs/docs/examples；`flowagent --version` = 0.9.0；`flowagent check --strict` 通过。
  - [English] Full pytest (308 passed, 3 skipped, 95.8% branch); Ruff clean; build OK; wheel metadata verified; sdist assets verified; CLI 0.9.0; strict check OK.

- **验证结论 / Validation Result:**
  - [中文] 评审 §9 验收标准逐项满足（否定审批不进批准分支、缺 score 失败关闭、超时按时返回且副作用一次、子图同父 thread 恢复、goto 唯一有效、真实 202、版本一致、wheel 可用）。
  - [English] All acceptance criteria from review §9 verified.

## 5. 预防措施 / Prevention

- **监控/告警 / Monitoring/Alerting:**
  - [中文] 生产接入后对审批分支命中分布做审计告警（否定语义命中批准=0）；对超时后重试次数断言为 0。
  - [English] Audit approval-branch hit distribution (negative-sentence-to-approved must be 0); alert on any retry-after-timeout.

- **测试补充 / Test Coverage:**
  - [中文] 已补 49 个行为测试；后续新能力须附带反例（否定/畸形/越界输入）与失败关闭断言。
  - [English] 49 behavior tests added; future features must ship negative/malformed/boundary cases with fail-closed assertions.

- **规范改进 / Process Improvement:**
  - [中文] CI 新增 build-and-install job（wheel 安装后验证命令与版本）；评审探针脚本可沉淀为回归测试集；"宽容默认"需在安全评审中显式豁免。
  - [English] New CI build-and-install job; review probes should be folded into regression suites; permissive defaults now require explicit security sign-off.

---

## 原始问题 / Original Issue

**用户提示词 / User Prompt:**
```
按照project_review与我讨论修复方案并修复
```
（依据：docs/PROJECT_REVIEW_2026-09-15.md，结论 DONE_WITH_CONCERNS，10 项已确认问题）
