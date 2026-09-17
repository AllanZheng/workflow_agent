# flowagent 评审修复设计（2026-09-15）

> 依据：`docs/PROJECT_REVIEW_2026-09-15.md`（结论 DONE_WITH_CONCERNS，10 项已确认问题）
> 状态：已经用户逐项确认方案（范围：全部 10 项；关键决策见各节）
> 目标：修复全部 10 项评审问题，补充行为测试，版本升至 0.9.0

## 0. 总体约定

- **单一版本源**：`flowagent/__init__.py` 的 `__version__`；pyproject 改为
  `dynamic = ["version"]` + `[tool.setuptools.dynamic] version = {attr = ...}`。
  本次修复完成（含新能力）后版本从 0.8.0 升为 **0.9.0**，CHANGELOG 同步。
- **所有新增/修改代码写中文注释**，说明"为什么/关键意图"。
- **TDD**：每个问题先写失败测试再修复（补齐评审 §5 指出的行为测试缺口）。
- **环境**：重建 `.venv`（系统 Python 3.12.7）；pytest 按 CI 方式设置
  `PYTHONUTF8=1`，必要时 `--basetemp` 指向工作区。
- **明确不做**（评审第 4 阶段新能力项，超出"修复"范畴）：API 认证/租户隔离/
  thread_id 访问控制、真实 LLM / Docker / 并发部署测试。

## 1. P1 语义修复

### 4.1 否定式人工回复被判为批准

**根因 / Root cause**：示例配置用 `contains` 子串匹配审批词（"不批准"包含"批准"），
引擎对人工输入不做 options 校验。

**方案 / Solution**：
1. `configs/approval.yaml`、`configs/refund_approval.yaml`：`contains` → `eq`，
   未知值落入 `default`（两份配置 default 均为安全侧 reject/rejected）。
2. 引擎层（`nodes.py` human 分支）：节点声明了 `options` 且恢复值不在其中时，
   **携带错误提示重新 `interrupt()`**——不前进、不崩溃，人工重新作答；
   未声明 `options` 保持自由文本。未知值永远无法进入业务分支（失败关闭）。
3. `options` 语义由"提示用"升级为"声明即强校验"，`config.py` 注释同步。

**测试**："不批准 / 暂不批准 / 乱输入"不进 approve；无效输入触发二次 interrupt。

### 4.2 LLM 结构化输出缺字段默认放行

**根因**：`parse_json` 只验 JSON 合法性不验字段；路由字段缺失时 `get_path`
返回 `None`，数值比较退化为字符串比较后落入 default（放行分支）。

**方案**：
1. `NodeConfig` 新增可选 `output_schema`：轻量字段规范，YAML 友好：
   ```yaml
   output_schema:
     score: {type: integer, min: 1, max: 10}   # 未标 optional 默认必填
     reason: {type: string}
   ```
   支持 `type: string|integer|number|boolean`、`min/max`、`enum`、`optional: true`。
   llm 节点 `parser: json` 时解析后校验，违规抛 `OutputParseError`
   （可被 retries / on_error 捕获；默认失败关闭）。
2. `router.py` 硬化：`gt/ge/lt/le` 遇字段 `None` 或不可数值化时规则**不命中**
   （禁止 `"None" < "7"` 退化比较）。`eq/ne` 保持宽容语义不变。
3. `customer_service.yaml`、`refund_approval.yaml` 质检节点补 `output_schema`。

**测试**：缺字段 / 类型错 / 越界 → 失败或降级；`None` 字段数值比较不命中。

### 4.3 timeout 不缩短耗时

**根因**：`with ThreadPoolExecutor` 退出即 `shutdown(wait=True)`，超时后仍等
线程跑完；且超时后重试会重复执行副作用。

**方案**：
1. `_with_timeout`：超时路径改为显式 `pool.shutdown(wait=False,
   cancel_futures=True)` 后抛 `TimeoutError`，调用按时返回。
   文档如实声明：Python 线程无法强杀，后台线程可能仍完成副作用。
2. `_run_with_reliability`：`TimeoutError` **不参与自动重试**，立即上抛
   （超时通常意味着外部调用慢/挂起，重试会造成副作用重复）。

**测试**：0.05s timeout + 0.6s 工具 → 实际耗时 ≈ timeout；配置 retries 时
超时后工具**只执行一次**。

### 4.4 子图 interrupt 被吞

**根因**：workflow 节点只提取子图 `vars`，丢弃 `__interrupt__`；且子引擎每次
`run` 生成随机 thread_id，中断后本就无法恢复。

**方案**（手动传播，保留现有架构）：
1. workflow 节点内用 `langgraph.config.get_config()` 取父 thread_id，派生稳定
   子 thread_id：`{parent_tid}::{node_name}`。
2. 执行逻辑：
   - 先查子引擎 `get_state(sub_tid)`：有 pending interrupt → 父图刚被恢复，
     本节点 `interrupt()` 取回父级 resume 值并**转发给 `sub_engine.resume`**；
   - 否则 `sub_engine.run(ctx, thread_id=sub_tid)`；
   - 子图返回值仍含 `__interrupt__`（多人工节点场景）→ 原样 re-raise
     `interrupt()` 并循环，直到子图真正结束；
   - 子图完成后按现有规则合并 vars（`output_key` 或直并）。
3. re-raise 的 payload 在原子图 interrupt 载荷上补充 `workflow_node` 上下文。

**测试**：父 → 子（含 human）→ 父 pending；同一父 thread_id resume → 子图继续
→ 父图完成（对应评审验收标准）。

### 4.5 on_error.goto 未生效

**根因**：`build_graph` 只编译 `next` 边；`node.on_error.goto` 从未加入图边，
也不参与引用校验。

**方案**：
1. `engine.py`：节点配 `on_error.goto` 时，`next` 为字符串/列表也编译为
   **条件边**（error.node 命中当前节点 → goto；否则原目标；列表目标保持
   并行扇出，条件函数返回列表）。
2. 权威优先级：`node.on_error.goto` > `route.on_error`（router 需感知
   node.on_error；文档写明只需配一处）。
3. `config.py next_targets` 纳入 `on_error.goto` → 非法目标加载期报错。

**测试**：str / list / route 三种 next 形态 + 优先级 + 非法目标被拒。

## 2. P2 修复

### 4.6 HTTP 202 假信号

- `examples/api_server.py`：等待人工时返回 `JSONResponse(status_code=202)`，
  删除 body 中伪 `status_code` 字段；OpenAPI 显式声明 200/202/400/404/409；
  FastAPI `version` 改用 `flowagent.__version__`。
- `tests/test_api_server.py` 同步改断言真实 202。

### 4.7 配置校验放行必败值

`config.py` 增加：`recursion_limit ≥ 1`；`max_loops ≥ 1`；配置 `max_loops`
的规则必须有 `loop_exhaust` 或路由级 `default`；`regex` 操作符编译期验证；
并行 `next` 列表非空（min_length=1）。

### 4.8 版本与发布一致性

- 单一版本源（见 §0）；`[project.scripts] flowagent = "flowagent.cli:main"`
  （确认/补齐 `main()` 入口）；`MANIFEST.in` 收录 configs / docs / examples /
  README / CHANGELOG / LICENSE；pyproject 增加可选组
  `sqlite = ["langgraph-checkpoint-sqlite>=2.0"]`。
- CI 新增 job：build → 安装 wheel → `flowagent --version`、版本一致性、
  `flowagent check` 严格校验。
- api_server / OpenAPI 版本统一读 `flowagent.__version__`。

### 4.9 工具白名单默认放行

`tools.py`：`FLOWAGENT_TOOL_MODULES` 未设置时默认白名单 `["flowagent.tools"]`
（内置工具可用，外部模块一律拒绝）；显式设 `*` 才全放行（可信本地用途）；
README 增加"可信配置 vs 不可信输入"安全边界说明。使用外部工具的现有测试
显式设置环境变量。

## 3. P3 文档漂移（4.10）

- `__init__.py` 导出 `TemplateError`（加入 `__all__`），与 docs/API.md 对齐。
- README：命令改为 PowerShell / Bash 双兼容写法（`'{"key": "value"}'`
  单引号包双引号 JSON、去除反斜杠续行）；补 console script 用法。

## 4. 产出物与验收

1. 代码修复 + 新增约 25–30 个行为测试（先失败后通过）。
2. `CHANGELOG.md` 0.9.0 条目。
3. 本设计文档。
4. 双语复盘文档：`docs/problems/review_fixes/`（用户已确认默认路径）。

**验收对齐评审 §9**：否定审批值永不进批准分支；缺/非法 score 不放行；
timeout 按时返回且副作用至多一次；子图 human 可暂停父流程并以同一父
thread_id 恢复；`on_error.goto` 唯一有效且过引用校验；等待人工时真实
HTTP 202；版本四处一致；wheel 安装后 CLI 可用。
