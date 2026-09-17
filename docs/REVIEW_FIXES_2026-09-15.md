# flowagent 评审问题修复方案与验证报告

> 日期：2026-09-15 ｜ 版本：0.9.0 ｜ 依据：[PROJECT_REVIEW_2026-09-15.md](PROJECT_REVIEW_2026-09-15.md)（结论 DONE_WITH_CONCERNS）
>
> 状态：**10 项已确认问题全部修复**；真实 LLM（glm-5.3-flash @ vLLM）端到端验证通过；
> 另修复验证过程中暴露的 2 项测试鲁棒性缺陷。

---

## 1. 修复原则

统一**失败关闭（fail-closed）**：未知审批值不放行、畸形 LLM 输出不通过、
字段缺失不命中路由、超时不重试、默认拒绝外部工具模块。
防御落在两层——**引擎层强校验**（配置写错也安全）+ **配置层精确语义**（示例即最佳实践）。

## 2. 修复清单（对应评审 §4.1–4.10）

| # | 级别 | 问题 | 根因 | 修复 | 主要改动文件 |
|---|------|------|------|------|--------------|
| 4.1 | P1 | 否定式审批被判批准 | 路由 `contains` 子串匹配 + options 不校验 | 示例改 `eq` + 默认驳回；options 声明即强校验，无效答复携带错误提示重新 `interrupt` | `nodes.py`、`cli.py`、`configs/{approval,refund_approval}.yaml` |
| 4.2 | P1 | LLM 缺字段默认通过 | 只验 JSON 合法性不验字段 | 新增 `output_schema`（type/min/max/enum/optional）字段校验；数值路由操作符遇缺失不命中 | `config.py`、`parsers.py`、`router.py`、`nodes.py`、3 份示例配置 |
| 4.3 | P1 | timeout 不按时返回 | `with` 退出即 `shutdown(wait=True)` | 超时 `shutdown(wait=False, cancel_futures=True)`；超时不参与自动重试；`TimeoutError` 规范化 | `nodes.py` |
| 4.4 | P1 | 子图人工中断被吞 | 只读子 vars、子线程 ID 随机 | 中断提升到父图重新 `interrupt`；子线程 ID 稳定派生 `{parent}::{node}`；同父 thread 恢复闭环 | `nodes.py` |
| 4.5 | P1 | on_error.goto 未生效 | 从未编译进图 | 三种 `next` 形态均编译错误改道边；`node.on_error.goto` 优先；目标参与引用校验 | `engine.py`、`router.py`、`config.py` |
| 4.6 | P2 | 伪 HTTP 202 | 状态码只写在 body | 真实 `JSONResponse(202)`；OpenAPI 声明 200/202/400/404/409 | `examples/api_server.py` |
| 4.7 | P2 | 校验允许必败值 | 缺下界与编译期检查 | `recursion_limit≥1`、`max_loops≥1` + 出口要求、regex 编译校验、并行列表非空 | `config.py` |
| 4.8 | P2 | 版本三处不一致 | pyproject/CLI/文档各自维护 | 单一版本源 `__version__` 动态读取；console script；`MANIFEST.in`；CI build-and-install job；sqlite extras | `pyproject.toml`、`MANIFEST.in`（新建）、`ci.yml`、`__init__.py` |
| 4.9 | P2 | 工具白名单默认放行 | 未配置即不限 | 默认仅内置 `flowagent.tools`；显式 `*` 才全放行；README 安全边界章节 | `tools.py`、`README.md` |
| 4.10 | P3 | 文档漂移 | 导出遗漏、命令不跨平台 | 导出 `TemplateError`；README 命令 PowerShell/Bash 双兼容 | `__init__.py`、`README.md`、`TECHNICAL_DOC.md` |

## 3. 真实 LLM 验证暴露并修复的额外问题

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| A | 配置真实端点后 3 个功能测试变红 | 套件隐式依赖"无 LLM 凭据"环境 | conftest autouse fixture 默认钉死 Mock（`test_real_llm.py` 豁免） |
| B | 无 `PYTHONUTF8=1` 时子进程输出解码崩溃 | 示例脚本管道输出 GBK | `examples/basic_usage.py` 补 stdout UTF-8 重配置（与 CLI 同款） |

## 4. 真实 LLM 端到端验证（glm-5.3-flash）

| 场景 | 结果 |
|------|------|
| customer_service 全流程 | 分类→退款→质检 `output_schema` 通过（score=7） |
| approval HITL（sqlite 跨进程） | 暂停 → **"不批准"被 options 拦截重新询问** → 批准闭环 |
| refund_approval 综合示例 | 真实风险评级 → 人工批准 → 质检 7/10 |
| report_writer 循环护栏 | 真实质检 7→9（`lt 8` 数值路由正确）后发布 |
| `tests/test_real_llm.py` | 3/3 通过 |
| 全量套件（真实环境 / 无环境） | 311 passed ／ 308 passed + 3 skipped |

## 5. 最终验证快照（2026-09-15）

```
pytest                 308 passed, 3 skipped（分支覆盖 95.8%）
ruff (E,F,W,B,UP)      全部通过
check --strict ×4      全部通过
python -m build        wheel + sdist 成功，元数据 0.9.0
flowagent --version    0.9.0
```

## 6. 测试增量

259 → 308 个（+49 行为测试）：审批否定反例、options 重新询问、畸形 LLM 输出、
路由数值硬化、超时耗时与副作用次数、子图恢复闭环、错误路由三形态、HTTP 状态码、
校验拒绝、白名单默认拒绝、版本一致性。全部遵循"反例 + 失败关闭断言"模式。

## 7. 未覆盖范围（评审 §8，第四阶段新能力，非缺陷修复）

- API 认证 / 租户隔离 / thread_id 访问控制
- 真实 LLM 的 Docker 化部署、并发与长稳压测

需要时另行立项（走 brainstorming → 设计 → 计划流程）。

## 8. 文档索引

- 评审报告：[PROJECT_REVIEW_2026-09-15.md](PROJECT_REVIEW_2026-09-15.md)
- 设计文档：[superpowers/specs/2026-09-15-review-fixes-design.md](superpowers/specs/2026-09-15-review-fixes-design.md)
- 实施计划：[superpowers/plans/2026-09-15-review-fixes.md](superpowers/plans/2026-09-15-review-fixes.md)
- 复盘（评审 10 项）：[problems/review_fixes/review_fixes_20260915.md](problems/review_fixes/review_fixes_20260915.md)
- 复盘（测试鲁棒性）：[problems/tests/suite_env_robustness_20260915.md](problems/tests/suite_env_robustness_20260915.md)
- 变更记录：[../CHANGELOG.md](../CHANGELOG.md) §0.9.0
