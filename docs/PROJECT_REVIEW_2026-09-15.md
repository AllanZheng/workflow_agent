# flowagent 项目审查与测试报告

> 审查日期：2026-09-15  
> 审查范围：当前项目快照、核心引擎、示例工作流、FastAPI 示例、测试与发布配置  
> 审查方式：静态代码审查、完整 pytest、覆盖率、Ruff、构建检查、CLI/API 运行探针  
> 结论：`DONE_WITH_CONCERNS`

## 1. 执行摘要

项目结构清晰，核心模块职责划分合理，自动化测试数量和覆盖率也处于较好水平。完整测试在隔离环境中得到 `259 passed, 3 skipped`，分支覆盖率为 `96.6%`，Ruff、Python 编译检查、示例配置严格校验和 Python 包构建均通过。

但项目目前不适合直接用于真实审批或生产服务。审查确认了多个测试未覆盖的语义缺陷，其中最需要优先处理的是：

1. 人工输入“**不批准**”会被识别为“批准”。
2. LLM 返回缺少关键字段的 JSON 时，质检路由可能默认放行。
3. `timeout` 不会真正缩短调用耗时，且可能造成副作用重复执行。
4. 子工作流中的人工中断会被父流程静默吞掉。
5. `on_error.goto` 配置没有按文档描述生效。

这些问题集中在审批语义、LLM 输出信任边界、错误恢复和组合能力上。它们不会被高语句覆盖率自动发现，需要补充针对业务语义和失败模式的测试。

## 2. 项目概况

flowagent 是一个配置驱动的 LangGraph 工作流引擎，主要能力包括：

- YAML 工作流定义与静态校验；
- LLM、模板、工具、人工介入和子工作流节点；
- 条件路由、并行扇出、循环护栏；
- 重试、超时和错误降级；
- 内存或 SQLite 检查点；
- 同步、异步、流式和 FastAPI 示例接口。

核心代码约 938 条可执行语句，测试覆盖 CLI、配置、引擎、路由、持久化、人工介入、子图和 API 等主要模块。

## 3. 测试结果

### 3.1 自动化测试

执行命令：

```powershell
python -m pytest --cov=flowagent --cov-report=term-missing
```

结果：

```text
259 passed, 3 skipped, 1 warning in 20.90s
TOTAL  938 statements, 27 missed, branch coverage 96.6%
```

3 个跳过项均位于 `tests/test_real_llm.py`，需要 `OPENAI_API_KEY` 和真实 LLM 服务，不属于本次离线测试失败。

### 3.2 其他检查

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| Ruff | 通过 | 使用 CI 中的 `E,F,W,B,UP` 规则集 |
| Python 编译检查 | 通过 | `flowagent/`、`examples/`、`scripts/` 均可编译 |
| 依赖一致性 | 通过 | 隔离环境中 `pip check` 未发现冲突 |
| 示例配置严格校验 | 通过 | 5 个 YAML 均通过 `check --strict` |
| CLI 离线执行 | 通过 | `customer_service.yaml` 在 Mock 模式下完成 |
| FastAPI 基础接口 | 通过 | `/docs` 和 `/openapi.json` 返回 200 |
| wheel/sdist 构建 | 通过但有发布问题 | 构建产物版本及内容与项目当前能力不一致 |

### 3.3 环境说明

仓库内现有 `.venv` 指向已不存在的 `D:\Program_files\Python\python.exe`，无法直接使用。本次测试在工作区临时隔离环境中完成，结束后已清理该环境。

首次 pytest 执行还受到受限系统临时目录影响。将 `--basetemp` 指向工作区并按 CI 设置 UTF-8 后，环境类错误全部消失，最终测试全绿。

## 4. 已确认问题

### 4.1 P1：否定式人工回复会被判为批准

**置信度：10/10**

涉及位置：

- [`configs/approval.yaml`](../configs/approval.yaml) 第 37 行；
- [`configs/refund_approval.yaml`](../configs/refund_approval.yaml) 第 48 行；
- [`flowagent/router.py`](../flowagent/router.py) 第 44 至 47 行。

审批配置使用子串判断：

```yaml
- when: {op: contains, value: 批准}
  goto: approve
```

路由实现等价于：

```python
return value in field
```

因此，“不批准”“拒绝批准”“暂不批准”都包含“批准”，会进入批准分支。

运行探针结果：

```text
STATUS finished APPROVED_BRANCH True DECISION_MATCH True
```

即系统保存的人工决定确实是“不批准”，最终却执行了批准分支。如果该流程连接真实退款或发布工具，会造成错误授权。

**建议：**

- 审批值使用 `eq` 精确比较；
- API 和 CLI 对 `options` 做强校验，而不是只提示；
- 未知值、空值和否定表达默认拒绝；
- 增加“不批准”“不拒绝”“批准后撤回”等反例测试。

### 4.2 P1：LLM 结构化输出缺少字段时可能默认通过

**置信度：10/10**

涉及位置：

- [`flowagent/parsers.py`](../flowagent/parsers.py) 第 25 至 39 行；
- [`flowagent/router.py`](../flowagent/router.py) 第 10 至 15 行、第 69 至 79 行；
- [`configs/customer_service.yaml`](../configs/customer_service.yaml) 第 99 至 117 行；
- [`configs/report_writer.yaml`](../configs/report_writer.yaml) 第 51 至 66 行。

JSON 解析器只验证文本能否解析为 JSON，不验证字段、类型和取值范围。路由字段不存在时，`get_path` 返回 `None`；比较逻辑随后退化为字符串比较，规则可能不命中并进入默认通过分支。

运行探针使用缺少 `score` 的 LLM 输出：

```json
{"reason": "missing score"}
```

结果：

```text
MISSING_SCORE_RESULT approved PARSED {'reason': 'missing score'}
```

**建议：**

- 为 JSON 节点增加 Pydantic 模型或 JSON Schema；
- 对 `score` 校验必填、数值类型和范围；
- 质检、风控和审批路由采用失败关闭策略，即解析异常或字段缺失时进入人工复核或拒绝分支；
- 不依赖提示词保证 LLM 输出格式。

### 4.3 P1：timeout 不会真正限制执行时间

**置信度：10/10**

涉及位置：[`flowagent/nodes.py`](../flowagent/nodes.py) 第 55 至 61 行。

当前实现：

```python
with ThreadPoolExecutor(max_workers=1) as pool:
    future = pool.submit(func, *args)
    return future.result(timeout=timeout)
```

`future.result` 会按时抛出 `TimeoutError`，但退出 `ThreadPoolExecutor` 上下文时默认执行 `shutdown(wait=True)`，因此仍会等待工具线程结束。

运行探针配置工具执行 0.6 秒、超时 0.05 秒：

```text
TIMEOUT_ELAPSED 0.613 ERROR TimeoutError
```

对有副作用的工具影响更大：线程超时后仍可能完成扣款、发送通知或写入外部系统，随后重试可能重复执行相同操作。

**建议：**

- 明确区分“停止等待”和“停止任务”；
- 优先使用支持取消的异步客户端和协作式超时；
- 对不可取消任务考虑独立进程；
- 对有副作用的工具要求幂等键，默认禁止自动重试非幂等操作；
- 超时测试必须断言实际耗时和副作用次数。

### 4.4 P1：子工作流中的人工中断被静默吞掉

**置信度：10/10**

涉及位置：[`flowagent/nodes.py`](../flowagent/nodes.py) 第 133 至 139 行。

父流程调用子引擎后只读取：

```python
sub_vars = dict(sub_final.get("vars") or {})
```

子流程返回的 `__interrupt__` 没有传播到父流程。实测将 `approval.yaml` 作为 workflow 节点执行时：

```text
PARENT_PENDING None
PARENT_SNAPSHOT_NEXT ()
```

父流程没有等待人工输入，检查点反而显示流程已经结束，输出中只保留了人工节点之前产生的部分变量。

**建议：**

- 使用 LangGraph 原生子图组合能力，让 interrupt 和 checkpoint namespace 正确传播；
- 或显式识别子图 `__interrupt__`，把中断状态、子线程标识和恢复上下文提升到父图；
- 增加“父图 → 子图 → human → resume → 父图继续”的端到端测试。

### 4.5 P1：on_error.goto 未按配置执行

**置信度：10/10**

涉及位置：

- [`flowagent/config.py`](../flowagent/config.py) 第 67 至 73 行；
- [`flowagent/nodes.py`](../flowagent/nodes.py) 第 199 至 217 行；
- [`flowagent/engine.py`](../flowagent/engine.py) 第 100 至 110 行。

配置模型声明：

```yaml
on_error:
  goto: fallback
  set:
    result: degraded
```

但异常处理只写入 `set` 和 `error`，图仍按节点的正常 `next` 继续。`on_error.goto` 既没有被加入编译边，也没有被引用完整性校验。

运行探针结果：

```text
ON_ERROR_TRACE ['bad'] FALLBACK_VAR None
```

配置中的 fallback 没有执行。不存在的 `goto` 目标也能通过配置加载。

**建议：**

- 统一错误路由模型，只保留一个权威的错误目标字段；
- 编译时将该目标加入图边和引用检查；
- 避免要求用户同时在 `node.on_error.goto` 与 `next.on_error` 重复配置；
- 增加正常 `next` 与错误 `goto` 不同的回归测试。

### 4.6 P2：FastAPI 返回码与接口说明不一致

**置信度：10/10**

涉及位置：

- [`examples/api_server.py`](../examples/api_server.py) 第 71 至 82 行、第 107 至 121 行；
- [`tests/test_api_server.py`](../tests/test_api_server.py) 第 32 至 38 行。

接口说明声称等待人工时返回 HTTP 202，但实现只是把 `202` 写进 JSON：

```python
result["status_code"] = 202
return result
```

运行结果：

```text
API_HTTP_STATUS 200 BODY_STATUS_CODE 202 BODY_STATUS waiting_human
```

生成的 OpenAPI 也只声明 `200` 和 `422`。客户端若根据 HTTP 状态判断同步完成或异步等待，会得到错误信号。

**建议：** 使用 FastAPI `Response`/`JSONResponse` 设置真实状态码，并在 OpenAPI 中声明 200、202、400、404、409。

### 4.7 P2：配置校验允许必然在运行期失败的值

**置信度：10/10**

涉及位置：[`flowagent/config.py`](../flowagent/config.py) 第 36 至 64 行、第 128 至 142 行、第 153 至 193 行。

已确认两个例子：

```text
recursion_limit=0  -> LOAD accepted -> ValueError: recursion_limit must be at least 1
max_loops=0 且无退出目标 -> LOAD accepted -> KeyError: None
```

这与 `check --strict`“配置合法即可运行”的用户预期不符。

**建议：**

- `recursion_limit >= 1`；
- `max_loops >= 1`；
- 配置循环护栏时必须存在 `loop_exhaust` 或 `default`；
- 校验 `on_error.goto`、路由正则和空的并行目标列表。

### 4.8 P2：版本与发布产物不一致

**置信度：10/10**

涉及位置：

- [`pyproject.toml`](../pyproject.toml) 第 8 至 22 行；
- [`flowagent/__init__.py`](../flowagent/__init__.py) 第 9 行；
- [`examples/api_server.py`](../examples/api_server.py) 第 52 至 56 行；
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) 第 21 至 32 行。

当前存在三个版本号：

| 位置 | 版本 |
| --- | --- |
| Python 包元数据 | `0.1.0` |
| `flowagent.__version__` | `0.8.0` |
| FastAPI OpenAPI | `0.3.0` |

实际构建产物为：

```text
flowagent-0.1.0-py3-none-any.whl
flowagent-0.1.0.tar.gz
```

同时还存在以下发布缺口：

- 没有 `[project.scripts]`，安装后不会生成 `flowagent` 命令；
- sdist 未包含 README 所引用的 `configs/`、`docs/` 和 `examples/`；
- CI 只运行源码测试，没有构建并安装 wheel 后再测试；
- `langgraph-checkpoint-sqlite` 出现在 `requirements.txt`，但未出现在 pyproject 的基础或可选依赖中。

**建议：** 建立单一版本源，并让 CI 执行 build、wheel 安装、版本一致性、入口命令和资产存在性测试。

### 4.9 P2：工具白名单采用默认放行

**置信度：8/10；风险取决于 YAML 是否可能来自不可信来源。**

涉及位置：[`flowagent/tools.py`](../flowagent/tools.py) 第 10 至 22 行、第 66 至 79 行。

未设置 `FLOWAGENT_TOOL_MODULES` 时，任何可导入的 Python callable 都可以被配置加载。只读探针确认：

```text
load_tool("shutil:rmtree") -> shutil.rmtree
```

当前 FastAPI 示例不接收用户上传的 YAML，因此没有确认直接的远程利用路径。但只要上层系统允许用户、LLM 或外部配置源控制 YAML，就可能变成任意代码调用或文件删除能力。

**建议：**

- 服务模式默认拒绝外部工具，只允许显式注册；
- 将“可信配置”和“不可信输入”的边界写入安全文档；
- 不要把 LLM 生成的模块名或函数名直接传给 `load_tool`；
- 为生产环境提供强制白名单开关，而不是依赖运维自觉设置环境变量。

### 4.10 P3：文档和跨平台命令存在漂移

**置信度：10/10**

- [`docs/API.md`](API.md) 第 17 行声称可从顶层导入 `TemplateError`，但 [`flowagent/__init__.py`](../flowagent/__init__.py) 没有导出它；
- [`README.md`](../README.md) 同时使用 Windows `.venv/Scripts`、Bash `export` 和反斜杠续行；
- README 中的单引号 JSON 命令在当前 Windows PowerShell 下无法直接运行，出现 `Expecting property name enclosed in double quotes`；
- API 文档使用 `flowagent ...` 命令，但包没有注册 console script。

## 5. 为什么高覆盖率没有发现这些问题

当前测试更擅长验证“代码路径是否执行”，对业务语义和时间语义的验证不足：

- 超时测试只断言抛出 `TimeoutError`，没有断言调用是否在 timeout 附近返回；
- API 测试注释写 202，却明确断言真实 HTTP 状态为 200；
- 子图测试覆盖模板和 LLM 节点，但没有覆盖子图内部的 human 节点；
- 人工审批测试只覆盖“批准”和“拒绝”，没有覆盖“不批准”等否定表达；
- LLM JSON 测试覆盖合法 JSON 解析，没有覆盖字段缺失、类型错误、越界值和额外字段；
- `on_error` 测试通常让 `next.on_error`、`default` 与 `node.on_error.goto` 指向同一节点，因此无法发现 `goto` 未被使用。

建议在保留覆盖率指标的同时，引入按风险分类的行为测试：审批、支付、权限、LLM 控制流、超时、副作用和恢复流程都应包含失败关闭及反例用例。

## 6. 建议修复顺序

### 第一阶段：阻止错误放行

1. 审批值改为精确枚举并强制校验 options；
2. 为 LLM JSON 输出增加 schema；
3. 关键路由改为失败关闭；
4. 补充否定输入和畸形 LLM 输出测试。

### 第二阶段：修复可靠性语义

1. 重新设计 timeout 和非幂等重试；
2. 修复 `on_error.goto` 的图编译与校验；
3. 支持子图 interrupt/resume 的完整传播；
4. 增加耗时、副作用次数和跨层恢复测试。

### 第三阶段：修复接口与发布流程

1. 返回真实 HTTP 202，并完善 OpenAPI；
2. 统一版本来源；
3. 注册 CLI 入口；
4. 补全 sdist/wheel 资产与依赖声明；
5. 在 CI 中测试构建后的安装包。

### 第四阶段：生产安全加固

1. 工具注册默认拒绝；
2. 为 API 增加认证、租户隔离和 thread_id 访问控制；
3. 明确可信 YAML 边界；
4. 执行真实 LLM、Docker 和并发部署测试。

## 7. 当前优势

- 模块边界清楚，配置、执行、路由、节点和解析器职责容易理解；
- 大部分配置错误能够在加载阶段给出中文错误信息；
- 测试数量充足，且覆盖 Windows/Linux、Python 3.10/3.12 的 CI 设计；
- Mock LLM 让主要示例能够离线复现；
- HITL、检查点、异步与流式能力都有独立测试；
- Ruff 和覆盖率已进入 CI，基础工程质量较好。

## 8. 本次未覆盖范围

以下项目未在本次审查中执行：

- 真实 OpenAI 兼容 LLM 调用；
- Docker 镜像完整构建与容器运行；
- Python 3.10 本地运行；
- 浏览器端视觉、响应式和可访问性测试；
- Git 分支差异、提交历史和回归范围审查，因为当前目录没有可用 Git 元数据。

这些限制不影响本报告中已通过运行探针确认的问题，但可能仍有未发现的部署或供应商兼容性问题。

## 9. 验收标准建议

修复完成后，至少应满足：

- “不批准”及所有未知审批值永远不会进入批准分支；
- 缺少或非法 `score` 的 LLM 输出进入失败或人工复核分支；
- 0.05 秒 timeout 的测试在合理误差内返回，且工具副作用最多执行一次；
- 子图 human 节点可以暂停父流程，并使用同一父 thread_id 完成恢复；
- `on_error.goto` 是唯一、有效且经过引用校验的错误路由来源；
- 等待人工时实际 HTTP 状态为 202，OpenAPI 同步声明；
- wheel 元数据、运行时版本、CHANGELOG 和 OpenAPI 版本一致；
- 从构建后的 wheel 安装后，CLI、示例资产和声明的可选能力可正常使用。

