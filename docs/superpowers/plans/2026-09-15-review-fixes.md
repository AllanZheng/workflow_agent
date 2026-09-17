# flowagent 评审修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `docs/PROJECT_REVIEW_2026-09-15.md` 确认的全部 10 项问题（5×P1 + 4×P2 + 1×P3），补齐行为测试，版本升至 0.9.0。

**Architecture:** 保持现有模块边界（config/router/nodes/engine/tools/parsers 各司其职），在引擎层加"失败关闭"语义（options 强校验、output_schema、路由数值硬化、超时不重试、子图中断手动传播、on_error.goto 编译进图）。设计依据：`docs/superpowers/specs/2026-09-15-review-fixes-design.md`。

**Tech Stack:** Python 3.10+/3.12, LangGraph 1.2.x, Pydantic v2, FastAPI, pytest。

**注意：本目录不是 git 仓库**（评审 §8 已确认），所有"Commit"步骤省略；如需版本管理请先 `git init`（需用户确认，不要擅自执行）。

**测试命令约定**（所有任务统一）：
```bash
cd e:/Git/agents/workflow/workflow
PYTHONUTF8=1 .venv/Scripts/python -m pytest <目标> -v --basetemp=.pytest_tmp
```

---

### Task 0: 重建环境与基线验证

**Files:** 无代码改动（`.venv/` 重建）

- [ ] **Step 0.1: 删除损坏的 .venv 并重建**

当前 `.venv` 指向不存在的 `D:\Program_files\Python\python.exe`（评审 §3.3）。

```bash
cd e:/Git/agents/workflow/workflow
rm -rf .venv
python -m venv .venv
```

- [ ] **Step 0.2: 安装依赖（含 API 可选组与构建工具）**

```bash
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install fastapi uvicorn httpx build
.venv/Scripts/python -m pip install -e . --no-deps   # 注册包元数据（版本一致性测试需要）
```

- [ ] **Step 0.3: 基线全量测试**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest --basetemp=.pytest_tmp
```
Expected: `259 passed, 3 skipped`（与评审一致）。任何偏差先停下排查。

---

### Task 1: 4.1 审批误判（contains→eq + options 强校验）

**Files:**
- Modify: `flowagent/nodes.py:125-131`（human 分支）
- Modify: `configs/approval.yaml:37,39`、`configs/refund_approval.yaml:48`
- Modify: `flowagent/config.py:96`（options 注释）
- Test: `tests/test_human.py`（追加）

- [ ] **Step 1.1: 写失败测试（引擎层 options 强校验）**

追加到 `tests/test_human.py`：

```python
class TestHumanOptionEnforcement:
    """4.1：声明 options 即强校验——无效答复不进入业务分支，重新询问。"""

    OPT_SPEC = {
        "workflow": {"name": "opt", "start": "ask", "recursion_limit": 20},
        "nodes": {
            "ask": {
                "type": "human",
                "message": "请回复 批准 或 拒绝：{ticket}",
                "options": ["批准", "拒绝"],
                "output_key": "decision",
                "next": {
                    "field": "decision",
                    "rules": [{"when": {"op": "eq", "value": "批准"}, "goto": "END"}],
                    "default": "redo",
                },
            },
            "redo": {
                "type": "template",
                "template": "已驳回:{ticket}",
                "output_key": "msg",
                "next": "END",
            },
        },
    }

    def test_invalid_answer_reinterrupts_with_error_hint(self):
        eng = WorkflowEngine(load_spec(self.OPT_SPEC), MockLLM())
        eng.run({"ticket": "T-9"}, thread_id="ie1")
        # “不批准”不在 options 内：必须再次暂停并携带错误提示，而不是放行
        again = eng.resume("不批准", thread_id="ie1")
        payload = WorkflowEngine.waiting_on(again)
        assert payload is not None
        assert "不批准" in payload.get("error", "")
        assert "msg" not in again["vars"]

    def test_second_attempt_with_valid_answer_completes(self):
        eng = WorkflowEngine(load_spec(self.OPT_SPEC), MockLLM())
        eng.run({"ticket": "T-10"}, thread_id="ie2")
        eng.resume("不批准", thread_id="ie2")      # 无效 → 重新询问
        final = eng.resume("批准", thread_id="ie2")  # 有效 → 正常完成
        assert final["vars"]["decision"] == "批准"
        assert "msg" not in final["vars"]

    def test_no_options_keeps_free_text(self):
        spec = {
            "workflow": {"name": "free", "start": "ask", "recursion_limit": 20},
            "nodes": {
                "ask": {
                    "type": "human",
                    "message": "任意回复",
                    "output_key": "d",
                    "next": "END",
                }
            },
        }
        eng = WorkflowEngine(load_spec(spec), MockLLM())
        eng.run({}, thread_id="ft1")
        final = eng.resume("不批准，理由略", thread_id="ft1")
        assert final["vars"]["d"] == "不批准，理由略"
```

- [ ] **Step 1.2: 运行确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_human.py -k TestHumanOptionEnforcement -v --basetemp=.pytest_tmp
```
Expected: `test_invalid_answer_reinterrupts_with_error_hint` FAIL（无效答复被写入 vars 并路由）。

- [ ] **Step 1.3: 实现 human 分支强校验**

`flowagent/nodes.py` human 分支替换为：

```python
    if node.type == "human":
        question = render_template(node.message, ctx)
        payload = {"node": node_name, "question": question}
        if node.options:
            payload["options"] = list(node.options)
        # 声明了 options 即强校验（失败关闭）：无效答复绝不进入业务分支，
        # 而是携带错误提示重新 interrupt，让人工重新作答；
        # interrupt() 抛出的 GraphBubbleUp 由外层原样上抛（见 _run_with_reliability）
        while True:
            answer = interrupt(payload)
            if not node.options or answer in node.options:
                return {node.output_key or "human_decision": answer}
            payload = {
                **payload,
                "error": f"输入 {answer!r} 不在可选范围 {list(node.options)}，请重新答复",
            }
```

`flowagent/config.py:96` options 注释改为：

```python
    options: list[str] = []  # 可选答复：声明即强校验，无效答复会重新询问（失败关闭）
```

- [ ] **Step 1.4: 运行测试通过**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_human.py -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS。

- [ ] **Step 1.5: 写失败测试（配置层 eq 语义，approval.yaml 端到端）**

追加到 `tests/test_workflows.py`（该文件已有 approval 流测试，导入 `project_root` fixture 使用）：

```python
class TestApprovalNegativeSemantics:
    """4.1：否定式答复绝不进入批准分支（approval.yaml 使用 eq）。"""

    def test_bu_pizhun_routes_to_reject(self, project_root):
        from flowagent import WorkflowEngine, load_spec_file

        spec = load_spec_file(project_root / "configs" / "approval.yaml")
        eng = WorkflowEngine(spec, MockLLM())
        eng.run({"ticket": "大额退款", "amount": 9}, thread_id="neg1")
        final = eng.resume("不批准", thread_id="neg1")
        # 已驳回模板被渲染，批准模板不存在于输出
        assert "已驳回" in final["vars"]["final"]
        assert "已批准" not in final["vars"]["final"]
```

（若 `tests/test_workflows.py` 顶部缺 `MockLLM` 导入则补充 `from flowagent.llm import MockLLM`。）

- [ ] **Step 1.6: 运行确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_workflows.py -k TestApprovalNegativeSemantics -v --basetemp=.pytest_tmp
```
Expected: FAIL（contains 匹配“不批准”→ 已批准）。

- [ ] **Step 1.7: 修改两份配置 contains→eq**

`configs/approval.yaml` ask 节点路由：

```yaml
    next:
      field: decision
      rules:
        - when: {op: eq, value: 批准}
          goto: approve
        - when: {op: eq, value: 拒绝}
          goto: reject
      default: reject   # 未知值一律驳回（失败关闭）
```

`configs/refund_approval.yaml` approval 节点路由同改（`contains` → `eq`，default 保持 `rejected`）。

- [ ] **Step 1.8: 全量相关测试**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_human.py tests/test_human_chain.py tests/test_workflows.py tests/test_cli.py -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS（含 CLI 既有 resume 批准/拒绝用例）。

---

### Task 2: 4.2 LLM 输出信任边界（output_schema + 路由数值硬化）

**Files:**
- Modify: `flowagent/config.py`（新增 `OutputFieldSpec`、`NodeConfig.output_schema`、加载期校验）
- Modify: `flowagent/parsers.py`（新增 `validate_schema`）
- Modify: `flowagent/nodes.py` llm 分支（解析后校验）
- Modify: `flowagent/router.py:10-15,30-41`（数值比较硬化）
- Modify: `configs/customer_service.yaml`、`configs/refund_approval.yaml`、`configs/report_writer.yaml`（质检节点补 schema）
- Test: `tests/test_parsers.py`、`tests/test_router.py`、`tests/test_llm.py`（追加）

- [ ] **Step 2.1: 写失败测试（validate_schema）**

追加到 `tests/test_parsers.py`：

```python
class TestValidateSchema:
    """4.2：output_schema 校验——缺字段/类型错/越界/枚举外都抛 OutputParseError。"""

    def test_missing_required_field(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"reason": "x"}, {"score": {}})

    def test_optional_field_missing_ok(self):
        from flowagent.parsers import validate_schema
        validate_schema({"score": 8}, {"score": {}, "reason": {"optional": True}})

    def test_wrong_type(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"score": "8"}, {"score": {"type": "integer"}})

    def test_out_of_range(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"score": 11}, {"score": {"type": "integer", "max": 10}})
        with pytest.raises(OutputParseError):
            validate_schema({"score": 0}, {"score": {"type": "integer", "min": 1}})

    def test_enum(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"level": "中"}, {"level": {"enum": ["高", "低"]}})

    def test_top_level_not_object(self):
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema([1, 2], {"score": {}})

    def test_boolean_not_integer(self):
        # Python 的 bool 是 int 子类，必须显式排除
        from flowagent.parsers import validate_schema
        with pytest.raises(OutputParseError):
            validate_schema({"ok": True}, {"ok": {"type": "integer"}})
```

- [ ] **Step 2.2: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_parsers.py -k TestValidateSchema -v --basetemp=.pytest_tmp
```
Expected: FAIL（`validate_schema` 不存在，ImportError）。

- [ ] **Step 2.3: 实现 validate_schema**

追加到 `flowagent/parsers.py`：

```python
def validate_schema(value, schema: dict) -> None:
    """按 output_schema 校验已解析的 JSON 输出（失败关闭）。

    只支持轻量字段规范（type/min/max/enum/optional，见 config.OutputFieldSpec），
    不实现完整 JSON Schema——配置驱动的 YAML 里够用且可读。
    校验失败抛 OutputParseError：让节点走 retries/on_error/失败路径，
    绝不允许畸形输出静默落入“默认通过”分支。
    """
    if not isinstance(value, dict):
        raise OutputParseError(f"output_schema 要求顶层 JSON 对象，实际为 {type(value).__name__}")
    for key, rule in schema.items():
        item = value.get(key)
        if item is None:
            if getattr(rule, "optional", False):
                continue
            raise OutputParseError(f"输出缺少必填字段 {key!r}")
        rtype = getattr(rule, "type", "string")
        # 注意：bool 是 int 的子类，数值类型必须显式排除布尔
        if rtype == "string" and not isinstance(item, str):
            raise OutputParseError(f"字段 {key!r} 应为 string，实际为 {type(item).__name__}")
        if rtype == "boolean" and not isinstance(item, bool):
            raise OutputParseError(f"字段 {key!r} 应为 boolean，实际为 {type(item).__name__}")
        if rtype == "integer" and (isinstance(item, bool) or not isinstance(item, int)):
            raise OutputParseError(f"字段 {key!r} 应为 integer，实际为 {type(item).__name__}")
        if rtype == "number" and (isinstance(item, bool) or not isinstance(item, (int, float))):
            raise OutputParseError(f"字段 {key!r} 应为 number，实际为 {type(item).__name__}")
        rmin, rmax = getattr(rule, "min", None), getattr(rule, "max", None)
        if rmin is not None and isinstance(item, (int, float)) and not isinstance(item, bool) and item < rmin:
            raise OutputParseError(f"字段 {key!r}={item!r} 低于下限 {rmin}")
        if rmax is not None and isinstance(item, (int, float)) and not isinstance(item, bool) and item > rmax:
            raise OutputParseError(f"字段 {key!r}={item!r} 超过上限 {rmax}")
        renum = getattr(rule, "enum", None)
        if renum is not None and item not in renum:
            raise OutputParseError(f"字段 {key!r}={item!r} 不在枚举 {renum} 内")
```

- [ ] **Step 2.4: 写失败测试（NodeConfig.output_schema + 节点级校验 + 路由硬化）**

追加到 `tests/test_llm.py`：

```python
class TestOutputSchemaNode:
    """4.2：llm 节点声明 output_schema 后，畸形输出失败关闭。"""

    def _spec(self, mock: str):
        return load_spec({
            "workflow": {"name": "sc", "start": "qa"},
            "nodes": {
                "qa": {
                    "type": "llm",
                    "prompt": "打分",
                    "parser": "json",
                    "output_key": "qa",
                    "mock_response": mock,
                    "output_schema": {
                        "score": {"type": "integer", "min": 1, "max": 10},
                        "reason": {"type": "string"},
                    },
                    "next": "END",
                }
            },
        })

    def test_missing_score_fails_closed(self):
        eng = WorkflowEngine(self._spec('{"reason": "missing"}'), MockLLM())
        with pytest.raises(OutputParseError):
            eng.run({})

    def test_missing_score_degrades_via_on_error(self):
        spec = self._spec('{"reason": "missing"}')
        spec.nodes["qa"].on_error = None  # 由下方 dict 重建更直观
        spec_dict = {
            "workflow": {"name": "sc2", "start": "qa"},
            "nodes": {
                "qa": {
                    "type": "llm", "prompt": "打分", "parser": "json",
                    "output_key": "qa", "mock_response": '{"reason": "m"}',
                    "output_schema": {"score": {"type": "integer"}},
                    "on_error": {"goto": "manual", "set": {"qa": None}},
                    "next": "END",
                },
                "manual": {
                    "type": "template", "template": "转人工复核",
                    "output_key": "final", "next": "END",
                },
            },
        }
        final = WorkflowEngine(load_spec(spec_dict), MockLLM()).run({})
        assert final["vars"]["final"] == "转人工复核"

    def test_schema_with_non_json_parser_rejected_at_load(self):
        with pytest.raises(ConfigError):
            load_spec({
                "workflow": {"name": "bad", "start": "n"},
                "nodes": {
                    "n": {
                        "type": "llm", "prompt": "x", "parser": "text",
                        "output_key": "o", "output_schema": {"a": {}},
                        "next": "END",
                    }
                },
            })
```

追加到 `tests/test_router.py`：

```python
class TestRouterNumericHardening:
    """4.2：数值操作符遇 None/不可数值化字段时规则不命中（禁止退化字符串比较）。"""

    def test_lt_with_none_field_no_match(self):
        assert evaluate_op("lt", None, 7) is False
        assert evaluate_op("ge", None, 7) is False

    def test_lt_with_non_numeric_string_no_match(self):
        assert evaluate_op("lt", "abc", 7) is False

    def test_lt_numeric_still_works(self):
        assert evaluate_op("lt", 5, 7) is True
        assert evaluate_op("lt", "5", 7) is True  # 可数值化字符串保持宽容
```

- [ ] **Step 2.5: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_llm.py tests/test_router.py -k "TestOutputSchemaNode or TestRouterNumericHardening" -v --basetemp=.pytest_tmp
```
Expected: FAIL。

- [ ] **Step 2.6: 实现 config + nodes + router 改动**

`flowagent/config.py`：

```python
class OutputFieldSpec(BaseModel):
    """output_schema 中单个字段的校验规则（未标 optional 默认必填）。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "integer", "number", "boolean"] = "string"
    min: float | None = None
    max: float | None = None
    enum: list | None = None
    optional: bool = False
```

`NodeConfig` 增加字段（放在 `labels` 之后）：

```python
    # llm 节点输出校验：parser=json 时解析后按此校验字段（失败关闭）
    output_schema: dict[str, OutputFieldSpec] = {}
```

`_check_references` 内追加（放在 classifier 校验之后）：

```python
            if node.output_schema and node.parser != "json":
                raise ValueError(f"节点 {name} 的 output_schema 仅适用于 parser: json")
```

`flowagent/nodes.py` llm 分支最后一行改为：

```python
        parsed = parse_output(content, node.parser, node.labels)
        # 声明了 output_schema 的输出必须过字段校验：缺字段/类型错/越界
        # 都视为解析失败（失败关闭），交给 retries/on_error 或直接失败
        if node.output_schema:
            validate_schema(parsed, node.output_schema)
        return {node.output_key: parsed}
```

（文件顶部 `from .parsers import parse_output` 改为 `from .parsers import parse_output, validate_schema`。）

`flowagent/router.py`：

```python
def _coerce(a, b):
    """eq/ne 用宽容比较：先试数值，失败退化为字符串（保持既有语义）。"""
    try:
        return float(a), float(b)
    except (TypeError, ValueError):
        return str(a), str(b)


def _numeric(a, b):
    """gt/ge/lt/le 专用：任一侧无法数值化返回 None（规则不命中）。

    此前 None/非数值会退化成字符串比较（如 "None" < "7"），导致字段缺失时
    规则意外命中或漏掉后落入默认放行分支——数值语义必须失败关闭。
    """
    try:
        return float(a), float(b)
    except (TypeError, ValueError):
        return None
```

`evaluate_op` 的四个数值分支改为：

```python
    if op in ("gt", "ge", "lt", "le"):
        pair = _numeric(field, value)
        if pair is None:
            return False  # 字段缺失/不可数值化：规则不命中
        a, b = pair
        return {"gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b}[op]
```

- [ ] **Step 2.7: 运行测试通过**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_parsers.py tests/test_llm.py tests/test_router.py -v --basetemp=.pytest_tmp
```

- [ ] **Step 2.8: 三份配置补 output_schema**

`configs/customer_service.yaml` review 节点、`configs/refund_approval.yaml` qa 节点、`configs/report_writer.yaml` review 节点，在 `parser: json` 之后加：

```yaml
    output_schema:
      score: {type: integer, min: 1, max: 10}
      reason: {type: string}
```

- [ ] **Step 2.9: 回归**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/ -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS（含既有质检循环用例，mock 输出本就合法）。

---

### Task 3: 4.5 on_error.goto 生效

**Files:**
- Modify: `flowagent/config.py:109-125`（next_targets 纳入 on_error.goto）
- Modify: `flowagent/router.py:61-81`（make_router 感知 node.on_error.goto）
- Modify: `flowagent/engine.py:100-110`（固定/并行 next 也编译条件边）
- Test: `tests/test_reliability.py`（追加）

- [ ] **Step 3.1: 写失败测试**

追加到 `tests/test_reliability.py`：

```python
class TestOnErrorGoto:
    """4.5：node.on_error.goto 是权威错误路由，三种 next 形态都要生效。"""

    FLAKY_GOTO = {
        "type": "tool",
        "tool": "tests.flaky:flaky_tool",
        "args": {"key": "og", "fail_times": 99},
        "output_key": "r",
        "retry_delay": 0,
        "on_error": {"goto": "fallback", "set": {"r": "降级"}},
    }

    def test_str_next_error_goes_to_goto(self):
        spec = {
            "workflow": {"name": "og1", "start": "n"},
            "nodes": {
                "n": {**self.FLAKY_GOTO, "next": "normal"},
                "normal": {"type": "template", "template": "正常", "output_key": "x", "next": "END"},
                "fallback": {"type": "template", "template": "兜底", "output_key": "y", "next": "END"},
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["r"] == "降级"
        assert final["vars"]["y"] == "兜底"   # 走了 goto 分支
        assert "x" not in final["vars"]       # 正常分支未执行

    def test_list_next_error_goes_to_goto(self):
        spec = {
            "workflow": {"name": "og2", "start": "n"},
            "nodes": {
                "n": {**self.FLAKY_GOTO, "next": ["a1", "a2"]},
                "a1": {"type": "template", "template": "A", "output_key": "x", "next": "END"},
                "a2": {"type": "template", "template": "B", "output_key": "w", "next": "END"},
                "fallback": {"type": "template", "template": "兜底", "output_key": "y", "next": "END"},
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["y"] == "兜底"
        assert "x" not in final["vars"] and "w" not in final["vars"]

    def test_route_next_goto_takes_precedence_over_route_on_error(self):
        node = {
            **self.FLAKY_GOTO,
            "next": {
                "field": "r",
                "rules": [{"when": {"op": "exists"}, "goto": "normal"}],
                "default": "normal",
                "on_error": "route_fallback",
            },
        }
        spec = {
            "workflow": {"name": "og3", "start": "n"},
            "nodes": {
                "n": node,
                "normal": {"type": "template", "template": "正常", "output_key": "x", "next": "END"},
                "fallback": {"type": "template", "template": "节点级", "output_key": "y", "next": "END"},
                "route_fallback": {"type": "template", "template": "路由级", "output_key": "z", "next": "END"},
            },
        }
        final = _engine(spec).run({})
        assert final["vars"]["y"] == "节点级"  # node.on_error.goto 优先
        assert "z" not in final["vars"]

    def test_invalid_goto_target_rejected_at_load(self):
        from flowagent.errors import ConfigError
        with pytest.raises(ConfigError):
            load_spec({
                "workflow": {"name": "og4", "start": "n"},
                "nodes": {
                    "n": {**self.FLAKY_GOTO, "next": "END"},  # goto: fallback 不存在
                },
            })
```

- [ ] **Step 3.2: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_reliability.py -k TestOnErrorGoto -v --basetemp=.pytest_tmp
```
Expected: 全部 FAIL（goto 未编译进图，前两个走了 normal；第四个 ConfigError 未抛）。

- [ ] **Step 3.3: 实现**

`flowagent/config.py` `next_targets` 末尾追加（return 之前）：

```python
        if self.on_error and self.on_error.goto:
            # on_error.goto 参与引用校验：非法目标在加载期报错（4.5）
            targets.append(self.on_error.goto)
        return targets
```

（把函数体改为先收集 `targets` 局部变量再统一 return，三种分支逻辑不变。）

`flowagent/router.py` `make_router` 增加参数：

```python
def make_router(node_name: str, route: RouteConfig, error_goto: str | None = None):
    """error_goto 为 node.on_error.goto：错误路由的唯一权威优先级。"""

    def router(state: dict) -> str:
        err = state.get("error")
        if err and err.get("node") == node_name:
            # 统一错误路由模型：node.on_error.goto > route.on_error（只需配一处）
            if error_goto:
                return error_goto
            if route.on_error:
                return route.on_error
        ...  # 其余逻辑不变
```

`flowagent/engine.py` build_graph 的边编译改为：

```python
def _fixed_or_error(node_name: str, normal_targets, error_goto: str):
    """固定 next（字符串/并行列表）的错误改道路由：正常走原目标，错误走 goto。"""
    def router(state: dict):
        err = state.get("error")
        if err and err.get("node") == node_name:
            return error_goto
        return normal_targets
    return router


# build_graph 内：
    for name, node in spec.nodes.items():
        nxt = node.next
        error_goto = node.on_error.goto if node.on_error else None
        if isinstance(nxt, str):
            if error_goto:
                # 配置了 on_error.goto：固定边升级为条件边（错误时改道）
                graph.add_conditional_edges(
                    name,
                    _fixed_or_error(name, nxt, error_goto),
                    path_map={nxt: _resolve(nxt), error_goto: _resolve(error_goto)},
                )
            else:
                graph.add_edge(name, _resolve(nxt))
        elif isinstance(nxt, list):
            if error_goto:
                graph.add_conditional_edges(
                    name,
                    _fixed_or_error(name, list(nxt), error_goto),
                    path_map={**{t: _resolve(t) for t in nxt},
                              error_goto: _resolve(error_goto)},
                )
            else:
                for target in nxt:  # 多条边 = 并行扇出，汇合点自动等待全部分支
                    graph.add_edge(name, _resolve(target))
        else:
            graph.add_conditional_edges(
                name,
                make_router(name, nxt, error_goto),
                path_map=_route_path_map(nxt, error_goto),
            )
```

`_route_path_map` 增加参数：

```python
def _route_path_map(route: RouteConfig, error_goto: str | None = None) -> dict:
    ...
    add(route.on_error)
    add(error_goto)  # node.on_error.goto 也可能成为条件边返回值
    return mapping
```

- [ ] **Step 3.4: 运行测试通过 + 回归**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_reliability.py tests/test_engine.py tests/test_router.py -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS。

---

### Task 4: 4.7 配置校验补强

**Files:**
- Modify: `flowagent/config.py`（WorkflowMeta 校验、RouteRule/RouteConfig 校验、next 非空、regex 编译）
- Test: `tests/test_config.py`（追加）

- [ ] **Step 4.1: 写失败测试**

追加到 `tests/test_config.py`：

```python
class TestStrictValidation:
    """4.7：加载期拒绝必然在运行期失败的值。"""

    def _load(self, spec):
        from flowagent import load_spec
        from flowagent.errors import ConfigError
        with pytest.raises(ConfigError):
            load_spec(spec)

    def test_recursion_limit_zero(self):
        self._load({"workflow": {"name": "v", "start": "n", "recursion_limit": 0},
                    "nodes": {"n": {"type": "template", "template": "x",
                                    "output_key": "o", "next": "END"}}})

    def test_max_loops_zero(self):
        self._load({"workflow": {"name": "v", "start": "n"},
                    "nodes": {"n": {"type": "template", "template": "x", "output_key": "o",
                                    "next": {"field": "o",
                                             "rules": [{"when": {"op": "exists"}, "goto": "n",
                                                        "max_loops": 0}],
                                             "default": "END"}}}})

    def test_max_loops_without_exhaust_or_default(self):
        self._load({"workflow": {"name": "v", "start": "n"},
                    "nodes": {"n": {"type": "template", "template": "x", "output_key": "o",
                                    "next": {"field": "o",
                                             "rules": [{"when": {"op": "exists"}, "goto": "n",
                                                        "max_loops": 2}]}}}})

    def test_bad_regex(self):
        self._load({"workflow": {"name": "v", "start": "n"},
                    "nodes": {"n": {"type": "template", "template": "x", "output_key": "o",
                                    "next": {"field": "o",
                                             "rules": [{"when": {"op": "regex", "value": "("},
                                                        "goto": "END"}],
                                             "default": "END"}}}})

    def test_empty_parallel_list(self):
        self._load({"workflow": {"name": "v", "start": "n"},
                    "nodes": {"n": {"type": "template", "template": "x",
                                    "output_key": "o", "next": []}}})
```

- [ ] **Step 4.2: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_config.py -k TestStrictValidation -v --basetemp=.pytest_tmp
```
Expected: 全部 FAIL（当前都接受）。

- [ ] **Step 4.3: 实现**

`WorkflowMeta` 增加校验（`_check_references` 里，start 校验之后）：

```python
        if self.workflow.recursion_limit < 1:
            raise ValueError("recursion_limit 至少为 1")
```

`NodeConfig` 增加 pydantic 字段校验（`from pydantic import ... field_validator` 补充导入）：

```python
    @field_validator("next")
    @classmethod
    def _next_not_empty(cls, v):
        """并行扇出列表与固定目标不允许为空（空列表编译期就会失败）。"""
        if isinstance(v, str) and not v:
            raise ValueError("next 不能为空字符串")
        if isinstance(v, list):
            if not v:
                raise ValueError("next 并行列表不能为空")
            if any(not isinstance(t, str) or not t for t in v):
                raise ValueError("next 并行列表的每个目标必须是非空字符串")
        return v
```

`_check_references` 的路由检查块（`isinstance(node.next, RouteConfig)` 分支）扩展为：

```python
            if isinstance(node.next, RouteConfig):
                route = node.next
                if not route.rules and not route.default:
                    raise ValueError(f"节点 {name} 的路由既无 rules 也无 default")
                if route.max_loops is not None and route.max_loops < 1:
                    raise ValueError(f"节点 {name} 的路由级 max_loops 至少为 1")
                for rule in route.rules:
                    if rule.max_loops is not None:
                        if rule.max_loops < 1:
                            raise ValueError(f"节点 {name} 的 max_loops 至少为 1")
                        if not (rule.loop_exhaust or route.default):
                            raise ValueError(
                                f"节点 {name} 配置 max_loops 的规则必须提供 "
                                f"loop_exhaust 或路由 default，否则护栏耗尽时无路可走"
                            )
                    if rule.when.op == "regex":
                        try:
                            re.compile(str(rule.when.value))
                        except re.error as e:
                            raise ValueError(f"节点 {name} 的路由 regex 无效: {e}") from e
```

- [ ] **Step 4.4: 运行测试通过 + 回归**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_config.py tests/test_subgraph_check.py -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS。

---

### Task 5: 4.3 timeout 修复（按时返回 + 超时不重试）

**Files:**
- Modify: `tests/flaky.py`（增加计数慢工具）
- Modify: `flowagent/nodes.py`（`_with_timeout`、`_run_with_reliability`）
- Test: `tests/test_reliability.py`（追加）

- [ ] **Step 5.1: 增加测试辅助工具**

`tests/flaky.py` 追加（沿用其既有计数字典约定；实现时先读该文件对齐风格）：

```python
def slow_counting_tool(key: str, seconds: float = 0.6) -> str:
    """慢工具（带副作用计数）：验证超时按时返回且副作用不重复。"""
    _calls[key] = _calls.get(key, 0) + 1
    time.sleep(float(seconds))
    return "done"
```

- [ ] **Step 5.2: 写失败测试**

追加到 `tests/test_reliability.py`：

```python
class TestTimeoutSemantics:
    """4.3：超时按时返回；超时不参与重试（副作用至多一次）。"""

    SLOW_NODE = {
        "type": "tool",
        "tool": "tests.flaky:slow_counting_tool",
        "args": {"key": "slow", "seconds": 0.6},
        "output_key": "r",
        "timeout": 0.05,
        "next": "END",
    }

    def test_timeout_returns_promptly(self):
        import time as _t
        spec = {"workflow": {"name": "t1", "start": "n"}, "nodes": {"n": dict(self.SLOW_NODE)}}
        start = _t.monotonic()
        with pytest.raises(TimeoutError):
            _engine(spec).run({})
        elapsed = _t.monotonic() - start
        # 此前 shutdown(wait=True) 会让耗时 ≈ 0.6s；现在必须在 timeout 附近返回
        assert elapsed < 0.3, f"超时未按时返回，耗时 {elapsed:.3f}s"

    def test_timeout_does_not_retry_side_effect(self):
        spec = {"workflow": {"name": "t2", "start": "n"},
                "nodes": {"n": {**self.SLOW_NODE, "retries": 3, "retry_delay": 0}}}
        with pytest.raises(TimeoutError):
            _engine(spec).run({})
        # 超时后不重试：副作用工具只执行一次
        assert flaky.calls("slow") == 1

    def test_timeout_degrades_via_on_error(self):
        spec = {"workflow": {"name": "t3", "start": "n"},
                "nodes": {"n": {**self.SLOW_NODE,
                                "on_error": {"goto": "END", "set": {"r": "超时降级"}}}}}
        final = _engine(spec).run({})
        assert final["vars"]["r"] == "超时降级"
```

- [ ] **Step 5.3: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_reliability.py -k TestTimeoutSemantics -v --basetemp=.pytest_tmp
```
Expected: 前两个 FAIL（耗时 0.6s+ / 重试导致多次副作用）。

- [ ] **Step 5.4: 实现**

`flowagent/nodes.py` 顶部导入增加 `from concurrent.futures import ThreadPoolExecutor`（已有）并补充 `import concurrent.futures`。`_with_timeout` 替换为：

```python
def _with_timeout(func, args: tuple, timeout: float | None):
    """带超时的函数执行；timeout 为 None 时直接调用。

    超时语义 = 停止等待：到点即 shutdown(wait=False) 返回 TimeoutError，
    不等待工具线程结束。Python 线程无法强制终止，工具线程可能仍在后台
    完成并产生副作用——因此上层约定超时不自动重试（见 _run_with_reliability）。
    """
    if timeout is None:
        return func(*args)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowagent-node")
    future = pool.submit(func, *args)
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as e:
        # 3.10 上 concurrent.futures.TimeoutError 与内建 TimeoutError 不同类，
        # 统一规范化为内建 TimeoutError，保证上层 except 语义跨版本一致
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"执行超过 {timeout}s，已停止等待（工具线程可能仍在后台运行）") from e
    pool.shutdown(wait=True)
    return result
```

`_run_with_reliability` 的 except 链改为：

```python
        except TimeoutError:
            # 超时不是瞬态故障：工具可能仍在后台执行并产生副作用，
            # 自动重试会造成副作用重复，一律立即上抛（4.3）
            raise
        except Exception:
            if attempt >= attempts:
                raise
            if node.retry_delay > 0:
                time.sleep(node.retry_delay * (node.backoff ** (attempt - 1)))
```

- [ ] **Step 5.5: 运行测试通过 + 回归**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_reliability.py tests/test_nodes.py -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS。注意既有超时测试若断言 `concurrent.futures.TimeoutError` 需同步改为内建 `TimeoutError`。

---

### Task 6: 4.4 子图 interrupt 手动传播

**Files:**
- Modify: `flowagent/nodes.py:133-139`（workflow 分支）+ 新增 `_current_thread_id` 辅助
- Test: `tests/test_subgraph.py`（追加）

- [ ] **Step 6.1: 确认 langgraph.config.get_config 可用**

```bash
.venv/Scripts/python -c "from langgraph.config import get_config; print('ok')"
```
Expected: `ok`。若导入失败，改用 `from langgraph.run_manager import ...` 按装着版本调整（实现时确认）。

- [ ] **Step 6.2: 写失败测试**

追加到 `tests/test_subgraph.py`：

```python
CHILD_WITH_HUMAN = """\
workflow: {name: child_hitl, start: ask}
nodes:
  ask:
    type: human
    message: '子图人工确认：{x}'
    options: [批准, 拒绝]
    output_key: decision
    next:
      field: decision
      rules:
        - when: {op: eq, value: 批准}
          goto: done
      default: denied
  done:
    type: template
    template: '子图批准:{x}'
    output_key: child_out
    next: END
  denied:
    type: template
    template: '子图拒绝:{x}'
    output_key: child_out
    next: END
"""


class TestSubgraphHumanPropagation:
    """4.4：子图 human 中断提升到父图，同一父 thread_id 完成恢复。"""

    def _parent_spec(self, child_path):
        return load_spec({
            "workflow": {"name": "parent_hitl", "start": "sub"},
            "nodes": {
                "sub": {
                    "type": "workflow",
                    "workflow_file": str(child_path),
                    "output_key": "sub",
                    "next": "END",
                }
            },
        })

    def test_subgraph_human_pauses_parent(self, tmp_path):
        child = _write(tmp_path / "child_hitl.yaml", CHILD_WITH_HUMAN)
        eng = WorkflowEngine(self._parent_spec(child), MockLLM())
        final = eng.run({"x": "任务B"}, thread_id="ph1")
        payload = WorkflowEngine.waiting_on(final)
        # 父图必须暂停（而非静默完成），且能看到子图的问题与上下文
        assert payload is not None
        assert "子图人工确认" in payload["question"]
        assert payload.get("workflow_node") == "sub"

    def test_resume_with_same_parent_thread_completes(self, tmp_path):
        child = _write(tmp_path / "child_hitl.yaml", CHILD_WITH_HUMAN)
        eng = WorkflowEngine(self._parent_spec(child), MockLLM())
        eng.run({"x": "任务C"}, thread_id="ph2")
        final = eng.resume("批准", thread_id="ph2")
        # 父图完成，子图输出按 output_key 包装
        assert final["vars"]["sub"]["child_out"] == "子图批准:任务C"

    def test_resume_reject_routes_in_subgraph(self, tmp_path):
        child = _write(tmp_path / "child_hitl.yaml", CHILD_WITH_HUMAN)
        eng = WorkflowEngine(self._parent_spec(child), MockLLM())
        eng.run({"x": "任务D"}, thread_id="ph3")
        final = eng.resume("拒绝", thread_id="ph3")
        assert final["vars"]["sub"]["child_out"] == "子图拒绝:任务D"
```

- [ ] **Step 6.3: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_subgraph.py -k TestSubgraphHumanPropagation -v --basetemp=.pytest_tmp
```
Expected: 全部 FAIL（父图不暂停、直接部分完成）。

- [ ] **Step 6.4: 实现**

`flowagent/nodes.py` 新增辅助函数（放在 `_sub_workflow_engine` 之后）：

```python
def _current_thread_id() -> str | None:
    """读取当前 LangGraph 执行上下文的 thread_id（用于派生稳定的子图线程标识）。

    在图内节点执行时可用；单元测试直接调用 execute_node 等场景下返回 None。
    """
    try:
        from langgraph.config import get_config
        cfg = get_config() or {}
        return (cfg.get("configurable") or {}).get("thread_id")
    except Exception:  # noqa: BLE001 - 无执行上下文时退回旧行为（随机 id）
        return None
```

workflow 分支替换为：

```python
    if node.type == "workflow":
        from .engine import WorkflowEngine  # 延迟导入避免环

        engine = _sub_workflow_engine(node.workflow_file, llm=llm, checkpointer=checkpointer)
        # 稳定子线程 ID：父 thread_id + 节点名派生。此前每次 run 随机生成，
        # 子图中断后检查点根本无法再次定位（4.4 的隐性根因之一）
        parent_tid = _current_thread_id()
        sub_tid = f"{parent_tid}::{node_name}" if parent_tid else uuid.uuid4().hex[:8]
        snapshot = engine.get_state(sub_tid)
        pending = WorkflowEngine.pending_payload(snapshot)
        if pending is not None:
            # 父图恢复后本节点重放：子图仍挂起中断。
            # 先 interrupt() 取回父级 resume 值，再转发给子图
            value = interrupt(_enrich_payload(pending, node_name, sub_tid))
            sub_final = engine.resume(value, thread_id=sub_tid)
        else:
            sub_final = engine.run(dict(ctx), thread_id=sub_tid)
        # 子图可能再次中断（多人工节点）：持续把中断提升到父图，直到子图结束
        while WorkflowEngine.waiting_on(sub_final) is not None:
            payload = WorkflowEngine.waiting_on(sub_final)
            value = interrupt(_enrich_payload(payload, node_name, sub_tid))
            sub_final = engine.resume(value, thread_id=sub_tid)
        sub_vars = dict(sub_final.get("vars") or {})
        if node.output_key:
            return {node.output_key: sub_vars}
        return sub_vars
```

（`uuid` 需在 nodes.py 顶部导入：`import uuid`。）

`_enrich_payload` 辅助函数：

```python
def _enrich_payload(payload, node_name: str, sub_tid: str):
    """把子图 interrupt 载荷提升到父图时补充定位上下文（不覆盖原字段）。"""
    if isinstance(payload, dict):
        enriched = dict(payload)
        enriched.setdefault("workflow_node", node_name)
        enriched.setdefault("sub_thread_id", sub_tid)
        return enriched
    return payload
```

- [ ] **Step 6.5: 运行测试通过 + 回归**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_subgraph.py tests/test_subgraph_check.py tests/test_workflows.py -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS（含 refund_approval 端到端：risk 子图无 human，行为不变）。

---

### Task 7: 4.6 API 真实 202 状态码

**Files:**
- Modify: `examples/api_server.py`
- Test: `tests/test_api_server.py`

- [ ] **Step 7.1: 写失败测试（先读现有 test_api_server.py 对齐风格，修改断言）**

现有测试断言 HTTP 200 + body.status_code==202（评审 §5 指出的"注释写 202 却断言 200"）。修改/追加：

```python
class TestHttpStatusCodes:
    """4.6：等待人工时真实 HTTP 202，OpenAPI 同步声明。"""

    def test_create_run_waiting_human_is_202(self, client):
        resp = client.post("/runs", json={"inputs": {"ticket": "大额退款", "amount": 1},
                                          "thread_id": "api202"})
        assert resp.status_code == 202
        assert resp.json()["status"] == "waiting_human"
        assert "status_code" not in resp.json()  # 伪字段已删除

    def test_resume_waiting_again_is_202(self, client):
        client.post("/runs", json={"inputs": {"ticket": "t", "amount": 1},
                                   "thread_id": "api203"})
        # 恢复无效值（options 强校验）→ 仍等待人工 → 202
        resp = client.post("/runs/api203/resume", json={"value": "不批准"})
        assert resp.status_code == 202

    def test_openapi_declares_202(self, client):
        spec = client.get("/openapi.json").json()
        assert "202" in spec["paths"]["/runs"]["post"]["responses"]

    def test_openapi_version_matches_package(self, client):
        import flowagent
        spec = client.get("/openapi.json").json()
        assert spec["info"]["version"] == flowagent.__version__
```

（`client` fixture 沿用现有文件里的 TestClient 构造方式。）

- [ ] **Step 7.2: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_api_server.py -v --basetemp=.pytest_tmp
```
Expected: 新增用例 FAIL（真实 200 / body 伪字段 / OpenAPI 无 202）。

- [ ] **Step 7.3: 实现**

`examples/api_server.py`：

```python
import flowagent
from fastapi.responses import JSONResponse

_RESP = {
    200: {"description": "运行完成"},
    202: {"description": "等待人工输入（用 /runs/{thread_id}/resume 恢复）"},
    400: {"description": "工作流执行错误"},
    404: {"description": "运行不存在"},
    409: {"description": "没有待恢复的人工节点"},
}

app = FastAPI(
    title="flowagent API",
    description="配置驱动的 LangGraph 工作流智能体服务示例",
    version=flowagent.__version__,  # 版本单一来源（4.8）
)


def _respond(result: dict) -> JSONResponse:
    """真实 HTTP 语义：等待人工=202（异步未完成），完成=200。"""
    status = 202 if result["status"] == "waiting_human" else 200
    return JSONResponse(status_code=status, content=result)


@app.post("/runs", responses=_RESP)
def create_run(body: RunIn):
    """启动一次运行；若流程包含 human 节点且命中，返回 202 与待办问题。"""
    ...
    return _respond(_result(final, thread_id))


@app.post("/runs/{thread_id}/resume", responses=_RESP)
def resume_run(thread_id: str, body: ResumeIn):
    ...
    return _respond(_result(final, thread_id))
```

（`GET /runs/{thread_id}` 也可以声明 responses，保持一致。删除两处 `result["status_code"] = ...`。）

- [ ] **Step 7.4: 运行测试通过**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_api_server.py -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS。

---

### Task 8: 4.8 版本统一与发布完整性

**Files:**
- Modify: `flowagent/__init__.py:9`（0.8.0 → 0.9.0）
- Modify: `pyproject.toml`
- Create: `MANIFEST.in`
- Modify: `.github/workflows/ci.yml`（新增 build-test job）
- Test: `tests/test_scaffold.py`（追加版本一致性）

- [ ] **Step 8.1: 写失败测试**

追加到 `tests/test_scaffold.py`：

```python
class TestVersionAndPackaging:
    """4.8：版本单一来源 + 包元数据完整。"""

    def test_installed_metadata_matches_runtime_version(self):
        from importlib.metadata import version
        import flowagent
        assert version("flowagent") == flowagent.__version__

    def test_console_script_registered(self):
        from importlib.metadata import entry_points
        eps = entry_points(group="console_scripts")
        assert any(ep.name == "flowagent" and ep.value.startswith("flowagent")
                   for ep in eps)

    def test_version_is_090_or_newer(self):
        import flowagent
        major, minor = (int(x) for x in flowagent.__version__.split(".")[:2])
        assert (major, minor) >= (0, 9)
```

- [ ] **Step 8.2: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_scaffold.py -k TestVersionAndPackaging -v --basetemp=.pytest_tmp
```
Expected: FAIL（无 console script；版本 0.8.0；metadata 0.1.0）。

- [ ] **Step 8.3: 实现**

`flowagent/__init__.py`：`__version__ = "0.9.0"`

`pyproject.toml` 相关段改为：

```toml
[tool.setuptools]
packages = ["flowagent"]

[project]
name = "flowagent
dynamic = ["version"]   # ← 修正为 name = "flowagent"（注意不要引入语法错误）
description = "配置驱动的 LangGraph 工作流智能体引擎"
requires-python = ">=3.10"
dependencies = [
    "langgraph>=1.2",
    "langchain-openai>=1.0.0",
    "pyyaml>=6.0.1",
    "pydantic>=2.7",
]

[tool.setuptools.dynamic]
version = {attr = "flowagent.__version__"}  # 版本单一来源：flowagent/__init__.py

[project.scripts]
flowagent = "flowagent.cli:main"  # 安装后生成 flowagent 命令（文档已引用）

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]
api = ["fastapi>=0.110", "uvicorn>=0.29", "httpx>=0.27"]
sqlite = ["langgraph-checkpoint-sqlite>=2.0.0"]  # FLOWAGENT_CHECKPOINT=sqlite 所需
```

`MANIFEST.in`（新建）：

```
include README.md
include CHANGELOG.md
include LICENSE
include requirements.txt
recursive-include configs *.yaml
recursive-include docs *.md
recursive-include examples *.py
```

`.github/workflows/ci.yml` 追加 job：

```yaml
  build-and-install:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build distributions
        run: |
          python -m pip install --upgrade pip build
          python -m build
      - name: Install wheel and verify
        run: |
          pip install dist/*.whl
          python -c "import flowagent, importlib.metadata as m; assert m.version('flowagent') == flowagent.__version__"
          flowagent check configs/customer_service.yaml --strict
```

- [ ] **Step 8.4: 重新安装元数据并运行测试**

```bash
.venv/Scripts/python -m pip install -e . --no-deps
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_scaffold.py -v --basetemp=.pytest_tmp
```
Expected: PASS。

---

### Task 9: 4.9 工具白名单默认拒绝

**Files:**
- Modify: `flowagent/tools.py:10-22`
- Modify: `tests/conftest.py`（套件默认白名单 fixture）
- Test: `tests/test_tools.py`（追加；实现时先读现有 allowlist 测试避免冲突）

- [ ] **Step 9.1: 写失败测试**

追加到 `tests/test_tools.py`：

```python
class TestToolDefaultDeny:
    """4.9：默认只放行内置工具；外部模块需显式白名单；* 显式全放行。"""

    def test_external_module_denied_by_default(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_TOOL_MODULES", raising=False)
        with pytest.raises(ConfigError):
            load_tool("shutil:rmtree")

    def test_builtin_tools_allowed_by_default(self, monkeypatch):
        monkeypatch.delenv("FLOWAGENT_TOOL_MODULES", raising=False)
        assert callable(load_tool("flowagent.tools:knowledge_base"))

    def test_star_allows_everything(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "*")
        assert callable(load_tool("shutil:rmtree"))  # 仅验证可加载，不执行

    def test_explicit_prefix_still_works(self, monkeypatch):
        monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "tests")
        assert callable(load_tool("tests.flaky:flaky_tool"))
        with pytest.raises(ConfigError):
            load_tool("shutil:rmtree")
```

- [ ] **Step 9.2: 确认失败**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_tools.py -k TestToolDefaultDeny -v --basetemp=.pytest_tmp
```
Expected: `test_external_module_denied_by_default` FAIL（默认放行）。

- [ ] **Step 9.3: 实现**

`tests/conftest.py` 追加（在现有 fixture 之后）：

```python
@pytest.fixture(autouse=True)
def _suite_tool_allowlist(monkeypatch):
    """测试套件默认白名单：内置工具 + tests 辅助模块。

    工具默认拒绝（4.9）后，tests.flaky:* 等测试辅助工具需要显式放行；
    默认拒绝行为本身由 test_tools.py 用 monkeypatch.delenv 覆盖验证。
    """
    monkeypatch.setenv("FLOWAGENT_TOOL_MODULES", "flowagent.tools,tests")
```

`flowagent/tools.py`：

```python
_DEFAULT_ALLOWED = ("flowagent.tools",)  # 默认白名单：仅内置工具模块


def _check_tool_allowlist(module_name: str) -> None:
    """工具模块白名单：FLOWAGENT_TOOL_MODULES=前缀1,前缀2。

    默认失败关闭（4.9）：未配置时只放行内置工具模块 flowagent.tools，
    外部模块（如 shutil）一律拒绝——防止不可信 YAML 直接调用任意 Python callable。
    显式设为 "*" 才全放行（仅限可信本地环境手动开启）。
    """
    allowed = os.getenv("FLOWAGENT_TOOL_MODULES")
    if allowed is not None and allowed.strip() == "*":
        return
    if not allowed:
        prefixes = list(_DEFAULT_ALLOWED)
    else:
        prefixes = [p.strip() for p in allowed.split(",") if p.strip()]
    if prefixes and not any(
        module_name == p or module_name.startswith(p + ".") for p in prefixes
    ):
        raise ConfigError(
            f"工具模块 {module_name} 不在白名单内。默认只放行内置工具；"
            f"如需外部工具请设置 FLOWAGENT_TOOL_MODULES=<模块前缀逗号分隔>"
            f"（可信环境可设 * 全放行）"
        )
```

- [ ] **Step 9.4: README 安全边界小节**

`README.md` 增加：

```markdown
## 安全边界：可信配置 vs 不可信输入

- 工具白名单默认失败关闭：未设置 `FLOWAGENT_TOOL_MODULES` 时只允许内置工具
  （`flowagent.tools:*`），YAML 中的外部 `module:function` 引用会被拒绝。
- 只在配置来源可信（你自己维护的 YAML）且确需外部工具时，显式设置：
  `FLOWAGENT_TOOL_MODULES=mytools,company.utils`；本地可信环境可设 `*` 全放行。
- 永远不要把 LLM 生成的模块/函数名直接写进 YAML 再交给引擎加载。
- 提供服务时，不要让最终用户上传 YAML（当前 API 示例只读取服务端本地配置）。
```

- [ ] **Step 9.5: 运行测试 + 全量回归**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/ -v --basetemp=.pytest_tmp
```
Expected: 全部 PASS。若 `scripts/smoke_v2.py` 引用了外部工具，在其文件头注释补充设置 `FLOWAGENT_TOOL_MODULES` 的说明（先 grep 确认）。

---

### Task 10: 4.10 文档与导出修复

**Files:**
- Modify: `flowagent/__init__.py`（导出 TemplateError）
- Modify: `README.md`（命令跨平台兼容）
- Modify: `docs/API.md`（options 语义、如涉及）

- [ ] **Step 10.1: 导出 TemplateError**

`flowagent/__init__.py`：

```python
from .errors import ConfigError, FlowAgentError, OutputParseError, TemplateError, ToolError
```

`__all__` 中 `"ToolError"` 之后插入 `"TemplateError",`。

测试（追加到 `tests/test_scaffold.py`）：

```python
def test_template_error_exported():
    import flowagent
    assert flowagent.TemplateError is not None
    assert "TemplateError" in flowagent.__all__
```

- [ ] **Step 10.2: README 命令修复**

（先完整读 README，逐块修正：）
- 第 28-34 行的反斜杠续行命令改为单行（PowerShell 不支持 `\` 续行）；
- 所有 JSON 输入统一为 `'{"key": "value"}'`（单引号包双引号，PowerShell/Bash 双兼容，去掉转义反斜杠）；
- 快速开始块补充安装后 console script 用法：`flowagent run configs/approval.yaml ...` 与 `python -m flowagent ...` 等价；
- `export VAR=...` 标注为 Bash 写法，PowerShell 用 `$env:VAR = "..."`。

- [ ] **Step 10.3: 验证**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest tests/test_scaffold.py -v --basetemp=.pytest_tmp
.venv/Scripts/python -c "from flowagent import TemplateError; print('ok')"
```

---

### Task 11: 收尾——CHANGELOG、全量验证、复盘文档

**Files:**
- Modify: `CHANGELOG.md`（0.9.0 条目）
- Create: `docs/problems/review_fixes/review_fixes_20260915.md`（双语复盘）

- [ ] **Step 11.1: CHANGELOG 0.9.0**

```markdown
## 0.9.0（2026-09-15）

评审修复（docs/PROJECT_REVIEW_2026-09-15.md 全部 10 项）。

### 修复（P1：错误放行与可靠性语义）

- **审批否定语义**：示例配置 contains→eq；human 节点声明 options 即强校验，
  无效答复携带错误提示重新询问，绝不进入业务分支。
- **LLM 输出信任边界**：新增 `output_schema` 字段校验（type/min/max/enum/optional），
  畸形输出抛 OutputParseError（失败关闭）；数值路由操作符遇 None/不可数值化
  字段不再退化字符串比较。
- **timeout 真正按时返回**：超时即停止等待（不再阻塞至工具线程结束）；
  超时不参与自动重试，避免副作用重复执行。
- **子图人工中断传播**：workflow 节点把子图 __interrupt__ 提升到父图，
  子线程 ID 由父 thread_id 派生，同一父 thread_id 即可完成恢复。
- **on_error.goto 生效**：固定/并行/路由三种 next 形态都编译错误改道边；
  node.on_error.goto 优先于 route.on_error；目标参与引用校验。

### 修复（P2/P3）

- API 等待人工返回真实 HTTP 202，OpenAPI 声明 200/202/400/404/409。
- 配置校验补强：recursion_limit≥1、max_loops≥1 且必须有出口、regex 编译期
  校验、并行 next 非空。
- 版本单一来源（flowagent.__version__）；注册 console script；sdist 收录
  configs/docs/examples；CI 新增构建安装验证；sqlite 可选依赖组。
- 工具白名单默认失败关闭（仅内置模块），显式 `*` 才全放行。
- 导出 TemplateError；README 命令 PowerShell/Bash 双兼容。

### 测试

- 新增行为测试 N 个（审批反例、畸形输出、超时耗时与副作用、子图恢复、
  错误路由、HTTP 状态码、校验拒绝、白名单、版本一致性）。
```

- [ ] **Step 11.2: 全量验证（verification-before-completion）**

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest --cov=flowagent --cov-report=term-missing --basetemp=.pytest_tmp
.venv/Scripts/python -m ruff check flowagent/ tests/ examples/ scripts/ --select E,F,W,B,UP --ignore E501
.venv/Scripts/python -m build
.venv/Scripts/flowagent --version
.venv/Scripts/flowagent check configs/customer_service.yaml --strict
```
Expected: 测试全绿（≥ 285 passed）、Ruff 无告警、构建成功、CLI 版本 0.9.0、严格校验通过。

- [ ] **Step 11.3: 复盘文档**

按 zheng-code-generate-fix-rule 模板写双语复盘（10 项问题的现象/根因/方案/验证/预防，
保存用户提示词），保存到 `docs/problems/review_fixes/review_fixes_20260915.md`。

---

## Self-Review 记录

1. **Spec coverage**：设计文档 §1-§4 的 10 项 → Task 1(4.1)、2(4.2)、3(4.5)、4(4.7)、5(4.3)、6(4.4)、7(4.6)、8(4.8)、9(4.9)、10(4.10)、11(CHANGELOG/复盘/验收) 全覆盖；设计 §0 的中文注释与 TDD 约定贯穿各任务。
2. **Placeholder scan**：Task 8 的 pyproject 块有一处 `name = "flowagent` 笔误已标注修正说明；Task 10.2 README 修复需实现时读原文逐块处理（列出了 4 条明确规则，非占位）。
3. **Type consistency**：`make_router(node_name, route, error_goto=None)`、`_route_path_map(route, error_goto=None)`、`validate_schema(value, schema)`、`_enrich_payload(payload, node_name, sub_tid)`、`OutputFieldSpec` 字段名与测试引用一致。
