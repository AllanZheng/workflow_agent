# 问题复盘 / Issue Postmortem: 测试套件环境鲁棒性（真实 LLM 环境变量泄漏 + Windows GBK 管道编码）

## 1. 问题现象 / Symptoms

- **错误信息 / Error Message:**
  - [中文] 配置真实 LLM 环境变量（`OPENAI_API_KEY` + `FLOWAGENT_LLM=real`）后跑全量套件，
    3 个用例失败：`test_build_graph_auto_creates_llm`（工厂返回 ChatOpenAI 而非 MockLLM）、
    `TestInputHints::test_typo_key_warns_but_proceeds` / `test_missing_declared_key_warns`
    （真实网络调用导致退出码/告警断言不确定）。另：不带 `PYTHONUTF8=1` 时
    `test_basic_usage_example_runs` 报 `TypeError: argument of type 'NoneType' is not iterable`
    （根因是子进程读取线程 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0`）。
  - [English] With real-LLM env vars exported, 3 tests fail (factory returns real client;
    network-dependent CLI tests become nondeterministic). Separately, without
    `PYTHONUTF8=1`, the subprocess reader thread dies with `UnicodeDecodeError`
    (byte 0xc0 = GBK) making `stdout` None.

- **触发条件 / Trigger Condition:**
  - [中文] 宿主机 shell 导出了真实端点变量后跑 pytest；或在中文 Windows（GBK）上
    不带 `PYTHONUTF8=1` 直接跑套件。
  - [English] Running pytest with real-endpoint env vars in the shell; or running the
    suite on Chinese Windows (GBK) without `PYTHONUTF8=1`.

- **影响范围 / Impact Scope:**
  - [中文] 仅测试套件与示例脚本，引擎本身无缺陷；但违背文档"配置 OPENAI_API_KEY
    后启用真实测试（套件应保持绿）"的承诺，且 CI/本地结果可复现性受损。
  - [English] Test suite and example script only — no engine defect; but violates the
    documented promise that the suite stays green with a real key configured, and
    hurts reproducibility.

## 2. 根因分析 / Root Cause Analysis

- **根因 / Root Cause:**
  - [中文] 功能测试隐式依赖"环境里没有 LLM 凭据"（Mock 模式），但从未显式钉死；
    套件里跑完整工作流并断言退出码/告警的用例（TestInputHints、test_scaffold）
    在真实端点可达时会走网络。示例脚本输出中文未经 `reconfigure(encoding="utf-8")`，
    Windows 管道下子进程按 GBK 编码写出。
  - [English] Functional tests implicitly assumed "no LLM credentials in env" without
    pinning it; tests that run full workflows (input hints, scaffold) hit the network
    when a real endpoint is reachable. The example script never reconfigured stdout,
    so piped output on Windows used GBK.

- **直接原因 / Direct Cause:**
  - [中文] `test_build_graph_auto_creates_llm` 只删 `OPENAI_API_KEY` 未删
    `FLOWAGENT_LLM`；TestInputHints 无环境隔离；basic_usage.py 缺 CLI 同款
    stdout 重配置；scaffold 测试按 UTF-8 读取子进程输出但未约束子进程写码。
  - [English] Partial env cleanup in one test; no isolation in input-hints tests;
    missing stdout reconfigure in the example; reader assumes UTF-8 without
    constraining the writer.

- **间接原因 / Contributing Factors:**
  - [中文] 历次验证都在 `PYTHONUTF8=1` 的 shell 里跑，掩盖了编码依赖；
    套件从未在"配置了真实 key"的声明场景下做过整体验证。
  - [English] All prior runs happened under `PYTHONUTF8=1`, masking the encoding
    dependency; the suite had never been validated end-to-end in its documented
    "key configured" scenario.

## 3. 解决方案 / Solution

- **修复思路 / Fix Strategy:**
  - [中文] 环境决定论收敛到一处：conftest 增加 autouse fixture 默认钉死 Mock 模式
    （删 `OPENAI_API_KEY`、设 `FLOWAGENT_LLM=auto`），真实 LLM 验证文件显式豁免
    （其 skipif 在收集期读环境，不受影响）——与既有 `_suite_tool_allowlist` 同模式。
    示例脚本补齐 CLI 同款 stdout UTF-8 重配置，从源头消除编码歧义。
  - [English] Centralize determinism: an autouse conftest fixture pins mock mode
    (delete key, force auto), with the real-LLM test file explicitly exempted —
    mirroring the existing tool-allowlist fixture. Add CLI-style stdout reconfigure
    to the example to remove encoding ambiguity at the source.

- **具体改动 / Changes Applied:**
  - [中文] `tests/conftest.py`（新增 `_suite_mock_llm` autouse fixture，含豁免逻辑与
    中文注释）；`examples/basic_usage.py`（`main()` 开头 stdout 重配置 UTF-8）；
    `CHANGELOG.md`（0.9.0 测试节补记）。
  - [English] conftest autouse fixture with opt-out; example stdout reconfigure;
    CHANGELOG entry.

- **为什么这样改 / Why This Fix:**
  - [中文] 单点收敛优于逐测试打补丁（新用例自动免疫）；豁免机制保证
    "配 key 即启用真实验证"的文档承诺继续成立；示例修复对直接运行示例的
    用户同样生效，而非只讨好测试。
  - [English] One chokepoint beats per-test patches (new tests inherit immunity);
    the opt-out preserves the documented "key enables real tests" contract; and the
    example fix serves real users, not just the test harness.

## 4. 验证结果 / Validation

- **验证步骤 / Validation Steps:**
  - [中文] 场景 A（真实端点 glm-5.3-flash @ 172.18.127.124:8901，含
    `FLOWAGENT_LLM=real`）：全量 311 passed；场景 B（无任何 LLM 环境变量、
    无 `PYTHONUTF8`）：全量 308 passed + 3 skipped；Ruff 按 CI 规则
    （`--select E,F,W,B,UP --ignore E501`）通过。
  - [English] Scenario A (real endpoint with FLOWAGENT_LLM=real): 311 passed;
    scenario B (no env, no PYTHONUTF8): 308 passed + 3 skipped; Ruff clean under
    the exact CI rule set.

- **验证结论 / Validation Result:**
  - [中文] 套件在两种环境下均确定全绿；`tests/test_real_llm.py` 针对真实
    glm-5.3-flash 端点 3/3 通过；示例脚本无 PYTHONUTF8 依赖。
  - [English] Suite is deterministic and green in both environments; real-LLM tests
    pass 3/3 against live glm-5.3-flash; example no longer depends on PYTHONUTF8.

## 5. 预防措施 / Prevention

- **监控/告警 / Monitoring/Alerting:**
  - [中文] CI 增加一条"带假 key + 错误端点"的冒烟矩阵项，验证套件隔离生效
    （功能测试必须离线通过而非网络超时）。
  - [English] Add a CI smoke leg with a fake key + unreachable endpoint to prove
    functional tests stay offline-fast rather than timing out on the network.

- **测试补充 / Test Coverage:**
  - [中文] 后续新增"跑完整工作流"的用例一律默认继承 Mock 钉死；需要真实网络的
    断言必须放入 test_real_llm.py 并接受 skip 语义。
  - [English] New workflow-running tests inherit the mock pin by default; any
    network-dependent assertion must live in test_real_llm.py with skip semantics.

- **规范改进 / Process Improvement:**
  - [中文] Windows 中文环境的子进程/管道输出一律显式声明编码（写读双方），
    不依赖调用方 shell 变量；进入 CHANGELOG 的承诺（如"配 key 后套件保持绿"）
    必须有对应的整体验证场景。
  - [English] Always declare encodings explicitly for subprocess/pipe I/O on Chinese
    Windows (both sides); every promise recorded in CHANGELOG needs a matching
    end-to-end validation scenario.

---

## 原始问题 / Original Issue

**用户提示词 / User Prompt:**
```
连接 http://172.18.127.124:8091/v1 glm-5.3-flash 再测试下
端口8901,重新测试下
```
