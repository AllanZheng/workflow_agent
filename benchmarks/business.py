"""业务套件：入口触发、群控制参数/分类、会商三阶段。

用例来自 benchmarks/cases/business.jsonl（冻结集，M0 记录内容 hash）。
三类用例：
- entry_chat：终端工具可见时模型是否触发群分析（入口召回/误触发）。
- group_analysis：生产 parse 提示 + run_group_analysis；按实际 fetch 参数评控制参数。
- consultation：run_function_stage 三阶段 × 知识模式；schema/知识状态/引用校验。

有意改变行为（基线 vs 目标规则不同）用 expect_baseline / expect_target 分别断言，
不要求基线兼容已决定修正的旧行为（评测方案 §8.1）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchmarks._compat import SAMPLE_AGENT_ROOT, bootstrap_paths
from benchmarks.drivers import BenchMcp, build_group_terminal_tool

bootstrap_paths()

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from benchmarks.baseline_v3 import PARSE_ARGS_SYSTEM  # noqa: E402

CASES_FILE = SAMPLE_AGENT_ROOT / "benchmarks" / "cases" / "business.jsonl"

# 入口选择系统提示（冻结；两引擎共用，不混入提示优化）。
ENTRY_SYSTEM_PROMPT = (
    "你是企业群助手。能力：日常问答；群消息分析与统计（需要群、时间范围、分类口径，"
    "缺失时向用户澄清）。可用的群分析工具会直接完成取数、分类统计与报告。"
)

# 会商基座提示（stage 指令与 schema 由 function_stage.build_system_prompt 追加）。
CONSULT_BASE_PROMPT = "你是风险会商助手，依据给定材料产出结构化结论。"

_WIKI_TOOLS = [
    {"name": "hybrid_search", "description": "知识库混合检索",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "get_knowledge", "description": "读取知识文档",
     "input_schema": {"type": "object", "properties": {"knowledge_id": {"type": "string"}}}},
]
# payload 形状对齐 function_stage.summarize_knowledge：hybrid_search 行取
# data[].knowledge_id/knowledge_title/id；命中判定 = data 非空。
_WIKI_HIT = {"data": [
    {"knowledge_id": "kb-001", "knowledge_title": "设备巡检制度", "id": "kb-001-c3",
     "text": "巡检频率：核心机房每日一次，普通机房每周一次。"}]}
_WIKI_EMPTY = {"data": []}

_RISK_MATERIALS = (
    "材料：1) 10月15日核心交换机 CPU 持续 95%，告警未及时处置；"
    "2) 机房巡检记录缺失三日；3) 值班表与实际在岗人员不一致。"
)


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- 入口选择
async def run_entry_chat(engine: Any, case: dict[str, Any]) -> dict[str, Any]:
    target = type("T", (), {"conversation_id": case["target"]["conversation_id"]})()
    mcp = BenchMcp(
        [{"name": "fetch_group_analysis_input", "description": "拉取群消息分析输入",
          "input_schema": {"type": "object"}}],
        results={"fetch_group_analysis_input": case["fetch_payload"]},
    )
    # 终端工具自带独立 llm（分类/报告）；usage 与聊天轮合并统计。
    tool_llm = engine.build_llm()
    tool = build_group_terminal_tool(tool_llm, mcp, target, requested_message=case["message"])
    result = await engine.chat(
        case["message"], system_prompt=ENTRY_SYSTEM_PROMPT,
        mcp_client=mcp, local_tools=[tool],
    )
    # 入口触发只认群分析终端工具执行（terminal_metadata）；裸 fetch 探查不算完整业务入口。
    triggered = bool(result.get("terminal_metadata"))
    fetch_only = (not triggered) and bool(mcp.calls)
    tool_usage = tool_llm.usage_summary()
    record = {
        "id": case["id"], "kind": case["kind"], "engine": engine.name,
        "entry_expect": case["expect"], "entry_actual": "trigger" if triggered else "no_trigger",
        "fetch_only_probe": fetch_only,
        "reply": (result.get("reply") or "")[:500], "error": result.get("error"),
        "model_calls": (result.get("model_calls") or 0) + tool_usage["model_calls"],
        "tokens_in": (result.get("tokens_in") or 0) + (tool_usage["tokens_in"] or 0),
        "tokens_out": (result.get("tokens_out") or 0) + (tool_usage["tokens_out"] or 0),
        "latency_ms": result.get("latency_ms"),
    }
    record["case_status"] = "pass" if record["entry_actual"] == record["entry_expect"] else "fail"
    return record


# ---------------------------------------------------------------- 群分析
def _parse_args_via_llm(llm: Any, message: str) -> dict[str, Any]:
    """生产 parse_args 节点语义：LLM 输出 JSON → parsed 字典（缺失键补默认）。"""
    prompt = (
        f"此前对话：\n\n当前用户消息：{message}\n\n"
        "请输出解析结果 JSON，包含 group_query、group_id、start、end、days、taxonomy_override。"
    )
    reply = llm.inner.invoke([SystemMessage(content=PARSE_ARGS_SYSTEM), HumanMessage(content=prompt)])
    text = re.sub(r"^```(json)?|```$", "", str(reply.content).strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {}
    return {
        "group_query": str(parsed.get("group_query") or ""),
        "group_id": parsed.get("group_id"),
        "start": str(parsed.get("start") or ""),
        "end": str(parsed.get("end") or ""),
        "days": parsed.get("days") if isinstance(parsed.get("days"), int) and not isinstance(parsed.get("days"), bool) else None,
        "taxonomy_override": str(parsed.get("taxonomy_override") or ""),
    }


def _engine_expects(case: dict[str, Any], engine_name: str) -> dict[str, Any]:
    """按引擎选择期望规则：基线与目标规则不同的行为分别断言（有意改变项）。"""
    if engine_name == "baseline-v3" and case.get("expect_baseline") is not None:
        return case["expect_baseline"]
    return case.get("expect_target") or {}


def _score_control_params(case: dict[str, Any], result: dict[str, Any], engine_name: str) -> dict[str, Any]:
    """按实际 fetch 参数评控制参数；clarify 行为=未取数且给出澄清文本。"""
    expects = _engine_expects(case, engine_name)
    calls = [(n, a) for n, a in result.get("fetch_calls") or [] if n == "fetch_group_analysis_input"]
    if expects.get("behavior") == "clarify":
        ok = not calls and not result.get("error") and bool((result.get("reply") or "").strip())
        return {"control_params": None, "control_params_ok": ok,
                "control_detail": "clarify" if ok else "expected-clarify-got-execution"}
    if not calls:
        return {"control_params": None, "control_params_ok": False,
                "control_detail": "fetch 未被调用（意外）"}
    _, args = calls[0]
    detail = {"conversation_id": args.get("conversation_id"), "start": args.get("start"),
              "end": args.get("end"), "days": args.get("days")}
    ok = True
    if "conversation_id" in expects:
        ok = ok and args.get("conversation_id") == expects["conversation_id"]
    for key in ("start", "end", "days"):
        if key in expects:
            ok = ok and args.get(key) == expects[key]
    analysis = result.get("analysis_result") or {}
    taxonomy = ((analysis.get("effective_taxonomy") or {}).get("categories")) or []
    if "taxonomy" in expects:
        ok = ok and taxonomy == expects["taxonomy"]
    return {"control_params": detail, "control_params_ok": ok,
            "control_taxonomy": taxonomy, "control_detail": "ok" if ok else "mismatch"}


async def run_group_analysis(engine: Any, case: dict[str, Any]) -> dict[str, Any]:
    parse_llm = engine.build_llm()
    parsed = _parse_args_via_llm(parse_llm, case["message"])
    target = type("T", (), {"conversation_id": case["target"]["conversation_id"]})() if case.get("target") else None
    result = await engine.group_analysis(
        message=case["message"], parsed=parsed, target=target,
        fetch_script={"tools": {
            "fetch_group_analysis_input": case["fetch_payload"],
            "platform_resolve_group": case.get("resolve_payload", {"status": "matched", "conversation_id": 42}),
        }},
    )
    record: dict[str, Any] = {
        "id": case["id"], "kind": case["kind"], "engine": engine.name,
        "error": result.get("error"), "parsed": parsed,
        "latency_ms": result.get("latency_ms"),
        "model_calls": result.get("model_calls"), "tokens_in": result.get("tokens_in"),
        "tokens_out": result.get("tokens_out"),
    }
    record.update(_score_control_params(case, result, engine.name))
    analysis = result.get("analysis_result") or {}
    coverage = analysis.get("coverage") or {}
    record["coverage"] = {
        "classified": coverage.get("classified"), "failed": coverage.get("classification_failed"),
        "unprocessed": None,
    }
    record["business_status"] = analysis.get("status")
    record["analysis_statistics"] = analysis.get("statistics")
    record["reply"] = (result.get("reply") or "")[:800]

    # 分类计数比对（旧引擎不暴露逐条结果，按类计数评分并记录）。
    labels = case.get("labels")
    if labels and record["analysis_statistics"]:
        expected_counts: dict[str, int] = {}
        for label in labels.values():
            expected_counts[label] = expected_counts.get(label, 0) + 1
        actual_counts = {c["name"]: c["count"] for c in record["analysis_statistics"]["categories"]}
        record["classification_counts"] = {"expected": expected_counts, "actual": actual_counts}
        counts_ok = all(actual_counts.get(k, 0) == v for k, v in expected_counts.items()) and \
            sum(actual_counts.values()) == sum(expected_counts.values())
        record["labels_counts_ok"] = counts_ok
    # 用例级判定：无错误 + 控制参数正确（+ 计数正确，如有标注）。
    checks = [not record["error"], record.get("control_params_ok", False)]
    if record.get("labels_counts_ok") is not None:
        checks.append(record["labels_counts_ok"])
    record["case_status"] = "pass" if all(checks) else "fail"
    return record


# ---------------------------------------------------------------- 会商
def _wiki_mcp(case: dict[str, Any]) -> BenchMcp | None:
    mode = (case.get("task", {}).get("inputs") or {}).get("knowledge_mode", "off")
    if mode == "off":
        return None
    results = {"hybrid_search": _WIKI_HIT, "get_knowledge": _WIKI_HIT}
    state = case.get("wiki_state", "ok")
    if state == "empty":
        results = {"hybrid_search": _WIKI_EMPTY, "get_knowledge": _WIKI_EMPTY}
    elif state == "failed":
        from flowagent.mcp_edge_client import McpEdgeError
        results = {"hybrid_search": McpEdgeError("MCP tool returned an error"), "get_knowledge": McpEdgeError("x")}
    return BenchMcp(_WIKI_TOOLS, results=results)


def _score_consultation(case: dict[str, Any], result: dict[str, Any], engine_name: str) -> dict[str, Any]:
    expects = _engine_expects(case, engine_name)
    function_result = result.get("function_result") or {}
    output = function_result.get("output") or {}
    knowledge = function_result.get("knowledge") or {}
    record: dict[str, Any] = {
        "business_status": function_result.get("status"),
        "knowledge_status": knowledge.get("status"),
    }
    citations = knowledge.get("citations") or []
    valid_ids = {"kb-001"}
    valid = sum(1 for c in citations if c.get("knowledge_id") in valid_ids)
    record["citations_total"] = len(citations)
    record["citations_valid"] = valid

    checks = [not result.get("error")]
    if expects.get("schema_ok"):
        checks.append(function_result.get("status") == "ok")
        if case.get("task", {}).get("stage") in ("draft", "final"):
            checks.append(bool(output.get("title")) and bool(output.get("body_markdown")))
        else:
            checks.append(isinstance(output.get("tasks"), list))
    if expects.get("knowledge_status"):
        checks.append(knowledge.get("status") == expects["knowledge_status"])
    if citations:
        checks.append(valid == len(citations))
    record["case_status"] = "pass" if all(checks) else "fail"
    return record


async def run_consultation(engine: Any, case: dict[str, Any]) -> dict[str, Any]:
    task = dict(case["task"])
    if task["stage"] == "tasks" and "assignable_members" not in task["inputs"]:
        task["inputs"]["assignable_members"] = [{"user_id": 1, "name": "张三"}, {"user_id": 2, "name": "李四"}]
    mcp = _wiki_mcp(case)
    result = await engine.consultation(
        task=task, message=case.get("message") or _RISK_MATERIALS,
        system_prompt=CONSULT_BASE_PROMPT, mcp_client=mcp,
    )
    record = {
        "id": case["id"], "kind": case["kind"], "engine": engine.name,
        "error": result.get("error"), "latency_ms": result.get("latency_ms"),
        "model_calls": result.get("model_calls"), "tokens_in": result.get("tokens_in"),
        "tokens_out": result.get("tokens_out"),
        "reply": (result.get("reply") or "")[:500],
        "assignees": [t.get("assignee_user_id") for t in (result.get("function_result") or {}).get("output", {}).get("tasks", [])],
    }
    if mcp is not None:
        record["mcp_calls"] = [n for n, _ in mcp.calls]
    record.update(_score_consultation(case, result, engine.name))
    return record


# ---------------------------------------------------------------- 总入口
async def run_suite(
    engine: Any,
    manifest: dict[str, Any],
    *,
    limit: int | None = None,
    case_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = load_cases()
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c["id"] in wanted]
    if limit:
        cases = cases[:limit]

    runners = {"entry_chat": run_entry_chat, "group_analysis": run_group_analysis,
               "consultation": run_consultation}
    records = []
    for case in cases:
        runner = runners[case["kind"]]
        try:
            record = await runner(engine, case)
        except Exception as exc:  # noqa: BLE001 - 单用例失败保留并继续
            record = {"id": case["id"], "kind": case["kind"], "engine": engine.name,
                      "case_status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        print(f"[business] {case['id']} -> {record.get('case_status')} "
              f"(err={bool(record.get('error'))})", flush=True)

    summary = {"num": len(records), "cases_file": str(CASES_FILE)}
    return records, summary
