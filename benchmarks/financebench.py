"""FinanceBench 套件：150 题开卷财务问答。

数据与判分器复用工作区 financebench/run_financebench.py（题目内嵌证据、
judge 提示与 1/0.5/0 口径、heuristic 下界），保证与历史成绩可比；
答题方换成本仓基准引擎（baseline-v3 = ManagedAgentRuntime 单轮问答）。

系统提示 = 旧 qa_flow.yaml answer 节点原文；温度 0.2 与历史一致。
历史参考（@18008）：旧 YAML 引擎 0.8133 / 0.6633（2026-10-02 两次）；
auto_cc 对照终版 0.88。端点已换 18007，数值不可与旧端点直接混比。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from benchmarks._compat import bootstrap_paths, resolve_secret

FINANCE_SYSTEM_PROMPT = (
    "你是资深企业财务分析师。依据题目给出的材料回答问题：\n"
    "先在材料中定位相关数据，简要列出演算过程（涉及的数字与计算），\n"
    "最后一行以\"答案：\"开头给出结论与关键数字；数字保留原文口径；\n"
    "无法从材料得出时明确说明。"
)


def setup_judge_env(manifest: dict[str, Any]) -> dict[str, Any]:
    """在 import run_financebench 之前注入 judge 环境变量（该模块 import 时读取）。"""
    judge = manifest.get("judge") or {}
    if judge.get("base_url"):
        os_environ_set("FINANCEBENCH_JUDGE_BASE_URL", judge["base_url"])
        os_environ_set("FINANCEBENCH_JUDGE_MODEL_ID", judge.get("model_id", ""))
        os_environ_set(
            "FINANCEBENCH_JUDGE_API_KEY",
            resolve_secret(judge.get("api_key"), env_default="FINANCEBENCH_JUDGE_API_KEY") or "EMPTY",
        )
        return {"base_url": judge["base_url"], "model_id": judge.get("model_id"), "enabled": True}
    import os

    os.environ.pop("FINANCEBENCH_JUDGE_BASE_URL", None)
    return {"enabled": False}


def os_environ_set(key: str, value: str) -> None:
    import os

    os.environ[key] = value


async def run_suite(
    engine: Any,
    manifest: dict[str, Any],
    *,
    limit: int | None = None,
    case_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """逐题执行：构造题目 → 引擎作答 → judge/heuristic 判分 → 记录。"""
    bootstrap_paths(with_financebench=True)
    judge_info = setup_judge_env(manifest)
    import run_financebench as fb  # noqa: PLC0415 - 复用模块（判分口径冻结）

    data_path = Path(fb.DATA_DIR) / "financebench.json"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    if case_ids:
        wanted = set(case_ids)
        rows = [r for r in rows if r.get("financebench_id") in wanted]
    if limit:
        rows = rows[:limit]

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question, has_context = fb.build_question(row)
        started = time.perf_counter()
        result = await engine.chat(question, system_prompt=FINANCE_SYSTEM_PROMPT, history=[])
        seconds = round(time.perf_counter() - started, 2)
        reply = (result.get("reply") or "").strip()
        error = result.get("error")
        truth = row.get("answer", "")

        heuristic = 0.0
        judge_score = None
        if error:
            reply = ""
        if reply:
            heuristic = fb.heuristic_grade(reply, truth)
            judge_score = await asyncio.to_thread(
                fb.llm_judge, reply, truth, row.get("question", "")
            )
        final = judge_score if judge_score is not None else heuristic

        records.append({
            "id": row.get("financebench_id"),
            "idx": idx,
            "company": row.get("company"),
            "doc_name": row.get("doc_name"),
            "question_type": row.get("question_type"),
            "question": row.get("question"),
            "ground_truth": truth,
            "reply": reply,
            "error": error,
            "has_context": has_context,
            "heuristic_score": heuristic,
            "judge_score": judge_score,
            "final_score": final,
            "seconds": seconds,
            "model_calls": result.get("model_calls"),
            "tokens_in": result.get("tokens_in"),
            "tokens_out": result.get("tokens_out"),
            "latency_ms": result.get("latency_ms"),
            "engine": engine.name,
            "case_status": "answered" if (reply or error) else "not_run",
        })
        print(
            f"[financebench] {idx + 1}/{len(rows)} {row.get('financebench_id')} "
            f"final={final} error={bool(error)} {seconds}s",
            flush=True,
        )

    scored = [r["final_score"] for r in records]
    summary = {
        "avg_score": round(sum(scored) / len(scored), 4) if scored else None,
        "num": len(records),
        "judge_enabled": judge_info["enabled"],
        "judge_succeeded": sum(1 for r in records if r["judge_score"] is not None),
        "judge_failed": sum(1 for r in records if r["judge_score"] is None and r["reply"]),
        "empty_or_error": sum(1 for r in records if r["error"] or not r["reply"]),
        "data_file": str(data_path),
    }
    return records, summary
