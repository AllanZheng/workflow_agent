"""指标汇总：纯函数，输入为各套件统一归一化的用例记录列表。

对应最终技术方案 §9.3 指标表：
- 入口召回 / 误触发率：入口选择类用例（entry_selection）。
- 控制参数正确率：群/窗口/口径全部一致才计对；按路径分组。
- 分类质量：与标注比对；failed/unprocessed 单列，不在成功子集上宣称总体。
- 引用/支持度：结构引用有效与事实支持分开；无依据建议单列。
- 业务成功：workflow/autonomous 分开；各终态分列。
- 直连可用率：独立统计，不计入业务成功。
- 性能/成本：p50/p95、首个有效正文、调用数/token（缺失为 null）。

约定：record 为 dict，至少含 id、suite、status(用例判定 pass/fail/not_run)、
engine、execution_path、latency_ms；其余字段按指标可选。
"""
from __future__ import annotations

import math
from typing import Any


def _num(values: list[float]) -> list[float]:
    return [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]


def percentile(values: list[float], q: float) -> float | None:
    """线性插值分位数；样本不足或空集返回 None，不虚构。"""
    nums = sorted(_num(values))
    if not nums:
        return None
    if len(nums) == 1:
        return float(nums[0])
    pos = (len(nums) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return float(nums[low])
    return float(nums[low] + (nums[high] - nums[low]) * (pos - low))


def entry_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """入口召回与误触发；entry_expect ∈ {trigger, no_trigger}，entry_actual 来自工具调用轨迹。"""
    pos = [r for r in records if r.get("entry_expect") == "trigger"]
    neg = [r for r in records if r.get("entry_expect") == "no_trigger"]
    pos_hit = [r for r in pos if r.get("entry_actual") == "trigger"]
    neg_hit = [r for r in neg if r.get("entry_actual") == "trigger"]
    return {
        "entry_recall": round(len(pos_hit) / len(pos), 4) if pos else None,
        "entry_recall_n": [len(pos_hit), len(pos)],
        "false_trigger_rate": round(len(neg_hit) / len(neg), 4) if neg else None,
        "false_trigger_n": [len(neg_hit), len(neg)],
    }


def control_param_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """群/窗口/口径全部一致才计对；按 workflow/autonomous 路径分组。"""
    out: dict[str, Any] = {"overall": None, "by_path": {}}
    pool = [r for r in records if r.get("control_params") is not None]
    if pool:
        out["overall"] = round(sum(1 for r in pool if r.get("control_params_ok")) / len(pool), 4)
    for path in ("workflow", "autonomous"):
        sub = [r for r in pool if r.get("execution_path") == path]
        if sub:
            out["by_path"][path] = round(sum(1 for r in sub if r.get("control_params_ok")) / len(sub), 4)
    return out


def classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """分类准确率与覆盖：分母=classified；failed/unprocessed 单列。"""
    rows = []
    for r in records:
        cov = r.get("coverage") or {}
        if cov.get("classified") is None:
            continue
        labelled = r.get("classification_labels")
        acc = None
        if labelled:
            correct = sum(1 for m in labelled if m.get("correct"))
            total = sum(1 for m in labelled if m.get("expected") is not None)
            acc = round(correct / total, 4) if total else None
        rows.append({
            "id": r.get("id"), "accuracy": acc,
            "classified": cov.get("classified"), "failed": cov.get("failed"),
            "unprocessed": cov.get("unprocessed"), "denominator": cov.get("classified"),
        })
    accs = _num([row["accuracy"] for row in rows])
    return {
        "cases": rows,
        "macro_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
        "total_classified": sum(row["classified"] or 0 for row in rows),
        "total_failed": sum(row["failed"] or 0 for row in rows),
        "total_unprocessed": sum(row["unprocessed"] or 0 for row in rows),
    }


def citation_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """引用合法率（结构性）与事实支持度分开统计。"""
    structural = [(r.get("citations_valid"), r.get("citations_total")) for r in records
                  if r.get("citations_total") is not None]
    support = [(r.get("facts_supported"), r.get("facts_total")) for r in records
               if r.get("facts_total") is not None]
    valid = sum(v for v, _ in structural if v is not None)
    total = sum(t for _, t in structural if t is not None)
    sup = sum(v for v, _ in support if v is not None)
    sup_total = sum(t for _, t in support if t is not None)
    return {
        "citation_validity": round(valid / total, 4) if total else None,
        "citation_n": [valid, total],
        "fact_support": round(sup / sup_total, 4) if sup_total else None,
        "fact_support_n": [sup, sup_total],
        "unsupported_advice_cases": sum(1 for r in records if r.get("unsupported_advice")),
    }


def business_success_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """业务成功：路径 × 终态分列；直连单列，不计入成功。"""
    out: dict[str, Any] = {"by_path": {}, "terminal_status": {}, "fallback": {}}
    for r in records:
        path = r.get("execution_path") or "unknown"
        status = r.get("business_status") or r.get("status")
        out["by_path"].setdefault(path, {}).setdefault(str(status), 0)
        out["by_path"][path][str(status)] += 1
        out["terminal_status"][str(status)] = out["terminal_status"].get(str(status), 0) + 1
        if r.get("fallback_attempted"):
            key = "success" if r.get("fallback_reply") else "failed"
            out["fallback"][key] = out["fallback"].get(key, 0) + 1
    return out


def performance_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """端到端 p50/p95、首个有效正文、调用数与 token（缺失为 null）。"""
    lat = [r.get("latency_ms") for r in records]
    first_text = [r.get("first_text_ms") for r in records]
    model_calls = _num([r.get("model_calls") or 0 for r in records])
    mcp_calls = _num([r.get("mcp_calls") or 0 for r in records])
    tokens_in = _num([r.get("tokens_in") for r in records])
    tokens_out = _num([r.get("tokens_out") for r in records])
    return {
        "latency_p50_ms": percentile(lat, 0.50),
        "latency_p95_ms": percentile(lat, 0.95),
        "first_text_p50_ms": percentile(first_text, 0.50),
        "model_calls_total": int(sum(model_calls)) if model_calls else 0,
        "mcp_calls_total": int(sum(mcp_calls)) if mcp_calls else 0,
        "tokens_in_total": int(sum(tokens_in)) if tokens_in else None,
        "tokens_out_total": int(sum(tokens_out)) if tokens_out else None,
        "tokens_missing_cases": sum(1 for r in records if r.get("tokens_in") is None),
    }


def capacity_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """容量：单位预算内 classified 数量、partial 原因、单批耗时（群分析专用）。"""
    classified = [r for r in records if (r.get("coverage") or {}).get("classified") is not None]
    batch_ms = _num([r.get("batch_ms") for r in classified])
    return {
        "classified_total": sum((r.get("coverage") or {}).get("classified") or 0 for r in classified),
        "partial_reasons": sorted({p for r in records for p in (r.get("partial_reasons") or [])}),
        "batch_ms_avg": round(sum(batch_ms) / len(batch_ms), 1) if batch_ms else None,
        "within_deadline_cases": sum(1 for r in records if r.get("within_deadline")),
    }


def suite_summary(suite: str, engine: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """统一出口：套件级汇总。scoring 字段是套件特化（reward/avg_score/…），原样保留。"""
    passed = sum(1 for r in records if r.get("case_status") == "pass")
    failed = sum(1 for r in records if r.get("case_status") == "fail")
    not_run = sum(1 for r in records if r.get("case_status") in (None, "not_run"))
    summary: dict[str, Any] = {
        "suite": suite, "engine": engine,
        "cases": {"pass": passed, "fail": failed, "not_run": not_run, "total": len(records)},
    }
    if any(r.get("entry_expect") for r in records):
        summary["entry"] = entry_metrics(records)
    if any(r.get("control_params") is not None for r in records):
        summary["control_params"] = control_param_metrics(records)
    if any((r.get("coverage") or {}).get("classified") is not None for r in records):
        summary["classification"] = classification_metrics(records)
        summary["capacity"] = capacity_metrics(records)
    if any(r.get("citations_total") is not None for r in records):
        summary["citations"] = citation_metrics(records)
    if any(r.get("business_status") for r in records):
        summary["business"] = business_success_metrics(records)
    summary["performance"] = performance_metrics(records)
    # 套件原生评分（tau2 reward、financebench avg_score 等）。
    for key in ("reward", "avg_score", "pass_count", "judge_enabled", "judge_succeeded"):
        vals = [r.get(key) for r in records if r.get(key) is not None]
        if vals:
            summary[key] = vals[0] if len(vals) == 1 else vals
    return summary
