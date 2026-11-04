"""基准 CLI：套件选择、manifest 装载、产物落盘。

用法（sample_agent 根目录，基准 venv python）：
  python benchmarks/run.py --engine baseline-v3 --suite financebench \
      --manifest benchmarks/manifests/baseline.json [--limit N]

产物：artifacts/benchmarks/<run-id>/ 下 manifest.json（含实际版本与数据 hash）、
cases.jsonl（每题/每用例记录）、summary.json（指标汇总）。产物目录不入库。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks._compat import load_manifest, repo_head, resolve_secret, sha256_file  # noqa: E402
from benchmarks import metrics  # noqa: E402
from benchmarks.drivers import build_engine  # noqa: E402

SUITES = ("tau2", "financebench", "business", "faults")


def run_id(engine: str, suite: str) -> str:
    return f"{engine}-{suite}-{time.strftime('%Y%m%d-%H%M%S')}"


def enrich_manifest(manifest: dict[str, Any], engine_name: str, suite: str) -> dict[str, Any]:
    """把实际版本、数据 hash 写入运行时 manifest（不回写源文件）。"""
    manifest = dict(manifest)
    manifest["run"] = {
        "engine": engine_name, "suite": suite,
        "sample_agent_revision": repo_head("sample_agent"),
        "agents_platform_revision": repo_head("agents_platform"),
        "python": sys.version.split()[0],
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    business_cases = _ROOT / "benchmarks" / "cases" / "business.jsonl"
    if business_cases.is_file():
        manifest["run"]["business_cases_sha256"] = sha256_file(business_cases)
    return manifest


async def execute(engine_name: str, suite: str, manifest: dict[str, Any],
                  limit: int | None, case_ids: list[str] | None) -> tuple[list, dict]:
    engine = build_engine(engine_name, manifest)
    if suite == "financebench":
        from benchmarks.financebench import run_suite
    elif suite == "tau2":
        from benchmarks.tau2_adapter import run_suite
    elif suite == "business":
        from benchmarks.business import run_suite
    elif suite == "faults":
        raise SystemExit("faults 套件：用例已冻结于 cases/faults.jsonl，随 M1 原生内核测试执行（M0 未运行）")
    else:
        raise SystemExit(f"unknown suite: {suite} (choose from {SUITES})")
    return await run_suite(engine, manifest, limit=limit, case_ids=case_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="sample_agent V3 基准入口")
    parser.add_argument("--engine", required=True, choices=["baseline-v3", "native"])
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="缺省读 manifest suites.<suite>.limit / num_tasks")
    parser.add_argument("--case-ids", nargs="*", default=None, help="仅跑指定用例/题目 ID")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if manifest["engine"] != args.engine:
        raise SystemExit(f"manifest.engine={manifest['engine']} 与 --engine={args.engine} 不一致")
    if args.limit is None:
        suite_cfg = manifest.get("suites", {}).get(args.suite, {})
        args.limit = suite_cfg.get("limit") or suite_cfg.get("num_tasks")

    rid = run_id(args.engine, args.suite)
    out_dir = _ROOT / "artifacts" / "benchmarks" / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    records, native_summary = asyncio.run(
        execute(args.engine, args.suite, manifest, args.limit, args.case_ids)
    )
    wall_seconds = round(time.perf_counter() - started, 1)

    rich_manifest = enrich_manifest(manifest, args.engine, args.suite)
    summary = metrics.suite_summary(args.suite, args.engine, records)
    summary.update(native_summary)
    summary["wall_seconds"] = wall_seconds

    (out_dir / "manifest.json").write_text(
        json.dumps(rich_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (out_dir / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps({k: summary.get(k) for k in
                      ("suite", "engine", "num", "num_tasks", "avg_reward", "avg_score",
                       "pass_count", "wall_seconds")}, ensure_ascii=False))
    print(f"[run.py] 产物目录: {out_dir}")


if __name__ == "__main__":
    main()
