"""准出检查：候选产物 vs acceptance.json 冻结门槛（M4）。

用法（sample_agent 根目录）：
    python benchmarks/check_acceptance.py [--acceptance benchmarks/manifests/acceptance.json]

读取 artifacts/benchmarks/ 下最新 native 各套件 summary.json，对照冻结门槛输出
逐项 pass/fail；阻断项（确定性门槛未过）单独列出。缺产物 = not_run（不算通过）。
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def latest(suite: str) -> dict | None:
    dirs = sorted(glob.glob(str(ROOT / "artifacts" / "benchmarks" / f"native-{suite}-*")))
    if not dirs:
        return None
    summary = Path(dirs[-1]) / "summary.json"
    return json.loads(io.open(summary, encoding="utf-8").read()) if summary.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", default=str(ROOT / "benchmarks/manifests/acceptance.json"))
    args = parser.parse_args()
    acceptance = json.loads(io.open(args.acceptance, encoding="utf-8").read())

    checks: list[tuple[str, str, bool, str]] = []  # (suite, item, ok, detail)

    tau2 = latest("tau2")
    if tau2 is None:
        checks.append(("tau2", "avg_reward", False, "not_run（无候选产物）"))
    else:
        target = acceptance["tau2"]["target_min_avg_reward"]
        actual = tau2.get("avg_reward")
        checks.append(("tau2", "avg_reward", actual is not None and actual >= target,
                       f"actual={actual} target>={target}"))

    fb = latest("financebench")
    if fb is None:
        checks.append(("financebench", "avg_score", False, "not_run（无候选产物）"))
    else:
        target = acceptance["financebench"]["target_min_avg_score"]
        actual = fb.get("avg_score")
        checks.append(("financebench", "avg_score", actual is not None and actual >= target,
                       f"actual={actual} target>={target}"))

    business = latest("business")
    if business is None:
        checks.append(("business", "case_pass", False, "not_run（无候选产物）"))
    else:
        cases = business.get("cases") or {}
        total = cases.get("total") or 0
        passed = cases.get("pass") or 0
        ratio = passed / total if total else 0.0
        checks.append(("business", "case_pass_ratio", ratio >= acceptance["business"]["case_pass_min_ratio"],
                       f"actual={passed}/{total} target>={acceptance['business']['case_pass_min_ratio']}"))
        entry = business.get("entry") or {}
        recall = entry.get("entry_recall")
        ftr = entry.get("false_trigger_rate")
        checks.append(("business", "entry_recall", recall is not None and
                       recall >= acceptance["business"]["entry_recall_min"],
                       f"actual={recall} target>={acceptance['business']['entry_recall_min']}"))
        checks.append(("business", "false_trigger_rate", ftr is not None and
                       ftr <= acceptance["business"]["false_trigger_max"],
                       f"actual={ftr} target<={acceptance['business']['false_trigger_max']}"))
        control = (business.get("control_params") or {}).get("overall")
        checks.append(("business", "control_param_accuracy", control is not None and
                       control >= acceptance["business"]["control_param_accuracy_min"],
                       f"actual={control} target>={acceptance['business']['control_param_accuracy_min']}"))

    failed = [c for c in checks if not c[2]]
    for suite, item, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {suite}.{item}: {detail}")
    print()
    print(f"准入结论：{'通过' if not failed else '未通过'}（{len(checks) - len(failed)}/{len(checks)}）")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
