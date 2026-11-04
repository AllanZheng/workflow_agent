"""只收集 pytest 节点，验证五区互斥且覆盖全集；不执行测试或连接服务。"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


EXPRESSIONS = {
    "live": "live",
    "installation": "installation and not live",
    "process": "process and not live and not installation",
    "db": "db and not live and not installation and not process",
    "offline": "not live and not installation and not process and not db",
}


def compare_partitions(all_nodes: set[str], partitions: dict[str, set[str]]) -> dict:
    overlaps = {f"{left} & {right}": sorted(partitions[left] & partitions[right])
                for left, right in itertools.combinations(partitions, 2)}
    union = set().union(*partitions.values())
    missing, extra = sorted(all_nodes - union), sorted(union - all_nodes)
    # 空全集不能证明任何测试经过核对；单个空分区仍是合法结果。
    return {"ok": bool(all_nodes) and not missing and not extra and not any(overlaps.values()),
            "empty_collection": not bool(all_nodes),
            "counts": {name: len(nodes) for name, nodes in partitions.items()},
            "overlaps": overlaps, "missing": missing, "extra": extra}


def collect(root: Path, expression: str | None) -> set[str]:
    command = [sys.executable, "-m", "pytest", "tests", "--collect-only", "-o", "addopts=", "-q"]
    if expression:
        command.extend(["-m", expression])
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", check=False)
    # pytest 对空分区返回 5；其余失败不可被误认为空集。
    if completed.returncode not in (0, 5):
        raise RuntimeError(f"pytest collection failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}")
    nodes = {line.strip().replace("\\", "/") for line in completed.stdout.splitlines()
             if line.startswith(("tests/", "tests\\")) and "::" in line}
    if completed.returncode == 0 and not nodes:
        raise RuntimeError("pytest reported successful collection without parseable node IDs")
    if completed.returncode == 5 and nodes:
        raise RuntimeError("pytest reported an empty partition but returned node IDs")
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="保存节点列表与校验结果的 JSON 路径")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    all_nodes = collect(root, None)
    partitions = {name: collect(root, expression) for name, expression in EXPRESSIONS.items()}
    report = compare_partitions(all_nodes, partitions)
    report.update({"total": len(all_nodes), "expressions": EXPRESSIONS,
                   "all_nodes": sorted(all_nodes),
                   "partitions": {name: sorted(nodes) for name, nodes in partitions.items()}})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "total": report["total"],
                      "empty_collection": report["empty_collection"], "counts": report["counts"],
                      "overlap_counts": {key: len(value) for key, value in report["overlaps"].items()},
                      "missing": len(report["missing"]), "extra": len(report["extra"]),
                      "output": str(target)}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
