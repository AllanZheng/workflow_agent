"""基准驱动的路径与环境自举。

设计要点（来自 M0 测绘，2026-10-19）：
- flowagent 以源码目录注入 sys.path，不要求安装（与 flowagent_suite 历史驱动一致）。
- tau2 / financebench 数据与判分器复用工作区 test/auto_cc_benchmarks，同样路径注入。
- vLLM 端点与密钥来自 manifest；api_key 允许环境变量覆盖，凭据不入库。
- PYTHONUTF8=1：中文提示与记录在 Windows 控制台/文件均按 UTF-8 处理。
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

# 三条根路径：sample_agent（本仓库）、auto_cc_benchmarks（评测资产工作区）、tau2 vendor。
SAMPLE_AGENT_ROOT = Path(__file__).resolve().parents[1]
BENCH_WORKSPACE = SAMPLE_AGENT_ROOT.parent / "test" / "auto_cc_benchmarks"
TAU2_SRC = BENCH_WORKSPACE / "vendor" / "tau2-bench" / "src"


def bootstrap_paths(*, with_tau2: bool = False, with_financebench: bool = False) -> None:
    """把需要的源码目录插入 sys.path（幂等）。"""
    os.environ.setdefault("PYTHONUTF8", "1")
    for path in (SAMPLE_AGENT_ROOT,):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    if with_financebench:
        text = str(BENCH_WORKSPACE / "financebench")
        if text not in sys.path:
            sys.path.insert(0, text)
        text = str(BENCH_WORKSPACE)
        if text not in sys.path:
            sys.path.insert(0, text)
    if with_tau2:
        for path in (BENCH_WORKSPACE, TAU2_SRC):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """读 manifest 并做最低限度结构校验；凭据字段只接受 *_env 形式或显式明文（本地私有运行）。"""
    data = json.loads(io.open(path, encoding="utf-8").read())
    for key in ("engine", "model", "suites"):
        if key not in data:
            raise SystemExit(f"manifest missing required key: {key}")
    return data


def resolve_secret(value: str | None, *, env_default: str | None = None) -> str | None:
    """manifest 里 api_key_env 指向环境变量；直接值仅用于本地 vLLM 的占位密钥。"""
    if value is None:
        return os.environ.get(env_default) if env_default else None
    if value.startswith("env:"):
        return os.environ.get(value[4:])
    return value


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_head(repo: str) -> str | None:
    """记录仓库 HEAD，便于结果溯源；不在 git 环境时返回 None。"""
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(SAMPLE_AGENT_ROOT.parent / repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001 - 离线机器可能无 git 或超时
        return None
