"""命令行入口：flowagent run --config config.json 与 serve。

退出码：completed/needs_input/partial=0，degraded=2，failed=1，取消=130。
信封/用法错误 2。旧 YAML 命令随 M4 退役。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__
from .errors import ConfigError

logger = logging.getLogger("flowagent.cli")

_EXIT_CODES = {
    "completed": 0, "needs_input": 0, "partial": 0,
    "degraded": 2, "failed": 1, "cancelled": 130,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="flowagent", description="业务工作流智能体（V3 原生内核）")
    parser.add_argument("--version", action="version", version=f"flowagent {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="单次执行（JSON 输入/结果）")
    p_run.add_argument("--config", dest="config_file", required=True)
    p_run.add_argument("--profile", help="profile 名称（默认取 default_profile）")
    p_run.add_argument("--input", default="{}", help="JSON 请求串")
    p_run.add_argument("--input-file", help="从 JSON 文件读取请求（优先）")

    p_serve = sub.add_parser("serve", help="启动原生 HTTP 服务（单 worker）")
    p_serve.add_argument("--config", dest="config_file", required=True)
    p_serve.add_argument("--profile")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            return _cmd_serve(args)
        return _cmd_run_native(args)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2


def _cmd_run_native(args) -> int:
    """单次执行：stdout 只输出结果 JSON；运行期 INFO 日志降噪。"""
    import asyncio

    from .app import build_runner
    from .config import load_profile_config, select_profile
    from .core.contracts import RunInput

    logging.getLogger("flowagent").setLevel(logging.WARNING)
    config = load_profile_config(args.config_file)
    profile = select_profile(config, args.profile)
    runner = build_runner(profile)
    if args.input_file:
        payload = json.loads(open(args.input_file, encoding="utf-8").read())
    else:
        payload = json.loads(args.input or "{}")
    result = asyncio.run(runner.run(RunInput(
        workflow=str(payload.get("workflow", "chat")),
        inputs=dict(payload.get("inputs") or {}),
        history=[dict(item) for item in payload.get("history") or []],
        correlation_id=payload.get("correlation_id"),
        continuation=payload.get("continuation"),
    )))
    print(json.dumps(result.to_public_dict(), ensure_ascii=False, indent=2, default=str))
    return _EXIT_CODES.get(result.status.value, 1)


def _cmd_serve(args) -> int:
    """serve：原生 HTTP 服务（fastapi 在 api extra）。"""
    from .adapters.http import build_http_app

    app = build_http_app({"config_path": args.config_file, "profile_name": args.profile})
    print(f"[flowagent] native http://{args.host}:{args.port}", file=sys.stderr)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
