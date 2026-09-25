"""命令行入口：python -m flowagent check|show|run|resume <config.yaml>。

退出码约定：
- 0 正常结束
- 1 运行期失败（未降级的节点异常、恢复失败等）
- 2 用法/配置错误（文件不存在、YAML/JSON 非法、校验不通过）
- 3 工作流已暂停，等待人工输入（human 节点 interrupt），可用 resume 恢复
"""

import argparse
import json
import sys
import uuid

from . import __version__
from .config import NodeConfig, check_spec, load_spec_file
from .engine import WorkflowEngine
from .errors import ConfigError, FlowAgentError


def _describe_next(node: NodeConfig) -> str:
    nxt = node.next
    if isinstance(nxt, str):
        return nxt
    if isinstance(nxt, list):
        return " + ".join(nxt) + "（并行）"
    parts = [f"{r.when.op}({r.when.value})?->{r.goto}" for r in nxt.rules]
    if nxt.default:
        parts.append(f"default->{nxt.default}")
    return " | ".join(parts)


def _print_trace(final: dict) -> None:
    print("\n=== 执行轨迹 ===")
    for i, step in enumerate(final.get("trace") or [], 1):
        status = step.get("status")
        if status == "error":
            err = step.get("error", {})
            print(f"  {i}. [错误] {step['node']}: {err.get('type')}: {err.get('message')}")
        else:
            outputs = "; ".join(f"{k}={v}" for k, v in (step.get("outputs") or {}).items())
            print(f"  {i}. {step['node']} -> {outputs}")


def _print_vars(final: dict) -> None:
    print("\n=== 输出变量 ===")
    for key, value in (final.get("vars") or {}).items():
        print(f"  {key} = {value}")


def _print_progress(prev_len: int, final: dict) -> int:
    """流式模式下打印新增的轨迹条目，返回新的轨迹长度。"""
    trace = final.get("trace") or []
    for step in trace[prev_len:]:
        if step.get("status") == "error":
            err = step.get("error", {})
            print(f"  ▸ [错误] {step['node']}: {err.get('type')}: {err.get('message')}")
        else:
            outputs = "; ".join(f"{k}={v}" for k, v in (step.get("outputs") or {}).items())
            print(f"  ▸ {step['node']} -> {outputs}")
    return len(trace)


def _cmd_check(args) -> int:
    try:
        spec = load_spec_file(args.config)
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2
    sub = sum(1 for n in spec.nodes.values() if n.type == "workflow")
    print(f"✓ 配置合法: {spec.workflow.name}")
    print(f"  节点 {len(spec.nodes)} 个（含子图 {sub} 个），起始: {spec.workflow.start}")
    for name, node in spec.nodes.items():
        rel = f"{name} -> {_describe_next(node)}"
        flag = "（子图）" if node.type == "workflow" else ""
        print(f"  [{node.type:^8}] {rel}{flag}")

    warnings = check_spec(spec)
    if warnings:
        print(
            "\n⚠ 引用分析：以下变量被模板/路由引用，但没有产出节点，"
            "也未在 workflow.inputs 中声明："
        )
        for w in warnings:
            print(f"  - {w}")
        print("  （若为运行时输入，请在 workflow.inputs 中声明以通过 --strict 校验）")
        if args.strict:
            print("\n[失败] --strict 模式下存在未解析引用。", file=sys.stderr)
            return 1
    else:
        print("✓ 引用分析通过：所有被引用变量均有产出来源或已声明为输入。")
    return 0


def _cmd_show(args) -> int:
    try:
        spec = load_spec_file(args.config)
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2
    engine = WorkflowEngine(spec, checkpointer=False)
    print(f"工作流: {spec.workflow.name} — {spec.workflow.description}")
    print(f"起始节点: {spec.workflow.start} | 递归上限: {spec.workflow.recursion_limit}")
    print("\n节点与流转:")
    for name, node in spec.nodes.items():
        print(f"  [{node.type:^8}] {name:<14} -> {_describe_next(node)}")
    print("\nMermaid 图源码:")
    print(engine.mermaid())
    return 0


def _load_inputs(args) -> dict:
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            inputs = json.load(f)
    else:
        inputs = json.loads(args.input)
    if not isinstance(inputs, dict):
        raise ValueError("输入必须是 JSON 对象")
    return inputs


def _finish(final: dict, thread_id: str) -> int:
    pending = WorkflowEngine.waiting_on(final)
    if pending is not None:
        question = pending.get("question") if isinstance(pending, dict) else pending
        print("\n⏸ 工作流已暂停，等待人工输入。")
        print(f"  问题: {question}")
        # options 强校验被拒后会带 error 提示：把原因展示给恢复方
        if isinstance(pending, dict) and pending.get("error"):
            print(f"  [无效输入] {pending['error']}")
        print(f"  恢复命令: python -m flowagent resume <config> --thread-id {thread_id} --value <你的回复>")
        return 3
    if final.get("error"):
        print(f"\n[注意] 节点 {final['error']['node']} 执行出错，已按 on_error 降级继续。")
    _print_trace(final)
    _print_vars(final)
    return 0


def _hint_inputs(spec, inputs: dict) -> None:
    """对照 workflow.inputs 声明提示可疑的运行输入（拼写错误高频坑），仅告警不阻断。"""
    declared = set(spec.workflow.inputs or [])
    if not declared:
        return
    extra = sorted(set(inputs) - declared)
    missing = sorted(declared - set(inputs))
    if extra:
        print(
            f"[输入提示] 以下输入未在 workflow.inputs 中声明: {extra}"
            "（若为拼写错误请修正；仍会原样传入）",
            file=sys.stderr,
        )
    if missing:
        print(
            f"[输入提示] 已声明但本次未提供: {missing}"
            "（相关模板占位符将保持原样输出）",
            file=sys.stderr,
        )


def _cmd_run(args) -> int:
    try:
        spec = load_spec_file(args.config)
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2
    try:
        inputs = _load_inputs(args)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(f"[输入错误] {e}", file=sys.stderr)
        return 2
    _hint_inputs(spec, inputs)

    with WorkflowEngine(spec) as engine:
        thread_id = args.thread_id or uuid.uuid4().hex
        llm_name = type(engine.llm).__name__
        print(f"▶ 运行工作流 {spec.workflow.name}（LLM={llm_name}，thread_id={thread_id}）")

        try:
            if args.stream:
                print("\n=== 实时执行（流式） ===")
                final = None
                shown = 0
                for chunk in engine.stream(inputs, thread_id=thread_id):
                    final = chunk
                    shown = _print_progress(shown, chunk)
                if final is None:
                    final = engine.get_state(thread_id).values
            else:
                final = engine.run(inputs, thread_id=thread_id)
        except FlowAgentError as e:
            print(f"[运行失败] {e}", file=sys.stderr)
            return 1
        if getattr(args, "trace_out", None):
            _write_trace(args.trace_out, final)
        if getattr(args, "output", None) == "json":
            pending = WorkflowEngine.waiting_on(final)
            payload = {
                "thread_id": thread_id,
                "status": "waiting_human" if pending is not None else "finished",
                "pending": pending,
                "vars": final.get("vars") or {},
                "error": final.get("error"),
                "trace": final.get("trace") or [],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 3 if pending is not None else 0
        return _finish(final, thread_id)

def _write_trace(path: str, final: dict) -> None:
    """执行轨迹导出为 JSONL（每行一个轨迹事件，UTF-8）。"""
    import json as _json

    try:
        with open(path, "w", encoding="utf-8") as f:
            for step in final.get("trace") or []:
                f.write(_json.dumps(step, ensure_ascii=False) + "\n")
        print(f"[轨迹已导出] {path}")
    except OSError as e:
        print(f"[轨迹导出失败] {e}", file=sys.stderr)


def _cmd_resume(args) -> int:
    try:
        spec = load_spec_file(args.config)
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2
    with WorkflowEngine(spec) as engine:
        # 预检：Command(resume=...) 落在未知/已结束的线程上会静默重新开跑，
        # 这里显式拦截，避免误操作覆盖原流程。
        snapshot = engine.get_state(args.thread_id)
        if not snapshot.values:
            print(
                f"[恢复失败] 未找到 thread-id={args.thread_id} 的检查点。\n"
                "  提示：跨进程恢复需要 PostgreSQL 检查点"
                "（FLOWAGENT_CHECKPOINT=postgres，且与暂停时使用同一数据库文件）。",
                file=sys.stderr,
            )
            return 1
        payload = WorkflowEngine.pending_payload(snapshot)
        if payload is None:
            # 无挂起中断即已结束。注意不能用 snapshot.next 判断：
            # human 节点内二次 interrupt（options 强校验重新询问）后 next 为空，
            # 但 tasks 里仍有待恢复中断（与 API 侧判定保持一致）
            print(
                f"[恢复失败] thread-id={args.thread_id} 的运行已结束，没有待恢复的人工节点。",
                file=sys.stderr,
            )
            return 1

        options = payload.get("options") if isinstance(payload, dict) else None
        value = _resolve_resume_value(args.value, payload, options)
        if value is None:
            return 2  # 交互式输入被取消

        print(f"▶ 恢复工作流 {spec.workflow.name}（thread_id={args.thread_id}）")
        try:
            final = engine.resume(value, thread_id=args.thread_id)
        except FlowAgentError as e:
            print(f"[恢复失败] {e}", file=sys.stderr)
            return 1
        except Exception as e:  # noqa: BLE001 - 线程不存在等 LangGraph 运行时错误
            print(
                f"[恢复失败] {type(e).__name__}: {e}\n"
                "  提示：检查 --thread-id；跨进程恢复需要 PostgreSQL 检查点\n"
                "  （FLOWAGENT_CHECKPOINT=postgres，且与暂停时使用同一数据库文件）。",
                file=sys.stderr,
            )
            return 1
        return _finish(final, args.thread_id)

def _resolve_resume_value(raw_value, payload, options):
    """确定恢复值：--value 优先；缺省时交互式问答。

    选项不匹配时仅告警（引擎层会强校验：无效值触发重新询问，不进入业务分支）。
    """
    if raw_value is not None:
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value  # 允许直接传裸字符串
        if options and value not in options and str(value) not in options:
            print(
                f"[提示] 输入 {value!r} 不在选项 {options} 中，工作流将要求重新作答。",
                file=sys.stderr,
            )
        return value
    # 交互式
    question = payload.get("question") if isinstance(payload, dict) else payload
    print("\n⏸ 工作流正在等待人工输入")
    print(f"  问题: {question}")
    if options:
        print(f"  可选: {' / '.join(options)}")
    try:
        answer = input("  请输入: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[已取消] 未提供恢复输入。", file=sys.stderr)
        return None
    return answer


SCAFFOLD_TEMPLATE = """\
# 由 `flowagent new` 生成的脚手架：可直接运行，亦可按需增删节点。
# 运行：python -m flowagent run %%PATH%% --input '{"text": "任意文本"}'
# 校验：python -m flowagent check --strict %%PATH%%

workflow:
  name: %%NAME%%
  description: 我的工作流（脚手架）
  start: prepare
  recursion_limit: 50
  inputs: [text]

nodes:
  prepare:
    type: template
    description: 预处理输入
    template: "已接收输入：{text}"
    output_key: summary
    next: classify

  classify:
    type: llm
    description: LLM 判断文本情绪
    system: 你是情绪分析器，只输出：正面 或 负面。
    prompt: 判断以下文本的情绪，只输出 正面/负面 之一：{text}
    parser: classifier
    labels: [正面, 负面]
    output_key: sentiment
    mock_response: 正面
    next:
      field: sentiment
      rules:
        - when: {op: eq, value: 正面}
          goto: praise
      default: comfort

  praise:
    type: template
    description: 正面分支
    template: "很高兴看到积极的内容：{summary}"
    output_key: reply
    next: END

  comfort:
    type: template
    description: 负面分支
    template: "别灰心！{summary}"
    output_key: reply
    next: END
"""


def _cmd_new(args) -> int:
    from pathlib import Path

    path = Path(args.path)
    if path.exists():
        print(f"[创建失败] 文件已存在: {path}", file=sys.stderr)
        return 2
    name = args.name or path.stem
    path.parent.mkdir(parents=True, exist_ok=True)
    content = SCAFFOLD_TEMPLATE.replace("%%PATH%%", path.as_posix()).replace(
        "%%NAME%%", name
    )
    path.write_text(content, encoding="utf-8")
    print(f"✓ 已生成工作流脚手架: {path}")
    print(f"  下一步: python -m flowagent check --strict {path.as_posix()}")
    print(
        f"  运行:   python -m flowagent run {path.as_posix()} "
        "--input '{\"text\": \"今天真开心\"}'"
    )
    return 0


def main(argv=None) -> int:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(
        prog="flowagent", description="配置驱动的 LangGraph 工作流智能体引擎"
    )
    parser.add_argument("--version", action="version", version=f"flowagent {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="校验工作流配置（含静态引用分析，不运行）")
    p_check.add_argument("config")
    p_check.add_argument(
        "--strict", action="store_true", help="存在未解析变量引用时以退出码 1 失败"
    )

    p_new = sub.add_parser("new", help="生成工作流配置脚手架")
    p_new.add_argument("path", help="目标 YAML 路径（已存在则拒绝）")
    p_new.add_argument("--name", help="工作流名称（默认取文件名）")

    p_show = sub.add_parser("show", help="查看工作流结构与 Mermaid 图")
    p_show.add_argument("config")

    p_run = sub.add_parser("run", help="运行工作流")
    p_run.add_argument("config")
    p_run.add_argument("--input", default="{}", help='JSON 格式输入，如 {"ticket": "..."}')
    p_run.add_argument("--input-file", help="从 JSON 文件读取输入（优先于 --input）")
    p_run.add_argument("--thread-id", help="检查点线程ID（相同ID可续跑/恢复）")
    p_run.add_argument("--stream", action="store_true", help="流式输出实时执行进度")
    p_run.add_argument("--trace-out", help="执行轨迹导出为 JSONL 文件")
    p_run.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="输出格式：json 为机器可读结果（等待人工时退出码 3）",
    )

    p_resume = sub.add_parser("resume", help="恢复因 human 节点暂停的工作流")
    p_resume.add_argument("config")
    p_resume.add_argument("--thread-id", required=True)
    p_resume.add_argument(
        "--value",
        help="人工输入（字符串或 JSON）；省略时进入交互式问答",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "new":
            return _cmd_new(args)
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "show":
            return _cmd_show(args)
        if args.command == "resume":
            return _cmd_resume(args)
        return _cmd_run(args)
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2



if __name__ == "__main__":
    sys.exit(main())
