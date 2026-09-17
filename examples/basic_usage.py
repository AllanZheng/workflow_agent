"""flowagent Python API 入门示例（离线 Mock，无需 API key）。

运行：python examples/basic_usage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowagent import MockLLM, WorkflowEngine, load_spec_file


def main() -> None:
    # Windows 管道输出默认 GBK，重配为 UTF-8（与 CLI 同款），避免下游按 UTF-8 读取时解码失败
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover
            pass

    # 1) 编译一次，多次运行
    engine = WorkflowEngine(
        load_spec_file(Path(__file__).resolve().parents[1] / "configs" / "customer_service.yaml"),
        MockLLM(),  # 省略此参数则按环境变量自动选择真实/Mock
    )
    final = engine.run({"ticket": "商品屏幕碎了，要求退款"}, thread_id="basic-1")
    print("类别:", final["vars"]["category"])
    print("答复:", final["vars"]["final_reply"])

    # 2) 导出 Mermaid 图源码（可粘贴到支持 Mermaid 的 Markdown 渲染）
    print("\n--- Mermaid ---")
    print(engine.mermaid())

    # 3) 流式执行：逐超步观察状态
    print("\n--- 流式 ---")
    for state in engine.stream({"ticket": "软件无法登录"}, thread_id="basic-2"):
        done = state.get("trace") or []
        print(f"  已完成 {len(done)} 个节点: {[s['node'] for s in done]}")


if __name__ == "__main__":
    main()
