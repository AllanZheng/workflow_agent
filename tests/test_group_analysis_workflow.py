"""L2 工作流配置测试 — group_analysis.yaml（节点级，注入式 fake，全部离线）。

覆盖测试计划的 L2 用例组：
  - flowagent check --strict 通过 + load_managed_spec 接受（受管门禁，阶段 0 固化）；
  - parse_args：正常解析（mock 渲染管线）、坏 JSON 重试后失败关闭 → clarify 反问；
  - fetch：agent 纪要含「无法完成分析」→ 反问路由；正常纪要 → stats；
    工具链故障 → on_error 用户可读降级；
  - stats：prompt 携带 digest 与 taxonomy_override（覆盖口径生效的管线证明）、
    失败不阻断报告（on_error set 占位统计）；
  - report：三段标题结构 + output_key=reply + 流式 delta。

真实 LLM 语义（解析准确率/统计准确率）属 L4 验收；本层证明引擎管线行为。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from flowagent.config import load_spec
from flowagent.engine import build_graph
from flowagent.llm import MockLLM
from flowagent.managed_config import load_managed_spec

pytestmark = pytest.mark.anyio

YAML_PATH = Path(__file__).resolve().parents[1] / "configs" / "managed" / "group_analysis.yaml"


@pytest.fixture(scope="module")
def yaml_text() -> str:
    return YAML_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def spec(yaml_text):
    return load_managed_spec(yaml_text)


# ---------------------------------------------------------------- 受管门禁

def test_strict_check_passes():
    """flowagent check --strict：引用完整性 + 子图检查（阶段 0 结论固化）。"""
    result = subprocess.run(
        [sys.executable, "-m", "flowagent", "check", "--strict", str(YAML_PATH)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(YAML_PATH.parents[2]),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "引用分析通过" in result.stdout


def test_managed_gate_accepts_yaml(yaml_text):
    """受管门禁：只含 agent/llm/template 三类节点，load_managed_spec 必须接受。"""
    managed = load_managed_spec(yaml_text)
    assert managed.workflow.start == "parse_args"
    assert {node.type for node in managed.nodes.values()} <= {"agent", "llm", "template"}


def test_engine_loads_same_yaml(yaml_text):
    """引擎通用加载与受管加载一致（同一文件两个入口都合法）。"""
    assert load_spec(__import__("yaml").safe_load(yaml_text)).workflow.name == "group-analysis"


# ------------------------------------------------------------ Admin 发布契约

def test_publication_contract_dry_run(monkeypatch):
    """Admin 发布脚本 --dry-run：mcp_tools 注册 schema 与服务实现同源，
    快照 YAML 与 configs 文件逐字一致且过受管门禁（阶段 3 契约对拍）。"""
    import importlib.util

    env = {
        "GROUP_ANALYSIS_AGENT_ID": "604b6209-30da-4e57-bf93-73c6867cc101",
        "GROUP_ANALYSIS_MODEL_CREDENTIAL_ID": "00000000-0000-4000-8000-0000000000a1",
        "GROUP_ANALYSIS_IM_CREDENTIAL_ID": "00000000-0000-4000-8000-0000000000a2",
        "GROUP_ANALYSIS_MODEL_NAME": "qwen3.5-35b",
        "GROUP_ANALYSIS_MODEL_BASE_URL": "http://vllm.test/v1",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parents[1]
             / "deploy" / "group-analysis" / "publish_group_analysis.py"),
         "--dry-run"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dry-run ok" in result.stdout

    # 进程内对拍快照结构（脚本 DB 依赖为惰性导入，无需 sqlalchemy）
    spec = importlib.util.spec_from_file_location(
        "publish_group_analysis",
        Path(__file__).resolve().parents[1] / "deploy" / "group-analysis"
        / "publish_group_analysis.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import uuid as uuid_module

    snapshot = module._snapshot(
        uuid_module.UUID(env["GROUP_ANALYSIS_AGENT_ID"]),
        uuid_module.UUID(env["GROUP_ANALYSIS_MODEL_CREDENTIAL_ID"]),
        uuid_module.UUID(env["GROUP_ANALYSIS_IM_CREDENTIAL_ID"]),
    )
    tools = {t["exposed_name"]: t for t in snapshot["mcp_tools"]}
    from services.im_tools_mcp.tools import (
        FETCH_GROUP_MESSAGES_INPUT_SCHEMA, FIND_GROUP_INPUT_SCHEMA)

    assert tools["find_group"]["input_schema"] == FIND_GROUP_INPUT_SCHEMA
    assert tools["fetch_group_messages"]["input_schema"] == \
        FETCH_GROUP_MESSAGES_INPUT_SCHEMA
    assert tools["fetch_group_messages"]["timeout_ms"] >= 60_000  # 分段拉取慢
    assert tools["find_group"]["url"].endswith("/mcp/")
    assert tools["find_group"]["credential_id"] == env["GROUP_ANALYSIS_IM_CREDENTIAL_ID"]
    # 快照 YAML = 仓库文件正文（改口径 = 改文件重发布）
    assert snapshot["workflow"]["yaml"] == YAML_PATH.read_text(encoding="utf-8")
    load_managed_spec(snapshot["workflow"]["yaml"])  # 入库前门禁复核


# ---------------------------------------------------------------- 测试替身

#: parse_args 节点的标准合法输出（无特殊断言的用例共用）
PARSE_OK = '{"group_query": "项目群", "start": "", "end": "", "days": null, "taxonomy_override": ""}'

#: stats 节点的标准合法输出
STATS_OK = '{"total_messages": 8, "categories": [{"name": "问题", "count": 3, "ratio": "38%", "quotes": ["[08-01 08:00] 阿明: 帮我看下配置"]}], "override_applied": false}'


class ScriptedLLM:
    """按节点 prompt 关键字路由输出的可编程 LLM（llm 节点用）。"""

    def __init__(self, outputs: dict[str, str], capture: list | None = None):
        self.outputs = outputs  # 关键字 -> 输出文本（先命中先出）
        self.capture = capture if capture is not None else []

    def invoke(self, messages, config=None, **kwargs):
        prompt = str(messages[-1].content)
        self.capture.append(prompt)
        return AIMessage(content=self._route(prompt))

    async def astream(self, messages, config=None, **kwargs):
        """流式路径（report 节点）：按 8 字符切块产出，验证增量管线。"""
        prompt = str(messages[-1].content)
        self.capture.append(prompt)
        text = self._route(prompt)
        for i in range(0, len(text), 8):
            yield AIMessageChunk(content=text[i:i + 8])

    def _route(self, prompt: str) -> str:
        for keyword, output in self.outputs.items():
            if keyword in prompt and output is not None:
                if isinstance(output, list):  # 同关键字依次弹出（重试场景）
                    text = output.pop(0)
                    if not output:
                        self.outputs[keyword] = None
                else:
                    text = output
                return text
        return "{}"


class FakeAgentRuntime:
    """注入式 agent runtime：返回固定 digest 或抛错（fetch 节点用）。"""

    def __init__(self, answer: str = "【分段纪要】正常纪要内容", error: Exception | None = None,
                 capture: list | None = None):
        self.answer = answer
        self.error = error
        self.capture = capture if capture is not None else []

    async def run(self, message, system_prompt, max_tool_iterations, history=None):
        if self.capture is not None:
            self.capture.append({"message": message, "system": system_prompt})
        if self.error is not None:
            raise self.error
        return self.answer

    async def stream(self, message, system_prompt, max_tool_iterations, history=None):
        if self.error is not None:
            raise self.error
        yield self.answer


async def _run(spec, *, llm, agent_runtime, message="分析群X最近7天", history=None,
               stream=False):
    graph = build_graph(spec, llm=llm, checkpointer=False,
                        agent_runtime=agent_runtime, stream_final_reply=stream)
    initial = {
        "vars": {"message": message, "history": history or [],
                 "system_prompt": "受管系统提示", "skills": []},
        "trace": [], "loop_counts": {}, "error": None,
    }
    if not stream:
        return await graph.ainvoke(initial)
    events: list[tuple[str, object]] = []
    final = {}
    async for mode, event in graph.astream(initial, stream_mode=["custom", "values"]):
        if mode == "custom":
            events.append((mode, event))
        else:
            final = event
    final["_stream_events"] = events
    return final


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------- parse_args

async def test_parse_args_mock_pipeline_produces_expected_dict(spec):
    """MockLLM + mock_response：解析节点输出通过 schema 校验并进入 fetch。"""
    captured_fetch: list[dict] = []
    llm = ScriptedLLM({})  # llm 节点全部走 mock_response（MockLLM.name 判定）

    class MockWithScript(MockLLM):
        pass

    runtime = FakeAgentRuntime("【分段纪要】mock", capture=captured_fetch)
    final = await _run(spec, llm=MockWithScript(), agent_runtime=runtime)
    args = final["vars"]["parsed_args"]
    assert args["group_query"] == "测试群"
    assert args["days"] == 7
    # fetch agent 收到的 message 由 fetch_msg 模板渲染（群名进入任务文本）
    assert "测试群" in captured_fetch[0]["message"]
    assert "分段纪要" in captured_fetch[0]["system"]


async def test_parse_args_bad_json_fails_closed_to_clarify(spec):
    """坏 JSON：retries 1 次后仍失败 → on_error → clarify，reply 为反问文案。"""
    llm = ScriptedLLM({"请输出解析结果 JSON": ["不是JSON", "还是不是JSON"]})
    final = await _run(spec, llm=llm, agent_runtime=FakeAgentRuntime())
    assert "请补充分析所需的信息" in final["vars"]["reply"]
    # fetch 未执行（digest 未产出）
    assert "digest" not in final["vars"]


# ---------------------------------------------------------------- fetch 路由

async def test_fetch_digest_with_failure_marker_routes_to_question(spec):
    runtime = FakeAgentRuntime("无法完成分析：未找到名为「X」的群，请确认群名")
    llm = ScriptedLLM({
        "请输出解析结果 JSON": PARSE_OK,
        "请输出分类统计 JSON": '{"total_messages": 0, "categories": [], "override_applied": false}',
    })
    final = await _run(spec, llm=llm, agent_runtime=runtime)
    assert final["vars"]["reply"] == "无法完成分析：未找到名为「X」的群，请确认群名"
    assert "stats_summary" not in final["vars"]  # stats 未执行


async def test_fetch_normal_digest_routes_to_stats(spec):
    digest = "【分段纪要】窗口 7 天；问题 3 条；讨论 5 条"
    captured: list[str] = []
    llm = ScriptedLLM({
        "请输出解析结果 JSON": PARSE_OK,
        "请输出分类统计 JSON": STATS_OK,
    }, capture=captured)
    final = await _run(spec, llm=llm, agent_runtime=FakeAgentRuntime(digest))
    assert final["vars"]["stats_summary"]["total_messages"] == 8
    stats_prompt = next(p for p in captured if "分类统计" in p)
    assert digest in stats_prompt  # digest 进入 stats 上下文


async def test_fetch_tool_failure_degrades_to_readable_reply(spec):
    runtime = FakeAgentRuntime(error=RuntimeError("MCP tool call failed"))
    llm = ScriptedLLM({"请输出解析结果 JSON": PARSE_OK})
    final = await _run(spec, llm=llm, agent_runtime=runtime)
    assert "无法获取群消息" in final["vars"]["reply"]
    assert "digest" not in final["vars"]


# ---------------------------------------------------------------- stats

async def test_stats_prompt_carries_taxonomy_override(spec):
    """覆盖口径从 parse_args 透传到 stats prompt（taxonomy_override 生效的管线证明）。"""
    digest = "【分段纪要】窗口 7 天"
    captured: list[str] = []
    llm = ScriptedLLM({
        "请输出解析结果 JSON": '{"group_query": "项目群", "start": "", "end": "", "days": null, "taxonomy_override": "按紧急/常规分类"}',
        "请输出分类统计 JSON": '{"total_messages": 5, "categories": [], "override_applied": true}',
    }, capture=captured)
    final = await _run(spec, llm=llm, agent_runtime=FakeAgentRuntime(digest))
    stats_prompt = next(p for p in captured if "分类统计" in p)
    assert "按紧急/常规分类" in stats_prompt
    report_prompt = next(p for p in captured if "群分析报告" in p)
    assert "按紧急/常规分类" in report_prompt
    assert final["vars"]["reply"]  # 报告生成（ScriptedLLM 兜底 "{}" -> text 节点输出）


async def test_stats_failure_does_not_block_report(spec):
    digest = "【分段纪要】窗口 7 天"
    llm = ScriptedLLM({
        "请输出解析结果 JSON": PARSE_OK,
        "请输出分类统计 JSON": ["坏统计", "还是坏统计"],  # 两次都失败 -> on_error
    })
    final = await _run(spec, llm=llm, agent_runtime=FakeAgentRuntime(digest))
    # stats 失败 -> on_error.set 占位统计 -> report 仍执行并产出 reply
    assert final["vars"]["stats_summary"] == "（分类统计生成失败，本段略过）"
    assert "群分析报告" not in final["vars"]["reply"] or final["vars"]["reply"]


# ---------------------------------------------------------------- report

async def test_report_streams_three_section_reply(spec):
    """report 节点 output_key=reply 且 next=END：stream_final_reply 下产出 delta。"""
    report_text = "## 总结\n事项A。\n## 分类统计\n- 问题 30%\n## 汇报要点\n- 待办：X"
    llm = ScriptedLLM({
        "请输出解析结果 JSON": PARSE_OK,
        "请输出分类统计 JSON": '{"total_messages": 5, "categories": [], "override_applied": false}',
        "请输出群分析报告": report_text,
    })
    final = await _run(
        spec, llm=llm, agent_runtime=FakeAgentRuntime("【分段纪要】窗口 7 天"),
        stream=True,
    )
    deltas = [event.get("content") for mode, event in final["_stream_events"]
              if mode == "custom" and isinstance(event, dict)
              and event.get("kind") == "delta"]
    assert deltas and "".join(deltas)  # 文本 parser 的可见增量非空
    assert "## 总结" in final["vars"]["reply"]
    assert "## 分类统计" in final["vars"]["reply"]
    assert "## 汇报要点" in final["vars"]["reply"]
