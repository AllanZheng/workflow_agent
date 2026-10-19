"""平台 V3 适配器（M3-1）协议/上下文/回执/路由行为测试（V01/V04–V07）。

覆盖范围（技术方案 §7.1 平台适配契约）：
- V01 信封校验：config_digest 损坏 / 旧快照版本 / 冻结引用不一致 / 执行与追踪标识不一致
  → 一律 422，且模型与授权零调用。
- V04 回执：只记实际发生的使用（同一工具两次调用去重为一条；目录不算使用）。
- V05 业务映射：memory_context 门禁、function_task 路由 consultation_stage、
  analysis_target 透传、request_metadata 白名单。
- V06 服务路由：cancel 未知任务 / reset 幂等 / capabilities 契约 / 内部令牌鉴权 / 默认开放。
- V07 上下文组装：无活动请求报错、预算推导（deadline 预留 + 策略上限）、
  history 机器标记剥离、continuation 显式 > 历史重建 > 新请求。

全部离线：模型流（OpenAI 兼容替身）、模型授权端点（计数替身）、MCP 会话（进程内假工厂）。
信封构造复用 tests/test_managed_config_v3.py 的 _snapshot/_envelope（最小合法 V3 信封）。
"""
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from flowagent.adapters.platform import capabilities_runtime
from flowagent.adapters.platform import context as platform_context
from flowagent.adapters.platform.context import (
    budget_from_request,
    to_run_input,
)
from flowagent.adapters.platform.protocol import ExecutionRequest, snapshot_digest
from flowagent.app import build_platform_service
from flowagent.core.contracts import render_clarification_marker
from tests.test_managed_config_v3 import _envelope, _snapshot

# V3 快照内使用的固定 UUID（与基线 _snapshot 的编号风格一致，互不冲突）。
TOOL1_ID = "11000000-0000-0000-0000-000000000001"
TOOL1_VER = "12000000-0000-0000-0000-000000000001"
SRV1_VER = "13000000-0000-0000-0000-000000000001"
TOOL2_ID = "11000000-0000-0000-0000-000000000002"
TOOL2_VER = "12000000-0000-0000-0000-000000000002"
SRV2_VER = "13000000-0000-0000-0000-000000000002"
SKILL1_ID = "31000000-0000-0000-0000-000000000001"
SKILL1_META = "32000000-0000-0000-0000-000000000001"
SKILL1_BODY = "33000000-0000-0000-0000-000000000001"


# ---------------------------------------------------------------- 测试替身

class _FakeChatTransport:
    """OpenAI 兼容流式端点替身：每次 create() 按脚本吐一轮 chunk（普通函数返回异步生成器）。"""

    def __init__(self, turns):
        self.turns = [list(turn) for turn in turns]
        self.requests = []  # 计数器：记录每次真实到达模型的请求
        # 客户端按 chat.completions.create(...) 取方法 → 以 SDK 形状暴露
        self.sdk = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def create(self, **request):
        self.requests.append(request)
        chunks = self.turns.pop(0) if self.turns else []

        async def _agen():
            for chunk in chunks:
                yield chunk

        return _agen()


def _text_chunk(text):
    """正文增量 chunk（无 usage）。"""
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=text, tool_calls=None), finish_reason=None)], usage=None)


def _tool_chunk(call_id, name, arguments):
    """工具调用 chunk：聚合后 completed 事件携带单个工具调用。"""
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                index=0, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))]),
        finish_reason="tool_calls")], usage=None)


def _final_chunk(prompt_tokens=9, completion_tokens=9):
    """收尾 chunk：finish=stop 且携带 usage（token 记账来源）。"""
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


def _make_mcp_session_factory(advertised, *, structured, cm_calls, tool_calls):
    """进程内 MCP 会话工厂：目录可见性可控；调用记录别名参数与会话头。"""

    @asynccontextmanager
    async def _factory(endpoint, headers, timeout):
        cm_calls.append({"endpoint": endpoint, "headers": dict(headers), "timeout": timeout})

        async def _list_tools():
            return SimpleNamespace(tools=[
                SimpleNamespace(name=name, description="目录工具", input_schema={"type": "object"})
                for name in advertised
            ])

        async def _call_tool(name, arguments):
            tool_calls.append({"name": name, "arguments": dict(arguments)})
            return SimpleNamespace(
                is_error=False, structured_content=dict(structured), content=[])

        yield SimpleNamespace(list_tools=_list_tools, call_tool=_call_tool)

    return _factory


def _install_offline_stack(monkeypatch, *, turns, session_factory=None, skill_transport=None):
    """离线假栈：计数授权替身 + 能力组装注入模型流/MCP/Skill 传输。

    返回 (transport, counter)；counter 统计 build_capabilities 与 authorize 调用次数。
    """
    transport = _FakeChatTransport(turns)
    counter = {"build": 0, "authorize": 0}

    class _StubAuthorizer(capabilities_runtime.ModelAuthorizer):
        """授权替身：忽略端点参数，仅计数（fail-open 的纯计数版）。"""

        def __init__(self, *args, **kwargs):
            pass

        async def authorize(self, **kwargs):
            counter["authorize"] += 1
            return {"credential_generation": 0, "request_id": None}

    real_build = platform_context.build_capabilities

    def _patched(request, **kwargs):
        counter["build"] += 1
        return real_build(
            request, model_transport=transport.sdk,
            skill_transport=skill_transport, mcp_session_factory=session_factory)

    # routes 经 platform_context.build_capabilities 模块属性在调用期取函数 → 打在 context 模块上。
    monkeypatch.setattr(platform_context, "build_capabilities", _patched)
    monkeypatch.setattr(platform_context, "DemoModelAuthorizer", _StubAuthorizer)
    return transport, counter


# ---------------------------------------------------------------- 信封加工

def _reseal(envelope):
    """快照变更后按规则重算 config_digest：快照内与信封层两处必须同时更新。"""
    snapshot = envelope["agent_config"]["snapshot"]
    digest = snapshot_digest(snapshot)
    snapshot["config_digest"] = digest
    envelope["agent_config"]["config_digest"] = digest
    return envelope


def _tools_envelope():
    """带 2 个固定工具 + 1 个 Skill（均未使用）的 V3 信封；MCP 凭据齐全。"""
    snapshot = copy.deepcopy(_snapshot())
    snapshot["tools"] = [
        {
            "tool_id": TOOL1_ID, "tool_version_id": TOOL1_VER, "server_version_id": SRV1_VER,
            "remote_name": "search_notes",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            "description": "固定工具一", "schema_digest": "a" * 64,
        },
        {
            "tool_id": TOOL2_ID, "tool_version_id": TOOL2_VER, "server_version_id": SRV2_VER,
            "remote_name": "fetch_calendar",
            "input_schema": {"type": "object", "properties": {"day": {"type": "string"}}},
            "description": "固定工具二", "schema_digest": "b" * 64,
        },
    ]
    snapshot["config"]["tools"] = [
        {"tool_id": TOOL1_ID, "tool_version_id": TOOL1_VER},
        {"tool_id": TOOL2_ID, "tool_version_id": TOOL2_VER},
    ]
    snapshot["skills"] = [{
        "skill_id": SKILL1_ID, "metadata_version_id": SKILL1_META,
        "body_version_id": SKILL1_BODY, "name": "巡检制度", "summary": "设备巡检制度检索指南",
    }]
    snapshot["config"]["skills"] = [{
        "skill_id": SKILL1_ID, "metadata_version_id": SKILL1_META, "body_version_id": SKILL1_BODY,
    }]
    envelope = _envelope(_reseal({"agent_config": {"snapshot": snapshot}})["agent_config"]["snapshot"])
    # 快照带工具 → 信封必须携带执行期 MCP 配置（协议校验硬约束）。
    envelope["agent_config"]["mcp_endpoint"] = "https://mcp.test/mcp"
    envelope["agent_config"]["tool_access_token"] = "tool-secret"
    return envelope


def _client(**kwargs):
    return TestClient(build_platform_service(**kwargs))


def _respond(client, envelope, headers=None):
    return client.post("/execution-v1/respond", json=envelope, headers=headers or {})


# ---------------------------------------------------------------- V01 信封校验

def test_v01a_corrupted_digest_rejected_with_zero_model_usage(monkeypatch):
    """V01(a)：信封 config_digest 损坏 → 422，且模型/授权/能力组装零调用。"""
    envelope = _envelope()
    envelope["agent_config"]["config_digest"] = "0" * 64
    transport, counter = _install_offline_stack(
        monkeypatch, turns=[[_text_chunk("不应被调用"), _final_chunk()]])

    response = _respond(_client(), envelope)

    assert response.status_code == 422, response.text
    assert "digest" in response.json()["detail"]
    assert transport.requests == []  # 模型一次都未被请求
    assert counter == {"build": 0, "authorize": 0}  # 能力组装与预授权零调用


@pytest.mark.parametrize("schema_version", [1, 2])
def test_v01b_old_snapshot_version_rejected(monkeypatch, schema_version):
    """V01(b)：V1/V2 快照（schema_version 1/2）→ 422 snapshot_version_not_supported。"""
    envelope = _envelope()
    envelope["agent_config"]["snapshot"]["schema_version"] = schema_version
    _reseal(envelope)  # 变更快照必须重算摘要；版本门禁先于摘要校验触发
    transport, counter = _install_offline_stack(monkeypatch, turns=[])

    response = _respond(_client(), envelope)

    assert response.status_code == 422, response.text
    assert "snapshot_version_not_supported" in response.json()["detail"]
    assert counter == {"build": 0, "authorize": 0}


def test_v01c_frozen_tool_reference_mismatch_rejected(monkeypatch):
    """V01(c)：快照工具版本被改但摘要正确重算 → 冻结引用不一致 422（稳定报错）。"""
    envelope = _tools_envelope()
    envelope["agent_config"]["snapshot"]["tools"][0]["tool_version_id"] = (
        "12999999-0000-0000-0000-000000000099")
    _reseal(envelope)  # 摘要自洽 → 只有冻结交叉引用失配
    transport, _ = _install_offline_stack(monkeypatch, turns=[])

    response = _respond(_client(), envelope)

    assert response.status_code == 422, response.text
    assert "tool version references mismatch" in response.json()["detail"]


def test_v01d_execution_trace_identifier_mismatch_rejected(monkeypatch):
    """V01(d)：execution_id 与 trace.trace_id 不一致 → 422。"""
    envelope = _envelope()
    envelope["trace"]["trace_id"] = "ff4b6209-30da-4e57-bf93-73c6867cc101"
    transport, counter = _install_offline_stack(monkeypatch, turns=[])

    response = _respond(_client(), envelope)

    assert response.status_code == 422, response.text
    assert "must match" in response.json()["detail"]
    assert counter == {"build": 0, "authorize": 0}


# ---------------------------------------------------------------- V04 回执

def test_v04_receipt_records_actual_pinned_tool_usage_with_dedup(monkeypatch):
    """V04：实际使用回执——钉住工具两次调用去重为一条；未用工具与未装载 Skill 不入回执。"""
    envelope = _tools_envelope()
    alias1 = capabilities_runtime.tool_alias(TOOL1_ID)
    alias2 = capabilities_runtime.tool_alias(TOOL2_ID)
    cm_calls, tool_calls = [], []
    session_factory = _make_mcp_session_factory(
        [alias1, alias2], structured={"result": "巡检记录 3 条"},
        cm_calls=cm_calls, tool_calls=tool_calls)
    turns = [
        [_tool_chunk("c1", alias1, '{"query": "巡检"}')],
        [_tool_chunk("c2", alias1, '{"query": "巡检"}')],  # 同工具再次调用 → 回执必须去重
        [_text_chunk("已检索两轮，结论如下。"), _final_chunk()],
    ]
    _install_offline_stack(monkeypatch, turns=turns, session_factory=session_factory)

    response = _respond(_client(), envelope)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reply"] == "已检索两轮，结论如下。"
    receipt = body["metadata"]["agent_config_applied"]
    assert receipt["schema_version"] == 3
    # 两次成功调用 → 一条去重记录，携带钉住的版本引用
    tools_used = receipt["called_tool_versions"]
    assert len(tools_used) == 1, tools_used
    assert tools_used[0]["tool_id"] == TOOL1_ID
    assert tools_used[0]["tool_version_id"] == TOOL1_VER
    assert tools_used[0]["server_version_id"] == SRV1_VER
    # 已配置未使用的工具二不入回执；Skill 目录不算使用
    assert tools_used[0]["tool_id"] != TOOL2_ID
    assert receipt["loaded_skill_versions"] == []
    # 调用确实发送了别名，且会话头携带工具身份（X-Agent-Tool-ID/版本）
    assert [entry["name"] for entry in tool_calls] == [alias1, alias1]
    identity = [entry["headers"] for entry in cm_calls if "X-Agent-Tool-ID" in entry["headers"]]
    assert len(identity) == 2
    assert all(h["X-Agent-Tool-ID"] == TOOL1_ID and h["X-Agent-Tool-Version-ID"] == TOOL1_VER
               for h in identity)
    assert cm_calls[0]["headers"]["Authorization"].startswith("Bearer ")


# ---------------------------------------------------------------- V05 业务映射

def test_v05a_memory_context_without_read_permission_rejected(monkeypatch):
    """V05(a)：有 memory_context 而快照 memory.read=False → 应 422 拒绝。"""
    envelope = _envelope()  # 基线快照即 memory.read=False / config.memory_read=False
    envelope["memory_context"] = {
        "schema_version": 1, "status": "ok",
        "rendered_text": "用户偏好中文回复", "included_ids": ["m1"], "items": [],
    }
    _install_offline_stack(
        monkeypatch, turns=[[_text_chunk("好的"), _final_chunk()]])

    response = _respond(_client(), envelope)

    assert response.status_code == 422, response.text


def test_v05b_function_task_routes_to_consultation_stage(monkeypatch):
    """V05(b)：function_task 信封路由 consultation_stage，返回 function_result 与稿件正文。"""
    envelope = _envelope()
    envelope["function_task"] = {
        "function_key": "risk_consultation", "stage": "draft",
        "output_schema": "consultation_document_v1", "inputs": {"knowledge_mode": "off"},
    }
    document = {"title": "会商纪要", "summary": "风险摘要", "body_markdown": "## 会商稿件", "risks": []}
    transport, _ = _install_offline_stack(
        monkeypatch, turns=[[_text_chunk(json.dumps(document, ensure_ascii=False)), _final_chunk()]])

    response = _respond(_client(), envelope)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reply"] == "## 会商稿件"  # draft 阶段回复 = body_markdown
    assert body["metadata"]["recovery"]["requested_workflow"] == "consultation_stage"
    function_result = body["metadata"]["function_result"]
    assert function_result["status"] == "ok"
    assert function_result["output"]["title"] == "会商纪要"
    assert body["metadata"]["agent_config_applied"]["schema_version"] == 3
    assert transport.requests  # 会商模型确实被调用（离线替身）


def test_v05c_analysis_target_passed_into_chat_inputs():
    """V05(c)：analysis_target 信封在 chat 路径透传进 RunInput.inputs。"""
    envelope = _envelope()
    envelope["analysis_target"] = {"conversation_id": 5}

    run_input = to_run_input(ExecutionRequest.model_validate(envelope))

    assert run_input.workflow == "chat"
    assert run_input.inputs["analysis_target"].conversation_id == 5


def test_v05d_request_metadata_whitelist_only():
    """V05(d)：request_metadata 只透传约定白名单键，业务私有键被丢弃。"""
    envelope = _envelope()
    envelope["request_metadata"] = {"original_user_message": "X", "junk": "Y"}

    run_input = to_run_input(ExecutionRequest.model_validate(envelope))

    assert run_input.inputs["request_metadata"] == {"original_user_message": "X"}


# ---------------------------------------------------------------- V06 服务路由

def test_v06a_cancel_unknown_execution_reports_not_found():
    """V06(a)：cancel 未知 execution_id → 200 且明确 not_found_or_finished。"""
    response = _client().post("/execution-v1/cancel", json={"execution_id": "unknown"})

    assert response.status_code == 200
    body = response.json()
    assert body["cancelled"] is False
    assert body["reason"] == "not_found_or_finished"


def test_v06b_reset_is_idempotent_ok():
    """V06(b)：reset 幂等确认 → 200 {ok:true,...} 并回显 continuity_key/generation。"""
    response = _client().post(
        "/execution-v1/reset", json={"continuity_key": "c-1", "generation": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["continuity_key"] == "c-1"
    assert body["generation"] == 3


def test_v06c_capabilities_contract_on_both_versions():
    """V06(c)：能力清单与平台 strict 契约逐字段一致（v1 无 snapshot 段；v2 声明 V3）。"""
    client = _client()
    v1 = client.get("/capabilities")
    assert v1.status_code == 200
    body = v1.json()
    assert body["schema_version"] == 1 and body["framework_id"] == "workflow"
    assert body["control"]["cancel"] == "implemented"
    assert body["context"]["memory_context_versions"] == [1]
    v2 = client.get("/capabilities/v2")
    assert v2.status_code == 200
    body2 = v2.json()
    assert body2["schema_version"] == 2 and body2["snapshot"]["versions"] == [3]
    assert body2["control"]["cancel"] == "implemented"
    assert body2["tools"]["execution_scoped_mcp"] is True
    assert body2["skills"]["session_pinning"] is True
    assert body2["references"]["uuid"] is True
    assert body2["models"]["pre_call_authorization"] is True


def test_v06d_internal_token_enforced_on_capabilities_and_respond(monkeypatch):
    """V06(d)：设置 internal_token 后，缺失/错误 Bearer → 401；正确 Bearer → 200。"""
    client = TestClient(build_platform_service(internal_token="internal-token"))
    envelope = _envelope()
    _install_offline_stack(
        monkeypatch, turns=[[_text_chunk("你好呀"), _final_chunk()]])

    assert client.get("/capabilities").status_code == 401
    assert client.get(
        "/capabilities", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(
        "/capabilities", headers={"Authorization": "Bearer internal-token"}).status_code == 200
    assert _respond(client, envelope).status_code == 401
    assert _respond(
        client, envelope, headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = _respond(client, envelope, headers={"Authorization": "Bearer internal-token"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["reply"] == "你好呀"


def test_v06e_default_service_is_open_without_token():
    """V06(e)：默认（不设 internal_token）服务对 /capabilities 开放。"""
    response = _client().get("/capabilities")

    assert response.status_code == 200


# ---------------------------------------------------------------- V07 上下文组装

def test_v07a_context_factory_raises_without_active_request():
    """V07(a)：ContextVar 无活动请求时 context_factory 直接抛错（fail-closed）。"""
    factory = platform_context.context_factory()

    with pytest.raises(RuntimeError, match="no active request"):
        factory()


def test_v07b_budget_derives_deadline_reserve_and_policy_limits():
    """V07(b)：预算推导——deadline 60s → 主阶段 55s/兜底 5s；策略 model_calls=10 与默认取小。"""
    envelope = _envelope()
    envelope["deadline_ms"] = 60_000
    request = ExecutionRequest.model_validate(envelope)

    budget = budget_from_request(request)

    assert budget.deadline_ms == 60_000
    assert budget.main_deadline_ms == 55_000  # 60s - 5s 平台兜底预留
    assert budget.fallback_timeout_ms == 5_000
    assert budget.model_total == 24  # 无策略时默认上限
    assert budget.mcp_total == 32

    limited = budget_from_request(request, {"limits": {"model_calls": 10}})
    assert limited.model_total == 10  # 策略上限 10 与默认 24 取较小值


def test_v07c_history_clarification_markers_stripped():
    """V07(c)：进入模型的 history 剥离 v1 机器标记（存储原文保留不在此层）。"""
    envelope = _envelope()
    marker = render_clarification_marker({
        "version": 1, "workflow": "group_analysis", "reference_time": "2026-10-20T10:00:00",
        "timezone": "Asia/Shanghai", "clarify_count": 1,
    })
    envelope["history"] = [
        {"role": "user", "content": "统计上周风险"},
        {"role": "assistant", "content": f"还需要补充群名。\n{marker}"},
    ]

    run_input = to_run_input(ExecutionRequest.model_validate(envelope))

    contents = [item["content"] for item in run_input.inputs["history"]]
    assert contents == ["统计上周风险", "还需要补充群名。"]
    assert all("sample-agent:clarification" not in content for content in contents)
    assert run_input.history == run_input.inputs["history"]  # RunInput.history 同为剥离后文本


def test_v07d_explicit_continuation_wins_over_history_rebuild():
    """V07(d)：显式 continuation 载荷优先于 history 重建。"""
    envelope = _envelope()
    marker = render_clarification_marker({
        "version": 1, "workflow": "group_analysis", "reference_time": "t0",
        "timezone": "Asia/Shanghai", "clarify_count": 1,
    })
    envelope["history"] = [
        {"role": "user", "content": "统计上周风险"},
        {"role": "assistant", "content": f"需要补充群名。\n{marker}"},
    ]
    envelope["request_metadata"] = {"continuation": {
        "version": 1, "workflow": "group_analysis",
        "original_request": "原始请求", "clarify_count": 1,
    }}

    run_input = to_run_input(ExecutionRequest.model_validate(envelope))

    diagnostics = run_input.inputs["_platform_diagnostics"]
    assert diagnostics["continuation_source"] == "explicit"
    assert run_input.continuation["original_request"] == "原始请求"
    assert run_input.inputs["continuation"] == run_input.continuation


def test_v07d_history_rebuild_pairs_from_clarification_marker():
    """V07(d)：无显式载荷时按"澄清 assistant（v1 标记）→ user"链重建 continuation。"""
    envelope = _envelope()
    envelope["message"] = "群名是A群"  # 最新补充走 message，history 终止于澄清 assistant
    marker = render_clarification_marker({
        "version": 1, "workflow": "group_analysis", "reference_time": "t0",
        "timezone": "Asia/Shanghai", "clarify_count": 1,
    })
    envelope["history"] = [
        {"role": "user", "content": "统计上周风险"},
        {"role": "assistant", "content": f"需要补充群名。\n{marker}"},
    ]

    run_input = to_run_input(ExecutionRequest.model_validate(envelope))

    diagnostics = run_input.inputs["_platform_diagnostics"]
    assert diagnostics["continuation_source"] == "history_rebuild"
    assert diagnostics["rebuilt_pairs"] == 1
    assert run_input.continuation["clarify_count"] == 1
    assert run_input.continuation["workflow"] == "group_analysis"
    assert run_input.inputs["continuation"] == run_input.continuation


def test_v07d_marker_count_mismatch_falls_back_to_new_request():
    """V07(d)：标记计数与链内澄清数不符 → 按新请求处理并记 template_invalid 诊断。"""
    envelope = _envelope()
    marker = render_clarification_marker({
        "version": 1, "workflow": "group_analysis", "reference_time": "t0",
        "timezone": "Asia/Shanghai", "clarify_count": 3,  # 计数虚高：链内只有 2 个澄清
    })
    envelope["history"] = [
        {"role": "user", "content": "统计上周风险"},
        {"role": "assistant", "content": f"需要补充群名。\n{marker}"},
        {"role": "user", "content": "群名是A群"},
        {"role": "assistant", "content": f"还需要时间范围。\n{marker}"},
    ]

    run_input = to_run_input(ExecutionRequest.model_validate(envelope))

    diagnostics = run_input.inputs["_platform_diagnostics"]
    assert diagnostics["continuation_source"] == "new_request"
    assert diagnostics["template_invalid"] is True
    assert run_input.continuation is None
    assert "continuation" not in run_input.inputs


def test_v07d_new_request_flag_clears_explicit_and_rebuilt_continuation():
    """V07(d)：new_request=true 时清空显式与重建两类 continuation。"""
    envelope = _envelope()
    marker = render_clarification_marker({
        "version": 1, "workflow": "group_analysis", "reference_time": "t0",
        "timezone": "Asia/Shanghai", "clarify_count": 1,
    })
    envelope["history"] = [
        {"role": "user", "content": "统计上周风险"},
        {"role": "assistant", "content": f"需要补充群名。\n{marker}"},
    ]
    envelope["request_metadata"] = {
        "continuation": {"version": 1, "workflow": "group_analysis", "clarify_count": 1},
        "new_request": True,
    }

    run_input = to_run_input(ExecutionRequest.model_validate(envelope))

    diagnostics = run_input.inputs["_platform_diagnostics"]
    assert diagnostics["continuation_source"] == "new_request_explicit"
    assert run_input.continuation is None
    assert "continuation" not in run_input.inputs
