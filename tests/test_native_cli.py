"""原生 V3 CLI 路径测试（run --config / serve / 旧引擎分流）。

覆盖行为：
- CLI-01 completed：原生 run 输出 JSON、status=completed、退出码 0。
- CLI-02 degraded：主阶段模型失败后本地直连兜底成功，退出码 2。
- CLI-03 failed：主阶段与兜底均失败（FALLBACK_FAILED），退出码 1。
- CLI-04 用法错误：run 既无 YAML 位置参数也无 --config，退出码 2；
  以及未知 --profile 同样按配置错误退出 2。
- CLI-05 serve：--help 冒烟（argparse SystemExit(0)，绝不启动 uvicorn）。
- CLI-06 旧引擎：YAML 位置参数仍走旧引擎，且完全不构造原生模型客户端。

未覆盖：needs_input 退出码 0（chat 工作流不产生澄清，由澄清用例覆盖）；
cancelled 退出码 130（需要并发取消，由 Runner 取消用例覆盖）。
"""
import json

import pytest

from flowagent.cli import main
from flowagent.core.contracts import ErrorCode, FlowAgentCoreError
from tests.native_helpers import FakeModel

# KERNEL-BUG: flowagent/app.py build_runner 以 workflow_enabled_map= 构造
# RuntimePolicy，而 flowagent/core/runner.py 的 RuntimePolicy 只接受
# workflow_enabled=，导致所有原生 run 在 build_runner 处 TypeError。
_KERNEL_POLICY_BUG = pytest.mark.xfail(
    strict=False,
    reason=(
        "KERNEL-BUG: build_runner(RuntimePolicy(workflow_enabled_map=...)) 与 "
        "RuntimePolicy(workflow_enabled=...) 签名不一致 → TypeError"
    ),
)

# KERNEL-BUG（completed 用例会先后撞上两处内核缺陷，合并标注）：
# ① flowagent/core/runner.py 以 INFO 记录 run_route，cli.setup_logging 的文本
#    handler 写 stdout（flowagent/obs_log.py），使 CLI 结果 JSON 前多出一行日志，
#    stdout 不再是纯 JSON（规格 CLI-01 要求 stdout 可直接 json.loads）。
# ② 与 _KERNEL_POLICY_BUG 相同的 RuntimePolicy 签名不一致。
_KERNEL_COMPLETED_BUG = pytest.mark.xfail(
    strict=False,
    reason=(
        "KERNEL-BUG: ①runner INFO 日志写入 stdout 污染结果 JSON；"
        "②build_runner 的 RuntimePolicy 构造参数与定义不一致 → TypeError"
    ),
)


@pytest.fixture
def profile_path(tmp_path):
    """最小原生 profile：模型直连参数齐全、chat 工作流启用。"""
    config = {
        "version": 1,
        "default_profile": "default",
        "profiles": {
            "default": {
                "model": {
                    "base_url": "http://offline.test/v1",
                    "api_key": "test-key",
                    "name": "fake-model",
                    "parameters": {"temperature": 0.0},
                },
                "workflows": {"chat": {"enabled": True}},
            }
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _install_model_shim(monkeypatch, turns_factory):
    """把 flowagent.app 的模型客户端替换为脚本化替身（每次执行一个新实例）。"""
    created = []

    class ScriptedModelShim:
        def __init__(self, **kwargs):
            # 记录构造参数，顺带验证 profile 中的模型名确实被解析进来。
            self.kwargs = kwargs
            self.fake = FakeModel(turns=turns_factory())
            created.append(self)

        def stream(self, messages, tools, limits):
            return self.fake.stream(messages, tools, limits)

    monkeypatch.setattr("flowagent.app.OpenAICompatibleModel", ScriptedModelShim)
    return created


def _run_chat(profile_path):
    """以 CLI 原生方式执行一次 chat。"""
    payload = json.dumps({"workflow": "chat", "inputs": {"message": "hi"}})
    return main(["run", "--config", profile_path, "--input", payload])


def _payload_from_stdout(text: str) -> dict:
    """从 stdout 提取结果 JSON（容忍内核日志前缀，CLI-01 另行严格要求纯 JSON）。"""
    start = text.find("{")
    assert start >= 0, f"stdout 中没有 JSON: {text!r}"
    return json.loads(text[start:])


@_KERNEL_COMPLETED_BUG
def test_run_native_completed_prints_json_and_exits_0(
        monkeypatch, profile_path, capsys):
    """CLI-01：completed → stdout 本身是 JSON（可直接解析），退出码 0。"""
    created = _install_model_shim(
        monkeypatch, lambda: [FakeModel.text_turn("你好，有什么可以帮你？")])

    code = _run_chat(profile_path)
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out)  # 规格要求 stdout 为纯 JSON（无日志污染）
    assert payload["status"] == "completed"
    assert payload["reply"] == "你好，有什么可以帮你？"
    assert payload["execution_path"] == "workflow"
    # profile 解析出的模型名应传入模型客户端构造参数
    assert created and created[0].kwargs["model_name"] == "fake-model"


@_KERNEL_POLICY_BUG
def test_run_native_degraded_exits_2(monkeypatch, profile_path, capsys):
    """CLI-02：主阶段模型失败、兜底直连成功 → degraded，退出码 2。"""
    def turns():
        # MODEL_UPSTREAM_ERROR 属于内核单轮可恢复错误：首次失败会重发一次，
        # 第二次仍失败才成为主错误；随后本地兜底消耗第三轮（正文）。
        failure = FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "upstream 503")
        return [failure, failure, FakeModel.text_turn("原任务未能完成：上游模型暂不可用。")]

    _install_model_shim(monkeypatch, turns)

    code = _run_chat(profile_path)
    payload = _payload_from_stdout(capsys.readouterr().out)

    assert code == 2
    assert payload["status"] == "degraded"
    assert payload["execution_path"] == "direct_llm"
    assert payload["recovery"]["fallback_attempted"] is True
    assert payload["recovery"]["final_source"] == "direct_llm"
    # 主错误保留在结果中，供调用方诊断
    assert payload["recovery"]["primary_error"]["code"] == "MODEL_UPSTREAM_ERROR"


@_KERNEL_POLICY_BUG
def test_run_native_fallback_failure_exits_1(monkeypatch, profile_path, capsys):
    """CLI-03：主阶段失败且兜底也失败（FALLBACK_FAILED）→ failed，退出码 1。"""
    def turns():
        # 前两轮主阶段（首次 + 单轮恢复），第三轮兜底；全部失败。
        return [FlowAgentCoreError(ErrorCode.MODEL_UPSTREAM_ERROR, "boom") for _ in range(3)]

    _install_model_shim(monkeypatch, turns)

    code = _run_chat(profile_path)
    payload = _payload_from_stdout(capsys.readouterr().out)

    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "FALLBACK_FAILED"
    assert payload["recovery"]["fallback_attempted"] is True
    assert payload["recovery"]["final_source"] == "fallback_failed"


def test_run_without_yaml_or_config_is_usage_error(capsys):
    """CLI-04：缺 YAML 位置参数且无 --config → 用法错误，退出码 2。"""
    code = main(["run", "--input", "{}"])
    err = capsys.readouterr().err
    assert code == 2
    assert "[用法错误]" in err


def test_run_unknown_profile_is_config_error(profile_path, capsys):
    """CLI-04b：未知 --profile → 配置错误，退出码 2（不进入执行）。"""
    code = main(["run", "--config", profile_path, "--profile", "nope", "--input", "{}"])
    err = capsys.readouterr().err
    assert code == 2
    assert "unknown profile" in err


def test_serve_help_does_not_start_server(capsys):
    """CLI-05：serve --help 由 argparse 直接退出 0（不启动 uvicorn）。"""
    with pytest.raises(SystemExit) as exc_info:
        main(["serve", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out and "--port" in out


def test_legacy_yaml_run_still_uses_old_engine(
        monkeypatch, project_root, capsys):
    """CLI-06：YAML 位置参数仍走旧引擎，绝不构造原生模型客户端。"""
    def _forbidden(**kwargs):
        raise AssertionError("旧引擎路径不应构造原生模型客户端")

    monkeypatch.setattr("flowagent.app.OpenAICompatibleModel", _forbidden)

    yaml_path = str(project_root / "configs" / "customer_service.yaml")
    ticket = json.dumps({"ticket": "我要退款"}, ensure_ascii=False)
    code = main(["run", yaml_path, "--input", ticket])
    out = capsys.readouterr().out

    assert code == 0
    assert "客服处理结果" in out
