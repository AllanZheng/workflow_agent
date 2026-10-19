"""安装验收（M1-4）：构建 wheel、干净环境安装、子包齐全与三入口可用。

标记 installation：默认离线回归不执行；阶段验收显式运行：
    python -m pytest tests/test_native_installation.py -m installation -ra

边界说明（与测试方案 §5.10 一致）：
- 本阶段验证"非源码目录、非 editable、PYTHONPATH 清空"的 wheel 安装；
- 依赖隔离与 amd64 容器真实启动属 M4 最终制品验收；
- 这里用 --system-site-packages 借用已装依赖以适配离线机器，并在结论中注明。
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.installation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"  # 中文输出按 UTF-8 解码（Windows 默认 GBK 会炸）
    env.update(env_extra or {})
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env, timeout=300,
    )


def _entry_run(installed_wheel: dict, cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """以 wheel 优先的 PYTHONPATH 在源码目录外执行入口。"""
    return _run(cmd, cwd=cwd, env_extra={"PYTHONPATH": installed_wheel["pythonpath"]})


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory) -> dict:
    """构建 wheel 并安装到独立 target 目录；返回 {python, wheel, pythonpath}。

    离线机器阶段方案（与测试方案 §5.10 的偏差已注明）：
    - wheel 安装到 target 目录并置于 PYTHONPATH 首位——flowagent 解析自 wheel 而非源码；
    - 依赖由当前 venv 的 site-packages 提供（无独立依赖隔离，M4 最终制品再收紧）；
    - 入口脚本一律在源码目录外运行。
    """
    wheel_dir = tmp_path_factory.mktemp("wheel")
    build = _run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(wheel_dir)],
                 cwd=PROJECT_ROOT)
    assert build.returncode == 0, f"wheel build failed: {build.stderr[-2000:]}"
    wheels = list(wheel_dir.glob("flowagent-*.whl"))
    assert wheels, "flowagent wheel not produced"

    import site

    target = tmp_path_factory.mktemp("wheeltarget")
    install = _run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target),
                    str(wheels[0])], cwd=wheel_dir)
    assert install.returncode == 0, f"wheel install failed: {install.stderr[-2000:]}"
    # PYTHONPATH 首位 = wheel target；父 venv site-packages 提供依赖。
    sep = ";" if sys.platform == "win32" else ":"
    parent_site = sep.join(site.getsitepackages())
    pythonpath = f"{target}{sep}{parent_site}"
    return {"python": Path(sys.executable), "wheel": wheels[0], "pythonpath": pythonpath}


def test_wheel_contains_all_subpackages(installed_wheel) -> None:
    """wheel 内必须包含 V3 内核全部子包，且不含 tests/docs/benchmarks。"""
    wheel = installed_wheel["wheel"]
    names = zipfile.ZipFile(wheel).namelist()
    for required in (
        "flowagent/__init__.py", "flowagent/core/__init__.py",
        "flowagent/core/contracts.py", "flowagent/clients/__init__.py",
        "flowagent/workflows/__init__.py", "flowagent/adapters/__init__.py",
        "flowagent/app.py",
    ):
        assert required in names, f"wheel missing {required}"
    assert not any(name.startswith(("tests/", "benchmarks/", "docs/")) for name in names)


def test_python_entrypoint_from_outside_source(installed_wheel, tmp_path) -> None:
    """Python 入口：源码目录外 import 内核并跑一次完整执行（假模型）。"""
    script = tmp_path / "entry.py"
    script.write_text(
        "import asyncio\n"
        "from flowagent.core.contracts import RunInput, ModelEvent\n"
        "from flowagent.core.context import RunContext, default_budget\n"
        "from flowagent.core.runner import Runner\n"
        "\n"
        "class M:\n"
        "    async def stream(self, messages, tools, limits):\n"
        "        yield ModelEvent(kind='completed', text='ok', tool_calls=[], finish_reason='stop')\n"
        "\n"
        "async def main():\n"
        "    r = Runner(workflows={}, context_factory=lambda: RunContext(model=M(), budget=default_budget()))\n"
        "    result = await r.run(RunInput(workflow='chat', inputs={'message': 'hi'}))\n"
        "    assert result.status.value == 'completed' and result.reply == 'ok'\n"
        "    print('PYTHON-ENTRY-OK')\n"
        "\n"
        "asyncio.run(main())\n",
        encoding="utf-8",
    )
    run = _entry_run(installed_wheel, [str(installed_wheel["python"]), str(script)], tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "PYTHON-ENTRY-OK" in run.stdout


def test_cli_entrypoint_from_outside_source(installed_wheel, tmp_path) -> None:
    """CLI 入口：flowagent --version 与 run 子命令可用（服务不真实启动）。"""
    version = _entry_run(installed_wheel, [str(installed_wheel["python"]), "-m", "flowagent", "--version"], tmp_path)
    assert version.returncode == 0, version.stderr[-1000:]
    assert "flowagent" in version.stdout

    help_run = _entry_run(installed_wheel, [str(installed_wheel["python"]), "-m", "flowagent", "run", "--help"], tmp_path)
    assert help_run.returncode == 0
    assert "--config" in help_run.stdout

    help_serve = _entry_run(installed_wheel, [str(installed_wheel["python"]), "-m", "flowagent", "serve", "--help"], tmp_path)
    assert help_serve.returncode == 0


def test_http_entrypoint_from_outside_source(installed_wheel, tmp_path) -> None:
    """HTTP 入口：安装后的包可构建应用并响应 /health（不触发模型）。"""
    profile = tmp_path / "config.json"
    profile.write_text(
        '{"version": 1, "default_profile": "default", "profiles": {"default": {'
        '"model": {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "name": "m"},'
        '"workflows": {"chat": {"enabled": true}}, "limits": {"model_calls": 4}}}}',
        encoding="utf-8",
    )
    script = tmp_path / "http_entry.py"
    script.write_text(
        "from fastapi.testclient import TestClient\n"
        "from flowagent.adapters.http import build_http_app\n"
        "app = build_http_app({'config_path': r'%s', 'profile_name': 'default'})\n"
        "client = TestClient(app)\n"
        "r = client.get('/health')\n"
        "assert r.status_code == 200 and r.json()['ok'] is True\n"
        "print('HTTP-ENTRY-OK')\n" % profile,
        encoding="utf-8",
    )
    run = _entry_run(installed_wheel, [str(installed_wheel["python"]), str(script)], tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "HTTP-ENTRY-OK" in run.stdout
