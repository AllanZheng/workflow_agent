"""原生 V3 内核架构边界测试（纯静态 AST 分析，离线、无网络、无 LLM）。

覆盖测试方案行为 ID：I02（基础 import/AST 边界部分；wheel 全模块扫描与
amd64 镜像烟测属 native_installation，不在本文件范围）。

I02 在本文件内拆为以下可定位子项（ARCH-xx 仅为本文件内编号）：
- ARCH-01 core 包任意层级不得导入外部框架：
          fastapi / openai / mcp / langgraph / langchain / uvicorn。
          （pydantic 属允许依赖，不在禁止清单内。）
- ARCH-02 core 包不得反向依赖编排层：
          flowagent.workflows / flowagent.adapters / flowagent.execution_service /
          agents_platform / langchain / langgraph（任意层级）。
- ARCH-03 clients 对 openai / mcp 的依赖必须保持惰性：模块顶层零导入，
          函数体内保留导入（缺依赖不影响进程启动、core 不被 SDK 污染）。
- ARCH-04 workflows 只依赖 core，不得导入 adapters 或平台模块
          （flowagent.adapters / flowagent.execution_service / flowagent.platform /
          agents_platform，任意层级）。
- ARCH-05 M1 过渡契约：flowagent/__init__.py 仍导出旧引擎（config/engine/llm/state）；
          旧引擎在 M4 拆除前允许存在，本测试固定该现状，M4 时同步改写。
- ARCH-06 flowagent.core.__init__ 与 flowagent.app 不得导入 execution_service
          （新内核与旧执行服务的硬解耦边界）。
- ARCH-07 每个内核文件必须 < 1000 行（单文件可读性硬上限）。
- ARCH-08 静态检查自检：被扫描文件非空、可解析且确实含导入语句，
          防止边界断言因扫描目标为空而空转。

实现说明：只读取源码文本做 AST 解析，不导入被测模块，因此与依赖安装状态无关。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOWAGENT_DIR = REPO_ROOT / "flowagent"
CORE_DIR = FLOWAGENT_DIR / "core"
CLIENTS_DIR = FLOWAGENT_DIR / "clients"
WORKFLOWS_DIR = FLOWAGENT_DIR / "workflows"
ADAPTERS_DIR = FLOWAGENT_DIR / "adapters"

CORE_FILES = sorted(CORE_DIR.glob("*.py"))
CLIENT_FILES = sorted(CLIENTS_DIR.glob("*.py"))
WORKFLOW_FILES = sorted(WORKFLOWS_DIR.glob("*.py"))

# 内核文件清单（与任务 KERNEL MAP 对齐）：core/clients/workflows/adapters + 装配与入口。
KERNEL_FILES = (
    CORE_FILES
    + CLIENT_FILES
    + WORKFLOW_FILES
    + sorted(ADAPTERS_DIR.glob("*.py"))
    + [FLOWAGENT_DIR / "app.py", FLOWAGENT_DIR / "config.py", FLOWAGENT_DIR / "cli.py"]
)

# core 内任意层级都禁止的外部框架（SDK/服务框架/旧编排引擎）。
FRAMEWORKS_FORBIDDEN_IN_CORE = (
    "fastapi",
    "openai",
    "mcp",
    "langgraph",
    "langchain",
    "uvicorn",
)

# core 禁止反向依赖的上层模块（依赖方向必须单向：上层 -> core）。
# agents_platform / agent_platform 两种拼写的平台包名都在禁止之列（技术方案 §依赖边界）。
LAYERS_FORBIDDEN_IN_CORE = (
    "flowagent.workflows",
    "flowagent.adapters",
    "flowagent.execution_service",
    "agents_platform",
    "agent_platform",
    "langchain",
    "langgraph",
)

# workflows 只允许依赖 core，不得触碰适配层与平台模块。
PLATFORM_FORBIDDEN_IN_WORKFLOWS = (
    "flowagent.adapters",
    "flowagent.execution_service",
    "flowagent.platform",
    "agents_platform",
    "agent_platform",
)

SDK_MODULES_LAZY_IN_CLIENTS = ("openai", "mcp")


# --------------------------------------------------------------------------
# AST 辅助
# --------------------------------------------------------------------------
def _parse(path: Path) -> ast.Module:
    """读取并解析源码；解析失败会直接暴露为测试失败（语法边界也是边界）。"""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _package_parts(path: Path) -> list[str]:
    """模块所属包的绝对分段：flowagent/core/runner.py -> [flowagent, core]。

    __init__.py 去掉文件名后缀后同样得到其所在包目录。
    """
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return parts[:-1]


def _resolve_from(path: Path, node: ast.ImportFrom) -> list[str]:
    """把 ImportFrom 解析成绝对模块名列表（相对导入按文件所在包上溯 level-1 层）。

    `from . import x` 会把 x 展开为包内子模块名，便于后续前缀匹配。
    """
    if node.level:
        package = _package_parts(path)
        drop = node.level - 1
        package = package[: max(0, len(package) - drop)]
        base = ".".join(package + ([node.module] if node.module else []))
    else:
        base = node.module or ""
    modules = [base] if base else []
    if node.module is None:
        # 例：from . import config -> flowagent.config
        modules.extend(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return modules


def _import_entries(path: Path, *, module_body_only: bool = False):
    """产出 (绝对模块名, 是否位于模块顶层)。

    module_body_only=True 时只扫描模块体直接子节点，用于“顶层导入”断言；
    默认 ast.walk 扫描任意层级（含函数体、if、try 分支内部导入）。
    """
    tree = _parse(path)
    top_ids = {id(node) for node in tree.body}
    nodes = tree.body if module_body_only else list(ast.walk(tree))
    for node in nodes:
        is_top = id(node) in top_ids
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, is_top
        elif isinstance(node, ast.ImportFrom):
            for module in _resolve_from(path, node):
                yield module, is_top


def _matches(module: str, prefix: str) -> bool:
    """模块等于前缀或以 `前缀.` 开头（避免 mcp 误匹配 mcp_utils 之类的同名前缀）。"""
    return module == prefix or module.startswith(prefix + ".")


def _violations(
    path: Path, forbidden: tuple[str, ...], *, module_body_only: bool = False
) -> list[str]:
    """返回命中禁止清单的全部模块名（空列表表示边界成立）。"""
    return [
        module
        for module, _is_top in _import_entries(path, module_body_only=module_body_only)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]


def _nested_imports(path: Path, prefixes: tuple[str, ...]) -> list[str]:
    """返回函数/条件块等非模块顶层位置的导入（惰性导入证据）。"""
    return [
        module
        for module, is_top in _import_entries(path)
        if not is_top and any(_matches(module, prefix) for prefix in prefixes)
    ]


# --------------------------------------------------------------------------
# ARCH-01 / ARCH-02：core 包纯内核边界
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_arch01_core_has_no_framework_imports(path: Path) -> None:
    """core 任意层级不得导入 fastapi/openai/mcp/langgraph/langchain/uvicorn。"""
    assert _violations(path, FRAMEWORKS_FORBIDDEN_IN_CORE) == [], (
        f"{path.relative_to(REPO_ROOT)} 导入了外部框架，违反 core 纯内核边界"
    )


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_arch02_core_does_not_import_upper_layers(path: Path) -> None:
    """core 不得反向依赖 workflows/adapters/execution_service/平台模块（任意层级）。"""
    assert _violations(path, LAYERS_FORBIDDEN_IN_CORE) == [], (
        f"{path.relative_to(REPO_ROOT)} 反向依赖了编排层，依赖方向必须是上层 -> core"
    )


# --------------------------------------------------------------------------
# ARCH-03：clients 的 SDK 导入保持惰性
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path", [CLIENTS_DIR / "model.py", CLIENTS_DIR / "mcp.py"], ids=lambda p: p.name
)
def test_arch03_clients_have_no_top_level_sdk_imports(path: Path) -> None:
    """openai / mcp 只允许出现在函数体内，模块顶层导入会使缺依赖时无法启动。"""
    top_level = _violations(path, SDK_MODULES_LAZY_IN_CLIENTS, module_body_only=True)
    assert top_level == [], (
        f"{path.name} 在模块顶层导入了 SDK {top_level}；必须保持函数内惰性导入"
    )


@pytest.mark.parametrize(
    "path", [CLIENTS_DIR / "model.py", CLIENTS_DIR / "mcp.py"], ids=lambda p: p.name
)
def test_arch03_clients_still_import_sdk_lazily(path: Path) -> None:
    """同时确认函数内导入确实存在——否则上一条“无顶层导入”会因检查空转而失真。"""
    nested = _nested_imports(path, SDK_MODULES_LAZY_IN_CLIENTS)
    assert nested, f"{path.name} 未发现函数内 SDK 导入，惰性边界检查可能失效"


# --------------------------------------------------------------------------
# ARCH-04：workflows 不依赖适配层/平台模块
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_arch04_workflows_do_not_import_adapters_or_platform(path: Path) -> None:
    """workflows 只允许依赖 flowagent.core；适配层装配属于入口职责。"""
    assert _violations(path, PLATFORM_FORBIDDEN_IN_WORKFLOWS) == [], (
        f"{path.relative_to(REPO_ROOT)} 导入了适配层或平台模块"
    )


# --------------------------------------------------------------------------
# ARCH-05：M1 过渡期包入口仍导出旧引擎
# --------------------------------------------------------------------------
def test_arch05_package_entry_keeps_old_engine_imports_for_m1() -> None:
    """M1 现状固定：旧引擎在 M4 拆除前允许由包入口导出（config/engine/llm/state）。"""
    path = FLOWAGENT_DIR / "__init__.py"
    tree = _parse(path)
    module_level = {
        module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for module in _resolve_from(path, node)
    }
    expected = {"flowagent.config", "flowagent.engine", "flowagent.llm", "flowagent.state"}
    assert expected <= module_level, (
        f"包入口缺少旧引擎模块导出：{sorted(expected - module_level)}（M4 拆除时同步更新本断言）"
    )


# --------------------------------------------------------------------------
# ARCH-06：内核入口与 execution_service 解耦
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [CORE_DIR / "__init__.py", FLOWAGENT_DIR / "app.py"],
    ids=lambda p: str(p.relative_to(FLOWAGENT_DIR)).replace("\\", "/"),
)
def test_arch06_kernel_entrypoints_do_not_import_execution_service(path: Path) -> None:
    """新内核入口（core 包与 app 装配）不得拉入旧执行服务（任意层级）。"""
    forbidden = ("flowagent.execution_service", "execution_service")
    assert _violations(path, forbidden) == [], (
        f"{path.name} 导入了 execution_service，破坏新内核与旧执行服务的解耦边界"
    )


# --------------------------------------------------------------------------
# ARCH-07：内核单文件行数上限
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path", KERNEL_FILES,
    ids=lambda p: str(p.relative_to(FLOWAGENT_DIR)).replace("\\", "/"),
)
def test_arch07_kernel_file_under_1000_lines(path: Path) -> None:
    """内核文件必须 < 1000 行：超出即需拆分，保持单文件可通读。"""
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert line_count < 1000, (
        f"{path.relative_to(REPO_ROOT)} 共 {line_count} 行，超过内核单文件 1000 行上限"
    )


# --------------------------------------------------------------------------
# ARCH-08：静态检查自检（防空转）
# --------------------------------------------------------------------------
def test_arch08_scan_targets_are_nonempty_and_parseable() -> None:
    """扫描目标非空、可解析且确实含导入语句，否则上述边界断言毫无意义。"""
    assert CORE_FILES, "未发现 flowagent/core/*.py，ARCH-01/02 将空转"
    assert CLIENT_FILES, "未发现 flowagent/clients/*.py，ARCH-03 将空转"
    assert WORKFLOW_FILES, "未发现 flowagent/workflows/*.py，ARCH-04 将空转"

    core_names = {path.name for path in CORE_FILES}
    assert {"__init__.py", "contracts.py", "context.py", "agent.py", "runner.py"} <= core_names, (
        f"core 文件清单不完整：{sorted(core_names)}"
    )

    for path in KERNEL_FILES:
        tree = _parse(path)
        assert any(
            isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)
        ), f"{path.relative_to(REPO_ROOT)} 没有任何导入语句，边界检查会空转"
