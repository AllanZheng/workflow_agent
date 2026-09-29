"""工作流配置模型与加载器。

配置是一个 YAML/dict，顶层两段：
- workflow: 名称、描述、起始节点、递归上限
- nodes:    节点名字 -> 节点定义（type/prompt/next 等）

节点类型（type）：
- llm:      LLM 调用（system/prompt/parser/labels/mock_response）
- template: 确定性模板渲染
- tool:     Python 工具调用（tool: "module:function"，args 支持 $变量）
- human:    人工介入节点（LangGraph interrupt，配合 engine.resume 恢复）
- workflow: 子图节点，把另一份 YAML 作为节点嵌入（支持嵌套，防循环引用）

节点的流转通过 next 声明，支持三种形态：
- 字符串: 固定边            next: review
- 列表:   并行扇出          next: [compensation, notify_team]
- 路由:   条件边 + 循环护栏  next: {field: ..., rules: [...], default: ...}

可靠性字段（llm/tool/template 节点）：
- retries: 失败重试次数（默认 0）；retry_delay: 重试间隔秒（默认 1.0）
- timeout: 单次执行超时秒数（对 human 节点不生效，见 nodes.py）
"""

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from .errors import ConfigError

_MAX_SUBGRAPH_DEPTH = 8


class WhenCondition(BaseModel):
    """单条匹配条件：对路由字段应用操作符 op 与期望值 value。"""

    op: Literal[
        "eq", "ne", "gt", "ge", "lt", "le",
        "in", "contains", "regex", "exists", "true",
    ]
    value: Any = None


class OutputFieldSpec(BaseModel):
    """output_schema 中单个字段的校验规则（未标 optional 默认必填）。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "integer", "number", "boolean"] = "string"
    min: float | None = None
    max: float | None = None
    enum: list | None = None
    optional: bool = False


class RouteRule(BaseModel):
    """一条路由规则：when 命中则跳转到 goto。"""

    when: WhenCondition
    goto: str
    max_loops: int | None = None
    loop_exhaust: str | None = None


class RouteConfig(BaseModel):
    """条件边配置：按 field 取状态变量，依次匹配 rules，全不中走 default。"""

    model_config = ConfigDict(extra="forbid")

    field: str
    rules: list[RouteRule] = []
    default: str | None = None
    max_loops: int | None = None
    on_error: str | None = None


class OnErrorConfig(BaseModel):
    """节点失败降级：跳转到 goto，并写入 set 中的变量。"""

    model_config = ConfigDict(extra="forbid")

    goto: str
    set: dict[str, Any] = {}


class NodeConfig(BaseModel):
    """单个节点定义。type 决定必填字段。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm", "template", "tool", "human", "workflow", "agent"]
    description: str = ""
    # llm 节点
    system: str | None = None
    prompt: str | None = None
    parser: Literal["text", "json", "lines", "classifier"] = "text"
    labels: list[str] = []
    # llm 节点输出校验：parser=json 时解析后按此校验字段（失败关闭，评审 4.2）
    output_schema: dict[str, OutputFieldSpec] = {}
    mock_response: str | None = None
    # template 节点
    template: str | None = None
    # tool 节点
    tool: str | None = None
    args: dict[str, Any] = {}
    # human 节点
    message: str | None = None
    options: list[str] = []  # 可选答复：声明即强校验，无效答复会重新询问（失败关闭）
    # workflow 子图节点
    workflow_file: str | None = None
    # 受管 Agent 节点的工具循环上限；独立工作流的旧节点不会使用该字段。
    max_tool_iterations: int = 8
    # 可靠性：重试与超时（human 节点不适用）
    retries: int = 0
    retry_delay: float = 1.0
    backoff: float = 1.0  # 重试间隔倍率：第 n 次重试间隔 = retry_delay * backoff^(n-1)
    timeout: float | None = None
    # 公共
    output_key: str | None = None
    on_error: OnErrorConfig | None = None
    next: str | list[str] | RouteConfig

    # English: Handle next not empty operations.
    @field_validator("next")
    @classmethod
    def _next_not_empty(cls, v):
        """空目标在图编译/运行期必然失败，加载期直接拒绝（评审 4.7）。"""
        if isinstance(v, str) and not v:
            raise ValueError("next 不能为空字符串")
        if isinstance(v, list):
            if not v:
                raise ValueError("next 并行列表不能为空")
            if any(not isinstance(t, str) or not t for t in v):
                raise ValueError("next 并行列表的每个目标必须是非空字符串")
        return v

    # English: Handle next targets operations.
    @property
    def next_targets(self) -> list[str]:
        """该节点所有可能的后继目标（用于校验与图展示）。"""
        if isinstance(self.next, str):
            targets = [self.next]
        elif isinstance(self.next, list):
            targets = list(self.next)
        else:
            targets = []
            for rule in self.next.rules:
                targets.append(rule.goto)
                if rule.loop_exhaust:
                    targets.append(rule.loop_exhaust)
            if self.next.default:
                targets.append(self.next.default)
            if self.next.on_error:
                targets.append(self.next.on_error)
        if self.on_error and self.on_error.goto:
            # on_error.goto 参与引用校验：非法目标在加载期报错（评审 4.5）
            targets.append(self.on_error.goto)
        return targets


class WorkflowMeta(BaseModel):
    """workflow 元信息段。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    group_analysis_mode: Literal["legacy", "deterministic"] = "legacy"
    description: str = ""
    start: str
    recursion_limit: int = 50
    inputs: list[str] = []  # 声明的运行时输入（供 check 静态引用分析）
    # 检查点写入策略（LangGraph 1.x durability）：
    #   exit  — 每个超级步结束落盘（默认，稳妥）
    #   async — 异步落盘（吞吐优先，极端情况崩溃可能丢最后一步）
    #   all   — 每次状态变更都落盘（最稳，开销最大）
    durability: Literal["exit", "async", "all"] = "exit"


class WorkflowSpec(BaseModel):
    """完整工作流规格（含引用完整性校验）。"""

    model_config = ConfigDict(extra="forbid")

    workflow: WorkflowMeta
    nodes: dict[str, NodeConfig]

    # 检查`references`。
    # English: Check references.
    @model_validator(mode="after")
    def _check_references(self) -> "WorkflowSpec":
        nodes = self.nodes
        if not nodes:
            raise ValueError("nodes 不能为空")
        if self.workflow.start not in nodes:
            raise ValueError(f"start 指向不存在的节点: {self.workflow.start}")
        # 0 会在运行期被 LangGraph 直接拒绝，必须在加载期拦下（评审 4.7）
        if self.workflow.recursion_limit < 1:
            raise ValueError("recursion_limit 至少为 1")

        # 处理有效相关逻辑。
        # English: Handle valid operations.
        def valid(target: str) -> bool:
            return target == "END" or target in nodes

        for name, node in nodes.items():
            for target in node.next_targets:
                if not valid(target):
                    raise ValueError(f"节点 {name} 的 next 目标不存在: {target}")
            if node.type == "llm" and not node.prompt:
                raise ValueError(f"llm 节点 {name} 缺少 prompt")
            if node.type == "template" and node.template is None:
                raise ValueError(f"template 节点 {name} 缺少 template")
            if node.type == "tool" and not node.tool:
                raise ValueError(f"tool 节点 {name} 缺少 tool")
            if node.type == "human" and not node.message:
                raise ValueError(f"human 节点 {name} 缺少 message")
            if node.type == "workflow" and not node.workflow_file:
                raise ValueError(f"workflow 节点 {name} 缺少 workflow_file")
            if node.type in ("llm", "template", "tool", "agent") and not node.output_key:
                raise ValueError(f"节点 {name} 缺少 output_key")
            if node.type == "agent" and not (1 <= node.max_tool_iterations <= 8):
                raise ValueError(f"agent 节点 {name} 的 max_tool_iterations 必须在 1..8")
            if node.parser == "classifier" and not node.labels:
                raise ValueError(f"节点 {name} 使用 classifier 解析器但未配置 labels")
            if node.output_schema and node.parser != "json":
                raise ValueError(f"节点 {name} 的 output_schema 仅适用于 parser: json")
            if node.retries < 0:
                raise ValueError(f"节点 {name} 的 retries 不能为负")
            if node.retry_delay < 0:
                raise ValueError(f"节点 {name} 的 retry_delay 不能为负")
            if node.backoff < 1:
                raise ValueError(f"节点 {name} 的 backoff 不能小于 1")
            if node.timeout is not None and node.timeout <= 0:
                raise ValueError(f"节点 {name} 的 timeout 必须为正数")
            if isinstance(node.next, RouteConfig):
                route = node.next
                if not route.rules and not route.default:
                    raise ValueError(f"节点 {name} 的路由既无 rules 也无 default")
                if route.max_loops is not None and route.max_loops < 1:
                    raise ValueError(f"节点 {name} 的路由级 max_loops 至少为 1")
                for rule in route.rules:
                    if rule.max_loops is not None:
                        if rule.max_loops < 1:
                            raise ValueError(f"节点 {name} 的 max_loops 至少为 1")
                        if not (rule.loop_exhaust or route.default):
                            # 护栏耗尽时若既无 loop_exhaust 也无 default，运行期无路可走
                            raise ValueError(
                                f"节点 {name} 配置 max_loops 的规则必须提供 "
                                f"loop_exhaust 或路由 default"
                            )
                    if rule.when.op == "regex":
                        try:
                            re.compile(str(rule.when.value))
                        except re.error as e:
                            raise ValueError(f"节点 {name} 的路由 regex 无效: {e}") from e
        return self


# English: Load spec.
def load_spec(data: dict) -> WorkflowSpec:
    """从 dict 加载并校验工作流规格。"""
    try:
        return WorkflowSpec.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"工作流配置非法:\n{e}") from e


# 解析`sub`路径。
# English: Resolve sub path.
def _resolve_sub_path(base_dir: Path, ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else (base_dir / p)


_TOKEN_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?:\.[A-Za-z0-9_]+)*\}")


# English: Handle analyze references operations.
def analyze_references(spec: WorkflowSpec) -> list[str]:
    """静态引用分析：返回模板/提示词中引用了、但无产出来源的变量名（根段）。

    来源包括：workflow.inputs 声明的输入、各节点 output_key、on_error.set 键。
    workflow 子图未声明 output_key 时变量直并，来源未知，不参与判断。
    """
    produced: set[str] = set(spec.workflow.inputs or [])
    for node in spec.nodes.values():
        if node.output_key:
            produced.add(node.output_key)
        if node.on_error:
            produced.update(node.on_error.set.keys())
        if node.type == "human" and not node.output_key:
            produced.add("human_decision")

    refs: set[str] = set()
    for node in spec.nodes.values():
        for text in (node.template, node.prompt, node.system, node.message):
            if text:
                for match in _TOKEN_RE.finditer(text):
                    refs.add(match.group(1))
        if isinstance(node.next, RouteConfig):
            refs.add(node.next.field.split(".")[0])
        if node.type == "tool":
            for value in (node.args or {}).values():
                if isinstance(value, str) and value.startswith("$"):
                    refs.add(value[1:].split(".")[0])
    return sorted(refs - produced)


# English: Check spec.
def check_spec(spec: WorkflowSpec, _inherited: set[str] | None = None) -> list[str]:
    """完整性检查（供 CLI check）：静态引用分析 + 子图输入交叉检查。

    子图检查：子图声明的 workflow.inputs 加上其内部未解析引用，必须能被
    父级上下文（父级 inputs、各节点 output_key、on_error.set）供给。
    递归遍历所有嵌套层级，深层子图的输入链断裂同样会被报告
    （前缀标注完整链路）；每一层子图都应显式声明自己的 inputs。
    """
    warnings = analyze_references(spec)

    known: set[str] = set(_inherited or ()) | set(spec.workflow.inputs or [])
    for node in spec.nodes.values():
        if node.output_key:
            known.add(node.output_key)
        if node.on_error:
            known.update(node.on_error.set.keys())
        if node.type == "human" and not node.output_key:
            known.add("human_decision")

    for name, node in spec.nodes.items():
        if node.type != "workflow" or not node.workflow_file:
            continue
        try:
            sub = load_spec_file(node.workflow_file)
        except ConfigError:
            continue  # 文件级问题已由加载流程报告，避免重复
        required = set(sub.workflow.inputs or []) | set(analyze_references(sub))
        for missing in sorted(required - known):
            warnings.append(
                f"子图节点 {name}（{Path(node.workflow_file).name}）"
                f"需要变量 {missing!r}，父级上下文未提供"
            )
        # 递归检查更深层子图：本层可见变量（known）作为子层的外部上下文
        sub_chain = f"{name}<{sub.workflow.name}>"
        for sub_warning in check_spec(sub, _inherited=known):
            warnings.append(f"[链路 {sub_chain}] {sub_warning}")
    return warnings


# English: Load spec file.
def load_spec_file(path: str | Path, _stack: tuple = ()) -> WorkflowSpec:
    """从 YAML 文件加载工作流规格。

    额外职责：
    - 把 workflow 子图节点的 workflow_file 改写为绝对路径（运行期不再依赖 CWD）；
    - 递归校验子图配置可加载，检测循环引用与超深嵌套。
    """
    p = Path(path).resolve()
    if p in _stack:
        chain = " -> ".join(str(s) for s in (*_stack, p))
        raise ConfigError(f"检测到工作流循环引用: {chain}")
    if len(_stack) >= _MAX_SUBGRAPH_DEPTH:
        raise ConfigError(f"工作流嵌套深度超过 {_MAX_SUBGRAPH_DEPTH} 层: {p}")
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {p}")
    with open(p, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML 解析失败: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层应为映射，实际为: {type(data).__name__}")
    spec = load_spec(data)

    # 子图节点：改写为绝对路径并递归校验（循环引用/深度）
    for name, node in spec.nodes.items():
        if node.type != "workflow":
            continue
        sub_path = _resolve_sub_path(p.parent, node.workflow_file)
        node.workflow_file = str(sub_path)
        try:
            load_spec_file(sub_path, _stack + (p,))
        except ConfigError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ConfigError(f"节点 {name} 的子图加载失败: {sub_path} ({e})") from e
    return spec
