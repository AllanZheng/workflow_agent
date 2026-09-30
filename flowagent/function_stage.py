"""功能阶段 flow：按阶段契约产出结构化结果（风险会商的初稿 / 终稿 / 任务清单）。

要点：
- 复用受管工具循环，可加载 Skill、调用授权范围内的知识库工具；
- 最终输出必须是符合 output_schema 的 JSON，不合规时用修复提示重试 1 次；
- knowledge.status 由真实的工具调用结果统计，不采信模型自报；
- 引用只保留本次执行中知识库工具真实返回过的文档 / 分块。
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_runtime import ManagedAgentRuntime
from .errors import FunctionStageError

WIKI_TOOLS = frozenset({
    "list_knowledge_bases", "get_knowledge_base", "list_knowledge",
    "get_knowledge", "hybrid_search", "list_chunks",
})
_LEVELS = frozenset({"high", "medium", "low"})
_STAGE_SCHEMA = {"draft": "consultation_document_v1", "final": "consultation_document_v1",
                 "tasks": "consultation_tasks_v1"}
MAX_SUMMARY_CHARS = 300


class FunctionTask(BaseModel):
    """平台下发的阶段任务：仅由可信的平台内部路径设置。"""

    model_config = ConfigDict(extra="forbid")
    function_key: Literal["risk_consultation"]
    stage: Literal["draft", "final", "tasks"]
    output_schema: Literal["consultation_document_v1", "consultation_tasks_v1"]
    inputs: dict[str, Any] = Field(default_factory=dict)

    # 阶段与输出 schema 必须一一对应，避免平台把任务阶段按稿件格式解析。
    # English: The stage and output schema must correspond.
    @model_validator(mode="after")
    def _stage_matches_schema(self):
        if _STAGE_SCHEMA[self.stage] != self.output_schema:
            raise ValueError("function_task_schema_mismatch")
        return self

    # 知识依据模式：required / optional / off。
    # English: Knowledge mode of this stage.
    @property
    def knowledge_mode(self) -> str:
        mode = self.inputs.get("knowledge_mode", "off")
        return mode if mode in {"required", "optional", "off"} else "off"


class _OutputInvalid(ValueError):
    pass


# ---- 输出契约校验 ----

def _text(value: Any, field: str, *, required: bool = True) -> str:
    # 可选文本字段允许模型输出 null。
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise _OutputInvalid(f"{field} 必须是字符串")
    value = value.strip()
    if required and not value:
        raise _OutputInvalid(f"{field} 不能为空")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _OutputInvalid(f"{field} 必须是字符串数组")
    return [item for item in value if item.strip()]


# 校验稿件输出（consultation_document_v1）。
# English: Validate a consultation_document_v1 object.
def validate_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _OutputInvalid("输出必须是 JSON 对象")
    risks = []
    for item in value.get("risks") or []:
        if not isinstance(item, dict) or item.get("level") not in _LEVELS:
            raise _OutputInvalid("risks[].level 必须是 high / medium / low")
        risks.append({"level": item["level"], "title": _text(item.get("title"), "risks[].title"),
                      "detail": _text(item.get("detail", ""), "risks[].detail", required=False)})
    citations = value.get("knowledge_citations") or []
    if not isinstance(citations, list) or any(not isinstance(item, dict) for item in citations):
        raise _OutputInvalid("knowledge_citations 必须是对象数组")
    return {
        "title": _text(value.get("title"), "title"),
        "summary": _text(value.get("summary"), "summary")[:MAX_SUMMARY_CHARS],
        "body_markdown": _text(value.get("body_markdown"), "body_markdown"),
        "risks": risks,
        "evidence_refs": _string_list(value.get("evidence_refs"), "evidence_refs"),
        "knowledge_citations": [dict(item) for item in citations],
    }


# 校验任务清单输出（consultation_tasks_v1）。
# English: Validate a consultation_tasks_v1 object.
def validate_tasks(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _OutputInvalid("输出必须是 JSON 对象")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list):
        raise _OutputInvalid("tasks 必须是数组")
    tasks = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise _OutputInvalid("tasks[] 必须是对象")
        assignee = item.get("assignee_user_id")
        if assignee is not None and (type(assignee) is not int or assignee < 1):
            raise _OutputInvalid("tasks[].assignee_user_id 必须是正整数或 null")
        due = item.get("due_date")
        if due is not None and not isinstance(due, str):
            raise _OutputInvalid("tasks[].due_date 必须是字符串或 null")
        tasks.append({
            "content": _text(item.get("content"), "tasks[].content"),
            "assignee_user_id": assignee, "due_date": due or None,
            "source_risk": _text(item.get("source_risk", ""), "tasks[].source_risk", required=False),
            "evidence_refs": _string_list(item.get("evidence_refs"), "tasks[].evidence_refs"),
        })
    return {"summary": _text(value.get("summary", ""), "summary", required=False), "tasks": tasks}


_VALIDATORS = {"consultation_document_v1": validate_document,
               "consultation_tasks_v1": validate_tasks}


# 从模型文本中提取 JSON 对象：兼容 ```json 围栏与前后说明文字。
# English: Extract a JSON object from model text.
def extract_json_object(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    try:
        return json.loads(candidate)
    except ValueError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise _OutputInvalid("输出中没有 JSON 对象") from None
        try:
            return json.loads(candidate[start:end + 1])
        except ValueError:
            raise _OutputInvalid("输出不是合法 JSON") from None


# ---- 知识依据统计 ----

def _has_hits(payload: Any) -> bool:
    data = payload.get("data") if isinstance(payload, dict) else None
    return bool(data)


# 汇总知识库工具的真实调用：返回（状态，已见文档，已见分块）。
# English: Summarize real Wiki tool calls into (status, seen docs, seen chunks).
def summarize_knowledge(events: list[dict[str, Any]]) -> tuple[str, dict[str, dict], set[tuple[str, str]]]:
    wiki = [item for item in events if item.get("remote_name") in WIKI_TOOLS]
    if not wiki:
        return "not_used", {}, set()
    docs: dict[str, dict] = {}
    chunks: set[tuple[str, str]] = set()
    succeeded = [item for item in wiki if item.get("outcome") == "success"]
    for event in succeeded:
        remote, payload = event["remote_name"], event.get("payload")
        data = payload.get("data") if isinstance(payload, dict) else None
        rows = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            if remote == "hybrid_search" or remote == "list_chunks":
                knowledge_id = row.get("knowledge_id")
                if knowledge_id is not None:
                    docs.setdefault(str(knowledge_id), {"title": row.get("knowledge_title")})
                    if row.get("id") is not None:
                        chunks.add((str(knowledge_id), str(row["id"])))
            elif remote == "get_knowledge" and row.get("id") is not None:
                docs.setdefault(str(row["id"]), {
                    "title": row.get("title"), "kb_id": row.get("knowledge_base_id")})
            elif remote == "list_knowledge" and row.get("id") is not None:
                docs.setdefault(str(row["id"]), {"title": row.get("title")})
    if any(_has_hits(item.get("payload")) for item in succeeded):
        return "ok", docs, chunks
    if succeeded:
        return "empty", docs, chunks
    return "failed", docs, chunks


# 只保留本次执行中知识库工具真实返回过的引用，并用真实标题覆盖模型给的标题。
# English: Keep only citations backed by real tool results.
def filter_citations(citations: list[dict], docs: dict[str, dict],
                     chunks: set[tuple[str, str]]) -> list[dict]:
    kept: list[dict] = []
    for item in citations:
        knowledge_id = item.get("knowledge_id")
        if knowledge_id is None or str(knowledge_id) not in docs:
            continue
        chunk_id = item.get("chunk_id")
        if chunk_id not in (None, "") and (str(knowledge_id), str(chunk_id)) not in chunks:
            continue
        info = docs[str(knowledge_id)]
        kept.append({
            "kb_id": info.get("kb_id") or item.get("kb_id"),
            "knowledge_id": str(knowledge_id),
            "chunk_id": None if chunk_id in (None, "") else str(chunk_id),
            "title": info.get("title") or item.get("title"),
        })
    return kept


# ---- 提示词与执行 ----

_DOCUMENT_SCHEMA_TEXT = (
    '{"title":"标题","summary":"不超过300字的摘要","body_markdown":"Markdown 正文",'
    '"risks":[{"level":"high|medium|low","title":"风险","detail":"说明"}],'
    '"evidence_refs":["m123"],'
    '"knowledge_citations":[{"kb_id":"库ID","knowledge_id":"文档ID","chunk_id":"分块ID","title":"标题"}]}'
)
_TASKS_SCHEMA_TEXT = (
    '{"summary":"任务概述","tasks":[{"content":"任务内容","assignee_user_id":11,'
    '"due_date":"2026-10-01","source_risk":"来源风险","evidence_refs":["m123"]}]}'
)


# 组装系统提示词：快照提示词 + 阶段指令 + 输出格式 + 知识依据要求。
# English: Build the system prompt of one stage.
def build_system_prompt(base: str, task: FunctionTask) -> str:
    sections = [base.strip()] if base.strip() else []
    if task.output_schema == "consultation_document_v1":
        sections.append(
            "你在执行风险研判会商的" + ("初稿" if task.stage == "draft" else "终稿") + "阶段。"
            "只依据输入材料写作，不得编造事实；信息不足时明确标注。"
            "最终只输出一个 JSON 对象，格式：" + _DOCUMENT_SCHEMA_TEXT +
            "。evidence_refs 只能引用输入材料中出现过的消息引用。")
    else:
        assignable = task.inputs.get("assignable_members") or []
        sections.append(
            "你在执行风险研判会商的任务分解阶段。依据已发布稿件提出任务与负责人。"
            "负责人 assignee_user_id 只能从可分配人员中选择，无法确定时填 null。"
            f"可分配人员：{json.dumps(assignable, ensure_ascii=False)}。"
            "最终只输出一个 JSON 对象，格式：" + _TASKS_SCHEMA_TEXT + "。")
    mode = task.knowledge_mode
    if mode == "required":
        sections.append("本阶段必须先加载知识库查询 Skill 并检索知识库，引用真实检索到的制度依据。")
    elif mode == "optional":
        sections.append("可先加载知识库查询 Skill 检索制度依据；检索不到时不要编造依据。")
    return "\n\n".join(sections)


# 执行一个功能阶段：返回（回复文本，metadata 片段，最后一次运行的 runtime）。
# English: Run one function stage.
async def run_function_stage(request, *, llm, mcp_client, system_prompt: str, skills,
                             skill_loader, execution_id: str, generation: int):
    task: FunctionTask = request.function_task
    validator = _VALIDATORS[task.output_schema]
    prompt = build_system_prompt(system_prompt, task)
    runtime = ManagedAgentRuntime(
        llm, mcp_client, skills=skills, skill_loader=skill_loader,
        execution_id=execution_id, generation=generation,
        tolerate_tool_errors=True, record_tool_events=True,
    )
    message = request.message
    events: list[dict[str, Any]] = []
    output: dict[str, Any] | None = None
    for attempt in range(2):
        text = await runtime.run(message, prompt, max_tool_iterations=8, history=None)
        events.extend(runtime.usage.tool_events)
        try:
            output = validator(extract_json_object(text))
            break
        except _OutputInvalid as exc:
            # 修复提示只说明不合规原因，不重复输入材料。
            message = f"{request.message}\n\n[上一次输出不合规：{exc}。请只输出符合格式的一个 JSON 对象。]"
    if output is None:
        raise FunctionStageError("invalid_structured_output")

    status, docs, chunks = summarize_knowledge(events)
    citations: list[dict] = []
    if task.output_schema == "consultation_document_v1":
        citations = filter_citations(output["knowledge_citations"], docs, chunks)
        output["knowledge_citations"] = citations
        reply = output["body_markdown"]
    else:
        lines = [output["summary"]] if output["summary"] else []
        lines += [f"{index}. {item['content']}" for index, item in enumerate(output["tasks"], 1)]
        reply = "\n".join(lines) or "本次无任务"
    function_result = {"status": "ok", "output": output,
                       "knowledge": {"status": status, "citations": citations}}
    return reply, {"function_result": function_result}, runtime
