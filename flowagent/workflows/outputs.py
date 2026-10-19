"""业务输出 schema 与校验：会商文档/任务、知识引用与覆盖口径。

规则冻结来源：技术方案 §4.5 + 测试方案 §5.8（迁移 fixture 表）。
"""
from __future__ import annotations

import re
from typing import Any

MAX_SUMMARY_CHARS = 300
RISK_LEVELS = ("high", "medium", "low")
WIKI_TOOL_NAMES = frozenset({
    "list_knowledge_bases", "get_knowledge_base", "list_knowledge",
    "get_knowledge", "hybrid_search", "list_chunks",
})


class OutputInvalid(ValueError):
    """业务输出未通过 schema（进入一次无工具格式修复，耗尽交错误边界）。"""


def validate_consultation_document(value: Any) -> dict[str, Any]:
    """consultation_document_v1：title/summary/body/risks/evidence/citations。"""
    if not isinstance(value, dict):
        raise OutputInvalid("document must be object")
    title = str(value.get("title") or "").strip()
    summary = str(value.get("summary") or "").strip()[:MAX_SUMMARY_CHARS]
    body = str(value.get("body_markdown") or "").strip()
    if not title or not summary or not body:
        raise OutputInvalid("title/summary/body_markdown required")
    risks_out = []
    for risk in value.get("risks") or []:
        if not isinstance(risk, dict):
            raise OutputInvalid("risk must be object")
        level = risk.get("level")
        if level not in RISK_LEVELS:
            raise OutputInvalid(f"risk.level must be one of {RISK_LEVELS}")
        risk_title = str(risk.get("title") or "").strip()
        if not risk_title:
            raise OutputInvalid("risk.title required")
        risks_out.append({
            "level": level, "title": risk_title,
            "detail": str(risk.get("detail") or ""),
        })
    evidence = _string_list(value.get("evidence_refs"))
    citations = _citation_list(value.get("knowledge_citations"))
    return {
        "title": title, "summary": summary, "body_markdown": body,
        "risks": risks_out, "evidence_refs": evidence, "knowledge_citations": citations,
    }


def validate_consultation_tasks(value: Any) -> dict[str, Any]:
    """consultation_tasks_v1：任务数组（空数组=本次无任务）；负责人/期限规则。"""
    if not isinstance(value, dict):
        raise OutputInvalid("tasks output must be object")
    summary = str(value.get("summary") or "")
    tasks = value.get("tasks")
    if not isinstance(tasks, list):
        raise OutputInvalid("tasks must be array")
    tasks_out = []
    for task in tasks:
        if not isinstance(task, dict):
            raise OutputInvalid("task must be object")
        content = str(task.get("content") or "").strip()
        if not content:
            raise OutputInvalid("task.content required")
        tasks_out.append({
            "content": content,
            "assignee_user_id": _assignee(task.get("assignee_user_id")),
            "due_date": _due_date(task.get("due_date")),
            "source_risk": str(task.get("source_risk") or ""),
            "evidence_refs": _string_list(task.get("evidence_refs")),
        })
    return {"summary": summary, "tasks": tasks_out}


def _assignee(raw: Any) -> int | None:
    """正整数或 null；bool/0/负数拒绝（类型错误交格式修复）。"""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise OutputInvalid("assignee_user_id must be positive int or null")
    if isinstance(raw, int):
        if raw <= 0:
            raise OutputInvalid("assignee_user_id must be positive int or null")
        return raw
    if isinstance(raw, str) and raw.strip().isdigit() and int(raw) > 0:
        return int(raw)
    raise OutputInvalid("assignee_user_id must be positive int or null")


def _due_date(raw: Any) -> str | None:
    """字符串或 null；空串归一 null；无法解析的日期置空（不生成错误期限）。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    return None


def _string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if item is not None]


def _citation_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict) or item.get("knowledge_id") is None:
            continue
        out.append({
            "kb_id": item.get("kb_id"),
            "knowledge_id": str(item.get("knowledge_id")),
            "chunk_id": None if item.get("chunk_id") in (None, "") else str(item.get("chunk_id")),
            "title": item.get("title"),
        })
    return out


def summarize_knowledge_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """从真实工具轨迹汇总知识状态与可见文档/分块（模型自报不采信）。

    返回 {status, docs, chunks}；status ∈ ok/empty/failed/not_used。
    """
    def _resolved(event: dict[str, Any]) -> str:
        name = str(event.get("remote_name") or event.get("name") or "")
        return name.split("__", 1)[-1]

    wiki = [e for e in events if _resolved(e) in WIKI_TOOL_NAMES]
    if not wiki:
        return {"status": "not_used", "docs": {}, "chunks": set()}
    docs: dict[str, dict] = {}
    chunks: set[tuple[str, str]] = set()
    succeeded = [e for e in wiki if e.get("status") == "success"]
    has_hits = False
    for event in succeeded:
        payload = event.get("result_value")
        data = payload.get("data") if isinstance(payload, dict) else None
        rows = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        if rows:
            has_hits = True
        for row in rows:
            if not isinstance(row, dict):
                continue
            remote = _resolved(event)
            knowledge_id = row.get("knowledge_id")
            if remote in ("hybrid_search", "list_chunks") and knowledge_id is not None:
                docs.setdefault(str(knowledge_id), {"title": row.get("knowledge_title")})
                if row.get("id") is not None:
                    chunks.add((str(knowledge_id), str(row["id"])))
            elif remote in ("get_knowledge", "list_knowledge") and row.get("id") is not None:
                docs.setdefault(str(row["id"]), {"title": row.get("title")})
    if has_hits:
        return {"status": "ok", "docs": docs, "chunks": chunks}
    if succeeded:
        return {"status": "empty", "docs": docs, "chunks": chunks}
    return {"status": "failed", "docs": docs, "chunks": chunks}


def filter_citations(
    citations: list[dict[str, Any]], knowledge: dict[str, Any]
) -> list[dict[str, Any]]:
    """只保留本次真实取得证据支撑的引用；标题用真实标题覆盖。"""
    docs, chunks = knowledge.get("docs") or {}, knowledge.get("chunks") or set()
    kept = []
    for item in citations:
        knowledge_id = str(item.get("knowledge_id") or "")
        if knowledge_id not in docs:
            continue
        chunk_id = item.get("chunk_id")
        if chunk_id not in (None, "") and (knowledge_id, str(chunk_id)) not in chunks:
            continue
        info = docs[knowledge_id]
        kept.append({
            "kb_id": item.get("kb_id"),
            "knowledge_id": knowledge_id,
            "chunk_id": None if chunk_id in (None, "") else str(chunk_id),
            "title": info.get("title") or item.get("title"),
        })
    return kept
