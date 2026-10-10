"""Deterministic group read, validated classification and report assembly."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .mcp_edge_client import McpEdgeError
from .template import render_template

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ("问题", "通知", "讨论", "闲聊", "其他")


# 处理JSON`object`相关逻辑。
# English: Handle JSON object operations.
def _json_object(content: Any) -> dict:
    if isinstance(content, list):
        content = "".join(str(item) for item in content)
    value = json.loads(str(content))
    if not isinstance(value, dict):
        raise ValueError("model_json_object_required")
    return value


# 处理`invoke`JSON相关逻辑。
# English: Handle invoke JSON operations.
async def _invoke_json(llm, system: str, prompt: str, *, use_async: bool = False) -> dict:
    messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
    # V3 授权模型只提供异步接口（同步 invoke 会绕过授权），因此由调用方指定。
    if use_async:
        response = await llm.ainvoke(messages)
    else:
        response = await asyncio.to_thread(llm.invoke, messages)
    return _json_object(response.content)


# 处理`taxonomy`相关逻辑。
# English: Handle taxonomy operations.
def _taxonomy(override: str) -> tuple[str, ...]:
    if not override:
        return DEFAULT_CATEGORIES
    # 分类指令的末尾“分类”不能截断前面的名称；仅去除指令前后缀。
    source = re.sub(r"(?:分类|分析|统计)\s*[。.!！]?$", "", override.strip())
    source = re.sub(r"(?<=类)(?:分类|分析|统计)", "", source)
    source = re.sub(r"^(?:按[^类、，,/;；]{0,16}分(?:为)?|分类为|分类[:：]|分为|按)\s*", "", source)
    names = list(dict.fromkeys(re.findall(r"[\u4e00-\u9fffA-Za-z]{1,16}类", source)))
    if not 2 <= len(names) <= 8:
        raise ValueError("taxonomy_needs_explicit_categories")
    return tuple(names) + ("未归类",)


def requested_taxonomy(message: str) -> str:
    # 群消息前缀也含“按……分析”；跳过没有明确类别的片段，继续查找用户指令。
    for match in re.finditer(r"按[^。！？\n]{1,180}?(?:分类|分析|统计)", message):
        try:
            return "、".join(_taxonomy(match.group())[:-1])
        except ValueError:
            continue
    return ""


# 处理`batches`相关逻辑。
# English: Handle batches operations.
def _batches(messages: list[dict], max_chars: int = 12000,
             max_count: int = 100) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    length = 0
    for message in messages:
        size = len(message["text"])
        if size > max_chars:
            raise ValueError("single_message_too_large")
        if current and (length + size > max_chars or len(current) >= max_count):
            batches.append(current)
            current, length = [], 0
        current.append(message)
        length += size
    if current:
        batches.append(current)
    return batches


# 处理`analyze`群组相关逻辑。
# English: Handle analyze group operations.
async def analyze_group(request, spec, llm, mcp_client) -> tuple[str, dict]:
    """Resolve a named group through the platform, then read its bound ID."""
    target = request.analysis_target
    if mcp_client is None:
        raise ValueError("group_analysis_mcp_required")
    parse_node = spec.nodes["parse_args"]
    context = {"message": request.message, "history": request.history}
    parsed = await _invoke_json(
        llm, parse_node.system or "", render_template(parse_node.prompt or "", context),
    )
    # 旧工作流同样以明确的原始分类为准，避免模型遗漏或替换类别。
    explicit_taxonomy = requested_taxonomy(request.message)
    if explicit_taxonomy:
        parsed["taxonomy_override"] = explicit_taxonomy
    # report 节点延迟取值：提前返回的分支不要求 spec 中存在 report 节点。
    return await run_group_analysis(
        parsed, target, llm, mcp_client.call_tool,
        lambda: spec.nodes["report"].system or "",
    )


# 已解析参数之后的确定性流水线：定位群、取消息、分批归类、程序统计、生成报告。
# V2 路径与 V3 终端工具共用这份实现，计数与比例永远由程序计算。
# English: Deterministic pipeline after arguments are parsed (shared by V2 and the V3 tool).
async def run_group_analysis(parsed: dict, target, llm, call_tool, report_system,
                             *, use_async: bool = False) -> tuple[str, dict]:
    for key in ("start", "end", "taxonomy_override"):
        if not isinstance(parsed.get(key, ""), str):
            raise ValueError(f"invalid_{key}")
    days = parsed.get("days")
    if days is not None and (type(days) is not int or not 1 <= days <= 31):
        raise ValueError("invalid_days")
    target_id = target.conversation_id if target is not None else None
    if target_id is None:
        group_query = parsed.get("group_query", "")
        group_id = parsed.get("group_id")
        if not isinstance(group_query, str) or len(group_query.strip()) > 100:
            raise ValueError("invalid_group_query")
        if group_id is not None and (type(group_id) is not int or group_id < 1):
            raise ValueError("invalid_group_id")
        if not group_query.strip() and group_id is None:
            return "请选择要分析的群，或告诉我群名称。", {"analysis_result": None}
        selection = {"name_query": group_query.strip()}
        if group_id is not None:
            selection["conversation_id"] = group_id
        try:
            resolved = await call_tool("platform_resolve_group", selection)
        except McpEdgeError as exc:
            logger.warning("group resolution MCP failed: %s", exc)
            return "暂时无法查找群，请稍后重试。", {"analysis_result": None}
        if not isinstance(resolved, dict):
            raise ValueError("group_resolution_invalid")
        if resolved.get("status") == "not_found":
            return "没有找到你可访问的目标群，请确认群名。", {"analysis_result": None}
        if resolved.get("status") == "ambiguous":
            candidates = resolved.get("candidates") or []
            choices = "、".join(
                f"{item['name']}（ID {item['conversation_id']}）"
                for item in candidates
                if isinstance(item, dict) and isinstance(item.get("name"), str)
                and type(item.get("conversation_id")) is int
            )
            return f"找到多个同名或相近的群：{choices}。请说明要分析的群 ID。", {"analysis_result": None}
        target_id = resolved.get("conversation_id")
        if resolved.get("status") != "matched" or type(target_id) is not int or target_id < 1:
            raise ValueError("group_resolution_invalid")
    try:
        categories = _taxonomy(parsed.get("taxonomy_override") or "")
    except ValueError:
        return "请明确写出 2 到 8 个分类名称，例如“按技术类、管理类分析”。", {"analysis_result": None}
    arguments = {"conversation_id": target_id}
    for key in ("start", "end", "days"):
        if parsed.get(key) not in (None, ""):
            arguments[key] = parsed[key]
    try:
        raw = await call_tool("fetch_group_analysis_input", arguments)
    except McpEdgeError as exc:
        # 参数校验失败应交回模型修正，不能误报为 IM 服务不可用。
        if exc.code == "invalid_arguments":
            raise McpEdgeError(
                "Invalid group analysis time window. Use ISO 8601 start/end with timezone "
                "(Beijing example: 2026-10-10T13:00:00+08:00), start before end, "
                "at most 31 days; omit days for an absolute window.", code=exc.code,
            ) from exc
        return (
            "抱歉，暂时无法获取群消息，请稍后重试或联系管理员检查 im-tools-mcp。",
            {"analysis_result": {
                "conversation_id": target_id,
                "window_start": None, "window_end_exclusive": None,
                "requested_taxonomy": {"text": parsed.get("taxonomy_override") or ""},
                "effective_taxonomy": {}, "statistics": {}, "coverage": {},
                "status": "failed", "partial_reasons": ["im_tool_unavailable"],
            }},
        )
    if not isinstance(raw, dict) or raw.get("conversation_id") != target_id:
        raise ValueError("analysis_target_mismatch")
    messages = raw.get("messages")
    if not isinstance(messages, list):
        raise ValueError("analysis_messages_invalid")
    refs = [item.get("message_ref") for item in messages]
    if len(refs) != len(set(refs)) or any(not isinstance(ref, str) for ref in refs):
        raise ValueError("analysis_message_refs_invalid")

    assignments: dict[str, str] = {}
    failed = 0
    batch_summaries: list[str] = []
    for batch in _batches(messages):
        payload = [{"message_ref": item["message_ref"], "text": item["text"]}
                   for item in batch]
        prompt = ("按给定类别给每条消息恰好分配一类，并简述本批事项，只输出 JSON："
                  '{"assignments":[{"message_ref":"m1","category":"类别"}],"summary":"事项"}。'
                  # 演练标记只说明材料性质，不能掩盖演练中的状态通报、异常与待办。
                  "演练或虚构材料也按其内容分类，不能仅因是演练而归入其他类。"
                  "类别中若含风险类、通知类、待办类，分别依据异常隐患、状态通报、责任与时限选择主类。"
                  f"类别：{list(categories)}。消息：{json.dumps(payload, ensure_ascii=False)}")
        expected = {item["message_ref"] for item in batch}
        for attempt in range(2):
            try:
                result = await _invoke_json(llm, "你是群消息分类器，不编造消息引用。", prompt,
                                            use_async=use_async)
                items = result["assignments"]
                if not isinstance(items, list) or len(items) != len(batch):
                    raise ValueError("classification_count_mismatch")
                proposed = {item["message_ref"]: item["category"] for item in items}
                if (len(proposed) != len(items) or set(proposed) != expected
                        or any(category not in categories for category in proposed.values())):
                    raise ValueError("classification_invalid")
                assignments.update(proposed)
                summary = result.get("summary")
                if isinstance(summary, str) and summary.strip():
                    batch_summaries.append(summary[:1000])
                break
            except (KeyError, TypeError, ValueError):
                if attempt:
                    failed += len(batch)

    counts = Counter(assignments.values())
    total = len(assignments)
    examples: dict[str, list[str]] = {category: [] for category in categories}
    for item in messages:
        category = assignments.get(item["message_ref"])
        if category and len(examples[category]) < 2:
            examples[category].append(item["text"])
    statistics = {
        "total_messages": total,
        "categories": [{"name": category, "count": counts[category],
                        "ratio": round(counts[category] * 100 / total, 1) if total else 0,
                        "quotes": examples[category]} for category in categories],
    }
    report_messages = [SystemMessage(content=report_system() or ""),
                       HumanMessage(content=(
                           "请输出群分析报告。仅写总结和汇报要点；统计表由程序提供。"
                           # 报告保留演练事实，禁止模型把材料性质扩写为无依据的管理评价。
                           "汇总消息里的具体数值、设备状态、责任人和时限；明确演练性质。"
                           "不得添加材料未支持的沟通纪律、信息噪音或团队管理建议，也不能把演练异常描述为无风险。"
                           f"分类统计：{json.dumps(statistics, ensure_ascii=False)}。"
                           f"分批纪要：{json.dumps(batch_summaries, ensure_ascii=False)}"
                       ))]
    if use_async:
        overview = await llm.ainvoke(report_messages)
    else:
        overview = await asyncio.to_thread(llm.invoke, report_messages)
    candidate = str(overview.content)
    summary = candidate.split("## 分类统计", 1)[0].strip()
    if "## 汇报要点" in candidate:
        key_points = "## 汇报要点" + candidate.rsplit("## 汇报要点", 1)[1]
    else:
        key_points = "## 汇报要点\n- 本次分析无额外要点"
    if not summary.startswith("## 总结"):
        summary = "## 总结\n" + summary
    lines = ["## 分类统计"]
    for category in statistics["categories"]:
        lines.append(f"- {category['name']}：{category['count']} 条（{category['ratio']}%）")
        lines.extend(f"  - {quote}" for quote in category["quotes"])
    partial_reasons = list(raw.get("partial_reasons") or [])
    if raw.get("status") != "complete" and not partial_reasons:
        partial_reasons.append("input_incomplete")
    if failed:
        partial_reasons.append("classification_failed")
    if partial_reasons:
        summary += "\n\n⚠ 覆盖情况：" + "、".join(partial_reasons)
    reply = "\n\n".join((summary, "\n".join(lines), key_points.strip()))
    return reply, {"analysis_result": {
        "conversation_id": target_id,
        "window_start": raw["window"]["start"],
        "window_end_exclusive": raw["window"]["end_exclusive"],
        "requested_taxonomy": {"text": parsed.get("taxonomy_override") or ""},
        "effective_taxonomy": {"categories": list(categories)},
        "statistics": statistics,
        "coverage": {**(raw.get("counts") or {}), "classified": total,
                     "classification_failed": failed},
        "status": "partial" if partial_reasons else "complete",
        "partial_reasons": partial_reasons,
    }}
