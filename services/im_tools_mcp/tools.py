"""im-tools-mcp 工具契约与实现（契约的单一事实源）。

两个只读工具（技术方案 §3 契约表 + 评审补充）：
  - find_group(name_query)：名称模糊匹配候选群（≤10 条）；
  - fetch_group_messages(...)：分段拉取格式化语料。

对契约表的两处实现级补充（引擎零改动约束下的必要扩展，见实施记录）：
  - start/end 接受"记号"而非严格 date-time：ISO 日期、缺年日期（MM-DD / M月D日）、
    相对记号（now / now-Nd / now-Nh）。引擎没有时钟，相对语义由本服务解析；
  - 新增 segment_index 参数：分段由服务端确定性切分（segments_total 随响应返回），
    调用方按索引寻址分段，可在一轮内并行取多段（受管 agent 每轮可发多个工具调用，
    max_tool_iterations 上限 1..8 的护栏下靠并行覆盖 30 天窗口）。

脱敏承诺：输出只含 昵称/时间/文本，绝不包含 sender_id / subject_id / uuid 等
内部标识（技术方案 §6）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from .window import ANALYSIS_TIMEZONE
from typing import Any

import asyncio
import shutil
import tempfile
from pathlib import Path

from .attachments import MAX_FILE_BYTES, extract_file
from .config import Settings
from .im_client import ImClient, ImError
from .window import TimeWindow, resolve_window

#: 一次分析输入最多查询多少个附件元数据（每个附件一次 im 调用）
MAX_ATTACHMENT_LOOKUPS = 200

#: find_group 输入 schema —— Admin mcp_tools 注册快照对拍用同一来源
FIND_GROUP_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name_query": {
            "type": "string",
            "description": "群名称关键词，服务端做大小写不敏感的模糊匹配",
        },
    },
    "required": ["name_query"],
    "additionalProperties": False,
}

#: fetch_group_messages 输入 schema —— Admin mcp_tools 注册快照对拍用同一来源
FETCH_GROUP_MESSAGES_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conversation_id": {"type": "integer", "description": "群会话 ID"},
        "start": {
            "type": "string",
            "description": (
                "分析窗口起点记号：ISO 日期(时间)、缺年日期(MM-DD / M月D日)、"
                "或相对记号 now / now-Nd / now-Nh；与 end 同时缺省时按 days 取相对窗口"
            ),
        },
        "end": {
            "type": "string",
            "description": "分析窗口终点记号，格式同 start；缺省取 now",
        },
        "days": {
            "type": "integer",
            "minimum": 1,
            "maximum": 31,
            "description": "相对窗口天数（start/end 均未提供时生效，默认 7）",
        },
        "segment_hours": {
            "type": "integer",
            "minimum": 1,
            "maximum": 720,
            "default": 24,
            "description": "分段时长（小时）；窗口按此切分为 segments_total 段",
        },
        "max_chars": {
            "type": "integer",
            "minimum": 1000,
            "maximum": 60000,
            "default": 30000,
            "description": "单段语料字符上限，超出截断并置 truncated=true",
        },
        "segment_index": {
            "type": "integer",
            "minimum": 0,
            "description": "取第几段（0 起计，上限见响应 segments_total-1）",
        },
    },
    "required": ["conversation_id"],
    "additionalProperties": False,
}


FETCH_GROUP_ANALYSIS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conversation_id": {"type": "integer", "minimum": 1},
        "start": {"type": "string"},
        "end": {"type": "string"},
        "days": {"type": "integer", "minimum": 1, "maximum": 31},
    },
    "required": ["conversation_id"],
    "additionalProperties": False,
}

#: find_group 候选上限（契约表：最多返回 10 条）
FIND_GROUP_LIMIT = 10

READ_GROUP_ATTACHMENT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conversation_id": {"type": "integer", "minimum": 1},
        "attachment_id": {"type": "integer", "minimum": 1},
    },
    "required": ["conversation_id", "attachment_id"],
    "additionalProperties": False,
}


class ToolInputError(ValueError):
    """参数不符合 schema；code 固定 invalid_arguments。"""

    code = "invalid_arguments"


# 校验必需的`str`。
# English: Require str.
def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{key} 必须为非空字符串")
    return value.strip()


# English: Parse message ts.
def _parse_message_ts(raw: Any) -> datetime | None:
    """把 im 消息 created_at（UTC ISO，Z 或 +00:00 后缀）解析为 aware UTC。"""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# English: Handle sender nickname map operations.
def _sender_nickname_map(conversation: dict[str, Any]) -> dict[int, str]:
    """sender_id -> 群内昵称映射；昵称缺失退化为 username/full_name/编号。"""
    mapping: dict[int, str] = {}
    for member in conversation.get("members") or []:
        if not isinstance(member, dict):
            continue
        user_id = member.get("user_id")
        if not isinstance(user_id, int):
            continue
        name = (
            member.get("nickname")
            or member.get("full_name")
            or member.get("username")
        )
        mapping[user_id] = str(name).strip() if name else f"成员{user_id}"
    return mapping


# English: Format message.
def _format_message(message: dict[str, Any], nicknames: dict[int, str]) -> str | None:
    """单条消息 -> 语料行 ``[MM-DD HH:mm] 昵称: 内容``；不适用的返回 None。

    脱敏：只保留时间/昵称/文本；已删除消息跳过；非文本消息用占位符。
    """
    ts = _parse_message_ts(message.get("created_at"))
    if ts is None:
        return None
    sender_id = message.get("sender_id")
    nickname = nicknames.get(sender_id) if isinstance(sender_id, int) else None
    nickname = nickname or "未知成员"
    content = message.get("content")
    if message.get("is_deleted"):
        return None  # 已删除/撤回消息不进入语料
    message_type = str(message.get("message_type") or "text")
    label = None
    if message_type != "text":
        label = {
            "image": "[图片]",
            "file": f"[文件: {message.get('file_name') or '未命名'}]",
            "audio": "[语音]",
            "video": "[视频]",
        }.get(message_type, f"[{message_type}]")
    if not isinstance(content, str) or not content.strip():
        if message_type == "text":
            return None  # 空文本无分析价值
        text = label or ""
    else:
        body = " ".join(content.split())  # 压缩空白，防语料被换行撑爆
        # 非文本消息保留占位符（告知类型）+ 原文案（若有）
        text = f"{label} {body}" if label else body
    if not text:
        return None
    # 引用语料与分析窗口、IM 页面使用同一时区，显式标注以免误读 UTC。
    stamp = ts.astimezone(ANALYSIS_TIMEZONE).strftime("%m-%d %H:%M UTC+08:00")
    return f"[{stamp}] {nickname}: {text}"


# English: Find group.
async def find_group(
    im: ImClient, arguments: dict[str, Any], *, settings: Settings | None = None
) -> dict[str, Any]:
    """按名称关键词查找候选群；无匹配返回空列表（不是错误）。"""
    name_query = _require_str(arguments, "name_query").lower()
    matched: list[dict[str, Any]] = []
    async for page in im.iter_group_conversations():
        for item in page:
            if item.get("type") not in ("group", "channel"):
                continue  # 私聊会话不参与群名匹配
            name = item.get("name")
            if not isinstance(name, str):
                continue
            if name_query not in name.lower():
                continue
            matched.append({
                "conversation_id": item.get("id"),
                "name": name,
                "member_count": item.get("member_count"),
            })
            if len(matched) >= FIND_GROUP_LIMIT:
                return {"groups": matched, "count": len(matched)}
    return {"groups": matched, "count": len(matched)}


# English: Fetch group messages.
async def fetch_group_messages(
    im: ImClient, arguments: dict[str, Any], *, settings: Settings
) -> dict[str, Any]:
    """拉取指定群在时间窗内某一段的格式化语料（含覆盖率与分段元数据）。"""
    conversation_id = arguments.get("conversation_id")
    if not isinstance(conversation_id, int) or conversation_id <= 0:
        raise ToolInputError("conversation_id 必须为正整数")

    # 处理`int``arg`相关逻辑。
    # English: Handle int arg operations.
    def _int_arg(key: str, default: int, low: int, high: int) -> int:
        value = arguments.get(key, default)
        if value is None:
            value = default
        if not isinstance(value, int) or not low <= value <= high:
            raise ToolInputError(f"{key} 必须为 {low}..{high} 的整数")
        return value

    segment_hours = _int_arg("segment_hours", 24, 1, 720)
    max_chars = _int_arg("max_chars", 30000, 1000, 60000)
    segment_index = _int_arg("segment_index", 0, 0, 10000)
    days = arguments.get("days")
    if days is not None and (not isinstance(days, int) or not 1 <= days <= 31):
        raise ToolInputError("days 必须为 1..31 的整数")
    start = arguments.get("start")
    end = arguments.get("end")
    if start is not None and not isinstance(start, str):
        raise ToolInputError("start 必须为字符串记号")
    if end is not None and not isinstance(end, str):
        raise ToolInputError("end 必须为字符串记号")

    window: TimeWindow = resolve_window(
        start, end, days, max_days=settings.max_window_days
    )
    segments_total = window.segments_total(segment_hours)
    seg_start, seg_end = window.segment_bounds(segment_hours, segment_index)

    # 会话详情先取：既验证存在性/权限，也拿到 sender -> 昵称映射（脱敏出口）
    conversation = await im.get_conversation(conversation_id)
    nicknames = _sender_nickname_map(conversation)

    lines: list[str] = []
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    collected = 0
    reached_before_window = False
    async for page in im.iter_message_pages(
        conversation_id, max_pages=settings.max_pages_per_segment
    ):
        # 页间由新到老、页内升序 -> 倒序逐条从新到老筛选
        for message in reversed(page):
            ts = _parse_message_ts(message.get("created_at"))
            if ts is None:
                continue
            if ts >= seg_end:
                continue
            if ts < seg_start:
                # 消息按 id 单调对应时间，遇到早于窗口的消息即可停止翻页
                reached_before_window = True
                break
            line = _format_message(message, nicknames)
            if line is None:
                continue
            lines.append(line)
            collected += 1
            last_ts = ts if last_ts is None else max(last_ts, ts)
            first_ts = ts if first_ts is None else min(first_ts, ts)
        if reached_before_window:
            break

    # 收集顺序为新->老，语料输出为老->新
    lines.reverse()
    corpus = "\n".join(lines)
    truncated = len(corpus) > max_chars
    if truncated:
        corpus = corpus[:max_chars]

    return {
        "corpus": corpus,
        "truncated": truncated,
        "coverage": {
            "messages": collected,
            "first_ts": first_ts.isoformat() if first_ts else None,
            "last_ts": last_ts.isoformat() if last_ts else None,
        },
        "window": {"start": window.start.isoformat(), "end": window.end.isoformat()},
        "segment": {
            "index": segment_index,
            "total": segments_total,
            "start": seg_start.isoformat(),
            "end": seg_end.isoformat(),
        },
    }


# 获取群组`analysis`输入。
# English: Fetch group analysis input.
async def fetch_group_analysis_input(
    im: ImClient, arguments: dict[str, Any], *, settings: Settings
) -> dict[str, Any]:
    """Read one fixed window exactly once; report incomplete pagination honestly."""
    conversation_id = arguments.get("conversation_id")
    if type(conversation_id) is not int or conversation_id < 1:
        raise ToolInputError("conversation_id 必须为正整数")
    for key in ("start", "end"):
        if arguments.get(key) is not None and not isinstance(arguments[key], str):
            raise ToolInputError(f"{key} 必须为字符串")
    days = arguments.get("days")
    if days is not None and (type(days) is not int or not 1 <= days <= 31):
        raise ToolInputError("days 必须为 1..31 的整数")
    now = datetime.now(ANALYSIS_TIMEZONE)
    window = resolve_window(arguments.get("start"), arguments.get("end"), days,
                            now=now, max_days=settings.max_window_days)
    conversation = await im.get_conversation(conversation_id)
    if conversation.get("type") not in {"group", "channel"}:
        raise ToolInputError("目标不是群会话")
    nicknames = _sender_nickname_map(conversation)
    # message_id -> (时间, 语料行, 发送者 ID, 附件 ID 列表, 消息上的文件名)
    rows: dict[int, tuple[datetime, str, int | None, list[int], str | None]] = {}
    page_count = 0
    partial_reasons: list[str] = []
    for_limit = settings.max_pages_per_segment
    async for page in im.iter_message_pages(conversation_id, max_pages=for_limit):
        page_count += 1
        oldest_ts: datetime | None = None
        for message in page:
            message_id = message.get("id")
            ts = _parse_message_ts(message.get("created_at"))
            if type(message_id) is not int or ts is None:
                continue
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
            if not window.start <= ts < window.end or message_id in rows:
                continue
            line = _format_message(message, nicknames)
            if line:
                sender = message.get("sender_id")
                raw_attachments = message.get("attachment_ids")
                attachment_ids = [item for item in (raw_attachments if isinstance(raw_attachments, list) else [])
                                  if type(item) is int and item > 0]
                rows[message_id] = (ts, line, sender if type(sender) is int else None,
                                    attachment_ids, message.get("file_name"))
        # 分页从最新往最老翻；本页最老一条已早于窗口起点，说明更老的页
        # 不可能再有落在窗口内的消息，提前停止（评审性能问题 9a）。
        if oldest_ts is not None and oldest_ts < window.start:
            break
    if page_count >= for_limit:
        partial_reasons.append("page_limit")
    ordered = sorted(rows.items(), key=lambda item: (item[1][0], item[0]))
    messages: list[dict[str, Any]] = []
    used_chars = 0
    attachment_budget = MAX_ATTACHMENT_LOOKUPS
    for message_id, (ts, line, sender_id, attachment_ids, file_name) in ordered:
        if len(messages) >= 1000:
            partial_reasons.append("message_limit")
            break
        if used_chars + len(line) > 120000:
            partial_reasons.append("char_limit")
            break
        entry: dict[str, Any] = {"message_ref": f"m{message_id}", "timestamp": ts.isoformat(),
                                 "text": line, "sender_id": sender_id}
        attachments: list[dict[str, Any]] = []
        for attachment_id in attachment_ids:
            meta: dict[str, Any] = {}
            if attachment_budget > 0:
                attachment_budget -= 1
                try:
                    meta = await im.get_attachment(attachment_id)
                except ImError:
                    meta = {}  # 元数据不可得时只返回编号，读取时再由 im 判定权限
            attachments.append({
                "attachment_id": attachment_id,
                "filename": meta.get("original_name") or file_name or "",
                "mime_type": meta.get("mime_type") or "",
                "size": meta.get("file_size") if type(meta.get("file_size")) is int else None,
            })
        if attachments:
            entry["attachments"] = attachments
        messages.append(entry)
        used_chars += len(line)
    return {
        "conversation_id": conversation_id,
        "window": {"start": window.start.isoformat(),
                   "end_exclusive": window.end.isoformat()},
        "status": "partial" if partial_reasons else "complete",
        "partial_reasons": partial_reasons,
        "counts": {"fetched_unique": len(rows), "included": len(messages)},
        "messages": messages,
    }


# 读取群附件文本：鉴权由 im 的附件访问规则完成（统一账号令牌），抽取逻辑见 attachments.py。
# English: Read and extract text of a group attachment.
async def read_group_attachment(
    im: ImClient, arguments: dict[str, Any], *, settings: Settings
) -> dict[str, Any]:
    conversation_id = arguments.get("conversation_id")
    attachment_id = arguments.get("attachment_id")
    if type(conversation_id) is not int or conversation_id < 1:
        raise ToolInputError("conversation_id 必须为正整数")
    if type(attachment_id) is not int or attachment_id < 1:
        raise ToolInputError("attachment_id 必须为正整数")
    # 先确认当前身份是该群成员（非成员由 im 返回不可见 -> group_not_found）
    conversation = await im.get_conversation(conversation_id)
    if conversation.get("type") not in {"group", "channel"}:
        raise ToolInputError("目标不是群会话")
    base: dict[str, Any] = {"attachment_id": attachment_id}
    try:
        meta = await im.get_attachment(attachment_id)
    except ImError as exc:
        if exc.code != "attachment_forbidden":
            raise
        return {**base, "filename": "", "status": "failed", "text": "", "truncated": False,
                "chars": 0, "sha256": "", "format": "", "error_code": "attachment_forbidden"}
    filename = str(meta.get("original_name") or "")
    base["filename"] = filename
    size = meta.get("file_size")
    if type(size) is int and size > MAX_FILE_BYTES:
        return {**base, "status": "too_large", "text": "", "truncated": False, "chars": 0,
                "sha256": "", "format": Path(filename).suffix.lstrip(".").lower(),
                "error_code": "too_large"}
    workdir = Path(tempfile.mkdtemp(prefix="im-attachment-"))
    try:
        target = workdir / "download.bin"
        try:
            complete = await im.download_attachment(attachment_id, target, max_bytes=MAX_FILE_BYTES)
        except ImError as exc:
            if exc.code != "attachment_forbidden":
                raise
            return {**base, "status": "failed", "text": "", "truncated": False, "chars": 0,
                    "sha256": "", "format": "", "error_code": "attachment_forbidden"}
        if not complete:
            return {**base, "status": "too_large", "text": "", "truncated": False, "chars": 0,
                    "sha256": "", "format": Path(filename).suffix.lstrip(".").lower(),
                    "error_code": "too_large"}
        extracted = await asyncio.to_thread(extract_file, target, filename=filename)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {**base, **extracted}


#: 工具名 -> (schema, 实现协程)；app.py 的 list/call 回调均由此分派
TOOL_HANDLERS = {
    "find_group": (FIND_GROUP_INPUT_SCHEMA, find_group),
    "fetch_group_messages": (FETCH_GROUP_MESSAGES_INPUT_SCHEMA, fetch_group_messages),
    "fetch_group_analysis_input": (FETCH_GROUP_ANALYSIS_INPUT_SCHEMA, fetch_group_analysis_input),
    "read_group_attachment": (READ_GROUP_ATTACHMENT_INPUT_SCHEMA, read_group_attachment),
}

#: 面向用户的错误码说明（MCP is_error 结构化返回；不泄露内部细节）
_ERROR_HINTS = {
    "im_unreachable": "群消息服务暂时不可用，请稍后再试",
    "im_unauthorized": "群消息凭据无效，请联系管理员检查配置",
    "group_not_found": "群不存在或当前身份无权访问",
    "window_invalid": "时间范围无法识别，请换用明确日期（如 2026-09-01）",
    "window_too_large": "时间范围过大，请缩小到 31 天以内",
    "segment_out_of_range": "分段序号越界，请按响应中的 segments_total 取值",
    "invalid_arguments": "工具参数不符合约定",
    "im_error": "群消息服务返回异常，请稍后再试",
    "attachment_forbidden": "附件不存在或当前身份无权访问",
}


# English: Handle error payload operations.
def error_payload(code: str) -> dict[str, Any]:
    """构造 is_error 结构化载荷；未知码回退到 im_error。"""
    code = code if code in _ERROR_HINTS else "im_error"
    return {"error_code": code, "message": _ERROR_HINTS[code]}


__all__ = [
    "FIND_GROUP_INPUT_SCHEMA",
    "FETCH_GROUP_MESSAGES_INPUT_SCHEMA",
    "TOOL_HANDLERS",
    "ToolInputError",
    "error_payload",
    "find_group",
    "fetch_group_messages",
]
