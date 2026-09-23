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
from typing import Any

from .config import Settings
from .im_client import ImClient
from .window import TimeWindow, resolve_window

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

#: find_group 候选上限（契约表：最多返回 10 条）
FIND_GROUP_LIMIT = 10


class ToolInputError(ValueError):
    """参数不符合 schema；code 固定 invalid_arguments。"""

    code = "invalid_arguments"


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{key} 必须为非空字符串")
    return value.strip()


def _parse_message_ts(raw: Any) -> datetime | None:
    """把 im 消息 created_at（UTC ISO，Z 或 +00:00 后缀）解析为 aware UTC。"""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


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
    stamp = ts.astimezone(timezone.utc).strftime("%m-%d %H:%M")
    return f"[{stamp}] {nickname}: {text}"


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


async def fetch_group_messages(
    im: ImClient, arguments: dict[str, Any], *, settings: Settings
) -> dict[str, Any]:
    """拉取指定群在时间窗内某一段的格式化语料（含覆盖率与分段元数据）。"""
    conversation_id = arguments.get("conversation_id")
    if not isinstance(conversation_id, int) or conversation_id <= 0:
        raise ToolInputError("conversation_id 必须为正整数")

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
            if ts > seg_end:
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


#: 工具名 -> (schema, 实现协程)；app.py 的 list/call 回调均由此分派
TOOL_HANDLERS = {
    "find_group": (FIND_GROUP_INPUT_SCHEMA, find_group),
    "fetch_group_messages": (FETCH_GROUP_MESSAGES_INPUT_SCHEMA, fetch_group_messages),
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
}


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
