"""D-12：请求级有界日志缓冲（技术方案 §7.3）。

进程内、单请求生命周期，不落库、不跨请求持久化。超过 100 条或 64 KiB 时
截断并标记 `logs_truncated=true`；不记录密钥、Authorization、完整请求体。
"""

from __future__ import annotations

from dataclasses import dataclass, field

_MAX_ENTRIES = 100
_MAX_BYTES = 64 * 1024


@dataclass
class ExecutionLog:
    stage: str = ""
    entries: list[dict[str, object]] = field(default_factory=list)
    truncated: bool = False
    _bytes: int = field(default=0, repr=False)

    # 执行添加对应的流程。
    # English: Add the current operation.
    def add(self, stage: str, level: str, message: str, *, duration_ms: int | None = None) -> None:
        if self.truncated:
            return
        entry = {"stage": stage, "level": level, "message": message}
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        entry_size = len(message.encode("utf-8"))
        if len(self.entries) >= _MAX_ENTRIES or self._bytes + entry_size > _MAX_BYTES:
            self.truncated = True
            return
        self.entries.append(entry)
        self._bytes += entry_size

    # 处理元数据相关逻辑。
    # English: Convert to metadata.
    def to_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {"logs": list(self.entries)}
        if self.truncated:
            payload["logs_truncated"] = True
        return payload
