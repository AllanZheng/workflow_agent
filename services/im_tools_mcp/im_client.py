"""im REST 客户端：会话列表/详情 + 消息游标翻页（只读）。

零改 im 代码（D15）：只调既有公网接口，凭据用调用方注入的 Bearer 令牌。
每次请求级新建 httpx 客户端（与 McpEdgeClient 同款"不复用连接"哲学）。

错误语义（技术方案 §3）：所有失败都归一为 ImError(code)，code 取值：
  - im_unreachable：网络/超时/5xx —— 上游不可达或不可用；
  - im_unauthorized：401/403 —— 令牌无效或无权限；
  - group_not_found：会话不存在（404）或当前用户不是成员（403 归并到 unauthorized，
    由 im 的语义决定，这里对 /conversations/{id} 的 403 视为不可见 -> group_not_found）；
  - im_error：其余非预期响应（含响应体形状异常）。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

import httpx


class ImError(RuntimeError):
    """im 调用失败；对外只暴露稳定错误码，绝不携带令牌或响应原文。"""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


#: 单页消息条数（im 接口上限 100）
MESSAGE_PAGE_SIZE = 100
#: 会话列表单页条数（im 接口上限 100）
CONVERSATION_PAGE_SIZE = 100
#: 指定会话的路径（/conversations/{id} 或 /conversations/{id}/messages），
#: 用于把这类路径上的 403 归并为 group_not_found
_SPECIFIC_CONVERSATION_RE = re.compile(r"/conversations/\d+")


class ImClient:
    """请求级 im REST 客户端；用完必须 aclose（或用 async with）。"""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds,
        )
        self._owns_client = client is None

    async def __aenter__(self) -> "ImClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = await self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise ImError("im_unreachable", "无法连接 im 服务") from exc
        if response.status_code in (401, 403):
            # 指定会话的 403 = 当前身份看不到该会话，与 404 同语义；
            # 会话列表等入口的 401/403 才是令牌/权限问题
            if response.status_code == 403 and _SPECIFIC_CONVERSATION_RE.search(path):
                raise ImError("group_not_found", "会话不存在或不可见")
            raise ImError("im_unauthorized", "im 令牌无效或无权限")
        if response.status_code == 404:
            raise ImError("group_not_found", "会话不存在")
        if response.status_code >= 500:
            raise ImError("im_unreachable", "im 服务内部错误")
        if response.status_code != 200:
            raise ImError("im_error", f"im 返回非预期状态码 {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise ImError("im_error", "im 返回非 JSON 响应") from exc

    async def iter_group_conversations(
        self, *, max_pages: int = 10
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """按页遍历当前用户的会话列表（新 -> 旧），每页 yield 原始 items。"""
        for page in range(1, max_pages + 1):
            payload = await self._get(
                "/conversations/",
                {"page": page, "page_size": CONVERSATION_PAGE_SIZE},
            )
            items = payload.get("items") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                raise ImError("im_error", "会话列表响应形状异常")
            yield [item for item in items if isinstance(item, dict)]
            if not payload.get("has_more"):
                return

    async def get_conversation(self, conversation_id: int) -> dict[str, Any]:
        """会话详情（含成员昵称映射所需 members，detail=True 才带 nickname）。"""
        payload = await self._get(f"/conversations/{conversation_id}")
        if not isinstance(payload, dict):
            raise ImError("im_error", "会话详情响应形状异常")
        return payload

    async def iter_message_pages(
        self,
        conversation_id: int,
        *,
        max_pages: int = 100,
        page_size: int = MESSAGE_PAGE_SIZE,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """从最新一页开始向更老方向翻页（before_id 游标），页内按 id 升序。

        im 接口语义：默认页=最新一页，items 升序返回；before_id=当前页最老消息
        id 继续取更老一页；has_more 表示仍存在更老消息。
        """
        before_id: int | None = None
        for _ in range(max_pages):
            params: dict[str, Any] = {"page_size": page_size}
            if before_id is not None:
                params["before_id"] = before_id
            payload = await self._get(
                f"/conversations/{conversation_id}/messages", params
            )
            items = payload.get("items") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                raise ImError("im_error", "消息列表响应形状异常")
            page = [item for item in items if isinstance(item, dict)]
            if page:
                before_id = page[0].get("id")
                if not isinstance(before_id, int):
                    raise ImError("im_error", "消息条目缺少 id")
            yield page
            if not payload.get("has_more") or not page:
                return
