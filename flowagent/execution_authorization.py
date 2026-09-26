"""Per-model-request authorization against Demo, with fail-closed behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class AuthorizationError(RuntimeError):
    """Authorization failed or could not be verified; never contains remote text."""


@dataclass(frozen=True)
class AuthorizationDecision:
    credential_generation: int
    request_id: str | None = None


class ModelAuthorizer(Protocol):
    async def authorize(
        self, *, execution_id: str, model_version_id: str, generation: int,
    ) -> AuthorizationDecision: ...


class DemoModelAuthorizer:
    """HTTP client for Demo's execution-scoped model-authorization contract."""

    def __init__(self, url: str, execution_token: str, runtime_id: str, *,
                 timeout_seconds: float = 3.0, transport: httpx.AsyncBaseTransport | None = None):
        if not url or not execution_token or not runtime_id:
            raise ValueError("model authorization configuration is incomplete")
        if timeout_seconds <= 0:
            raise ValueError("authorization timeout must be positive")
        self._url = url
        self._execution_token = execution_token
        self._runtime_id = runtime_id
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def __repr__(self) -> str:
        return f"DemoModelAuthorizer(url={self._url!r}, timeout_seconds={self._timeout_seconds!r})"

    async def authorize(
        self, *, execution_id: str, model_version_id: str, generation: int,
    ) -> AuthorizationDecision:
        payload = {
            "execution_id": execution_id,
            "model_version_id": model_version_id,
            "generation": generation,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds, transport=self._transport,
            ) as client:
                response = await client.post(
                    self._url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._execution_token}",
                        "X-Runtime-ID": self._runtime_id,
                    },
                )
            if response.status_code < 200 or response.status_code >= 300:
                raise AuthorizationError("model_authorization_denied")
            body = response.json()
        except AuthorizationError:
            raise
        except Exception:
            raise AuthorizationError("model_authorization_unavailable") from None

        if not isinstance(body, Mapping) or body.get("authorized") is not True:
            raise AuthorizationError("model_authorization_denied")
        credential_generation = body.get("credential_generation")
        request_id = body.get("request_id")
        if (not isinstance(credential_generation, int) or isinstance(credential_generation, bool)
                or credential_generation < 0):
            raise AuthorizationError("model_authorization_response_invalid")
        if request_id is not None and not isinstance(request_id, str):
            raise AuthorizationError("model_authorization_response_invalid")
        return AuthorizationDecision(credential_generation, request_id)


class AuthorizedChatModel:
    """Small proxy that checks Demo immediately before each actual model request."""

    def __init__(self, model: Any, authorizer: ModelAuthorizer, *, execution_id: str,
                 model_version_id: str, generation: int):
        self._model = model
        self._authorizer = authorizer
        self._execution_id = execution_id
        self._model_version_id = model_version_id
        self._generation = generation

    def bind_tools(self, tools: list[dict[str, Any]]):
        return AuthorizedChatModel(
            self._model.bind_tools(tools), self._authorizer,
            execution_id=self._execution_id,
            model_version_id=self._model_version_id,
            generation=self._generation,
        )

    async def _authorize(self) -> None:
        await self._authorizer.authorize(
            execution_id=self._execution_id,
            model_version_id=self._model_version_id,
            generation=self._generation,
        )

    async def ainvoke(self, *args: Any, **kwargs: Any):
        await self._authorize()
        return await self._model.ainvoke(*args, **kwargs)

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        await self._authorize()
        async for chunk in self._model.astream(*args, **kwargs):
            yield chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
