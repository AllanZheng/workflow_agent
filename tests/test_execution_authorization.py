import asyncio

import httpx
import pytest

from flowagent.execution_authorization import AuthorizationError, DemoModelAuthorizer


def test_authorizer_posts_execution_generation_and_model_version_with_scoped_identity():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = request.read().decode("utf-8")
        return httpx.Response(200, json={
            "authorized": True, "credential_generation": 7, "request_id": "auth-1",
        })

    authorizer = DemoModelAuthorizer(
        "https://demo.test/internal/executions/f04b6209-30da-4e57-bf93-73c6867cc101/model-authorization",
        "execution-secret",
        "804b6209-30da-4e57-bf93-73c6867cc101",
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(authorizer.authorize(
        execution_id="f04b6209-30da-4e57-bf93-73c6867cc101",
        model_version_id="a04b6209-30da-4e57-bf93-73c6867cc101",
        generation=4,
    ))

    assert result.credential_generation == 7
    assert result.request_id == "auth-1"
    assert captured["url"].endswith("/model-authorization")
    assert captured["headers"]["authorization"] == "Bearer execution-secret"
    assert captured["headers"]["x-runtime-id"] == "804b6209-30da-4e57-bf93-73c6867cc101"
    assert captured["json"] == (
        '{"execution_id":"f04b6209-30da-4e57-bf93-73c6867cc101",'
        '"model_version_id":"a04b6209-30da-4e57-bf93-73c6867cc101","generation":4}'
    )
    assert "execution-secret" not in repr(authorizer)


@pytest.mark.parametrize("response", [
    httpx.Response(403, json={"authorized": False}),
    httpx.Response(503, json={"error": "backend-secret"}),
    httpx.Response(200, json={"authorized": "true"}),
])
def test_authorizer_fails_closed_without_reflecting_remote_error_body(response):
    authorizer = DemoModelAuthorizer(
        "https://demo.test/internal/model-authorization",
        "execution-secret", "804b6209-30da-4e57-bf93-73c6867cc101",
        transport=httpx.MockTransport(lambda request: response),
    )

    with pytest.raises(AuthorizationError) as exc:
        asyncio.run(authorizer.authorize(
            execution_id="f04b6209-30da-4e57-bf93-73c6867cc101",
            model_version_id="a04b6209-30da-4e57-bf93-73c6867cc101",
            generation=4,
        ))
    assert "execution-secret" not in str(exc.value)
    assert "backend-secret" not in str(exc.value)


def test_authorized_model_checks_before_each_bound_model_request():
    from flowagent.execution_authorization import AuthorizedChatModel

    events = []

    class Authorizer:
        async def authorize(self, **kwargs):
            events.append(("authorize", kwargs["generation"]))
            return object()

    class Model:
        def bind_tools(self, tools):
            events.append(("bind", len(tools)))
            return self

        async def ainvoke(self, messages):
            events.append(("request", messages))
            return "ok"

    model = AuthorizedChatModel(
        Model(), Authorizer(), execution_id="e1", model_version_id="v1", generation=8,
    ).bind_tools([{"name": "search"}])
    assert asyncio.run(model.ainvoke(["one"])) == "ok"
    assert asyncio.run(model.ainvoke(["two"])) == "ok"

    assert events == [
        ("bind", 1), ("authorize", 8), ("request", ["one"]),
        ("authorize", 8), ("request", ["two"]),
    ]


def test_authorized_model_does_not_send_request_after_pre_call_denial():
    from flowagent.execution_authorization import AuthorizedChatModel

    requests = []

    class DenyingAuthorizer:
        async def authorize(self, **kwargs):
            raise AuthorizationError("model_authorization_denied")

    class Model:
        async def ainvoke(self, messages):
            requests.append(messages)
            return "must not run"

    model = AuthorizedChatModel(
        Model(), DenyingAuthorizer(), execution_id="e1", model_version_id="v1", generation=1,
    )
    with pytest.raises(AuthorizationError):
        asyncio.run(model.ainvoke(["private prompt"]))
    assert requests == []
