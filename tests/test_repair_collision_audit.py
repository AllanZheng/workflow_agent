"""RUN-04: duplicate execution IDs through actual Sample execution-v1 routes."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from flowagent.adapters.platform import context as platform_context
from flowagent.app import build_platform_service
from tests.test_managed_config_v3 import _envelope
from tests.test_platform_v3_adapter import _text_chunk


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('first_stream', [False, True])
@pytest.mark.parametrize('duplicate_stream', [False, True])
async def test_run04_duplicate_execution_route_never_replaces_cancellable_first(
        monkeypatch, first_stream, duplicate_stream):
    entered, stopped = asyncio.Event(), asyncio.Event()
    calls, authorizations = [], []
    class Allow:
        def __init__(self, *args, **kwargs): pass
        async def authorize(self, **kwargs):
            authorizations.append(kwargs['execution_id'])
    async def model_stream():
        calls.append('started')
        entered.set()
        try:
            await asyncio.Event().wait()
            yield _text_chunk('late answer')
        finally:
            stopped.set()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: model_stream())))
    real_build = platform_context.build_capabilities
    monkeypatch.setattr(platform_context, 'DemoModelAuthorizer', Allow)
    monkeypatch.setattr(platform_context, 'build_capabilities',
                        lambda request: real_build(request, model_transport=sdk))
    envelope = _envelope()
    app = build_platform_service()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://sample.local') as client:
        def path(stream):
            return '/execution-v1/respond/stream' if stream else '/execution-v1/respond'
        first = asyncio.create_task(client.post(path(first_stream), json=envelope))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            duplicate = await asyncio.wait_for(client.post(path(duplicate_stream), json=envelope), 2)
            if duplicate_stream:
                events = [json.loads(line[6:]) for line in duplicate.text.splitlines() if line.startswith('data: ')]
                assert duplicate.status_code == 200
                assert [event['kind'] for event in events] == ['meta', 'error']
                assert events[-1]['error_code'] == 'invalid_input'
            else:
                assert duplicate.status_code == 422
                assert duplicate.json()['error']['code'] == 'invalid_input'
            assert not first.done()
            assert calls == ['started'] and len(authorizations) == 1
            cancel = await client.post('/execution-v1/cancel', json={'execution_id': envelope['execution_id'].upper()})
            assert cancel.json()['cancelled'] is True
            first_response = await asyncio.wait_for(first, 2)
            assert stopped.is_set()
            if first_stream:
                events = [json.loads(line[6:]) for line in first_response.text.splitlines() if line.startswith('data: ')]
                assert first_response.status_code == 200
                assert [event['kind'] for event in events] == ['meta', 'error']
                assert events[-1]['error_code'] == 'cancelled'
            else:
                assert first_response.status_code == 499
                assert first_response.json()['error']['code'] == 'cancelled'
            after = await client.post('/execution-v1/cancel', json={'execution_id': envelope['execution_id']})
            assert after.json()['cancelled'] is False
            assert calls == ['started'] and len(authorizations) == 1
        finally:
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
