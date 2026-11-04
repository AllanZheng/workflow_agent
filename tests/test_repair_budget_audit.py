"""BUD-01: exact MCP ceilings, concurrent reservation and fixed-run errors."""
import asyncio

import pytest

from flowagent.core.agent import ToolRegistry, _execute_remote
from flowagent.core.context import Budget
from flowagent.core.contracts import ErrorCode, FlowAgentCoreError, RunInput, RunResult, Status, WorkflowSpec
from flowagent.core.runner import Runner
from tests.native_helpers import FakeModel, FakeTools, make_context


@pytest.fixture
def anyio_backend():
    return 'asyncio'


def budget_for(limit):
    return Budget(model_total=24, mcp_total=limit, deadline_ms=10000,
                  main_deadline_ms=9000, fallback_timeout_ms=1000)


@pytest.mark.anyio
@pytest.mark.parametrize('limit', [0, 1, 2])
async def test_bud01_sequential_remote_calls_obey_exact_limit(limit):
    budget = budget_for(limit)
    budget.switch_path('fixed')
    tools = FakeTools()
    ctx = make_context(tools=tools, budget=budget)
    registry = ToolRegistry(remote=tools)
    for index in range(limit):
        result = await _execute_remote(ctx, registry, 'echo', {'index': index})
        assert result.status == 'success'
    with pytest.raises(FlowAgentCoreError) as exc:
        await _execute_remote(ctx, registry, 'echo', {'index': limit})
    assert exc.value.code is ErrorCode.TOOL_BUDGET_EXCEEDED
    assert len(tools.calls) == ctx.usage.mcp_attempts == budget.mcp_spent == limit


@pytest.mark.anyio
@pytest.mark.parametrize('limit', [0, 1, 2])
async def test_bud01_concurrent_remote_calls_use_all_and_only_available_slots(limit):
    budget = budget_for(limit)
    budget.switch_path('fixed')
    tools = FakeTools()
    ctx = make_context(tools=tools, budget=budget)
    registry = ToolRegistry(remote=tools)
    ready, release = asyncio.Event(), asyncio.Event()
    arrivals = 0
    async def contender(index):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 3:
            ready.set()
        await release.wait()
        try:
            return await _execute_remote(ctx, registry, 'echo', {'index': index})
        except FlowAgentCoreError as exc:
            return exc
    tasks = [asyncio.create_task(contender(index)) for index in range(3)]
    await asyncio.wait_for(ready.wait(), 2)
    release.set()
    results = await asyncio.gather(*tasks)
    successes = [result for result in results if not isinstance(result, Exception)]
    failures = [result for result in results if isinstance(result, FlowAgentCoreError)]
    assert len(successes) == len(tools.calls) == ctx.usage.mcp_attempts == budget.mcp_spent == limit
    assert len(failures) == 3 - limit
    assert all(error.code is ErrorCode.TOOL_BUDGET_EXCEEDED for error in failures)


@pytest.mark.anyio
@pytest.mark.parametrize('limit', [0, 1, 2])
async def test_bud01_contenders_only_one_wins_last_slot(limit):
    budget = budget_for(limit)
    budget.switch_path('fixed')
    for _ in range(max(0, limit - 1)):
        await budget.reserve_tool()
    tools = FakeTools()
    ctx = make_context(tools=tools, budget=budget)
    registry = ToolRegistry(remote=tools)
    ready, release = asyncio.Event(), asyncio.Event()
    arrivals = 0
    async def contender(index):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            ready.set()
        await release.wait()
        try:
            return await _execute_remote(ctx, registry, 'echo', {'index': index})
        except FlowAgentCoreError as exc:
            return exc
    tasks = [asyncio.create_task(contender(index)) for index in range(2)]
    await asyncio.wait_for(ready.wait(), 2)
    release.set()
    results = await asyncio.gather(*tasks)
    successes = [result for result in results if not isinstance(result, Exception)]
    failures = [result for result in results if isinstance(result, FlowAgentCoreError)]
    expected_successes = int(limit > 0)
    assert len(successes) == len(tools.calls) == ctx.usage.mcp_attempts == expected_successes
    assert len(failures) == 2 - expected_successes
    assert all(error.code is ErrorCode.TOOL_BUDGET_EXCEEDED for error in failures)
    assert budget.mcp_spent == limit


@pytest.mark.anyio
@pytest.mark.parametrize('limit', [0, 1, 2])
async def test_bud01_fixed_runner_preserves_tool_budget_error(limit):
    budget = budget_for(limit)
    tools = FakeTools()
    model = FakeModel(default_text='must not be called')
    ctx = make_context(tools=tools, model=model, budget=budget)
    async def workflow(inputs, context):
        registry = ToolRegistry(remote=context.tools)
        for index in range(limit + 1):
            await _execute_remote(context, registry, 'echo', {'index': index})
        return RunResult(run_id='', status=Status.COMPLETED, reply='unexpected')
    runner = Runner(workflows={'job': WorkflowSpec(name='job', run=workflow)},
                    context_factory=lambda: ctx, fallback_owner='caller')
    result = await runner.run(RunInput(workflow='job', inputs={'message': 'run'}))
    assert result.status is Status.FAILED and result.error.code is ErrorCode.TOOL_BUDGET_EXCEEDED
    assert result.recovery['primary_error']['code'] == 'TOOL_BUDGET_EXCEEDED'
    assert result.recovery['fallback_attempted'] is False and result.recovery['final_source'] == 'caller'
    assert len(tools.calls) == result.usage.mcp_attempts == budget.mcp_spent == limit
    assert model.call_count == 0 and not result.reply
