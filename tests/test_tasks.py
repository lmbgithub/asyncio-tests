import asyncio

import pytest

from aiolab import basics
from aiolab.errors import ConcurrencyError
from aiolab.tasks import (
    TaskRegistry,
    cancel_and_wait,
    first_result,
    run_all,
    run_all_settled,
)


async def boom(message: str = "boom", after: float = 0.0):
    if after:
        await asyncio.sleep(after)
    raise RuntimeError(message)


async def test_run_all_returns_results_in_input_order():
    factories = [(lambda i=i, d=d: basics.delay(i, d)) for i, d in enumerate([0.02, 0.0, 0.01])]
    assert await run_all(factories) == [0, 1, 2]


async def test_run_all_on_empty_input():
    assert await run_all([]) == []


async def test_run_all_cancels_siblings_on_failure():
    cancelled = []

    async def slow(name):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    with pytest.raises(ConcurrencyError):
        await run_all([lambda: slow("a"), lambda: slow("b"), lambda: boom(after=0.01)])
    assert sorted(cancelled) == ["a", "b"]


async def test_run_all_does_not_wait_for_the_slow_siblings():
    async def slow():
        await asyncio.sleep(10)

    _, elapsed = await basics.timed(
        _expect_failure([slow, lambda: boom(after=0.01)])
    )
    assert elapsed < 1.0


async def _expect_failure(factories):
    with pytest.raises(ConcurrencyError):
        await run_all(factories)


async def test_run_all_aggregates_every_failure():
    with pytest.raises(ConcurrencyError) as exc:
        await run_all([lambda: boom("one"), lambda: boom("two")])
    assert len(exc.value.errors) == 2
    assert {str(e) for e in exc.value.errors} == {"one", "two"}


async def test_concurrency_error_message_names_the_types():
    err = ConcurrencyError([RuntimeError("a"), ValueError("b")])
    assert "2 child task(s) failed" in str(err)
    assert "RuntimeError" in str(err) and "ValueError" in str(err)


async def test_run_all_limit_bounds_concurrency():
    in_flight = 0
    peak = 0

    async def work():
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return 1

    await run_all([work] * 10, limit=3)
    assert peak <= 3


async def test_run_all_propagates_caller_cancellation_and_stops_children():
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    task = asyncio.ensure_future(run_all([slow, slow]))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(cancelled) == 2


async def test_run_all_settled_reports_each_outcome_in_order():
    out = await run_all_settled([lambda: basics.delay("ok", 0.01), lambda: boom("bad")])
    assert out[0] == "ok"
    assert isinstance(out[1], RuntimeError)


async def test_run_all_settled_never_raises_on_child_failure():
    out = await run_all_settled([lambda: boom("a"), lambda: boom("b")])
    assert all(isinstance(o, RuntimeError) for o in out)


async def test_run_all_settled_propagates_caller_cancellation():
    async def slow():
        await asyncio.sleep(5)

    task = asyncio.ensure_future(run_all_settled([slow]))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_first_result_returns_the_winner():
    factories = [lambda: basics.delay("slow", 0.05), lambda: basics.delay("fast", 0.001)]
    assert await first_result(factories) == "fast"


async def test_first_result_cancels_the_losers():
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    assert await first_result([slow, lambda: basics.delay("fast", 0.001)]) == "fast"
    assert cancelled == [1]


async def test_first_result_ignores_an_early_failure():
    assert await first_result([lambda: boom("early"), lambda: basics.delay("ok", 0.02)]) == "ok"


async def test_first_result_raises_when_everything_fails():
    with pytest.raises(ConcurrencyError) as exc:
        await first_result([lambda: boom("a"), lambda: boom("b")])
    assert len(exc.value.errors) == 2


async def test_first_result_rejects_empty_input():
    with pytest.raises(ValueError):
        await first_result([])


async def test_cancel_and_wait_waits_for_cleanup_to_finish():
    finished = []

    async def with_cleanup():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            await asyncio.sleep(0.02)  # a slow `finally`
            finished.append(1)
            raise

    task = asyncio.ensure_future(with_cleanup())
    await asyncio.sleep(0.01)
    await cancel_and_wait([task])
    assert finished == [1]


async def test_cancel_and_wait_ignores_already_finished_tasks():
    task = asyncio.ensure_future(basics.delay(1, 0))
    await task
    await cancel_and_wait([task])
    assert task.result() == 1


async def test_registry_keeps_a_strong_reference():
    registry = TaskRegistry()
    registry.spawn(basics.delay(1, 0.01))
    assert len(registry) == 1
    await registry.drain()
    assert len(registry) == 0
    assert registry.completed == 1


async def test_registry_records_failures_instead_of_losing_them():
    registry = TaskRegistry()
    registry.spawn(boom("background"))
    await registry.drain()
    assert registry.failed == 1
    assert str(registry.errors[0]) == "background"


async def test_registry_aclose_cancels_outstanding_work():
    registry = TaskRegistry()
    registry.spawn(asyncio.sleep(5))
    await registry.aclose()
    assert len(registry) == 0
    assert registry.closed is True


async def test_registry_rejects_spawn_after_close():
    registry = TaskRegistry("svc")
    await registry.aclose()
    with pytest.raises(RuntimeError, match="svc is closed"):
        registry.spawn(asyncio.sleep(0))


async def test_registry_context_manager_drains_on_success():
    done = []
    async with TaskRegistry() as registry:
        registry.spawn(_append(done, 0.01))
    assert done == [1]


async def test_registry_context_manager_cancels_on_error():
    done = []
    with pytest.raises(RuntimeError):
        async with TaskRegistry() as registry:
            registry.spawn(_append(done, 5))
            raise RuntimeError("caller failed")
    assert done == []


async def _append(sink, seconds):
    await asyncio.sleep(seconds)
    sink.append(1)


async def test_registry_named_tasks_are_introspectable():
    registry = TaskRegistry()
    task = registry.spawn(asyncio.sleep(0), name="heartbeat")
    assert task.get_name() == "heartbeat"
    await registry.drain()
