import asyncio
import time

import pytest

from aiolab import basics


async def test_delay_returns_its_value():
    assert await basics.delay("x", 0) == "x"


async def test_timed_reports_a_positive_duration():
    result, elapsed = await basics.timed(basics.delay(1, 0.01))
    assert result == 1
    assert elapsed >= 0.009


async def test_sequential_takes_the_sum():
    factories = [(lambda: basics.delay(i, 0.02)) for i in range(3)]
    _, elapsed = await basics.timed(basics.run_sequential(factories))
    assert elapsed >= 0.055


async def test_concurrent_takes_the_max():
    factories = [(lambda: basics.delay(i, 0.02)) for i in range(3)]
    _, elapsed = await basics.timed(basics.run_concurrent(factories))
    assert elapsed < 0.05


async def test_concurrent_preserves_input_order_not_completion_order():
    delays = [0.03, 0.001, 0.02]
    factories = [(lambda i=i, d=d: basics.delay(i, d)) for i, d in enumerate(delays)]
    assert await basics.run_concurrent(factories) == [0, 1, 2]


async def test_concurrent_on_an_empty_list():
    assert await basics.run_concurrent([]) == []


async def test_to_thread_does_not_block_the_loop():
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    task = asyncio.ensure_future(ticker())
    await basics.to_thread(time.sleep, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # A `time.sleep` in the coroutine itself would have frozen the ticker.
    assert ticks > 1


async def test_is_awaited_correctly_rejects_a_function_object():
    async def f():
        return 1

    coro = f()
    assert basics.is_awaited_correctly(coro) is True
    assert basics.is_awaited_correctly(f) is False
    await coro  # an un-awaited coroutine warns at GC; that is the whole point


async def test_consume_preserves_order():
    assert await basics.consume([basics.delay(i, 0) for i in range(3)]) == [0, 1, 2]
