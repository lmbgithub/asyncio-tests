import asyncio
import threading

import pytest

from aiolab.errors import ConcurrencyError
from aiolab.futures import (
    CallbackBridge,
    as_completed_results,
    gather_with_index,
    make_future,
    settle,
    shielded,
    wait_any,
)


async def test_make_future_is_bound_to_the_running_loop():
    future = make_future()
    assert future.get_loop() is asyncio.get_running_loop()
    assert not future.done()
    future.cancel()


async def test_settle_resolves_once_and_reports_the_loser():
    future = make_future()
    assert settle(future, "first") is True
    assert settle(future, "second") is False
    assert await future == "first"


async def test_settle_can_carry_an_exception():
    future = make_future()
    settle(future, error=ValueError("nope"))
    with pytest.raises(ValueError):
        await future


async def test_settle_on_a_cancelled_future_does_not_raise():
    future = make_future()
    future.cancel()
    await asyncio.sleep(0)
    assert settle(future, "late") is False


async def test_callback_bridge_awaits_a_callback_api():
    bridge = CallbackBridge()
    asyncio.get_running_loop().call_later(0.01, bridge.resolve, "payload")
    assert await bridge == "payload"


async def test_callback_bridge_propagates_a_rejection():
    bridge = CallbackBridge()
    bridge.reject(ConnectionError("reset"))
    with pytest.raises(ConnectionError):
        await bridge


async def test_callback_bridge_ignores_a_late_second_callback():
    bridge = CallbackBridge()
    bridge.resolve("first")
    bridge.reject(RuntimeError("late failure"))
    assert await bridge == "first"


async def test_callback_bridge_from_another_thread():
    bridge = CallbackBridge(from_thread=True)

    def worker():
        bridge.resolve("from-thread")

    threading.Thread(target=worker).start()
    assert await asyncio.wait_for(bridge.future, 1.0) == "from-thread"


async def test_as_completed_results_carries_the_input_index():
    async def work(i, delay):
        await asyncio.sleep(delay)
        return i

    factories = [
        (lambda: work(0, 0.03)),
        (lambda: work(1, 0.001)),
        (lambda: work(2, 0.015)),
    ]
    out = await as_completed_results(factories)
    assert [index for index, _ in out] == [1, 2, 0]  # completion order
    assert dict(out) == {0: 0, 1: 1, 2: 2}


async def test_as_completed_results_reports_failures_as_values():
    async def bad():
        raise RuntimeError("x")

    out = await as_completed_results([bad])
    assert isinstance(out[0][1], RuntimeError)


async def test_as_completed_results_on_empty_input():
    assert await as_completed_results([]) == []


async def test_wait_any_returns_the_first_to_settle():
    slow = asyncio.ensure_future(asyncio.sleep(1, result="slow"))
    fast = asyncio.ensure_future(asyncio.sleep(0.001, result="fast"))
    winner = await wait_any([slow, fast])
    assert winner.result() == "fast"
    assert not slow.done()  # wait_any does not own them, so it does not cancel
    slow.cancel()


async def test_wait_any_times_out():
    task = asyncio.ensure_future(asyncio.sleep(1))
    with pytest.raises(asyncio.TimeoutError):
        await wait_any([task], timeout=0.01)
    task.cancel()


async def test_wait_any_rejects_empty_input():
    with pytest.raises(ValueError):
        await wait_any([])


async def test_gather_with_index_aggregates_every_failure():
    async def bad(msg):
        raise RuntimeError(msg)

    with pytest.raises(ConcurrencyError) as exc:
        await gather_with_index([lambda: bad("a"), lambda: bad("b"), lambda: bad("c")])
    assert len(exc.value.errors) == 3


async def test_gather_with_index_returns_ordered_results():
    async def work(i, delay):
        await asyncio.sleep(delay)
        return i

    out = await gather_with_index([lambda: work(0, 0.02), lambda: work(1, 0.0)])
    assert out == [0, 1]


async def test_shielded_returns_the_default_but_lets_the_work_finish():
    finished = []

    async def slow():
        await asyncio.sleep(0.05)
        finished.append(1)
        return "done"

    task = asyncio.ensure_future(slow())
    assert await shielded(task, timeout=0.01) is None
    assert finished == []
    assert await task == "done"  # the work was not cancelled
    assert finished == [1]


async def test_shielded_returns_the_value_when_it_is_fast_enough():
    assert await shielded(asyncio.sleep(0, result="quick"), timeout=1.0) == "quick"
