import asyncio

import pytest

from aiolab.errors import PoolClosedError
from aiolab.queues import WorkerPool, batched, drain_now, produce
from aiolab.streams import aiter_from


async def square(n: int) -> int:
    await asyncio.sleep(0.001)
    return n * n


async def test_pool_processes_everything_submitted():
    async with WorkerPool(square, workers=3, maxsize=4) as pool:
        for n in range(10):
            await pool.submit(n)
        await pool.join()
    assert sorted(pool.results) == [n * n for n in range(10)]
    assert pool.stats.completed == 10


async def test_pool_runs_workers_concurrently():
    in_flight = 0
    peak = 0

    async def handler(n):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return n

    async with WorkerPool(handler, workers=4, maxsize=8) as pool:
        for n in range(8):
            await pool.submit(n)
        await pool.join()
    assert peak == 4


async def test_a_failing_item_does_not_kill_its_worker():
    async def handler(n):
        if n == 3:
            raise ValueError("bad item")
        return n

    async with WorkerPool(handler, workers=1, maxsize=8) as pool:
        for n in range(6):
            await pool.submit(n)
        await pool.join()
    assert pool.stats.failed == 1
    assert pool.stats.completed == 5
    assert isinstance(pool.stats.errors[0], ValueError)


async def test_submit_blocks_when_the_queue_is_full():
    release = asyncio.Event()

    async def handler(n):
        await release.wait()
        return n

    pool = await WorkerPool(handler, workers=1, maxsize=2).start()
    await pool.submit(1)  # taken by the worker
    await pool.submit(2)
    await pool.submit(3)  # queue now full
    blocked = asyncio.ensure_future(pool.submit(4))
    await asyncio.sleep(0.01)
    assert not blocked.done()  # this is the backpressure

    release.set()
    await blocked
    await pool.aclose()


async def test_try_submit_sheds_load_and_counts_it():
    release = asyncio.Event()

    async def handler(n):
        await release.wait()
        return n

    pool = await WorkerPool(handler, workers=1, maxsize=2).start()
    await asyncio.sleep(0)
    accepted = [pool.try_submit(n) for n in range(5)]
    release.set()
    await pool.aclose()
    assert accepted.count(False) == pool.stats.rejected
    assert pool.stats.rejected > 0


async def test_aclose_drains_the_backlog_by_default():
    seen = []

    async def handler(n):
        await asyncio.sleep(0.005)
        seen.append(n)
        return n

    pool = await WorkerPool(handler, workers=2, maxsize=16).start()
    for n in range(8):
        await pool.submit(n)
    await pool.aclose()
    assert len(seen) == 8


async def test_aclose_without_drain_abandons_the_backlog():
    seen = []

    async def handler(n):
        await asyncio.sleep(0.05)
        seen.append(n)
        return n

    pool = await WorkerPool(handler, workers=1, maxsize=16).start()
    for n in range(8):
        await pool.submit(n)
    await asyncio.sleep(0)
    await pool.aclose(drain=False)
    assert len(seen) < 8
    assert pool.backlog > 0  # the abort is visible, not hidden


async def test_submitting_to_a_closed_pool_raises():
    pool = await WorkerPool(square, workers=1, maxsize=2).start()
    await pool.aclose()
    with pytest.raises(PoolClosedError):
        await pool.submit(1)
    with pytest.raises(PoolClosedError):
        pool.try_submit(1)


async def test_join_returns_even_when_a_worker_is_cancelled_mid_item():
    started = asyncio.Event()

    async def handler(n):
        started.set()
        await asyncio.sleep(5)
        return n

    pool = await WorkerPool(handler, workers=1, maxsize=4).start()
    await pool.submit(1)
    await started.wait()
    # Cancelling mid-item must still mark the item done, or join() hangs.
    await asyncio.wait_for(pool.aclose(drain=False), 1.0)


async def test_pool_context_manager_aborts_when_the_body_raises():
    pool = WorkerPool(square, workers=1, maxsize=4)
    with pytest.raises(RuntimeError):
        async with pool:
            await pool.submit(1)
            raise RuntimeError("caller failed")
    assert pool.closed is True


async def test_pool_rejects_an_unbounded_queue():
    with pytest.raises(ValueError, match="backpressure"):
        WorkerPool(square, workers=1, maxsize=0)


async def test_pool_rejects_zero_workers():
    with pytest.raises(ValueError):
        WorkerPool(square, workers=0)


async def test_pool_cannot_be_started_twice():
    pool = await WorkerPool(square, workers=1, maxsize=2).start()
    with pytest.raises(RuntimeError):
        await pool.start()
    await pool.aclose()


async def test_on_result_callback_fires_per_item():
    seen = []
    async with WorkerPool(square, workers=2, maxsize=8, on_result=seen.append) as pool:
        for n in range(4):
            await pool.submit(n)
        await pool.join()
    assert sorted(seen) == [0, 1, 4, 9]


async def test_produce_respects_the_bound():
    queue = asyncio.Queue(maxsize=2)
    task = asyncio.ensure_future(produce(queue, [1, 2, 3, 4]))
    await asyncio.sleep(0.01)
    assert not task.done()
    assert await drain_now(queue) == [1, 2]
    assert await task == 4


async def test_drain_now_on_an_empty_queue():
    assert await drain_now(asyncio.Queue()) == []


async def test_batched_groups_by_size():
    out = [b async for b in batched(aiter_from(range(7)), size=3)]
    assert out == [[0, 1, 2], [3, 4, 5], [6]]


async def test_batched_emits_the_tail_when_the_source_ends():
    out = [b async for b in batched(aiter_from([1, 2]), size=10)]
    assert out == [[1, 2]]


async def test_batched_flushes_on_timeout_without_dropping_an_item():
    async def trickle():
        yield 1
        yield 2
        await asyncio.sleep(0.05)
        yield 3

    out = [b async for b in batched(trickle(), size=10, timeout=0.01)]
    assert out == [[1, 2], [3]]  # nothing lost across the timeout


async def test_batched_emits_nothing_extra_when_idle():
    async def slow():
        await asyncio.sleep(0.03)
        yield 1

    out = [b async for b in batched(slow(), size=2, timeout=0.005)]
    assert out == [[1]]  # empty batches are never yielded


async def test_batched_rejects_a_zero_size():
    with pytest.raises(ValueError):
        [b async for b in batched(aiter_from([1]), size=0)]
