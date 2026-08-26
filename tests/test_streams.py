import asyncio

import pytest

from aiolab.basics import timed
from aiolab.streams import aiter_from, amap, collect, merge, take_until, throttle


async def double(n: int) -> int:
    await asyncio.sleep(0.001)
    return n * 2


async def test_aiter_from_yields_everything():
    assert await collect(aiter_from([1, 2, 3])) == [1, 2, 3]


async def test_collect_limit_stops_early_and_closes_the_source():
    closed = []

    async def source():
        try:
            for i in range(100):
                yield i
        finally:
            closed.append(1)

    assert await collect(source(), limit=3) == [0, 1, 2]
    assert closed == [1]


async def test_amap_preserves_order():
    async def variable(n):
        await asyncio.sleep(0.02 if n % 2 == 0 else 0.001)
        return n

    out = await collect(amap(aiter_from(range(6)), variable, concurrency=6))
    assert out == [0, 1, 2, 3, 4, 5]


async def test_amap_is_actually_concurrent():
    async def slow(n):
        await asyncio.sleep(0.02)
        return n

    _, elapsed = await timed(collect(amap(aiter_from(range(8)), slow, concurrency=8)))
    assert elapsed < 0.1  # serial would be ~160 ms


async def test_amap_bounds_concurrency():
    in_flight = 0
    peak = 0

    async def work(n):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return n

    await collect(amap(aiter_from(range(12)), work, concurrency=3))
    assert peak <= 3


async def test_amap_unordered_yields_the_fast_item_first():
    async def variable(n):
        await asyncio.sleep(0.001 if n == 2 else 0.03)
        return n

    out = await collect(amap(aiter_from([0, 1, 2]), variable, concurrency=3, ordered=False))
    assert out[0] == 2
    assert sorted(out) == [0, 1, 2]


async def test_amap_on_an_empty_source():
    assert await collect(amap(aiter_from([]), double)) == []


async def test_amap_propagates_the_mapped_error():
    async def bad(n):
        if n == 2:
            raise ValueError("bad element")
        return n

    with pytest.raises(ValueError):
        await collect(amap(aiter_from(range(5)), bad, concurrency=2))


async def test_amap_rejects_zero_concurrency():
    with pytest.raises(ValueError):
        await collect(amap(aiter_from([1]), double, concurrency=0))


async def test_amap_cancels_in_flight_work_when_the_consumer_stops():
    cancelled = []

    async def slow(n):
        try:
            await asyncio.sleep(1)
            return n
        except asyncio.CancelledError:
            cancelled.append(n)
            raise

    stream = amap(aiter_from(range(4)), slow, concurrency=4)
    task = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()
    assert len(cancelled) >= 1


async def test_merge_interleaves_by_arrival():
    async def fast():
        for i in range(3):
            await asyncio.sleep(0.005)
            yield f"f{i}"

    async def slow():
        for i in range(2):
            await asyncio.sleep(0.02)
            yield f"s{i}"

    out = await collect(merge(fast(), slow()))
    assert sorted(out) == ["f0", "f1", "f2", "s0", "s1"]
    assert out.index("f1") < out.index("s0")  # the fast source is not held back


async def test_merge_of_a_single_source():
    assert await collect(merge(aiter_from([1, 2]))) == [1, 2]


async def test_merge_of_nothing_terminates():
    assert await collect(merge()) == []


async def test_merge_handles_sources_of_different_lengths():
    out = await collect(merge(aiter_from([1]), aiter_from([2, 3, 4])))
    assert sorted(out) == [1, 2, 3, 4]


async def test_take_until_stops_at_an_item_boundary():
    event = asyncio.Event()

    async def ticker():
        i = 0
        while True:
            await asyncio.sleep(0.005)
            yield i
            i += 1

    async def stopper():
        await asyncio.sleep(0.03)
        event.set()

    asyncio.ensure_future(stopper())
    out = await collect(take_until(ticker(), event))
    assert out == list(range(len(out)))
    assert 1 <= len(out) <= 8


async def test_take_until_returns_immediately_if_already_set():
    event = asyncio.Event()
    event.set()
    assert await collect(take_until(aiter_from([1, 2, 3]), event)) == []


async def test_take_until_ends_with_the_source():
    assert await collect(take_until(aiter_from([1, 2]), asyncio.Event())) == [1, 2]


async def test_throttle_paces_the_stream_without_dropping():
    out, elapsed = await timed(collect(throttle(aiter_from(range(4)), per_second=200)))
    assert out == [0, 1, 2, 3]
    assert elapsed >= 0.01  # 3 waits at 5 ms after the initial token
