import asyncio

import pytest

from aiolab.sync import (
    Once,
    RaceyCounter,
    ReadWriteLock,
    ResourcePool,
    SafeCounter,
    TokenBucket,
    wait_for_event,
)


class FakeClock:
    """A monotonic clock the tests advance by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


async def test_the_race_is_real_and_needs_an_await():
    counter = RaceyCounter()
    await asyncio.gather(*(counter.increment() for _ in range(100)))
    assert counter.value < 100  # every increment but one is lost


async def test_the_lock_fixes_it():
    counter = SafeCounter()
    await asyncio.gather(*(counter.increment() for _ in range(100)))
    assert counter.value == 100


async def test_no_await_means_no_race():
    # The same read-modify-write without a suspension point is atomic with
    # respect to other coroutines, which is why not every counter needs a lock.
    value = 0

    async def increment():
        nonlocal value
        value = value + 1

    await asyncio.gather(*(increment() for _ in range(100)))
    assert value == 100


async def test_once_runs_the_initializer_a_single_time():
    once = Once()
    calls = []

    async def init():
        await asyncio.sleep(0.01)
        calls.append(1)
        return "conn"

    results = await asyncio.gather(*(once.run(init) for _ in range(5)))
    assert results == ["conn"] * 5
    assert once.calls == 1
    assert calls == [1]


async def test_once_takes_no_lock_after_completion():
    once = Once()
    await once.run(lambda: asyncio.sleep(0, result="x"))
    assert once.done is True
    assert await once.run(lambda: asyncio.sleep(0, result="y")) == "x"


async def test_once_stays_retryable_after_a_failed_initializer():
    once = Once()
    attempts = {"n": 0}

    async def init():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("no route to host")
        return "conn"

    with pytest.raises(ConnectionError):
        await once.run(init)
    assert once.done is False
    assert await once.run(init) == "conn"


async def test_pool_bounds_concurrency_to_its_size():
    pool = ResourcePool(["c1", "c2"])

    async def use():
        async with pool.borrow() as conn:
            await asyncio.sleep(0.01)
            return conn

    await asyncio.gather(*(use() for _ in range(6)))
    assert pool.peak_in_use == 2
    assert pool.available == 2


async def test_pool_returns_the_resource_when_the_body_raises():
    pool = ResourcePool(["c1"])
    with pytest.raises(RuntimeError):
        async with pool.borrow():
            raise RuntimeError("query failed")
    assert pool.available == 1


async def test_pool_returns_the_resource_when_the_body_is_cancelled():
    pool = ResourcePool(["c1"])

    async def use():
        async with pool.borrow():
            await asyncio.sleep(5)

    task = asyncio.ensure_future(use())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool.available == 1


async def test_pool_rejects_an_empty_resource_list():
    with pytest.raises(ValueError):
        ResourcePool([])


async def test_bucket_starts_full_and_bursts():
    bucket = TokenBucket(10, capacity=5, clock=FakeClock())
    assert [bucket.try_acquire() for _ in range(6)] == [True] * 5 + [False]


async def test_bucket_refills_at_exactly_the_rate():
    clock = FakeClock()
    bucket = TokenBucket(10, capacity=10, clock=clock, sleep=clock.sleep)
    for _ in range(10):
        bucket.try_acquire()
    assert bucket.tokens == 0
    clock.now += 0.5
    assert bucket.tokens == pytest.approx(5.0)


async def test_bucket_never_exceeds_capacity():
    clock = FakeClock()
    bucket = TokenBucket(10, capacity=3, clock=clock)
    clock.now += 100
    assert bucket.tokens == 3


async def test_bucket_acquire_waits_the_exact_deficit():
    clock = FakeClock()
    bucket = TokenBucket(5, capacity=1, clock=clock, sleep=clock.sleep)
    assert await bucket.acquire() == 0.0
    assert await bucket.acquire() == pytest.approx(0.2)
    assert clock.now == pytest.approx(0.2)


async def test_bucket_serialises_waiters_instead_of_stampeding():
    clock = FakeClock()
    bucket = TokenBucket(5, capacity=1, clock=clock, sleep=clock.sleep)
    await bucket.acquire()
    waits = await asyncio.gather(*(bucket.acquire() for _ in range(3)))
    # Each waiter pays for its own token; they do not all wake on the same one.
    assert sum(waits) == pytest.approx(0.6)


async def test_bucket_rejects_an_unsatisfiable_request():
    bucket = TokenBucket(5, capacity=2)
    with pytest.raises(ValueError, match="capacity"):
        await bucket.acquire(3)


async def test_bucket_rejects_a_non_positive_rate():
    with pytest.raises(ValueError):
        TokenBucket(0)
    with pytest.raises(ValueError):
        TokenBucket(1, capacity=0)


async def test_bucket_ignores_a_backwards_clock():
    clock = FakeClock()
    bucket = TokenBucket(10, capacity=10, clock=clock)
    for _ in range(10):
        bucket.try_acquire()
    clock.now -= 5  # NTP stepped the clock back
    assert bucket.tokens == 0


async def test_rwlock_allows_concurrent_readers():
    lock = ReadWriteLock()
    peak = 0

    async def reader():
        nonlocal peak
        async with lock.read():
            peak = max(peak, lock.readers)
            await asyncio.sleep(0.01)

    await asyncio.gather(*(reader() for _ in range(4)))
    assert peak == 4


async def test_rwlock_writer_excludes_readers():
    lock = ReadWriteLock()
    order = []

    async def writer():
        async with lock.write():
            order.append("w-start")
            await asyncio.sleep(0.02)
            order.append("w-end")

    async def reader():
        await asyncio.sleep(0.005)
        async with lock.read():
            order.append("r")

    await asyncio.gather(writer(), reader())
    assert order == ["w-start", "w-end", "r"]


async def test_rwlock_does_not_starve_a_waiting_writer():
    lock = ReadWriteLock()
    order = []

    async def reader(name, delay):
        await asyncio.sleep(delay)
        async with lock.read():
            order.append(name)
            await asyncio.sleep(0.02)

    async def writer():
        await asyncio.sleep(0.005)
        async with lock.write():
            order.append("writer")

    await asyncio.gather(reader("r1", 0.0), writer(), reader("r2", 0.01))
    # r2 arrives while the writer is queued, so it waits behind it.
    assert order == ["r1", "writer", "r2"]


async def test_rwlock_cancelled_writer_leaves_no_phantom_waiter():
    lock = ReadWriteLock()
    await lock.acquire_write()

    blocked = asyncio.ensure_future(lock.acquire_write())
    await asyncio.sleep(0.01)
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    await lock.release_write()

    # A leaked waiting-writer count would block this reader forever.
    await asyncio.wait_for(lock.acquire_read(), 0.5)
    assert lock.readers == 1


async def test_wait_for_event_true_when_set():
    event = asyncio.Event()
    asyncio.get_running_loop().call_later(0.01, event.set)
    assert await wait_for_event(event, 0.5) is True


async def test_wait_for_event_false_on_timeout():
    assert await wait_for_event(asyncio.Event(), 0.01) is False
