"""Level 3 — locks, semaphores, events, conditions, and rate limits.

The first thing to be clear about: an asyncio race needs an `await` in the
middle of a read-modify-write. Without a suspension point the operation is
atomic with respect to other coroutines, so half the mutexes people port over
from threaded code are guarding nothing. `RaceyCounter` here exists to make
that concrete — it is genuinely broken, and it is broken *only* because of the
`await` between its read and its write.

The second: a `Semaphore` bounds **concurrency**, not **rate**. Ten permits
against a 100-requests-per-second quota will blow through the quota happily as
soon as each request is fast. Those are different constraints and need
different objects, so `TokenBucket` is separate.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class RaceyCounter:
    """Deliberately broken. The teaching exhibit for "where races live".

    `read -> await -> write` is not atomic: every other coroutine gets to run
    at the `await`, reads the same stale value, and the increments are lost.
    """

    def __init__(self) -> None:
        self.value = 0

    async def increment(self) -> None:
        current = self.value
        await asyncio.sleep(0)  # the suspension point that creates the race
        self.value = current + 1


class SafeCounter:
    """The same operation, with the critical section actually held.

    `asyncio.Lock` is not reentrant: a coroutine that acquires it twice
    deadlocks against itself, and because it is a single-threaded deadlock it
    manifests as a task that quietly never completes rather than a traceback.
    """

    def __init__(self) -> None:
        self.value = 0
        self._lock = asyncio.Lock()

    async def increment(self) -> None:
        async with self._lock:
            current = self.value
            await asyncio.sleep(0)
            self.value = current + 1


class Once:
    """Run an async initializer exactly once, however many callers arrive.

    Double-checked: the fast path after initialisation takes no lock at all,
    and the slow path re-tests the flag *inside* the lock because everyone who
    queued on it while the first caller worked would otherwise re-run it.

    If the initializer raises, the flag stays unset — a failed connection setup
    must be retryable, not permanently poisoned. But the error is re-raised to
    every waiter, not just the unlucky one that happened to run it.
    """

    def __init__(self) -> None:
        self._done = False
        self._lock = asyncio.Lock()
        self._result: object = None
        self.calls = 0

    @property
    def done(self) -> bool:
        return self._done

    async def run(self, factory: Callable[[], Awaitable[T]]) -> T:
        if self._done:
            return self._result  # type: ignore[return-value]
        async with self._lock:
            if self._done:
                return self._result  # type: ignore[return-value]
            self.calls += 1
            self._result = await factory()
            self._done = True
        return self._result  # type: ignore[return-value]


class ResourcePool:
    """Bounded concurrency over a fixed set of reusable resources.

    A semaphore alone answers "may I proceed"; a pool also answers "with
    which connection". Handing back the resource happens in `finally`, so a
    task cancelled mid-use does not permanently shrink the pool — the single
    most common way a service degrades to a hang under load.
    """

    def __init__(self, resources: list[T]) -> None:
        if not resources:
            raise ValueError("ResourcePool requires at least one resource")
        self._free: asyncio.LifoQueue = asyncio.LifoQueue()
        for resource in resources:
            self._free.put_nowait(resource)
        self.size = len(resources)
        self.peak_in_use = 0
        self._in_use = 0

    @property
    def available(self) -> int:
        return self._free.qsize()

    async def acquire(self) -> T:
        resource = await self._free.get()
        self._in_use += 1
        self.peak_in_use = max(self.peak_in_use, self._in_use)
        return resource

    def release(self, resource: T) -> None:
        self._in_use -= 1
        self._free.put_nowait(resource)

    def borrow(self) -> "_Borrow[T]":
        """`async with pool.borrow() as conn:` — release is guaranteed."""
        return _Borrow(self)


class _Borrow:
    def __init__(self, pool: ResourcePool) -> None:
        self._pool = pool
        self._resource: object = None

    async def __aenter__(self):
        self._resource = await self._pool.acquire()
        return self._resource

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._pool.release(self._resource)


class TokenBucket:
    """Rate limiting: `rate` operations per second with a burst of `capacity`.

    The clock is injectable so the tests assert exact refill arithmetic instead
    of sleeping and hoping. Tokens are computed lazily from elapsed time rather
    than refilled by a background task: a timer task per limiter is a task leak
    waiting to happen, and it keeps the event loop awake on an idle service.

    Waiting is serialised under a lock. Without it, ten waiters all compute
    "0.1s until a token" from the same instant, all sleep 0.1s, and all wake to
    find one token — a thundering herd that violates the very limit being
    enforced.
    """

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = float(capacity if capacity is not None else rate)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._updated = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._updated
        if elapsed <= 0:
            # A non-monotonic clock must not mint tokens. Clamp, do not trust.
            self._updated = now
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._updated = now

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens

    def try_acquire(self, amount: float = 1.0) -> bool:
        """Non-blocking: take a token if one is there, otherwise say no."""
        self._refill()
        if self._tokens >= amount:
            self._tokens -= amount
            return True
        return False

    async def acquire(self, amount: float = 1.0) -> float:
        """Wait until `amount` tokens are available. Returns seconds waited."""
        if amount > self.capacity:
            raise ValueError(
                f"cannot acquire {amount} tokens from a bucket of capacity {self.capacity}"
            )
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return waited
                deficit = amount - self._tokens
                pause = deficit / self.rate
                waited += pause
                await self._sleep(pause)


class ReadWriteLock:
    """Many concurrent readers, one exclusive writer, writers not starved.

    The naive version lets a steady stream of readers hold the lock forever and
    a writer waits for a gap that a busy service never has. Here, a waiting
    writer blocks *new* readers; the readers already inside finish, then the
    writer runs. That is the trade: slightly slower reads, bounded write
    latency. An unbounded write latency is not a performance characteristic,
    it is an outage.
    """

    def __init__(self) -> None:
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._condition = asyncio.Condition()

    @property
    def readers(self) -> int:
        return self._readers

    async def acquire_read(self) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writer and self._waiting_writers == 0
            )
            self._readers += 1

    async def release_read(self) -> None:
        async with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    async def acquire_write(self) -> None:
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._writer and self._readers == 0
                )
            finally:
                # Decremented in `finally` so a cancelled writer cannot leave a
                # phantom waiter behind, which would block every future reader.
                self._waiting_writers -= 1
            self._writer = True

    async def release_write(self) -> None:
        async with self._condition:
            self._writer = False
            self._condition.notify_all()

    def read(self) -> "_RWContext":
        return _RWContext(self, write=False)

    def write(self) -> "_RWContext":
        return _RWContext(self, write=True)


class _RWContext:
    def __init__(self, lock: ReadWriteLock, *, write: bool) -> None:
        self._lock = lock
        self._write = write

    async def __aenter__(self) -> "ReadWriteLock":
        if self._write:
            await self._lock.acquire_write()
        else:
            await self._lock.acquire_read()
        return self._lock

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._write:
            await self._lock.release_write()
        else:
            await self._lock.release_read()


async def wait_for_event(event: asyncio.Event, timeout: float) -> bool:
    """Wait for an event, returning False on timeout instead of raising.

    `asyncio.Event` is the right primitive for "has this happened yet"; a
    polling loop over a boolean flag is the wrong one, and it is what people
    write when they have not met `Event`.
    """
    try:
        await asyncio.wait_for(event.wait(), timeout)
        return True
    except asyncio.TimeoutError:
        return False
