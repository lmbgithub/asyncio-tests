"""Level 1 — coroutines, awaiting, and what concurrency actually buys.

The single idea a reader has to internalise before anything else in this
package makes sense: `await` is a *suspension point*, not a thread. Between two
suspension points a coroutine runs alone and cannot be interrupted by another
coroutine — which is why some races that plague threaded code cannot happen
here, and why the ones that remain are always visible as an `await` sitting in
the middle of a read-modify-write.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")


async def delay(value: T, seconds: float = 0.0) -> T:
    """Sleep, then return `value`. The canonical stand-in for slow I/O."""
    await asyncio.sleep(seconds)
    return value


async def timed(awaitable: Awaitable[T]) -> tuple[T, float]:
    """Await something and report how long it took, in seconds.

    Uses `perf_counter`, not `time()`: the wall clock can step backwards on an
    NTP correction and produce a negative duration, which then poisons any
    percentile computed from it.
    """
    started = time.perf_counter()
    result = await awaitable
    return result, time.perf_counter() - started


async def run_sequential(factories: Sequence[Callable[[], Awaitable[T]]]) -> list[T]:
    """Await each coroutine one after another. Total time is the *sum*.

    Takes factories rather than coroutine objects because a coroutine can only
    be awaited once; handing the same list to both this function and
    `run_concurrent` — as the demo does — requires a fresh object per call.
    """
    return [await factory() for factory in factories]


async def run_concurrent(factories: Sequence[Callable[[], Awaitable[T]]]) -> list[T]:
    """Run everything at once. Total time is the *maximum*, results stay in order.

    `asyncio.gather` preserves argument order in its result list regardless of
    completion order. Relying on completion order here is a classic bug: it
    happens to be right when the work is uniform and silently wrong the moment
    one item is slower.
    """
    return list(await asyncio.gather(*(factory() for factory in factories)))


async def to_thread(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a blocking function without stalling the loop.

    A CPU-bound or blocking-I/O call made directly from a coroutine freezes
    *every* task on that loop, including the ones whose only job is to answer a
    health check. This is the escape hatch; `asyncio.sleep` is not a substitute
    for it, and `time.sleep` inside a coroutine is the bug it exists to prevent.
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


def is_awaited_correctly(obj: Any) -> bool:
    """True if `obj` is something `await` accepts.

    Exists for the error message: forgetting parentheses on an async call and
    awaiting the function object raises `TypeError: object function can't be
    used in 'await' expression`, which reads as a typing problem rather than
    the missing `()` it actually is.
    """
    return asyncio.iscoroutine(obj) or isinstance(obj, asyncio.Future) or hasattr(obj, "__await__")


async def consume(iterable: Iterable[Awaitable[T]]) -> list[T]:
    """Await an iterable of awaitables in order, returning their results."""
    return [await item for item in iterable]
