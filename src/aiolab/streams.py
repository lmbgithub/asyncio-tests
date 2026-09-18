"""Level 6 — async iterators, generators, and pipelines.

An async generator is the natural shape for "results that arrive over time":
the consumer's `async for` drives the producer, so a slow consumer is automatic
backpressure with no queue involved at all.

Two things bite people here.

**Async generators need explicit cleanup.** A `break` out of an `async for`
leaves the generator suspended; its `finally` runs whenever the garbage
collector gets round to it, possibly after the loop has closed, producing the
famous "Task was destroyed but it is pending" at interpreter shutdown. Every
generator in this module cleans up in `finally`, and `aclose()` is called
explicitly on sources this module owns.

**`async for` is sequential.** Mapping an async function over a stream with a
plain `async for` gives concurrency of exactly one. `amap` is the fix, and it
keeps output order while doing so.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def aiter_from(items: Iterable[T], *, delay: float = 0.0) -> AsyncIterator[T]:
    """Turn a plain iterable into an async iterator. The test fixture of choice."""
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


async def collect(source: AsyncIterator[T], *, limit: int | None = None) -> list[T]:
    """Drain an async iterator into a list, optionally stopping early.

    Closes the source when it stops early, rather than leaving a suspended
    generator for the collector to finalise at an arbitrary later moment.
    """
    out: list[T] = []
    try:
        async for item in source:
            out.append(item)
            if limit is not None and len(out) >= limit:
                break
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()
    return out


async def amap(
    source: AsyncIterator[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    concurrency: int = 4,
    ordered: bool = True,
) -> AsyncIterator[R]:
    """Apply an async function across a stream with real concurrency.

    `ordered=True` costs a small buffer — a finished item waits for its slower
    predecessor — and buys a deterministic output stream. Unordered is faster
    to first result and is the right choice only when nothing downstream cares
    about position. Choose it deliberately; do not inherit it by accident from
    whichever `as_completed` idiom you copied.

    Concurrency is bounded by construction: at most `concurrency` items are in
    flight, so a fast source cannot turn this into an unbounded fan-out.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    in_flight: list[asyncio.Task] = []
    exhausted = False
    iterator = source.__aiter__()
    try:
        while True:
            while not exhausted and len(in_flight) < concurrency:
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    exhausted = True
                    break
                in_flight.append(asyncio.ensure_future(fn(item)))

            if not in_flight:
                return

            if ordered:
                yield await in_flight.pop(0)
            else:
                done, _ = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    in_flight.remove(task)
                    yield task.result()
    finally:
        # A consumer that breaks out mid-stream must not leave live tasks behind.
        for task in in_flight:
            task.cancel()
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)


async def merge(*sources: AsyncIterator[T]) -> AsyncIterator[T]:
    """Interleave several async iterators, yielding items as they arrive.

    One pending `__anext__` per source, replaced as each one lands. The naive
    round-robin — pull one from each in turn — makes every source wait for the
    slowest, which is the exact opposite of what merging a set of live feeds is
    for.
    """
    pending: dict[asyncio.Task, AsyncIterator[T]] = {}
    for source in sources:
        iterator = source.__aiter__()
        pending[asyncio.ensure_future(iterator.__anext__())] = iterator
    try:
        while pending:
            done, _ = await asyncio.wait(
                list(pending), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                iterator = pending.pop(task)
                try:
                    item = task.result()
                except StopAsyncIteration:
                    continue
                yield item
                pending[asyncio.ensure_future(iterator.__anext__())] = iterator
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def take_until(source: AsyncIterator[T], event: asyncio.Event) -> AsyncIterator[T]:
    """Yield from a source until an event is set — a cooperative stop signal.

    Shutdown that works: the consumer stops at an item boundary instead of
    being cancelled halfway through handling one. Cancellation is the blunt
    instrument; an `Event` is the one you can reason about.
    """
    stop = asyncio.ensure_future(event.wait())
    iterator = source.__aiter__()
    nxt: asyncio.Task | None = None
    try:
        while not event.is_set():
            nxt = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({nxt, stop}, return_when=asyncio.FIRST_COMPLETED)
            if nxt not in done:
                return
            try:
                yield nxt.result()
            except StopAsyncIteration:
                return
            finally:
                nxt = None
    finally:
        stop.cancel()
        if nxt is not None:
            nxt.cancel()
            await asyncio.gather(nxt, return_exceptions=True)


async def throttle(source: AsyncIterator[T], *, per_second: float) -> AsyncIterator[T]:
    """Pace a stream to a fixed rate, without dropping anything.

    Rate limiting a *stream* means slowing it; rate limiting a *request* means
    queueing it. Both use the same bucket, but only one of them may drop, and
    that difference belongs to the caller, not to the limiter.
    """
    from .sync import TokenBucket

    bucket = TokenBucket(per_second, capacity=1.0)
    async for item in source:
        await bucket.acquire()
        yield item
