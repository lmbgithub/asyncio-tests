"""Level 5 — futures: the object under every await.

A `Task` is a `Future` with a coroutine driving it. A bare `Future` is what you
reach for when something *other* than a coroutine will supply the result: a
callback API, another thread, a protocol's `data_received`. Getting this seam
right is what lets a callback-shaped library be awaited without a polling loop.

The rules that matter:

- **Create futures with `loop.create_future()`**, not `asyncio.Future()`. A
  custom loop can return its own optimised subclass, and the constructor form
  silently binds to the running loop at construction time — wrong when the
  object outlives or predates the loop it will be resolved on.
- **Setting a result on a done future raises `InvalidStateError`.** Any bridge
  where a timeout and a callback can both fire needs a guard, or the late
  arrival crashes a thread nobody is watching.
- **Cross-thread resolution goes through `call_soon_threadsafe`.** Calling
  `set_result` from another thread corrupts loop state in ways that surface
  much later, somewhere else.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

from .errors import ConcurrencyError

T = TypeVar("T")


def make_future(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Future:
    """Create a future bound to the running loop, the supported way."""
    return (loop or asyncio.get_running_loop()).create_future()


def settle(
    future: asyncio.Future, result: T | None = None, error: BaseException | None = None
) -> bool:
    """Resolve a future, returning False if it was already done.

    Idempotent by design. A timeout path and a completion callback racing to
    settle the same future is normal; an `InvalidStateError` from the loser is
    not an error condition, it just means it lost.
    """
    if future.done():
        return False
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(result)
    return True


class CallbackBridge:
    """Turn a callback-style API into something awaitable.

    The pattern behind every `asyncio` protocol implementation: hand out a
    `resolve`/`reject` pair, hand the caller a future, and let the transport
    call whichever it reaches first.

    `from_thread=True` routes the resolution through `call_soon_threadsafe`,
    which is mandatory when the callback fires on a worker thread.
    """

    def __init__(self, *, from_thread: bool = False) -> None:
        self._loop = asyncio.get_running_loop()
        self._future = self._loop.create_future()
        self._from_thread = from_thread

    @property
    def future(self) -> asyncio.Future:
        return self._future

    def resolve(self, value: T) -> None:
        if self._from_thread:
            self._loop.call_soon_threadsafe(settle, self._future, value, None)
        else:
            settle(self._future, value)

    def reject(self, error: BaseException) -> None:
        if self._from_thread:
            self._loop.call_soon_threadsafe(settle, self._future, None, error)
        else:
            settle(self._future, None, error)

    def __await__(self):
        return self._future.__await__()


async def as_completed_results(
    factories: Iterable[Callable[[], Awaitable[T]]],
) -> list[tuple[int, T | BaseException]]:
    """Yield `(input_index, outcome)` in *completion* order.

    `asyncio.as_completed` gives back anonymous awaitables: you learn that
    something finished, not which something. For a fan-out over 200 hosts that
    is useless, so the index is carried through the task itself.
    """
    tasks: dict[asyncio.Task, int] = {}
    for index, factory in enumerate(factories):
        tasks[asyncio.ensure_future(factory())] = index
    if not tasks:
        return []

    out: list[tuple[int, T | BaseException]] = []
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                exc = task.exception()
                out.append((tasks[task], exc if exc is not None else task.result()))
    except asyncio.CancelledError:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    return out


async def wait_any(
    futures: Iterable[asyncio.Future], *, timeout: float | None = None
) -> asyncio.Future:
    """Return the first future to settle; leave the rest alone.

    Deliberately does *not* cancel the losers — the caller may still want them.
    That is the difference from `tasks.first_result`, which owns its children
    and therefore must clean them up. Ownership decides who cancels.
    """
    pending = set(futures)
    if not pending:
        raise ValueError("wait_any requires at least one future")
    done, _ = await asyncio.wait(
        pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
    )
    if not done:
        raise asyncio.TimeoutError(f"no future settled within {timeout}s")
    return next(iter(done))


async def gather_with_index(factories: Iterable[Callable[[], Awaitable[T]]]) -> list[T]:
    """`gather` with the failure aggregated instead of first-wins.

    `gather` raises the first exception and discards the rest. When a batch of
    50 uploads fails, "the first one raised ConnectionError" is far less useful
    than "38 failed, here they are" — the second tells you it is the network,
    not the file.
    """
    tasks = [asyncio.ensure_future(f()) for f in factories]
    await asyncio.gather(*tasks, return_exceptions=True)
    failures = [
        t.exception() for t in tasks if not t.cancelled() and t.exception() is not None
    ]
    if failures:
        raise ConcurrencyError([f for f in failures if f is not None])
    return [t.result() for t in tasks]


async def shielded(awaitable: Awaitable[T], *, timeout: float) -> T | None:
    """Give up waiting after `timeout`, but let the work finish.

    The right shape for "record this payment, but do not hold the request open"
    — cancelling a half-written transaction is worse than waiting. Returns
    `None` on timeout; the underlying task keeps running, so the caller must
    still own it somewhere (see `tasks.TaskRegistry`).
    """
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout)
    except asyncio.TimeoutError:
        return None
