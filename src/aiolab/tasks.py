"""Level 2 — tasks, cancellation, and structured concurrency.

Three decisions live in this module.

1. **A failed sibling must cancel the others.** `asyncio.gather(...)` without
   `return_exceptions` propagates the first exception to the caller and leaves
   every other task *running*, detached, writing to a database the caller
   believes it has finished with. `run_all` cancels the survivors, waits for
   them to actually finish unwinding, and only then raises.

2. **Fire-and-forget needs an owner.** `asyncio.create_task(coro())` without
   keeping the returned reference is a documented footgun: the loop holds only
   a weak reference, so the task can be garbage-collected mid-flight and the
   work simply never happens. `TaskRegistry` holds strong references and
   discards them on completion.

3. **Cancellation is not failure.** `CancelledError` is re-raised everywhere,
   never aggregated into `ConcurrencyError`. A caller who cancelled a group
   wants a `CancelledError`, not a report that its children mysteriously died.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Sequence, TypeVar

from .errors import ConcurrencyError

T = TypeVar("T")
Factory = Callable[[], Awaitable[T]]


async def cancel_and_wait(tasks: Sequence[asyncio.Task]) -> None:
    """Cancel tasks and wait until every one of them has actually stopped.

    The wait is the point. `task.cancel()` only *schedules* a `CancelledError`
    at the task's next suspension point; returning immediately after calling it
    leaves cleanup in `finally` blocks racing against whatever the caller does
    next — usually closing the connection pool those blocks are trying to use.
    """
    pending = [t for t in tasks if not t.done()]
    for task in pending:
        task.cancel()
    for task in pending:
        try:
            await task
        except asyncio.CancelledError:
            # Expected: this is the cancellation we just requested. Absorbing it
            # here is safe precisely because we are the one who asked for it.
            pass
        except Exception:
            # A task can still fail *while* unwinding. That failure belongs to
            # the error we are already reporting, not to a new one.
            pass


async def run_all(factories: Sequence[Factory[T]], *, limit: int | None = None) -> list[T]:
    """Run everything concurrently; results in input order; all-or-nothing.

    On the first failure the remaining children are cancelled and awaited, then
    a `ConcurrencyError` carrying every collected exception is raised. This is
    `asyncio.TaskGroup` semantics, written out because the floor here is Python
    3.10 — and because a reader who has only ever called `gather` should see
    what the group is actually doing.

    `limit` caps how many run at once. Without it, "concurrent" means "open one
    socket per input", which is fine for 10 inputs and an outage for 10,000.
    """
    if not factories:
        return []
    semaphore = asyncio.Semaphore(limit) if limit else None

    async def guarded(factory: Factory[T]) -> T:
        if semaphore is None:
            return await factory()
        async with semaphore:
            return await factory()

    tasks = [asyncio.ensure_future(guarded(f)) for f in factories]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except asyncio.CancelledError:
        # The *caller* was cancelled. Tear the children down before propagating,
        # or they outlive the scope that created them.
        await cancel_and_wait(tasks)
        raise

    errors = [t.exception() for t in tasks if t.done() and not t.cancelled()]
    failures = [e for e in errors if e is not None]
    if failures:
        await cancel_and_wait(tasks)
        raise ConcurrencyError(failures)
    return [t.result() for t in tasks]


async def run_all_settled(factories: Sequence[Factory[T]]) -> list[T | BaseException]:
    """Run everything and report each outcome, in input order. Never raises.

    The right shape for a fan-out where partial success is a real result — 200
    health checks, say. `gather(return_exceptions=True)` also swallows
    `CancelledError` into the result list, which makes a cancelled group look
    like a group that returned some odd values; here it propagates.
    """
    tasks = [asyncio.ensure_future(f()) for f in factories]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        await cancel_and_wait(tasks)
        raise

    out: list[T | BaseException] = []
    for task in tasks:
        if task.cancelled():
            out.append(asyncio.CancelledError())
        else:
            exc = task.exception()
            out.append(exc if exc is not None else task.result())
    return out


async def first_result(factories: Sequence[Factory[T]]) -> T:
    """Race several attempts; return the first success and cancel the losers.

    The losers are cancelled *and awaited* — see `cancel_and_wait`. A hedged
    request that leaves its slower twin running has not hedged anything, it has
    doubled the load.

    A loser that fails before the winner arrives is ignored: in a race, one
    failed candidate is not the race's failure. Only an all-failed race raises.
    """
    if not factories:
        raise ValueError("first_result requires at least one factory")

    tasks = [asyncio.ensure_future(f()) for f in factories]
    errors: list[BaseException] = []
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is None:
                    await cancel_and_wait(list(pending))
                    return task.result()
                errors.append(exc)
    except asyncio.CancelledError:
        await cancel_and_wait(tasks)
        raise
    raise ConcurrencyError(errors)


class TaskRegistry:
    """Owns background tasks so they cannot be garbage-collected mid-flight.

    The loop keeps only weak references to running tasks. A caller that does
    `asyncio.create_task(send_email())` and drops the handle has written code
    that usually works and occasionally does not send the email — the worst
    possible failure mode, because it is unreproducible.

    Also acts as a scope: `aclose()` cancels everything still running, so a
    shutdown path has one thing to await instead of a set of loose handles.
    """

    def __init__(self, name: str = "registry") -> None:
        self.name = name
        self._tasks: set[asyncio.Task] = set()
        self._closed = False
        self.completed = 0
        self.failed = 0
        self.errors: list[BaseException] = []

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def closed(self) -> bool:
        return self._closed

    def spawn(self, awaitable: Awaitable[T], *, name: str | None = None) -> asyncio.Task:
        """Start a background task and keep a strong reference to it."""
        if self._closed:
            # Close the coroutine rather than leaking it: an un-awaited
            # coroutine emits a RuntimeWarning from the GC, at a point in the
            # program with no connection to the rejected spawn.
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise RuntimeError(f"{self.name} is closed")
        task = asyncio.ensure_future(awaitable)
        if name:
            task.set_name(name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            self.completed += 1
        else:
            # Retrieving the exception here also suppresses the "Task exception
            # was never retrieved" warning the loop emits at GC time — which is
            # a real signal, so it is recorded rather than merely silenced.
            self.failed += 1
            self.errors.append(exc)

    async def drain(self) -> None:
        """Wait for everything currently running, including tasks they spawn."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        """Cancel every outstanding task and wait for it to unwind."""
        self._closed = True
        await cancel_and_wait(list(self._tasks))

    async def __aenter__(self) -> "TaskRegistry":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            await self.drain()
            self._closed = True
        else:
            await self.aclose()
