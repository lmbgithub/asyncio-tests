"""Level 4 — queues, worker pools, and backpressure.

`asyncio.Queue` is the seam between a producer that decides *when* work arrives
and a consumer that decides *how fast* it is done. Three things people get
wrong:

1. **An unbounded queue is not "no backpressure", it is "backpressure applied
   by the OOM killer".** A bounded queue makes a fast producer wait, which is
   the whole mechanism. `maxsize` is required here, not defaulted to zero.

2. **Sentinels versus `join()`.** Putting N `None`s on the queue to stop N
   workers works only if no worker has died; if one has, its sentinel is never
   consumed and shutdown hangs. This pool cancels workers after `join()`
   reports the backlog drained, which is correct whether or not workers died.

3. **`task_done()` belongs in `finally`.** A worker that raises between `get()`
   and `task_done()` leaves the counter permanently short, and `join()` waits
   forever for an item that has already been processed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .errors import PoolClosedError

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class PoolStats:
    """What the pool did. Failures are counted, never silently dropped."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    errors: list[BaseException] = field(default_factory=list, repr=False)

    @property
    def processed(self) -> int:
        return self.completed + self.failed


class WorkerPool(Generic[T, R]):
    """A fixed set of workers draining a bounded queue.

    `submit` blocks when the queue is full — that is the backpressure, and it
    is why the producer must be `await`ing it rather than calling `put_nowait`
    in a loop. `try_submit` is offered for the case where shedding load beats
    waiting, and it reports the shed rather than hiding it.

    A handler that raises does not kill its worker: the failure is recorded and
    the worker takes the next item. One poison message must not take down the
    consumer group.
    """

    def __init__(
        self,
        handler: Callable[[T], Awaitable[R]],
        *,
        workers: int = 4,
        maxsize: int = 32,
        on_result: Callable[[R], None] | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if maxsize < 1:
            raise ValueError(
                "maxsize must be >= 1; an unbounded queue has no backpressure"
            )
        self._handler = handler
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._workers: list[asyncio.Task] = []
        self._n_workers = workers
        self._on_result = on_result
        self._closed = False
        self.results: list[R] = []
        self.stats = PoolStats()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def backlog(self) -> int:
        return self._queue.qsize()

    async def start(self) -> WorkerPool[T, R]:
        if self._workers:
            raise RuntimeError("pool already started")
        self._workers = [
            asyncio.ensure_future(self._worker(i)) for i in range(self._n_workers)
        ]
        return self

    async def _worker(self, index: int) -> None:
        while True:
            item = await self._queue.get()
            try:
                result = await self._handler(item)
            except asyncio.CancelledError:
                # The item was taken but not processed. Mark it done anyway, or
                # a concurrent `join()` blocks forever on work nobody will redo.
                self._queue.task_done()
                raise
            except Exception as exc:
                self.stats.failed += 1
                self.stats.errors.append(exc)
                self._queue.task_done()
                continue
            self.stats.completed += 1
            self.results.append(result)
            if self._on_result is not None:
                self._on_result(result)
            self._queue.task_done()

    async def submit(self, item: T) -> None:
        """Enqueue, waiting for room. This is where a fast producer is slowed."""
        if self._closed:
            raise PoolClosedError("pool is closed")
        self.stats.submitted += 1
        await self._queue.put(item)

    def try_submit(self, item: T) -> bool:
        """Enqueue if there is room; otherwise shed the item and count it."""
        if self._closed:
            raise PoolClosedError("pool is closed")
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.stats.rejected += 1
            return False
        self.stats.submitted += 1
        return True

    async def join(self) -> None:
        """Wait until every submitted item has been processed."""
        await self._queue.join()

    async def aclose(self, *, drain: bool = True) -> PoolStats:
        """Stop the pool. With `drain=True` the backlog is finished first.

        `drain=False` is an abort, and it is honest about it: outstanding items
        are discarded and the count is visible in the backlog at the time of
        the call. Pretending an abort drained is how "we processed everything"
        ends up in a postmortem as a false statement.
        """
        self._closed = True
        if drain:
            await self.join()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        return self.stats

    async def __aenter__(self) -> WorkerPool[T, R]:
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose(drain=exc_type is None)


async def produce(queue: asyncio.Queue, items: list[T]) -> int:
    """Put every item on the queue, respecting its bound. Returns the count."""
    for item in items:
        await queue.put(item)
    return len(items)


async def drain_now(queue: asyncio.Queue) -> list[Any]:
    """Take everything currently queued without waiting for more.

    `get_nowait` until empty, rather than `while not queue.empty(): await get()`
    — that version has a race: another consumer can empty the queue between the
    check and the get, and this coroutine then blocks forever on a queue its
    caller believes it has finished with.
    """
    out: list[Any] = []
    while True:
        try:
            out.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return out


async def batched(
    source: AsyncIterator[T], *, size: int, timeout: float | None = None
) -> AsyncIterator[list[T]]:
    """Group a stream into batches of `size`, or whatever arrived by `timeout`.

    Size alone stalls the tail: the last four items of a 104-item stream sit in
    the buffer until the stream closes, which for a live feed is never. The
    timeout is what turns a batcher into something usable on a real feed, and
    a partial batch on timeout is a feature, not a degraded case.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    batch: list[T] = []
    iterator = source.__aiter__()
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            try:
                if timeout is None:
                    item = await pending
                else:
                    # `shield` keeps the in-flight `__anext__` alive across the
                    # timeout, so the item it eventually produces lands in the
                    # next batch. Timing out a read must never consume an
                    # element — that is silent data loss on a live feed.
                    item = await asyncio.wait_for(asyncio.shield(pending), timeout)
            except asyncio.TimeoutError:
                if batch:
                    yield batch
                    batch = []
                continue
            except StopAsyncIteration:
                pending = None
                if batch:
                    yield batch
                return
            pending = None
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch = []
    finally:
        # The generator can be closed mid-flight (a `break` in the consumer).
        # An orphaned `__anext__` task would then log "never retrieved".
        if pending is not None:
            pending.cancel()
