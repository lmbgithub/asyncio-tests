"""aiolab — asyncio from `async def` to structured concurrency, in seven levels.

Each module is one level, and each one is organised around the same question:
what does the obvious version of this get wrong?

    basics    coroutines, awaiting, sequential vs concurrent, blocking calls
    tasks     structured concurrency, cancellation, task ownership
    sync      locks, semaphores, events, conditions, rate limiting
    queues    bounded queues, worker pools, backpressure, batching
    futures   raw futures, callback bridges, completion order
    streams   async iterators, merging, concurrent map, cooperative stop
    retry     timeouts, jittered backoff, circuit breaking
"""

from __future__ import annotations

from .errors import AiolabError, ConcurrencyError, PoolClosedError, RetryError
from .basics import delay, run_concurrent, run_sequential, timed, to_thread
from .tasks import (
    TaskRegistry,
    cancel_and_wait,
    first_result,
    run_all,
    run_all_settled,
)
from .sync import (
    Once,
    RaceyCounter,
    ReadWriteLock,
    ResourcePool,
    SafeCounter,
    TokenBucket,
    wait_for_event,
)
from .queues import PoolStats, WorkerPool, batched, drain_now, produce
from .futures import (
    CallbackBridge,
    as_completed_results,
    gather_with_index,
    make_future,
    settle,
    shielded,
    wait_any,
)
from .streams import aiter_from, amap, collect, merge, take_until, throttle
from .retry import (
    CircuitBreaker,
    RetryPolicy,
    default_retryable,
    retry,
    retry_with_timeout,
    with_timeout,
)

__version__ = "0.1.0"

__all__ = [
    "AiolabError",
    "ConcurrencyError",
    "PoolClosedError",
    "RetryError",
    "delay",
    "timed",
    "to_thread",
    "run_sequential",
    "run_concurrent",
    "TaskRegistry",
    "cancel_and_wait",
    "first_result",
    "run_all",
    "run_all_settled",
    "Once",
    "RaceyCounter",
    "SafeCounter",
    "ReadWriteLock",
    "ResourcePool",
    "TokenBucket",
    "wait_for_event",
    "PoolStats",
    "WorkerPool",
    "batched",
    "drain_now",
    "produce",
    "CallbackBridge",
    "as_completed_results",
    "gather_with_index",
    "make_future",
    "settle",
    "shielded",
    "wait_any",
    "aiter_from",
    "amap",
    "collect",
    "merge",
    "take_until",
    "throttle",
    "CircuitBreaker",
    "RetryPolicy",
    "default_retryable",
    "retry",
    "retry_with_timeout",
    "with_timeout",
]
