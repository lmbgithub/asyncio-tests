# aiolab

A hands-on tour of asyncio, from `async def` to structured concurrency, in seven levels. Python standard library only, **150 tests**, all running under `pytest-asyncio` auto mode.

Most asyncio material stops at running two coroutines at once. The bugs that actually reach production come from four other places: a task nobody owns, a cancellation somebody swallowed, a queue with no bound, and a retry loop that does not handle `CancelledError`. Each module here is built around one of them, so the failure mode is visible and testable rather than described.

```
$ aiolab tasks
tasks — one child fails, the siblings are cancelled and awaited
  raised                   ConcurrencyError
  errors                   [RuntimeError('upstream 503')]
  siblings cancelled       ['a', 'b']
  elapsed                  11.5 ms  (not 10 s)

$ aiolab races
sync — 1000 increments, with and without the lock
  no lock                  1 / 1000  (lost updates)
  asyncio.Lock             1000 / 1000
```

That first number is the whole point of the second module: 999 of 1000 increments are lost, and they are lost _only_ because there is an `await` between the read and the write.

## The five decisions worth discussing

### 1. `gather` leaves the siblings running

```python
await asyncio.gather(fetch(a), fetch(b), write_to_db(c))
```

When `fetch(a)` raises, `gather` propagates that exception to the caller immediately — and `write_to_db(c)` is **still running**. The caller unwinds, closes its connection pool, logs a failure, and a detached task keeps writing to a database that the request it belonged to has already given up on.

`run_all` cancels the survivors, _awaits them to a stop_, and only then raises a `ConcurrencyError` carrying every error collected. The awaiting is not cosmetic: `task.cancel()` merely schedules a `CancelledError` at the task's next suspension point, so returning without awaiting races the children's `finally` blocks against whatever the caller does next.

This is `asyncio.TaskGroup` semantics. It is written out longhand because the floor here is Python 3.10, where `TaskGroup` does not exist — and because someone who has only ever called `gather` should be able to read what a task group is actually doing for them.

### 2. A fire-and-forget task can be garbage-collected mid-flight

```python
asyncio.create_task(send_receipt(order))   # handle dropped
```

The event loop keeps only a _weak_ reference to a running task. Drop the handle and the task may be collected before it finishes — code that usually works and occasionally does not send the receipt, which is the worst available failure mode because it is unreproducible.

`TaskRegistry` owns them: strong references held until completion, failures recorded rather than surfacing as "Task exception was never retrieved" at some unrelated GC point, and `aclose()` as a single awaitable shutdown for the whole scope.

### 3. A semaphore bounds concurrency, not rate

These get conflated constantly. `Semaphore(10)` against a 100-request-per-second quota will sail straight through the quota the moment each request gets fast — ten in flight at 1 ms each is 10,000/s. They are different constraints: `ResourcePool` bounds _how many at once_, `TokenBucket` bounds _how many per second_, and neither substitutes for the other.

`TokenBucket` computes tokens lazily from elapsed time rather than refilling from a background task — a timer per limiter is a task leak waiting to happen, and it keeps the loop awake on an idle service. Waiters are serialised under a lock, because ten waiters that all compute "0.1 s until a token" from the same instant will all wake to find a single token, which is a thundering herd violating the very limit being enforced.

### 4. An unbounded queue is backpressure applied by the OOM killer

`WorkerPool` requires `maxsize`; there is no unbounded default. A full queue makes `submit` wait, and that wait _is_ the backpressure travelling upstream. Where waiting is worse than shedding, `try_submit` sheds — and counts what it shed, because silent loss becomes an unreproducible complaint three weeks later.

Two smaller things the same module gets right:

- **`task_done()` lives in `finally`.** A worker that raises between `get()` and `task_done()` leaves the counter permanently short and `join()` waits forever for an item that was already processed.
- **Shutdown does not use sentinels.** Putting N `None`s on the queue to stop N workers only works if no worker has died; if one has, its sentinel is never consumed and shutdown hangs. Draining via `join()` and then cancelling is correct either way.

### 5. Retry loops are where cancellation discipline dies

`asyncio.CancelledError` inherits from `BaseException` precisely so that `except Exception` cannot eat it. Then someone widens the handler "to be safe", and the retry loop absorbs the cancellation: the service stops shutting down, and the deploy hangs on a task that will retry forever.

It is re-raised explicitly at every point in this library that catches broadly — in `retry`, in the worker pool, in the circuit breaker. Additional choices in that module:

- **`retry` takes a factory, not a coroutine.** A coroutine object can be awaited exactly once, so a retry helper accepting one can never actually retry — it fails with `RuntimeError: cannot reuse already awaited coroutine` on attempt two, under exactly the failure conditions nobody tested.
- **Full jitter, not plain exponential.** Clients backing off in lockstep at 1 s, 2 s, 4 s reconverge on every step and re-DDoS the service they are waiting for.
- **A retry policy without a per-attempt timeout retries a hung call zero times** — attempt one never returns, so the budget is never spent.
- **The circuit breaker counts _consecutive_ failures.** Cumulative counting trips the breaker on a service with a 1% error rate after a few hours of perfectly healthy operation.

## The seven levels

| Module       | Subject                                                | The thing worth reading                                            |
| ------------ | ------------------------------------------------------ | ------------------------------------------------------------------ |
| `basics.py`  | `async`/`await`, sequential vs concurrent, `to_thread` | `await` is a suspension point, not a thread                        |
| `tasks.py`   | Structured concurrency, cancellation, ownership        | `run_all`, `first_result`, `TaskRegistry`                          |
| `sync.py`    | Lock, Event, Condition, Semaphore, rate limiting       | `RaceyCounter` vs `SafeCounter`, writer-preferring `ReadWriteLock` |
| `queues.py`  | Bounded queues, worker pools, batching                 | Backpressure, drain-vs-abort, timeout batching without data loss   |
| `futures.py` | Raw futures, callback bridges, completion order        | `CallbackBridge`, cross-thread resolution, `shielded`              |
| `streams.py` | Async iterators and generators, pipelines              | `amap` with bounded concurrency, `merge`, cooperative `take_until` |
| `retry.py`   | Timeouts, jittered backoff, circuit breaking           | Cancellation-correct retry with an injected clock                  |

Ownership decides who cancels — the rule that separates two similar-looking functions in this library. `tasks.first_result` creates its children, so it cancels the losers. `futures.wait_any` was handed futures it does not own, so it leaves them alone.

## Quick start

```bash
pip install -e ".[dev]"

aiolab                      # every demo
aiolab queue                # one of them
pytest -q                   # 150 tests
python examples/pipeline_demo.py
```

The example wires every level into one pipeline — rate-limited, retried, timed-out fetches, a bounded concurrent map, timeout-flushed batching, a worker pool doing the writes, and a cooperative stop signal — against an in-process server that injects failures. No network, no keys:

```
$ python examples/pipeline_demo.py
requests issued      25
injected failures    12
batches written      3
titles written       12
stopped early        True
first batch          ['title-1', 'title-2', 'title-3', 'title-4']
```

## Library use

```python
from aiolab import RetryPolicy, TaskRegistry, retry_with_timeout, run_all

async def main():
    # All-or-nothing fan-out, bounded, siblings cancelled on first failure.
    pages = await run_all([lambda u=u: fetch(u) for u in urls], limit=8)

    # Retry with jittered backoff and a per-attempt deadline.
    body = await retry_with_timeout(
        lambda: fetch(url), per_attempt_timeout=2.0,
        policy=RetryPolicy(attempts=4, base_delay=0.2),
    )

    # Background work that cannot be garbage-collected mid-flight.
    async with TaskRegistry("requests") as bg:
        bg.spawn(write_audit_log(event))
```

## Tests

```bash
$ pytest -q
150 passed in 2.61s
```

| Suite             | Tests | Covers                                                                                   |
| ----------------- | ----- | ---------------------------------------------------------------------------------------- |
| `test_basics.py`  | 9     | Sequential vs concurrent timing, order preservation, `to_thread` not blocking the loop   |
| `test_tasks.py`   | 25    | Sibling cancellation, error aggregation, caller cancellation, registry ownership         |
| `test_sync.py`    | 24    | The race being real, `Once` under contention, exact bucket arithmetic, writer starvation |
| `test_queues.py`  | 21    | Backpressure blocking, load shedding, drain vs abort, batching across a timeout          |
| `test_futures.py` | 18    | Idempotent settling, cross-thread callbacks, completion order, `shield` semantics        |
| `test_streams.py` | 18    | Ordered/unordered `amap`, concurrency bounds, merge fairness, mid-stream cleanup         |
| `test_retry.py`   | 23    | Exact delay sequences, budget enforcement, cancellation, breaker state machine           |
| `test_cli.py`     | 12    | Every demo exits zero, public API surface, the example end to end                        |

Every timing-sensitive assertion that _can_ be made exact is made exact: the token bucket, the retry backoff and the circuit breaker all take an injected clock, so the tests assert `0.2` rather than sleeping and tolerating slop. The handful that remain — "concurrent is faster than sequential" — are stated with wide margins, because a test that fails on a loaded CI runner teaches everyone to ignore it.

The failure paths are the point of the suite. A worker cancelled between `get()` and `task_done()`; a `ReadWriteLock` writer cancelled while queued, leaving a phantom waiter that would block every future reader; a resource pool whose borrower is cancelled mid-use; a batcher timing out with a read already in flight. Each is a real hang, and each has a test that would catch it.

## Not included

- **A production task supervisor.** No restart policies, no exponential re-spawn, no health reporting. `TaskRegistry` is a scope, not a supervisor.
- **`TaskGroup` / `asyncio.timeout` / `ExceptionGroup`.** All 3.11+; the floor here is 3.10, and writing the semantics out is more instructive anyway. On 3.11+ prefer the stdlib versions in real code.
- **Threads and processes beyond `to_thread`.** No executor tuning, no `ProcessPoolExecutor` bridge, no GIL discussion.
- **Networking.** No transports, protocols, or `asyncio.streams`. Every module is testable with no socket, which is what keeps the suite fast and honest.
- **`uvloop` or any benchmark of it.** The numbers here are about _shape_ — sum versus max — not about loop implementations.

## License

MIT — see [LICENSE](LICENSE).
