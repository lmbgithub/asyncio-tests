"""Level 7 — timeouts, retries, and cancellation-correct backoff.

Retry code is where cancellation discipline usually breaks. The archetypal bug:

    while True:
        try:
            return await call()
        except Exception:
            await asyncio.sleep(delay)

`asyncio.CancelledError` is a `BaseException`, so it escapes that `except` — but
the same author usually writes `except BaseException` a week later "to be
safe", at which point the retry loop swallows the cancellation and the service
refuses to shut down. It is not caught here, ever, and the guard is explicit
rather than incidental.

Three more decisions:

- **Jitter is not optional.** Synchronised clients retrying at 1s, 2s, 4s
  reconverge on every step and re-DDoS the service they are waiting for. Full
  jitter — a uniform draw over `[0, backoff]` — decorrelates them.
- **Which errors are retryable is the caller's call.** Retrying a 400 is
  pointless; retrying a 503 is the whole idea. `retry_on` takes a predicate,
  and the default retries nothing surprising.
- **The sleeper and the RNG are injectable**, so the tests assert the exact
  delay sequence instead of measuring wall-clock and tolerating slop.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from .errors import RetryError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter, capped and bounded.

    `max_delay` matters more than it looks: without it, attempt 12 of an
    exponential policy sleeps for over an hour, and a caller who asked for
    "retry a few times" has built an indefinite hang.
    """

    attempts: int = 3
    base_delay: float = 0.05
    factor: float = 2.0
    max_delay: float = 5.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.factor < 1:
            raise ValueError("factor must be >= 1; a shrinking backoff is not a backoff")

    def delay_for(self, attempt: int, rand: Callable[[], float] = random.random) -> float:
        """Delay *before* attempt `attempt` (1-based). Attempt 1 never waits."""
        if attempt <= 1:
            return 0.0
        raw = min(self.max_delay, self.base_delay * (self.factor ** (attempt - 2)))
        return raw * rand() if self.jitter else raw


def default_retryable(exc: BaseException) -> bool:
    """Retry ordinary exceptions and timeouts; never retry programmer errors.

    A `TypeError` or `ValueError` will fail identically on every attempt, so
    retrying one only delays the traceback and triples the log volume.
    """
    if isinstance(exc, (TypeError, ValueError, KeyError, AttributeError)):
        return False
    return isinstance(exc, Exception)


async def retry(
    factory: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    *,
    retry_on: Callable[[BaseException], bool] = default_retryable,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call `factory()` until it succeeds or the attempt budget runs out.

    Takes a factory, not a coroutine: a coroutine object can be awaited exactly
    once, so a retry helper that accepts one can never actually retry — it
    raises `RuntimeError: cannot reuse already awaited coroutine` on attempt
    two, and only under the failure conditions nobody tested.
    """
    policy = policy or RetryPolicy()
    last: BaseException | None = None

    for attempt in range(1, policy.attempts + 1):
        delay = policy.delay_for(attempt, rand)
        if delay > 0:
            await sleep(delay)
        try:
            return await factory()
        except asyncio.CancelledError:
            # Never retried, never wrapped, never counted as a failure.
            raise
        except BaseException as exc:
            if not retry_on(exc):
                raise
            last = exc
            if on_retry is not None:
                on_retry(attempt, exc, delay)

    assert last is not None
    raise RetryError(policy.attempts, last)


async def with_timeout(
    factory: Callable[[], Awaitable[T]], seconds: float, *, default: T | None = None
) -> T | None:
    """Run with a deadline, returning `default` instead of raising on timeout.

    `wait_for` cancels the inner task and waits for it to unwind before raising
    — which is what makes it safe, and also why a timeout on an operation with
    a slow `finally` can take longer than the timeout. That is a property to
    know about, not a bug to work around.
    """
    try:
        return await asyncio.wait_for(factory(), seconds)
    except asyncio.TimeoutError:
        return default


async def retry_with_timeout(
    factory: Callable[[], Awaitable[T]],
    *,
    per_attempt_timeout: float,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Per-attempt deadline plus backoff — the combination worth shipping.

    A retry policy without a per-attempt timeout retries a *hung* call zero
    times: attempt one never returns, so the budget is never spent. The timeout
    is what makes the attempts count real.
    """

    async def attempt() -> T:
        return await asyncio.wait_for(factory(), per_attempt_timeout)

    return await retry(
        attempt,
        policy,
        retry_on=lambda exc: (
            isinstance(exc, asyncio.TimeoutError) or default_retryable(exc)
        ),
        sleep=sleep,
        rand=rand,
    )


class CircuitBreaker:
    """Stop calling a service that is already down.

    Retries help with a blip and hurt during an outage: every client retrying
    three times triples the load on a service that is failing *because* of
    load. After `threshold` consecutive failures the breaker opens and calls
    fail immediately for `reset_after` seconds, then one probe is allowed
    through (half-open). A success closes it; a failure re-opens it.

    Consecutive, not cumulative: a service with a 1% error rate would otherwise
    trip the breaker after a few hours of perfectly healthy operation.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        *,
        threshold: int = 3,
        reset_after: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        import time as _time

        self.threshold = threshold
        self.reset_after = reset_after
        self._clock = clock or _time.monotonic
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return self.CLOSED
        if self._clock() - self._opened_at >= self.reset_after:
            return self.HALF_OPEN
        return self.OPEN

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = self._clock()

    async def call(self, factory: Callable[[], Awaitable[T]]) -> T:
        if self.state == self.OPEN:
            raise RetryError(self._failures, RuntimeError("circuit is open"))
        try:
            result = await factory()
        except asyncio.CancelledError:
            # A cancelled call says nothing about the remote service's health.
            raise
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result
