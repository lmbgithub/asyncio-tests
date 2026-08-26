import asyncio

import pytest

from aiolab.errors import RetryError
from aiolab.retry import (
    CircuitBreaker,
    RetryPolicy,
    default_retryable,
    retry,
    retry_with_timeout,
    with_timeout,
)


class Recorder:
    """Captures the delay sequence instead of sleeping through it."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


def flaky(failures: int, exc=ConnectionError):
    state = {"n": 0}

    async def call():
        state["n"] += 1
        if state["n"] <= failures:
            raise exc(f"attempt {state['n']}")
        return "ok"

    call.state = state
    return call


async def test_succeeds_without_sleeping_on_the_first_attempt():
    sleeper = Recorder()
    assert await retry(flaky(0), sleep=sleeper) == "ok"
    assert sleeper.slept == []


async def test_retries_until_it_succeeds():
    call = flaky(2)
    sleeper = Recorder()
    assert await retry(call, RetryPolicy(attempts=5, jitter=False), sleep=sleeper) == "ok"
    assert call.state["n"] == 3
    assert len(sleeper.slept) == 2


async def test_backoff_is_exponential_and_capped():
    policy = RetryPolicy(attempts=8, base_delay=1.0, factor=2.0, max_delay=8.0, jitter=False)
    assert [policy.delay_for(i) for i in range(1, 8)] == [0.0, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


async def test_full_jitter_stays_within_the_backoff_window():
    policy = RetryPolicy(base_delay=1.0, jitter=True)
    assert policy.delay_for(3, rand=lambda: 0.0) == 0.0
    assert policy.delay_for(3, rand=lambda: 1.0) == 2.0
    assert 0.0 <= policy.delay_for(3) <= 2.0


async def test_the_attempt_budget_is_actually_enforced():
    call = flaky(99)
    with pytest.raises(RetryError) as exc:
        await retry(call, RetryPolicy(attempts=3), sleep=Recorder())
    assert call.state["n"] == 3
    assert exc.value.attempts == 3
    assert isinstance(exc.value.last, ConnectionError)


async def test_a_non_retryable_error_is_raised_immediately():
    call = flaky(99, exc=ValueError)
    with pytest.raises(ValueError):
        await retry(call, RetryPolicy(attempts=5), sleep=Recorder())
    assert call.state["n"] == 1


async def test_default_retryable_classification():
    assert default_retryable(ConnectionError()) is True
    assert default_retryable(asyncio.TimeoutError()) is True
    assert default_retryable(ValueError()) is False
    assert default_retryable(TypeError()) is False
    assert default_retryable(KeyboardInterrupt()) is False


async def test_cancellation_is_never_retried():
    attempts = {"n": 0}

    async def call():
        attempts["n"] += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry(call, RetryPolicy(attempts=5), sleep=Recorder())
    assert attempts["n"] == 1


async def test_cancelling_the_caller_mid_backoff_stops_the_retry():
    call = flaky(99)
    task = asyncio.ensure_future(retry(call, RetryPolicy(attempts=99, base_delay=1.0)))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_on_retry_hook_sees_every_failed_attempt():
    seen = []
    await retry(
        flaky(2),
        RetryPolicy(attempts=5, jitter=False, base_delay=0.5),
        sleep=Recorder(),
        on_retry=lambda attempt, exc, delay: seen.append((attempt, str(exc), delay)),
    )
    assert [s[0] for s in seen] == [1, 2]
    assert [s[2] for s in seen] == [0.0, 0.5]


async def test_retry_takes_a_factory_not_a_coroutine():
    # A coroutine object can only be awaited once, which is why a helper that
    # accepted one could never retry. The factory form does.
    call = flaky(1)
    assert await retry(call, sleep=Recorder()) == "ok"


async def test_policy_rejects_nonsense():
    with pytest.raises(ValueError):
        RetryPolicy(attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(factor=0.5)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=-1)


async def test_with_timeout_returns_the_default_on_expiry():
    async def slow():
        await asyncio.sleep(1)
        return "late"

    assert await with_timeout(slow, 0.01, default="fallback") == "fallback"


async def test_with_timeout_passes_the_value_through():
    assert await with_timeout(lambda: asyncio.sleep(0, result=5), 1.0) == 5


async def test_with_timeout_cancels_the_inner_call():
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    await with_timeout(slow, 0.01)
    assert cancelled == [1]


async def test_retry_with_timeout_retries_a_hung_call():
    attempts = {"n": 0}

    async def hangs_then_works():
        attempts["n"] += 1
        if attempts["n"] < 3:
            await asyncio.sleep(10)
        return "ok"

    result = await retry_with_timeout(
        hangs_then_works,
        per_attempt_timeout=0.01,
        policy=RetryPolicy(attempts=5, jitter=False),
        sleep=Recorder(),
    )
    assert result == "ok"
    assert attempts["n"] == 3


async def test_retry_with_timeout_gives_up_within_the_budget():
    async def never():
        await asyncio.sleep(10)

    with pytest.raises(RetryError):
        await retry_with_timeout(
            never,
            per_attempt_timeout=0.01,
            policy=RetryPolicy(attempts=2),
            sleep=Recorder(),
        )


async def test_breaker_opens_after_consecutive_failures():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(threshold=2, reset_after=1.0, clock=lambda: clock["t"])

    async def bad():
        raise ConnectionError("down")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            await breaker.call(bad)
    assert breaker.state == CircuitBreaker.OPEN


async def test_open_breaker_fails_fast_without_calling_through():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(threshold=1, reset_after=1.0, clock=lambda: clock["t"])
    calls = {"n": 0}

    async def bad():
        calls["n"] += 1
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await breaker.call(bad)
    with pytest.raises(RetryError, match="circuit is open"):
        await breaker.call(bad)
    assert calls["n"] == 1  # the second call never reached the service


async def test_breaker_half_opens_and_closes_on_success():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(threshold=1, reset_after=1.0, clock=lambda: clock["t"])

    async def bad():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await breaker.call(bad)
    clock["t"] = 1.5
    assert breaker.state == CircuitBreaker.HALF_OPEN
    assert await breaker.call(lambda: asyncio.sleep(0, result="up")) == "up"
    assert breaker.state == CircuitBreaker.CLOSED


async def test_breaker_counts_consecutive_not_cumulative_failures():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(threshold=3, reset_after=1.0, clock=lambda: clock["t"])

    async def bad():
        raise ConnectionError("blip")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            await breaker.call(bad)
    await breaker.call(lambda: asyncio.sleep(0, result="ok"))
    for _ in range(2):
        with pytest.raises(ConnectionError):
            await breaker.call(bad)
    assert breaker.state == CircuitBreaker.CLOSED  # the success reset the run


async def test_breaker_does_not_count_a_cancellation_as_a_failure():
    breaker = CircuitBreaker(threshold=1)

    async def cancelled():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await breaker.call(cancelled)
    assert breaker.state == CircuitBreaker.CLOSED


async def test_breaker_rejects_a_zero_threshold():
    with pytest.raises(ValueError):
        CircuitBreaker(threshold=0)
