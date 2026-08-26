"""`aiolab <demo>` — run one level and watch what it does.

Results to stdout, diagnostics to stderr, non-zero exit on failure, so the
whole thing can be gated in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Awaitable, Callable

from . import basics, streams
from .errors import ConcurrencyError
from .queues import WorkerPool
from .retry import RetryPolicy, retry
from .sync import RaceyCounter, SafeCounter, TokenBucket
from .tasks import TaskRegistry, run_all


def _line(label: str, value: object) -> None:
    print(f"  {label:<24} {value}")


async def demo_basics() -> None:
    print("basics — sequential vs concurrent, 4 x 50ms of fake I/O")
    factories: list[Callable[[], Awaitable[str]]] = [
        (lambda i=i: basics.delay(f"r{i}", 0.05)) for i in range(4)
    ]
    _, seq = await basics.timed(basics.run_sequential(factories))
    _, con = await basics.timed(basics.run_concurrent(factories))
    _line("sequential", f"{seq * 1000:.1f} ms  (the sum)")
    _line("concurrent", f"{con * 1000:.1f} ms  (the max)")
    _line("speedup", f"{seq / con:.2f}x")


async def demo_races() -> None:
    print("sync — 1000 increments, with and without the lock")
    racey = RaceyCounter()
    await asyncio.gather(*(racey.increment() for _ in range(1000)))
    safe = SafeCounter()
    await asyncio.gather(*(safe.increment() for _ in range(1000)))
    _line("no lock", f"{racey.value} / 1000  (lost updates)")
    _line("asyncio.Lock", f"{safe.value} / 1000")


async def demo_tasks() -> None:
    print("tasks — one child fails, the siblings are cancelled and awaited")
    cancelled: list[str] = []

    async def slow(name: str) -> str:
        try:
            await asyncio.sleep(10)
            return name
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    async def boom() -> str:
        await asyncio.sleep(0.01)
        raise RuntimeError("upstream 503")

    started = time.perf_counter()
    try:
        await run_all([lambda: slow("a"), lambda: slow("b"), boom])
    except ConcurrencyError as exc:
        _line("raised", type(exc).__name__)
        _line("errors", exc.errors)
    _line("siblings cancelled", sorted(cancelled))
    _line("elapsed", f"{(time.perf_counter() - started) * 1000:.1f} ms  (not 10 s)")


async def demo_queue() -> None:
    print("queues — 3 workers, bounded queue of 4, one poison item")

    async def handle(n: int) -> int:
        await asyncio.sleep(0.01)
        if n == 7:
            raise ValueError("cannot handle 7")
        return n * n

    async with WorkerPool(handle, workers=3, maxsize=4) as pool:
        for n in range(10):
            await pool.submit(n)
        await pool.join()
    _line("submitted", pool.stats.submitted)
    _line("completed", pool.stats.completed)
    _line("failed", pool.stats.failed)
    _line("errors", [str(e) for e in pool.stats.errors])


async def demo_streams() -> None:
    print("streams — concurrent map over a stream, order preserved")

    async def slow_square(n: int) -> int:
        await asyncio.sleep(0.05 if n % 2 else 0.01)
        return n * n

    source = streams.aiter_from(range(8))
    _, elapsed = await basics.timed(
        streams.collect(streams.amap(source, slow_square, concurrency=4))
    )
    out = await streams.collect(streams.amap(streams.aiter_from(range(8)), slow_square))
    _line("results", out)
    _line("elapsed", f"{elapsed * 1000:.1f} ms  (serial would be ~240 ms)")


async def demo_retry() -> None:
    print("retry — deterministic backoff, no wall-clock sleeping")
    slept: list[float] = []
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError(f"attempt {attempts['n']} refused")
        return "ok"

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    result = await retry(
        flaky,
        RetryPolicy(attempts=5, base_delay=0.1, jitter=False),
        sleep=fake_sleep,
        rand=lambda: 1.0,
    )
    _line("result", result)
    _line("attempts", attempts["n"])
    _line("delays", [round(s, 3) for s in slept])


async def demo_ratelimit() -> None:
    print("sync — token bucket, 5/s with a burst of 5, on an injected clock")
    now = {"t": 0.0}
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    bucket = TokenBucket(5, capacity=5, clock=lambda: now["t"], sleep=fake_sleep)
    for _ in range(5):
        await bucket.acquire()
    _line("burst", "5 immediate")
    waited = await bucket.acquire()
    _line("6th call waited", f"{waited:.3f} s  (1 / rate)")
    _line("clock advanced to", f"{now['t']:.3f} s")


DEMOS: dict[str, Callable[[], Awaitable[None]]] = {
    "basics": demo_basics,
    "races": demo_races,
    "tasks": demo_tasks,
    "queue": demo_queue,
    "streams": demo_streams,
    "retry": demo_retry,
    "ratelimit": demo_ratelimit,
}


async def _run(names: list[str]) -> None:
    async with TaskRegistry("cli"):
        for i, name in enumerate(names):
            if i:
                print()
            await DEMOS[name]()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aiolab", description="Runnable demonstrations of each asyncio level."
    )
    parser.add_argument(
        "demo",
        nargs="?",
        default="all",
        choices=[*DEMOS, "all"],
        help="which demonstration to run (default: all)",
    )
    args = parser.parse_args(argv)

    names = list(DEMOS) if args.demo == "all" else [args.demo]
    try:
        asyncio.run(_run(names))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # a demo raising is a real failure
        print(f"demo failed: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
