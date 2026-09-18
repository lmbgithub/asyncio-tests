"""A small end-to-end pipeline using every level of the library at once.

Fetch (rate-limited, retried, timed out) -> parse (bounded concurrency) ->
batch -> write, with a cooperative stop signal and a supervised background
task. No network, no keys: the "server" is a dict with an injected failure.

    python examples/pipeline_demo.py
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiolab import (
    RetryPolicy,
    TaskRegistry,
    TokenBucket,
    WorkerPool,
    aiter_from,
    amap,
    batched,
    retry,
    take_until,
)

PAGES = {f"/doc/{i}": f"title-{i}\nbody of document {i}" for i in range(1, 21)}


class FlakyServer:
    """Fails the first request to any path, and rate-limits to 20/s."""

    def __init__(self, *, seed: int = 7) -> None:
        self._seen: set[str] = set()
        self._rng = random.Random(seed)
        self.requests = 0
        self.failures = 0

    async def get(self, path: str) -> str:
        self.requests += 1
        await asyncio.sleep(self._rng.uniform(0.001, 0.01))
        if path not in self._seen:
            self._seen.add(path)
            self.failures += 1
            raise ConnectionError(f"{path}: connection reset")
        return PAGES[path]


async def main() -> int:
    server = FlakyServer()
    bucket = TokenBucket(50, capacity=10)
    stop = asyncio.Event()
    written: list[list[str]] = []

    async def fetch(path: str) -> str:
        await bucket.acquire()
        return await retry(
            lambda: server.get(path),
            RetryPolicy(attempts=3, base_delay=0.002, jitter=True),
        )

    async def parse(path: str) -> str:
        body = await fetch(path)
        return body.splitlines()[0]

    async def write(batch: list[str]) -> None:
        await asyncio.sleep(0.001)
        written.append(batch)

    async with TaskRegistry("pipeline") as registry:
        # A background watchdog that stops the pipeline early — the cooperative
        # shutdown path, not a cancellation.
        async def watchdog() -> None:
            while len(written) < 3:
                await asyncio.sleep(0.005)
            stop.set()

        registry.spawn(watchdog(), name="watchdog")

        titles = amap(aiter_from(PAGES), parse, concurrency=5)
        async with WorkerPool(write, workers=2, maxsize=4) as writer:
            async for batch in batched(take_until(titles, stop), size=4, timeout=0.5):
                await writer.submit(batch)
            await writer.join()

    print(f"requests issued      {server.requests}")
    print(f"injected failures    {server.failures}")
    print(f"batches written      {len(written)}")
    print(f"titles written       {sum(len(b) for b in written)}")
    print(f"stopped early        {stop.is_set()}")
    print(f"first batch          {written[0] if written else []}")

    assert server.failures > 0, "the demo is supposed to exercise the retry path"
    assert written, "nothing made it through the pipeline"
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
