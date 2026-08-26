"""The exception surface of the library.

One base exception per library, and a deliberate rule that runs through every
module here: `asyncio.CancelledError` is **never** wrapped, retried, logged as a
failure, or absorbed into an aggregate. In Python 3.8+ it inherits from
`BaseException` precisely so that `except Exception` does not eat it, and code
that catches it broadly turns a cancellation request into a hang.
"""

from __future__ import annotations


class AiolabError(Exception):
    """Base class for every error raised by this library."""


class ConcurrencyError(AiolabError):
    """One or more concurrent children failed.

    Python 3.11 has `ExceptionGroup` for this; the minimum supported version
    here is 3.10, so failures are aggregated explicitly. The list is kept in
    child-index order rather than completion order, because "which input
    failed" is the question a caller actually asks.
    """

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = list(errors)
        first = self.errors[0] if self.errors else None
        summary = ", ".join(sorted({type(e).__name__ for e in self.errors}))
        super().__init__(
            f"{len(self.errors)} child task(s) failed ({summary}); first: {first!r}"
        )


class PoolClosedError(AiolabError):
    """Work was submitted to a pool that is shutting down or already closed."""


class RetryError(AiolabError):
    """Every retry attempt was exhausted without a success."""

    def __init__(self, attempts: int, last: BaseException) -> None:
        self.attempts = attempts
        self.last = last
        super().__init__(f"failed after {attempts} attempt(s); last error: {last!r}")
