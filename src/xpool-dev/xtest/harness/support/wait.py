"""Bounded polling helpers shared by deterministic test harnesses."""

from __future__ import annotations

import time
from collections.abc import Callable


def remaining_seconds(deadline: float, operation: str) -> float:
    """Return the positive remainder of one monotonic deadline."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError(f"{operation} exceeded its deadline")
    return remaining


def wait_until(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> bool:
    """Poll until a predicate succeeds or the bounded deadline expires."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def wait_until_raise(callback: Callable[[], None], *, timeout_s: float = 1.0) -> None:
    """Poll a callback until it raises, making deadline expiry observable."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        callback()
        time.sleep(0.01)
    callback()
