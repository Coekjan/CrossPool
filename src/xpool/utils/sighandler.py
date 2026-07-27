"""Signal handler lifecycle helpers."""

from __future__ import annotations

import signal
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import FrameType


@contextmanager
def sighandle(
    signum: signal.Signals | int,
    handler: signal.Handlers | Callable[[int, FrameType | None], object],
) -> Generator[None, None, None]:
    """Temporarily install a signal handler and restore the previous handler.

    Args:
        signum: Signal number or ``signal.Signals`` enum value to handle.
        handler: Replacement handler accepted by ``signal.signal``.

    Side Effects:
        Mutates the process-wide Python signal handler for ``signum`` while the
        context is active, then restores the previous handler on exit.
    """

    previous_handler = signal.getsignal(signum)
    signal.signal(signum, handler)
    try:
        yield
    finally:
        if previous_handler is not None:
            signal.signal(signum, previous_handler)
