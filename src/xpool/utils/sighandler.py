"""Signal handler lifecycle helpers."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from types import FrameType

__all__ = ("defer_signal_exceptions", "sighandle")


@contextmanager
def defer_signal_exceptions() -> Generator[None, None, None]:
    """Deliver handler exceptions after a short resource-publication operation.

    Cooperative SIGINT/SIGTERM handlers still run immediately; only their raised
    exception is retained until the body completes. Enclose acquisition and the
    owner's assignment together. Readiness and cleanup do not belong in this
    context. A body failure takes precedence over deferred interruption.

    Ignored/default dispositions are unchanged. Worker threads use no replacement
    handlers, because Python delivers handlers on the main thread only. Installed
    handlers are restored before the first deferred exception is raised.
    """

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handlers: dict[int, Callable[[int, FrameType | None], object]] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        handler = signal.getsignal(signum)
        if handler is not None and not isinstance(handler, int):
            handlers[signum] = handler
    pending: BaseException | None = None

    def deliver(signum: int, frame: FrameType | None) -> None:
        nonlocal pending
        handler = handlers[signum]
        try:
            handler(signum, frame)
        except BaseException as error:
            if pending is None:
                pending = error

    with ExitStack() as restoration:
        for signum, handler in handlers.items():
            restoration.enter_context(sighandle(signum, deliver))
        yield
    if pending is not None:
        raise pending


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
