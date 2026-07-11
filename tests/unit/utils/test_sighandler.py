from __future__ import annotations

import signal
from types import FrameType

import pytest

from xpool.utils.sighandler import sighandle


def test_sighandle_installs_and_restores_handler() -> None:
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def handler(signum: int, frame: FrameType | None) -> None:
        return None

    try:
        with sighandle(signal.SIGUSR1, handler):
            assert signal.getsignal(signal.SIGUSR1) is handler
        assert signal.getsignal(signal.SIGUSR1) == previous_handler
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGUSR1, previous_handler)


def test_sighandle_restores_handler_after_exception() -> None:
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def handler(signum: int, frame: FrameType | None) -> None:
        return None

    try:
        with pytest.raises(RuntimeError, match="boom"):
            with sighandle(signal.SIGUSR1, handler):
                raise RuntimeError("boom")
        assert signal.getsignal(signal.SIGUSR1) == previous_handler
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGUSR1, previous_handler)
