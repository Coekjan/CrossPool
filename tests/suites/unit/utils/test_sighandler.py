from __future__ import annotations

import signal
from types import FrameType

import pytest

from xpool.utils.sighandler import sighandle


@pytest.mark.parametrize("raises", [False, True])
def test_sighandle_restores_handler_after_context_exit(raises: bool) -> None:
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def handler(signum: int, frame: FrameType | None) -> None:
        return None

    try:
        if raises:
            with pytest.raises(RuntimeError, match="boom"):
                with sighandle(signal.SIGUSR1, handler):
                    assert signal.getsignal(signal.SIGUSR1) is handler
                    raise RuntimeError("boom")
        else:
            with sighandle(signal.SIGUSR1, handler):
                assert signal.getsignal(signal.SIGUSR1) is handler
        assert signal.getsignal(signal.SIGUSR1) == previous_handler
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGUSR1, previous_handler)
