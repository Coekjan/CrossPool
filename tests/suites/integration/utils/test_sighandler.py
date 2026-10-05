from __future__ import annotations

import os
import signal
from types import FrameType

import pytest

from xpool.utils.sighandler import defer_signal_exceptions, sighandle


@pytest.mark.parametrize("body_failure", [False, True])
def test_signal_exception_waits_for_publication_and_preserves_operation_error(body_failure: bool) -> None:
    events: list[str] = []

    def interrupt(signum: int, frame: FrameType | None) -> None:
        events.append("cancel-requested")
        raise InterruptedError("cancelled")

    with sighandle(signal.SIGTERM, interrupt):
        with pytest.raises(ValueError if body_failure else InterruptedError):
            with defer_signal_exceptions():
                os.kill(os.getpid(), signal.SIGTERM)
                events.append("published")
                if body_failure:
                    raise ValueError("creation failed")
        assert signal.getsignal(signal.SIGTERM) is interrupt
    assert events == ["cancel-requested", "published"]
