from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import xpool.service.daemon.control
from xpool.service.daemon.control import ControlPlane
from xpool.service.wire import ControlPlaneWarning
from xtest.harness.support.config import install_test_config, reset_global_config, synthetic_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_first_retirement_trigger_seals_admission_and_owns_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    control = ControlPlane()
    monkeypatch.setattr(xpool.service.daemon.control, "monotonic", lambda: 1000.0)
    deadline = control.begin_close()

    assert control.admission_closed
    assert not control.closed
    assert deadline == 1000.0 + xpool.service.daemon.control.MPS_CLEANUP_TIMEOUT_S
    monkeypatch.setattr(xpool.service.daemon.control, "monotonic", lambda: 2000.0)
    assert control.begin_close(deadline=3000.0) == deadline
    control.close(deadline=4000.0)
    assert control.closed
    assert control.cleanup_deadline == deadline


@pytest.mark.parametrize("deadline", [None, 1001.0, 999.0], ids=["no-cleanup", "live-budget", "expired"])
def test_mps_cleanup_expiry_is_unavailable_without_hiding_other_timeouts(
    deadline: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = ControlPlane()
    error = TimeoutError("observation failed")

    def probe() -> None:
        raise error

    monkeypatch.setattr(control, "mps_scope", SimpleNamespace(cleanup_deadline=deadline, probe=probe))
    monkeypatch.setattr(xpool.service.daemon.control, "monotonic", lambda: 1000.0)
    if deadline is None or deadline > 1000.0:
        with pytest.raises(TimeoutError) as raised:
            control.refresh_mps_status(1000.0)
        assert raised.value is error
        assert control.mps_cache_result is None
    else:
        control.refresh_mps_status(1000.0)
        assert control.mps_cache_result is not None
        assert control.mps_cache_result.online is None


def test_global_warning_logs_first_entry_and_final_clearance(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    install_test_config(synthetic_config())
    first = ControlPlaneWarning(kind="stale_instance", device=0, message="first rank")
    second = ControlPlaneWarning(kind="stale_instance", device=0, message="second rank")
    warning_snapshots = iter(((first, second), (second,), ()))
    monkeypatch.setattr(
        xpool.service.daemon.control.ControlPlaneProjection,
        "capture",
        lambda **kwargs: SimpleNamespace(warnings=next(warning_snapshots)),
    )
    control = ControlPlane()

    with caplog.at_level(logging.INFO, logger="xpool.service.daemon.control"):
        counts = tuple(len(control.global_warnings(now)) for now in (0.0, 2.0, 4.0))

    assert counts == (2, 1, 0)
    assert [record.levelno for record in caplog.records] == [logging.WARNING, logging.INFO]
