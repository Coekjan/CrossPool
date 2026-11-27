from __future__ import annotations

from types import SimpleNamespace

import pytest

import xpool.service.daemon.control
from xpool.service.daemon.control import ControlPlane
from xtest.harness.support.config import reset_global_config

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


def test_passive_mps_observation_timeout_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    control = ControlPlane()

    def probe() -> None:
        raise TimeoutError("MPS control deadline expired during serialization")

    monkeypatch.setattr(control, "mps_scope", SimpleNamespace(probe=probe))
    control.refresh_mps_status()
    assert control.mps_status is not None
    assert control.mps_status.online is None
