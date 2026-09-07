from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import xpool.service.daemon.control
from tests.harness.support.config import install_test_config, reset_global_config, synthetic_config
from xpool.service.daemon.control import ControlPlane
from xpool.service.wire import ControlPlaneWarning

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_global_warning_logs_first_entry_and_final_clearance(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    install_test_config(synthetic_config())
    first = ControlPlaneWarning(kind="stale_instance", cuda_device=0, message="first rank")
    second = ControlPlaneWarning(kind="stale_instance", cuda_device=0, message="second rank")
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
