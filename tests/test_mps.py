from __future__ import annotations

import logging
import subprocess
import threading
from types import SimpleNamespace

from xpool.runtime import mps
from xpool.runtime.mps import MpsHealthMonitor, MpsPreflight


def test_mps_preflight_fails_when_control_binary_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda _name: None)

    status = MpsPreflight.detect()

    assert status.healthy is False
    assert status.control_binary_found is False
    assert status.control_daemon_reachable is False


def test_mps_preflight_checks_control_daemon(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/usr/bin/nvidia-cuda-mps-control")

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mps.subprocess, "run", fake_run)

    status = MpsPreflight.detect(timeout_s=1.0)

    assert status.healthy is True
    assert status.control_binary_found is True
    assert status.control_daemon_reachable is True
    assert calls[0][0] == (["/usr/bin/nvidia-cuda-mps-control"],)
    assert calls[0][1]["input"] == "get_server_list\n"
    assert calls[0][1]["timeout"] == 1.0


def test_mps_preflight_reports_unreachable_control_daemon(monkeypatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(
        mps.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="connection refused\n"),
    )

    status = MpsPreflight.detect()

    assert status.healthy is False
    assert status.control_binary_found is True
    assert status.control_daemon_reachable is False
    assert "connection refused" in status.message


def test_mps_preflight_reports_timeout(monkeypatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda _name: "/usr/bin/nvidia-cuda-mps-control")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-cuda-mps-control", timeout=0.1)

    monkeypatch.setattr(mps.subprocess, "run", timeout)

    status = MpsPreflight.detect(timeout_s=0.1)

    assert status.healthy is False
    assert status.control_daemon_reachable is False
    assert "did not respond" in status.message


def test_mps_health_monitor_refreshes_with_detector() -> None:
    statuses = [
        _mps_status(healthy=True, checked_at=1.0),
        _mps_status(healthy=False, checked_at=2.0),
    ]
    monitor = MpsHealthMonitor(detector=lambda: statuses.pop(0))

    assert monitor.snapshot().healthy is True
    assert monitor.refresh().healthy is False
    assert monitor.snapshot().checked_at_unix_s == 2.0


def test_mps_health_monitor_survives_detector_exception() -> None:
    calls = 0
    refreshed = threading.Event()

    def detector() -> MpsPreflight:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary detector failure")
        refreshed.set()
        return _mps_status(healthy=False, checked_at=2.0)

    monitor = MpsHealthMonitor(interval_s=0.01, detector=detector, initial=_mps_status(healthy=True, checked_at=1.0))

    monitor.start()
    assert refreshed.wait(timeout=1.0)
    monitor.stop()

    assert monitor.snapshot().healthy is False


def test_mps_health_monitor_can_restart_after_stop_timeout(caplog) -> None:
    first_entered = threading.Event()
    first_release = threading.Event()
    second_refreshed = threading.Event()
    calls = 0

    def detector() -> MpsPreflight:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            first_release.wait(timeout=1.0)
            return _mps_status(healthy=False, checked_at=2.0)
        second_refreshed.set()
        return _mps_status(healthy=True, checked_at=3.0)

    monitor = MpsHealthMonitor(interval_s=0.01, detector=detector, initial=_mps_status(healthy=True, checked_at=1.0))

    monitor.start()
    assert first_entered.wait(timeout=1.0)
    with caplog.at_level(logging.WARNING, logger="xpool.runtime.mps"):
        monitor.stop()

    assert "did not stop" in caplog.text
    monitor.start()
    assert second_refreshed.wait(timeout=1.0)
    first_release.set()
    monitor.stop()
    assert monitor.snapshot().checked_at_unix_s == 3.0


def _mps_status(*, healthy: bool, checked_at: float) -> MpsPreflight:
    return MpsPreflight(
        required=True,
        control_binary="/usr/bin/nvidia-cuda-mps-control",
        pipe_directory=None,
        control_binary_found=True,
        control_daemon_reachable=healthy,
        healthy=healthy,
        checked_at_unix_s=checked_at,
        message="MPS control daemon is reachable" if healthy else "MPS control daemon is not reachable",
    )
