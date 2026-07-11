"""CUDA MPS controller probe behavior."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from xpool.service.daemon import mps


def test_probe_reports_missing_control_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: None)

    result = mps.probe_mps_controller()

    assert result.online is False
    assert "not installed" in result.diagnostic


@pytest.mark.parametrize(
    ("stdout", "expected_detail"),
    [
        ("not-a-number\n", "invalid"),
        ("nan\n", "out-of-range"),
        ("0\n", "out-of-range"),
        ("101\n", "out-of-range"),
    ],
)
def test_probe_rejects_invalid_controller_output(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    expected_detail: str,
) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(
        mps.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )

    result = mps.probe_mps_controller()

    assert result.online is False
    assert expected_detail in result.diagnostic


def test_probe_reports_unreachable_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(
        mps.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="controller unavailable"),
    )

    result = mps.probe_mps_controller()

    assert result.online is False
    assert "controller unavailable" in result.diagnostic


def test_probe_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")

    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("nvidia-cuda-mps-control", mps.MPS_PROBE_TIMEOUT_S)

    monkeypatch.setattr(mps.subprocess, "run", raise_timeout)

    result = mps.probe_mps_controller()

    assert result.online is False
    assert "exceeded" in result.diagnostic


def test_probe_accepts_reachable_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(
        mps.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="100.0\n", stderr=""),
    )

    result = mps.probe_mps_controller()

    assert result.online is True
    assert "100%" in result.diagnostic
