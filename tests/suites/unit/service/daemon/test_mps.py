"""CUDA MPS controller probe behavior."""

from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from xpool.service.daemon import mps


@pytest.fixture(autouse=True)
def configured_mps_pipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "mps-pipe"))


@pytest.mark.parametrize("value", [None, "", "   "])
def test_probe_requires_explicit_pipe_directory(
    value: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if value is None:
        monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    else:
        monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", value)

    result = mps.probe_mps_controller()

    assert result.online is False
    assert "CUDA_MPS_PIPE_DIRECTORY" in result.diagnostic


def test_probe_lock_identity_uses_explicit_pipe_directory(tmp_path: Path) -> None:
    first = mps.mps_probe_lock_path(tmp_path / "first")
    second = mps.mps_probe_lock_path(tmp_path / "second")

    assert first.parent == mps.MPS_PROBE_LOCK_DIRECTORY
    assert first != second
    assert first == mps.mps_probe_lock_path(tmp_path / "first")


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


def test_probe_serializes_concurrent_queries_for_one_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def run_probe(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal active, max_active
        del args, kwargs
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with state_lock:
            active -= 1
        return SimpleNamespace(returncode=0, stdout="100.0\n", stderr="")

    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "pipe"))
    monkeypatch.setattr(mps, "MPS_PROBE_LOCK_DIRECTORY", tmp_path / "locks")
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps.subprocess, "run", run_probe)

    with ThreadPoolExecutor(max_workers=7) as executor:
        results = tuple(executor.map(lambda index: mps.probe_mps_controller(), range(14)))

    assert all(result.online for result in results)
    assert max_active == 1
