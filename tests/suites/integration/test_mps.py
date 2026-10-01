"""CUDA MPS controller probe locking behavior."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from xpool import mps


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
