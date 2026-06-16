"""CUDA MPS preflight helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

MPS_HEALTH_INTERVAL_S = 5.0
MPS_CONTROL_TIMEOUT_S = 2.0


class MpsPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool
    control_binary: str | None
    pipe_directory: str | None
    control_binary_found: bool
    control_daemon_reachable: bool
    healthy: bool
    checked_at_unix_s: float
    message: str

    @classmethod
    def detect(cls, *, timeout_s: float = MPS_CONTROL_TIMEOUT_S) -> "MpsPreflight":
        control_binary = shutil.which("nvidia-cuda-mps-control")
        pipe_directory = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        checked_at_unix_s = time.time()
        if control_binary is None:
            return cls(
                required=True,
                control_binary=None,
                pipe_directory=pipe_directory,
                control_binary_found=False,
                control_daemon_reachable=False,
                healthy=False,
                checked_at_unix_s=checked_at_unix_s,
                message="MPS is required but nvidia-cuda-mps-control was not found on PATH",
            )

        try:
            result = subprocess.run(
                [control_binary],
                input="get_server_list\n",
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return cls(
                required=True,
                control_binary=control_binary,
                pipe_directory=pipe_directory,
                control_binary_found=True,
                control_daemon_reachable=False,
                healthy=False,
                checked_at_unix_s=checked_at_unix_s,
                message=f"MPS control daemon did not respond within {timeout_s:.1f}s",
            )
        except OSError as exc:
            return cls(
                required=True,
                control_binary=control_binary,
                pipe_directory=pipe_directory,
                control_binary_found=True,
                control_daemon_reachable=False,
                healthy=False,
                checked_at_unix_s=checked_at_unix_s,
                message=f"MPS control daemon check failed: {exc}",
            )

        control_daemon_reachable = result.returncode == 0
        if control_daemon_reachable:
            message = "MPS control daemon is reachable"
        else:
            detail = (result.stderr or result.stdout).strip().splitlines()
            suffix = f": {detail[0]}" if detail else ""
            message = f"MPS control daemon is not reachable{suffix}"
        return cls(
            required=True,
            control_binary=control_binary,
            pipe_directory=pipe_directory,
            control_binary_found=True,
            control_daemon_reachable=control_daemon_reachable,
            healthy=control_daemon_reachable,
            checked_at_unix_s=checked_at_unix_s,
            message=message,
        )


class MpsHealthMonitor:
    """Caches and refreshes MPS health without blocking every health request."""

    def __init__(
        self,
        *,
        interval_s: float = MPS_HEALTH_INTERVAL_S,
        detector: Callable[[], MpsPreflight] = MpsPreflight.detect,
        initial: MpsPreflight | None = None,
    ) -> None:
        self._interval_s = interval_s
        self._detector = detector
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = initial or self._detector()

    def refresh(self) -> MpsPreflight:
        status = self._detector()
        with self._lock:
            self._status = status
        return status

    def snapshot(self) -> MpsPreflight:
        with self._lock:
            return self._status

    def assert_healthy(self) -> None:
        status = self.snapshot()
        if not status.healthy:
            raise RuntimeError(status.message)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="xpool-mps-health", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self._interval_s)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.refresh()
