"""CUDA MPS preflight helpers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

MPS_HEALTH_INTERVAL_S = 5.0
MPS_CONTROL_TIMEOUT_S = 2.0

LOGGER = logging.getLogger(__name__)


class MpsPreflight(BaseModel):
    """CUDA MPS health snapshot used by daemon startup and readiness checks."""

    model_config = ConfigDict(extra="forbid")

    required: bool = Field(description="Whether xpool requires CUDA MPS for this runtime mode.")
    control_binary: str | None = Field(description="Resolved nvidia-cuda-mps-control path, if found.")
    pipe_directory: str | None = Field(description="CUDA_MPS_PIPE_DIRECTORY value used by the control daemon.")
    control_binary_found: bool = Field(description="Whether nvidia-cuda-mps-control was found on PATH.")
    control_daemon_reachable: bool = Field(description="Whether the MPS control daemon answered get_server_list.")
    healthy: bool = Field(description="Overall MPS readiness result consumed by daemon health and readiness.")
    checked_at_unix_s: float = Field(description="Unix timestamp when this snapshot was produced.")
    message: str = Field(description="Human-readable health summary or failure reason.")

    @classmethod
    def detect(cls, *, timeout_s: float = MPS_CONTROL_TIMEOUT_S) -> "MpsPreflight":
        """Probe CUDA MPS control-daemon health.

        Args:
            timeout_s: Maximum seconds to wait for ``nvidia-cuda-mps-control``.

        Returns:
            Health snapshot describing control-binary presence, pipe directory,
            daemon reachability, and overall readiness.

        Side Effects:
            Reads environment variables, resolves a program on ``PATH``, and runs
            ``nvidia-cuda-mps-control`` with ``get_server_list``.
        """

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
        """Initialize an MPS health monitor.

        Args:
            interval_s: Seconds between background health refreshes.
            detector: Callable that returns a fresh MPS health snapshot.
            initial: Optional precomputed snapshot. If omitted, ``detector`` is
                called synchronously during initialization.

        Side Effects:
            May run the detector synchronously when ``initial`` is not provided.
        """

        self._interval_s = interval_s
        self._detector = detector
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._stopping = False
        self._status = initial or self._detector()

    def refresh(self) -> MpsPreflight:
        """Refresh and cache MPS health immediately.

        Returns:
            Newly detected MPS health snapshot.

        Side Effects:
            Runs the detector and replaces the cached status under a lock.
        """

        status = self._detector()
        with self._lock:
            self._status = status
        return status

    def snapshot(self) -> MpsPreflight:
        """Return the most recently cached MPS health snapshot.

        Returns:
            Cached MPS health snapshot.

        Side Effects:
            Takes the monitor lock but does not run the detector.
        """

        with self._lock:
            return self._status

    def assert_healthy(self) -> None:
        """Raise if the cached MPS health snapshot is unhealthy.

        Raises:
            RuntimeError: If the cached snapshot is not healthy.
        """

        status = self.snapshot()
        if not status.healthy:
            raise RuntimeError(status.message)

    def start(self) -> None:
        """Start background MPS health refreshes.

        Side Effects:
            Creates and starts a daemon thread when no live monitor thread
            exists. Repeated calls while running or stopping are no-ops.
        """

        with self._lock:
            if self._stopping:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._generation += 1
            generation = self._generation
            stop_event = threading.Event()
            self._stop = stop_event
            thread = threading.Thread(
                target=self._run,
                args=(generation, stop_event),
                name="xpool-mps-health",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        """Stop the background MPS health refresh thread.

        Side Effects:
            Signals the thread, waits up to one refresh interval, and logs if the
            thread does not stop in time.
        """

        with self._lock:
            thread = self._thread
            stop_event = self._stop
            if thread is None:
                return
            self._stopping = True
            self._generation += 1
            self._thread = None
        stop_event.set()
        thread.join(timeout=self._interval_s)
        if thread.is_alive():
            LOGGER.warning("MPS health monitor did not stop within %.1fs", self._interval_s)
            with self._lock:
                self._stopping = False
            return
        with self._lock:
            self._stopping = False

    def _run(self, generation: int, stop_event: threading.Event) -> None:
        while not stop_event.wait(self._interval_s):
            try:
                status = self._detector()
            except Exception:
                LOGGER.exception("MPS health refresh failed")
                continue
            with self._lock:
                if generation != self._generation:
                    return
                self._status = status
