"""Serialized CUDA MPS controller readiness probing."""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MPS_CONTROL_COMMAND = "nvidia-cuda-mps-control"
MPS_PROBE_LOCK_DIRECTORY = Path("/tmp") / f"xpool-mps-probe-locks-{os.getuid()}"
MPS_PROBE_TIMEOUT_S = 1.0


@dataclass(frozen=True, slots=True)
class MpsProbeResult:
    """Result of one bounded CUDA MPS controller probe.

    Attributes:
        online: Whether the configured MPS controller answered successfully.
        active_thread_percentage: Integral controller percentage when online.
        diagnostic: Stable human-readable reason for logs when offline.
    """

    online: bool
    active_thread_percentage: int | None
    diagnostic: str


def probe_mps_controller() -> MpsProbeResult:
    """Return whether the configured CUDA MPS control daemon is reachable.

    The probe queries controller state that exists before an MPS server is
    created, avoiding a readiness dependency on the first CUDA client.

    Returns:
        Structured controller availability, active-thread percentage, and
        diagnostic detail.

    Side Effects:
        Executes ``nvidia-cuda-mps-control`` with the process environment, so
        ``CUDA_MPS_PIPE_DIRECTORY`` selects the controller being checked.
    """

    raw_pipe_directory = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if raw_pipe_directory is None or not raw_pipe_directory.strip():
        return MpsProbeResult(False, None, "CUDA_MPS_PIPE_DIRECTORY is required for MPS controller readiness")
    pipe_directory = Path(raw_pipe_directory).expanduser().resolve(strict=False)
    executable = shutil.which(MPS_CONTROL_COMMAND)
    if executable is None:
        return MpsProbeResult(False, None, f"{MPS_CONTROL_COMMAND} is not installed or not on PATH")
    try:
        MPS_PROBE_LOCK_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
        with mps_probe_lock_path(pipe_directory).open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            completed = subprocess.run(
                [executable],
                input="get_default_active_thread_percentage\n",
                capture_output=True,
                text=True,
                timeout=MPS_PROBE_TIMEOUT_S,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return MpsProbeResult(False, None, f"MPS controller probe exceeded {MPS_PROBE_TIMEOUT_S:.1f}s")
    except OSError as exc:
        return MpsProbeResult(False, None, f"failed to execute serialized MPS controller probe: {exc}")
    output = completed.stdout.strip()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or output or f"exit code {completed.returncode}"
        return MpsProbeResult(False, None, f"MPS controller is unreachable: {detail}")
    try:
        active_thread_percentage = float(output)
    except ValueError:
        return MpsProbeResult(
            False,
            None,
            f"MPS controller returned an invalid active-thread percentage: {output!r}",
        )
    if (
        not math.isfinite(active_thread_percentage)
        or not active_thread_percentage.is_integer()
        or not 1 <= active_thread_percentage <= 100
    ):
        return MpsProbeResult(
            False,
            None,
            f"MPS controller returned an out-of-range or non-integral active-thread percentage: {output}",
        )
    percentage = int(active_thread_percentage)
    return MpsProbeResult(True, percentage, f"MPS controller is online with {percentage}% active threads")


def mps_probe_lock_path(pipe_directory: Path) -> Path:
    """Return the per-user lock that serializes one MPS controller endpoint."""

    digest = hashlib.sha256(os.fsencode(pipe_directory.resolve(strict=False))).hexdigest()
    return MPS_PROBE_LOCK_DIRECTORY / f"{digest}.lock"
