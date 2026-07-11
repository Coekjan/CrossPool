"""CUDA MPS controller readiness probing for the daemon."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

MPS_CONTROL_COMMAND = "nvidia-cuda-mps-control"
MPS_PROBE_TIMEOUT_S = 1.0


@dataclass(frozen=True, slots=True)
class MpsProbeResult:
    """Result of one bounded CUDA MPS controller probe.

    Attributes:
        online: Whether the configured MPS controller answered successfully.
        diagnostic: Stable human-readable reason for logs when offline.
    """

    online: bool
    diagnostic: str


type MpsStatusProvider = Callable[[], MpsProbeResult]


def probe_mps_controller() -> MpsProbeResult:
    """Return whether the configured CUDA MPS control daemon is reachable.

    The probe queries controller state that exists before an MPS server is
    created, avoiding a readiness dependency on the first CUDA client.

    Returns:
        Structured controller availability and diagnostic detail.

    Side Effects:
        Executes ``nvidia-cuda-mps-control`` with the process environment, so
        ``CUDA_MPS_PIPE_DIRECTORY`` selects the controller being checked.
    """

    executable = shutil.which(MPS_CONTROL_COMMAND)
    if executable is None:
        return MpsProbeResult(False, f"{MPS_CONTROL_COMMAND} is not installed or not on PATH")
    try:
        completed = subprocess.run(
            [executable],
            input="get_default_active_thread_percentage\n",
            capture_output=True,
            text=True,
            timeout=MPS_PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MpsProbeResult(False, f"MPS controller probe exceeded {MPS_PROBE_TIMEOUT_S:.1f}s")
    except OSError as exc:
        return MpsProbeResult(False, f"failed to execute MPS controller probe: {exc}")
    output = completed.stdout.strip()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or output or f"exit code {completed.returncode}"
        return MpsProbeResult(False, f"MPS controller is unreachable: {detail}")
    try:
        active_thread_percentage = float(output)
    except ValueError:
        return MpsProbeResult(False, f"MPS controller returned an invalid active-thread percentage: {output!r}")
    if not math.isfinite(active_thread_percentage) or not 1 <= active_thread_percentage <= 100:
        return MpsProbeResult(False, f"MPS controller returned an out-of-range active-thread percentage: {output}")
    return MpsProbeResult(True, f"MPS controller is online with {active_thread_percentage:g}% active threads")
