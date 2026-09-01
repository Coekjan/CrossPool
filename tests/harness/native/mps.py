"""Direct MPS server and client observations for topology qualification."""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass

MPS_CONTROL_COMMAND = "nvidia-cuda-mps-control"


@dataclass(frozen=True, slots=True)
class MpsServerObservation:
    """One live MPS server, its client PIDs, and active-thread percentage."""

    process_id: int
    client_process_ids: tuple[int, ...]
    active_thread_percentage: float


def query_mps_servers() -> tuple[MpsServerObservation, ...]:
    """Query every live MPS server through the configured controller."""

    server_ids = parse_process_ids(run_mps_control("get_server_list"), allow_empty=False)
    observations = []
    for server_id in server_ids:
        client_ids = parse_process_ids(run_mps_control(f"get_client_list {server_id}"), allow_empty=True)
        percentage_output = run_mps_control(f"get_active_thread_percentage {server_id}")
        try:
            percentage = float(percentage_output)
        except ValueError as error:
            raise RuntimeError(f"MPS server {server_id} returned invalid active-thread percentage") from error
        if not math.isfinite(percentage) or not 1 <= percentage <= 100:
            raise RuntimeError(f"MPS server {server_id} returned invalid active-thread percentage {percentage}")
        observations.append(MpsServerObservation(server_id, client_ids, percentage))
    return tuple(observations)


def run_mps_control(command: str) -> str:
    """Run one nonempty MPS control command and return stripped output."""

    if not command:
        raise ValueError("MPS control command must be nonempty")
    try:
        result = subprocess.run(
            [MPS_CONTROL_COMMAND],
            input=f"{command}\n",
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"failed to run MPS control command {command!r}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"MPS control command {command!r} failed: {detail}")
    return result.stdout.strip()


def parse_process_ids(output: str, *, allow_empty: bool) -> tuple[int, ...]:
    """Parse unique positive process IDs from controller output."""

    if not output:
        if allow_empty:
            return ()
        raise RuntimeError("MPS controller returned no process IDs")
    values = output.splitlines()
    if any(not value.isdigit() or int(value) <= 0 for value in values):
        raise RuntimeError(f"MPS controller returned invalid process IDs: {output!r}")
    process_ids = tuple(int(value) for value in values)
    if len(process_ids) != len(set(process_ids)):
        raise RuntimeError(f"MPS controller returned duplicate process IDs: {process_ids}")
    return process_ids
