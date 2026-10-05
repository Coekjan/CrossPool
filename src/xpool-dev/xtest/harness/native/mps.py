"""Direct MPS server and client observations for topology qualification."""

from __future__ import annotations

import math
from dataclasses import dataclass

from xpool.utils.mps import MpsEndpoint


@dataclass(frozen=True, slots=True)
class MpsServerObservation:
    """One live MPS server, its client PIDs, and active-thread percentage."""

    process_id: int
    client_process_ids: tuple[int, ...]
    active_thread_percentage: float


def query_mps_servers(endpoint: MpsEndpoint, *, deadline: float) -> tuple[MpsServerObservation, ...]:
    """Query live servers within the topology owner's unchanged deadline."""

    server_ids = endpoint.parse_process_ids(
        endpoint.run_control("get_server_list", deadline=deadline), allow_empty=False
    )
    observations = []
    for server_id in server_ids:
        client_ids = endpoint.parse_process_ids(
            endpoint.run_control(f"get_client_list {server_id}", deadline=deadline), allow_empty=True
        )
        percentage_output = endpoint.run_control(f"get_active_thread_percentage {server_id}", deadline=deadline)
        try:
            percentage = float(percentage_output)
        except ValueError as error:
            raise RuntimeError(f"MPS server {server_id} returned invalid active-thread percentage") from error
        if not math.isfinite(percentage) or not 1 <= percentage <= 100:
            raise RuntimeError(f"MPS server {server_id} returned invalid active-thread percentage {percentage}")
        observations.append(MpsServerObservation(server_id, client_ids, percentage))
    return tuple(observations)
