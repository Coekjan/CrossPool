"""Retry policy and readiness for public-command SGLang probes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tests.harness.runner.network import TcpEndpointConflict
from tests.harness.sglang.manifest import E2eManifest, E2eServingCase
from tests.harness.sglang.serving.attempt import ProbeAttempt, ProbeRun
from tests.harness.sglang.serving.graph import SglangGraphSettings
from tests.harness.sglang.serving.server import SglangServerProcess, SglangServerResult
from xpool.config import XpoolConfig

PROBE_ATTEMPTS = 3


def run_probe(
    manifest: E2eManifest,
    case: E2eServingCase,
    *,
    base_config: XpoolConfig,
    graph_settings: SglangGraphSettings,
    workdir: Path,
    workload: Callable[[list[SglangServerProcess]], tuple[SglangServerResult, ...]] | None = None,
) -> ProbeRun:
    """Run one rematerialized attempt, retrying confirmed bind conflicts only."""

    conflicts: list[TcpEndpointConflict] = []
    for attempt_number in range(1, PROBE_ATTEMPTS + 1):
        attempt = ProbeAttempt(
            manifest,
            case,
            base_config,
            graph_settings,
            workdir / f"attempt-{attempt_number}",
        )
        try:
            return attempt.run(workload)
        except TcpEndpointConflict as error:
            conflicts.append(error)
    summaries = []
    for attempt_number, conflict in enumerate(conflicts, start=1):
        addresses = ", ".join(f"{host}:{port}" for host, port in conflict.addresses)
        cause = conflict.__cause__
        summary = "unknown startup failure"
        if cause is not None:
            first_line = str(cause).splitlines()[0] if str(cause) else ""
            summary = f"{type(cause).__name__}: {first_line}".rstrip()
        summaries.append(f"attempt {attempt_number}: {addresses}; startup={summary}")
    raise RuntimeError(
        f"SGLang E2E endpoint conflicts exhausted after {PROBE_ATTEMPTS} attempts:\n" + "\n".join(summaries)
    ) from conflicts[-1]
