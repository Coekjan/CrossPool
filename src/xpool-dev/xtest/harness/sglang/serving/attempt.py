"""Test projection and qualification over the shared serving lifecycle."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from threading import Barrier

from xkit.serving.sglang.graph import SglangGraphSettings
from xkit.serving.sglang.system import XpoolServingSystem
from xpool.integrations.sglang.devkit import SglangGraphEvent
from xpool.model import ModelId
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving.alignment import ServingGraphArtifact, TokenOutput
from xtest.harness.sglang.serving.graph import read_graph_events
from xtest.harness.sglang.serving.launch import prepare
from xtest.harness.sglang.serving.server import SglangProbeServer, SglangServerResult

PROBE_TIMEOUT_SECONDS = 30 * 60


@dataclass(frozen=True, slots=True)
class ProbeRun:
    """Results and graph evidence from one complete SGLang attempt."""

    model_ids: tuple[ModelId, ...]
    observer_outdir: Path
    graph_settings: SglangGraphSettings
    results: tuple[SglangServerResult, ...]
    events: list[SglangGraphEvent]
    daemon_startup_seconds: float
    duration_seconds: float

    def serving_graph_artifact(self) -> ServingGraphArtifact:
        """Project this completed probe to portable serving-graph evidence."""

        return ServingGraphArtifact(
            graph_settings=self.graph_settings,
            outputs=tuple(TokenOutput(result.model_id, result.output_ids) for result in self.results),
        )


@dataclass(slots=True)
class ProbeAttempt:
    """Apply test policy, then gather test evidence over shared process ownership."""

    case: E2eServingCase
    base_config: ResolvedConfig
    graph_settings: SglangGraphSettings
    workdir: Path
    system: XpoolServingSystem | None = None

    @property
    def observer_outdir(self) -> Path:
        """Test-owned observer artifacts for this attempt."""

        return (self.workdir / "observers").resolve()

    def run(
        self,
        workload: Callable[[list[SglangProbeServer]], tuple[SglangServerResult, ...]] | None = None,
    ) -> ProbeRun:
        """Retain qualification evidence after complete shared local cleanup.

        Only startup endpoint conflicts escape as retryable conflicts. Inference
        or cleanup failures cannot be replayed merely because a port is occupied.
        The enclosing supervisor continues to own descendant-domain proof.
        """

        self.workdir.mkdir(parents=True, exist_ok=False)
        started_at = time.monotonic()
        try:
            launch = prepare(
                self.case,
                base_config=self.base_config,
                workdir=self.workdir,
                graph_settings=self.graph_settings,
            )
            self.system = XpoolServingSystem.start(
                launch, workdir=self.workdir, startup_timeout_seconds=PROBE_TIMEOUT_SECONDS
            )
            assert self.system.cluster is not None
            config = self.system.launch.config
            servers = [
                SglangProbeServer(
                    server,
                    self.workdir / "models" / server.model.model_id.uri_encode() / "inference.json",
                    config.debug.prefill_logit_observer.outdir if config.debug.prefill_logit_observer.enable else None,
                )
                for server in self.system.servers
            ]
            try:
                if workload is None:
                    request_barrier = Barrier(len(servers))
                    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                        results = tuple(executor.map(SglangProbeServer.result, servers, repeat(request_barrier)))
                else:
                    results = workload(servers)
            except BaseException as error:
                error.add_note(self.system.diagnostics())
                raise AssertionError(f"SGLang E2E inference failed: {error}") from error
            daemon_startup_seconds = self.system.cluster.daemon_startup_seconds
        finally:
            if self.system is not None:
                try:
                    self.system.close()
                except Exception as error:
                    raise RuntimeError(f"SGLang E2E cleanup failed: {error}") from error
        return ProbeRun(
            model_ids=tuple(model.model_id for model in launch.models),
            observer_outdir=self.observer_outdir,
            graph_settings=self.graph_settings,
            results=results,
            events=read_graph_events(self.observer_outdir),
            daemon_startup_seconds=daemon_startup_seconds,
            duration_seconds=time.monotonic() - started_at,
        )
