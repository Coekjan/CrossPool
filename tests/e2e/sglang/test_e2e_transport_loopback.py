from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tests.harness.sglang.environment import SglangTestEnvironment
from tests.harness.sglang.graph import assert_graph_events, read_graph_events
from tests.harness.sglang.offline_probe import GRAPH_SETTINGS, MODEL_ID
from tests.harness.sglang.probe import (
    ProbeRun,
    graph_settings_key,
    run_probe,
)
from tests.harness.sglang.transport_cluster import TransportLoopbackCluster

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.requires_config,
    pytest.mark.requires_model_weights(MODEL_ID),
    pytest.mark.timeout(600),
]


def test_sglang_transport_loopback_graph_modes(
    sglang_environment: SglangTestEnvironment,
    tmp_path: Path,
) -> None:
    runs: list[ProbeRun] = []
    for graph_settings in GRAPH_SETTINGS:
        key = graph_settings_key(graph_settings)
        workdir = tmp_path / f"{int(key[0])}-{int(key[1])}"
        event_outdir = workdir / "events"
        cluster = TransportLoopbackCluster.create(
            source_config=sglang_environment.config,
            model_id=MODEL_ID,
            model_path=sglang_environment.model_path,
            graph_observer_outdir=event_outdir,
            workdir=workdir / "cluster",
        )
        started_at = time.perf_counter()
        with cluster:
            try:
                result = run_probe(
                    graph_settings=graph_settings,
                    base_gpu_id=sglang_environment.base_gpu_id,
                    config_path=cluster.config_path,
                    event_outdir=event_outdir,
                    loopback_mode="transport",
                )
            except Exception as exc:
                raise AssertionError(f"{exc}\n{cluster.diagnostics()}") from exc
        observer_paths = sorted(event_outdir.glob("xpool.transport-observer.*.json"))
        assert len(observer_paths) == len(cluster.config.devices.atn_cuda_devices)
        for observer_path in observer_paths:
            observer = json.loads(observer_path.read_text(encoding="utf-8"))
            assert observer["sequence"] > 0
            assert observer["dropped"] == 0
            assert observer["incomplete"] == 0
            assert observer["records"]
        runs.append(
            ProbeRun(
                graph_settings=graph_settings,
                result=result,
                events=read_graph_events(event_outdir),
                base_gpu_id=sglang_environment.base_gpu_id,
                duration_s=time.perf_counter() - started_at,
            )
        )

    eager_output_ids = next(
        run.result["output_ids"] for run in runs if graph_settings_key(run.graph_settings) == (False, False)
    )
    for run in runs:
        assert_graph_events(run.graph_settings, run.events)
        assert run.result["output_ids"] == eager_output_ids

    print(
        "XPOOL_SGLANG_TRANSPORT_LOOPBACK_DURATIONS="
        + json.dumps(
            [
                {
                    "base_gpu_id": run.base_gpu_id,
                    "cuda_graph": run.graph_settings.cuda_graph,
                    "duration_s": run.duration_s,
                    "piecewise_cuda_graph": run.graph_settings.piecewise_cuda_graph,
                }
                for run in runs
            ],
            sort_keys=True,
        )
    )
