from __future__ import annotations

import json

import pytest

from tests.harness.sglang.environment import SglangTestEnvironment
from tests.harness.sglang.graph import assert_graph_events
from tests.harness.sglang.offline_probe import GRAPH_SETTINGS, MODEL_ID, ProbeResult
from tests.harness.sglang.probe import (
    GraphSettings,
    ProbeRun,
    graph_settings_key,
    run_probe_worker,
)

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.requires_config,
    pytest.mark.requires_model_weights(MODEL_ID),
    pytest.mark.timeout(300),
]


@pytest.fixture(scope="module")
def shim_loopback_runs(
    sglang_environment: SglangTestEnvironment,
    tmp_path_factory: pytest.TempPathFactory,
) -> list[ProbeRun]:
    """Run all four graph settings with direct shim loopback enabled."""

    return run_probe_worker(
        graph_settings_list=list(GRAPH_SETTINGS),
        base_gpu_id=sglang_environment.base_gpu_id,
        config_path=sglang_environment.config_path,
        tmp_path=tmp_path_factory.mktemp("sglang-shim-loopback"),
        loopback_mode="shim",
    )


def test_sglang_shim_loopback_graph_modes(shim_loopback_runs: list[ProbeRun]) -> None:
    results: dict[GraphSettings, ProbeResult] = {}
    runs_by_settings: dict[GraphSettings, ProbeRun] = {}
    for run in shim_loopback_runs:
        key = graph_settings_key(run.graph_settings)
        results[key] = run.result
        runs_by_settings[key] = run
        assert_graph_events(run.graph_settings, run.events)

    eager_output_ids = results[(False, False)]["output_ids"]
    for graph_settings in GRAPH_SETTINGS:
        assert results[graph_settings_key(graph_settings)]["output_ids"] == eager_output_ids

    print(
        "XPOOL_SGLANG_SHIM_LOOPBACK_DURATIONS="
        + json.dumps(
            [
                {
                    "base_gpu_id": run.base_gpu_id,
                    "cuda_graph": run.graph_settings.cuda_graph,
                    "duration_s": run.duration_s,
                    "piecewise_cuda_graph": run.graph_settings.piecewise_cuda_graph,
                }
                for run in (runs_by_settings[graph_settings_key(settings)] for settings in GRAPH_SETTINGS)
            ],
            sort_keys=True,
        )
    )
