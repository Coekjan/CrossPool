"""Manifest-driven model-serving topology, graph, and lifecycle evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from _pytest.mark.structures import ParameterSet

from tests.harness.sglang.manifest import E2E_MANIFEST_PATH, E2eManifest, E2eServingCase
from tests.harness.sglang.serving.alignment import (
    PREFILL_LOGITS_ARTIFACT_FILENAME,
    SERVING_GRAPH_ARTIFACT_FILENAME,
)
from tests.harness.sglang.serving.graph import SglangGraphMode
from tests.harness.sglang.serving.probe import run_probe
from tests.harness.support.native.observer import (
    assert_dp_attention_paths,
    assert_fabric_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
)
from tests.harness.support.sglang.graph import assert_run_graph_evidence
from xpool.config import XpoolConfig

pytest_plugins = ("tests.harness.support.config",)

MANIFEST = E2eManifest.load(E2E_MANIFEST_PATH)
SGLANG_DURATION_ARTIFACT_FILENAME = "sglang.duration.json"


def case_parameter(case: E2eServingCase, graph_mode: SglangGraphMode) -> ParameterSet:
    """Attach every manifest-derived requirement to one serving task."""

    model_ids = tuple(MANIFEST.model(placement.model).model_id for placement in case.models)
    marks = [
        pytest.mark.requires_cuda(min_devices=case.required_gpu_count),
        pytest.mark.requires_config,
        pytest.mark.requires_mps,
        pytest.mark.timeout(case.timeout_seconds),
        pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
    ]
    if len(case.graph_modes) >= 2:
        marks.append(pytest.mark.serving_graph_group(name=case.id, expected_case_count=len(case.graph_modes)))
    marks.extend(pytest.mark.requires_model_weights(model_id) for model_id in model_ids)
    return pytest.param(case, graph_mode, id=f"{case.id}-{graph_mode.value}", marks=marks)


@pytest.mark.parametrize(
    ("case", "graph_mode"),
    tuple(case_parameter(case, graph_mode) for case in MANIFEST.model_serving_cases for graph_mode in case.graph_modes),
)
def test_e2e_model_serving(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: XpoolConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove one manifest topology and graph mode through installed commands."""

    run = run_probe(
        MANIFEST,
        case,
        base_config=e2e_base_config,
        graph_settings=graph_mode.settings(),
        workdir=tmp_path,
    )
    assert_transport_observer_snapshots(
        run.launch.observer_outdir,
        expected_count=case.atnagent_count * len(case.models),
        site="atnagent",
    )
    assert_fabric_observer_snapshots(
        run.launch.observer_outdir,
        atnagent_count=case.atnagent_count,
        ffnagent_count=case.ffnagent_count,
        expected_topologies=tuple((model.atn_tp_size, model.atn_dp_size) for model in case.models),
    )
    if len(case.models) > 1:
        assert_two_model_executor_overlap(run.launch.observer_outdir)
    if any(model.atn_dp_size > 1 for model in case.models):
        assert_dp_attention_paths(run.launch.observer_outdir)
    assert_run_graph_evidence(run)
    if task_artifact_dir is not None and len(case.graph_modes) >= 2:
        run.serving_graph_artifact(case.id).write(task_artifact_dir / SERVING_GRAPH_ARTIFACT_FILENAME)
        prefill_logits_paths = tuple(
            result.prefill_logits_path for result in run.results if result.prefill_logits_path is not None
        )
        if len(prefill_logits_paths) > 1:
            raise AssertionError("one serving graph task cannot publish multiple prefill-logit artifacts")
        if prefill_logits_paths:
            shutil.copyfile(prefill_logits_paths[0], task_artifact_dir / PREFILL_LOGITS_ARTIFACT_FILENAME)
    durations = {
        "cuda_graph": run.graph_settings.cuda_graph,
        "daemon_startup_seconds": run.daemon_startup_seconds,
        "duration_seconds": run.duration_seconds,
        "piecewise_cuda_graph": run.graph_settings.piecewise_cuda_graph,
    }
    serialized_durations = json.dumps(durations, indent=2, sort_keys=True) + "\n"
    if task_artifact_dir is not None:
        (task_artifact_dir / SGLANG_DURATION_ARTIFACT_FILENAME).write_text(
            serialized_durations,
            encoding="utf-8",
        )
    print("XPOOL_SGLANG_MODEL_SERVING_DURATIONS=" + json.dumps(durations, sort_keys=True))
