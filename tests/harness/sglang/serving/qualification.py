"""Shared installed-serving evidence for routine and model qualification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from _pytest.mark.structures import ParameterSet

from tests.harness.sglang.manifest import E2eModel, E2eServingCase
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
from xpool.config import LatencySloConfig, XpoolConfig

SGLANG_DURATION_ARTIFACT_FILENAME = "sglang.duration.json"


def case_parameter(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    *,
    models: tuple[E2eModel, ...],
    compare_modes: bool,
) -> ParameterSet:
    """Attach every model and resource requirement to one serving task."""

    marks = [
        pytest.mark.requires_cuda(min_devices=case.required_gpu_count),
        pytest.mark.requires_config,
        pytest.mark.requires_mps,
        pytest.mark.timeout(case.timeout_seconds),
        pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
    ]
    if compare_modes:
        marks.append(pytest.mark.serving_graph_group(name=case.id, expected_case_count=len(case.graph_modes)))
    marks.extend(pytest.mark.requires_model_weights(model.model_id) for model in models)
    return pytest.param(case, graph_mode, id=f"{case.id}-{graph_mode.value}", marks=marks)


def run_serving_case(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: XpoolConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
    *,
    models: tuple[E2eModel, ...],
    serving_slo: LatencySloConfig,
    compare_modes: bool,
) -> None:
    """Prove one manifest topology and graph mode through installed commands."""

    run = run_probe(
        case,
        models=models,
        serving_slo=serving_slo,
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
    if task_artifact_dir is not None and compare_modes:
        run.serving_graph_artifact(case.id).write(task_artifact_dir / SERVING_GRAPH_ARTIFACT_FILENAME)
        prefill_logits_paths = tuple(
            result.prefill_logits_path for result in run.results if result.prefill_logits_path is not None
        )
        if len(prefill_logits_paths) > 1:
            raise AssertionError("one serving graph task cannot publish multiple prefill-logit artifacts")
        if prefill_logits_paths:
            shutil.copyfile(prefill_logits_paths[0], task_artifact_dir / PREFILL_LOGITS_ARTIFACT_FILENAME)
    durations = {
        "decode_backend": run.graph_settings.decode_backend,
        "daemon_startup_seconds": run.daemon_startup_seconds,
        "duration_seconds": run.duration_seconds,
        "prefill_backend": run.graph_settings.prefill_backend,
    }
    serialized_durations = json.dumps(durations, indent=2, sort_keys=True) + "\n"
    if task_artifact_dir is not None:
        (task_artifact_dir / SGLANG_DURATION_ARTIFACT_FILENAME).write_text(
            serialized_durations,
            encoding="utf-8",
        )
    print("XPOOL_SGLANG_MODEL_SERVING_DURATIONS=" + json.dumps(durations, sort_keys=True))
