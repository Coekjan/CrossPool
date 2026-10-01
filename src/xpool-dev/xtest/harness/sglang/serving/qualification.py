"""Shared installed-serving evidence for routine and model qualification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from _pytest.mark.structures import ParameterSet

from xkit import ResourceRequirements
from xkit.results import write_json
from xkit.serving.sglang.graph import SglangGraphMode
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving.alignment import (
    PREFILL_LOGITS_ARTIFACT_FILENAME,
    SERVING_GRAPH_ARTIFACT_FILENAME,
)
from xtest.harness.sglang.serving.probe import run_probe
from xtest.harness.support.native.observer import (
    assert_dp_attention_paths,
    assert_fabric_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
)
from xtest.harness.support.sglang.graph import assert_run_graph_evidence

SGLANG_DURATION_ARTIFACT_FILENAME = "sglang.duration.json"


def requirements_of(
    case: E2eServingCase,
    graph_mode: SglangGraphMode | None = None,
) -> ResourceRequirements:
    """Declare the complete deployment and local checkpoints for a serving row."""

    del graph_mode
    return ResourceRequirements(case.required_gpu_count, True, True, case.models)


def graph_rows(
    case: E2eServingCase,
    *,
    compare_modes: bool = False,
) -> tuple[ParameterSet, ...]:
    """Expand declared graph modes with native row IDs, timing and comparison groups."""

    marks = [
        pytest.mark.timeout(case.timeout_seconds),
        pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
    ]
    if compare_modes and len(case.graph_modes) > 1:
        marks.append(pytest.mark.serving_graph_group(expected_case_count=len(case.graph_modes)))
    return tuple(
        pytest.param(case, graph_mode, id=f"{case.id or case.deployment.stem}-{graph_mode.value}", marks=marks)
        for graph_mode in case.graph_modes
    )


def run_serving_case(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
    *,
    compare_modes: bool,
) -> None:
    """Prove one manifest topology and graph mode through installed commands."""

    run = run_probe(
        case,
        base_config=e2e_base_config,
        graph_settings=graph_mode.settings(),
        workdir=tmp_path,
    )
    assert_transport_observer_snapshots(
        run.observer_outdir,
        expected_count=case.atnagent_count * len(case.models),
        site="atnagent",
    )
    assert_fabric_observer_snapshots(
        run.observer_outdir,
        atnagent_count=case.atnagent_count,
        ffnagent_count=case.ffnagent_count,
        expected_topologies=tuple(
            (case.atnagent_count // model.atn_dp_size, model.atn_dp_size)
            for model_id in case.models
            for model in (case.deployment_config.model_by_id[model_id],)
        ),
    )
    if len(case.models) > 1:
        assert_two_model_executor_overlap(run.observer_outdir)
    if any(model.atn_dp_size > 1 for model in case.deployment_config.models):
        assert_dp_attention_paths(run.observer_outdir)
    assert_run_graph_evidence(run)
    if task_artifact_dir is not None and compare_modes:
        run.serving_graph_artifact().write(task_artifact_dir / SERVING_GRAPH_ARTIFACT_FILENAME)
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
    if task_artifact_dir is not None:
        write_json(task_artifact_dir / SGLANG_DURATION_ARTIFACT_FILENAME, durations)
    print("XPOOL_SGLANG_MODEL_SERVING_DURATIONS=" + json.dumps(durations, sort_keys=True))
