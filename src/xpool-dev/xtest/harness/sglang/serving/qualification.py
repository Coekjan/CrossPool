"""Installed-serving structure and same-prefix numerical qualification."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import httpx
import pytest
from _pytest.mark.structures import ParameterSet
from pydantic import JsonValue, TypeAdapter

from xkit import ResourceRequirements
from xkit.config import resolve_model_weights
from xkit.results import write_json
from xkit.serving.sglang.deterministic import SamplerReplayEvidence
from xkit.serving.sglang.graph import SglangGraphMode, SglangGraphSettings
from xpool.integrations.sglang.devkit.requests import LocalModelMetadata
from xpool.model import ModelId
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving.alignment import (
    DECODE_DISTRIBUTION_KL_LIMIT,
    PREFILL_DISTRIBUTION_KL_LIMIT,
    DistributionComparison,
    compare_distributions,
)
from xtest.harness.sglang.serving.attempt import ProbeRun
from xtest.harness.sglang.serving.probe import run_probe
from xtest.harness.sglang.serving.server import SglangProbeServer, SglangServerResult, generation_request
from xtest.harness.support.native.observer import (
    assert_dp_attention_paths,
    assert_fabric_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
)
from xtest.harness.support.sglang.graph import assert_run_graph_evidence

SGLANG_DURATION_ARTIFACT_FILENAME = "sglang.duration.json"


def requirements_of(case: E2eServingCase, graph_mode: SglangGraphMode | None = None) -> ResourceRequirements:
    """Declare the complete deployment and local checkpoints for a serving case."""

    del graph_mode
    return ResourceRequirements(case.required_device_count, True, case.models)


def graph_rows(case: E2eServingCase) -> tuple[ParameterSet, ...]:
    """Expand independent structural graph modes with their per-mode time budget."""

    return tuple(
        pytest.param(
            case,
            graph_mode,
            id=f"{case.id or case.deployment.stem}-{graph_mode.value}",
            marks=(
                pytest.mark.timeout(case.timeout_seconds),
                pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
            ),
        )
        for graph_mode in case.graph_modes
    )


def model_rows(case: E2eServingCase) -> tuple[ParameterSet, ...]:
    """Keep all modes of one model/topology in one item and one device lease."""

    deployments = len(required_graph_settings(case.graph_modes))
    return (
        pytest.param(
            case,
            id=f"{case.id or case.deployment.stem}-{'-vs-'.join(mode.value for mode in case.graph_modes)}",
            marks=(
                pytest.mark.timeout(case.timeout_seconds * deployments),
                pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds * deployments),
            ),
        ),
    )


def required_graph_settings(graph_modes: tuple[SglangGraphMode, ...]) -> tuple[SglangGraphSettings, ...]:
    """Order candidates after their Eager-Decode references, reusing each setting.

    Singleton cases retain their structural workload. Numerical cases start
    with ordinary Eager, whose continuation is replayed by all later runs.
    """

    if len(graph_modes) == 1:
        return (graph_modes[0].settings(),)
    settings = [SglangGraphMode.EAGER.settings()]
    for mode in graph_modes:
        candidate = mode.settings()
        for required in (replace(candidate, decode_backend="disabled"), candidate):
            if required not in settings:
                settings.append(required)
    return tuple(settings)


def assert_serving_evidence(case: E2eServingCase, run: ProbeRun) -> None:
    """Require observed graph, Transport and Fabric behavior for the deployment."""

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


def run_serving_case(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove one topology and graph mode through installed commands."""

    run = run_probe(case, base_config=e2e_base_config, graph_settings=graph_mode.settings(), workdir=tmp_path)
    assert_serving_evidence(case, run)
    durations = {
        "decode_backend": run.graph_settings.decode_backend,
        "daemon_startup_seconds": run.daemon_startup_seconds,
        "duration_seconds": run.duration_seconds,
        "prefill_backend": run.graph_settings.prefill_backend,
    }
    if task_artifact_dir is not None:
        write_json(task_artifact_dir / SGLANG_DURATION_ARTIFACT_FILENAME, durations)
    print("XPOOL_SGLANG_MODEL_SERVING_DURATIONS=" + json.dumps(durations, sort_keys=True))


def run_model_case(
    case: E2eServingCase,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Record Eager, replay continuations and isolate each Graph phase's verdict.

    Each deployment fully retires before the next starts. All verdicts execute
    inside pytest, and direct pytest retains numerical evidence under tmp_path.
    Single-mode model cases retain their structural serving contract.
    """

    if len(case.graph_modes) == 1:
        run_serving_case(case, case.graph_modes[0], e2e_base_config, tmp_path, task_artifact_dir)
        return
    if SglangGraphMode.EAGER not in case.graph_modes:
        raise ValueError("numerical qualification requires an Eager reference mode")
    root = task_artifact_dir if task_artifact_dir is not None else tmp_path / "numerical"
    root.mkdir(parents=True, exist_ok=True)
    vocabularies = {
        model_id: LocalModelMetadata.from_checkpoint(
            resolve_model_weights(e2e_base_config.config, model_id), text_prompts={}
        ).vocab_size
        for model_id in case.models
    }
    eager_settings = SglangGraphMode.EAGER.settings()
    results: dict[SglangGraphSettings, dict[ModelId, SglangServerResult]] = {}
    replay_path: Path | None = None
    durations = []
    for settings in required_graph_settings(case.graph_modes):

        def workload(servers: list[SglangProbeServer]) -> tuple[SglangServerResult, ...]:
            barrier = Barrier(len(servers))

            def generate(server: SglangProbeServer) -> SglangServerResult:
                model_id = server.process.model.model_id
                with httpx.Client(base_url=server.url(), timeout=60.0) as client:
                    client.post("/flush_cache").raise_for_status()
                result = server.generate(
                    generation_request(f"xpool-numerical-{model_id}", vocab_size=vocabularies[model_id]),
                    record_path=root / settings.id() / model_id.uri_encode() / "generation.json",
                    request_barrier=barrier,
                )
                if result.cached_tokens != 0:
                    raise AssertionError(f"numerical request did not start cold: {model_id}")
                return result

            with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                outputs = tuple(executor.map(generate, servers))
            return outputs

        run = run_probe(
            case,
            base_config=e2e_base_config,
            graph_settings=settings,
            workdir=tmp_path / settings.id(),
            workload=workload,
            deterministic_mode="record" if settings == eager_settings else "replay",
            replay_path=replay_path,
        )
        if settings == eager_settings:
            replay_path = run.sampling_record_path
            if replay_path is None:
                raise AssertionError("Eager did not produce a sampling record")
        else:
            for actual in run.results:
                reference = results[eager_settings][actual.model_id]
                if actual.input_ids != reference.input_ids or actual.output_ids != reference.output_ids:
                    raise AssertionError(f"sampler replay changed conditioning: {actual.model_id}")
                tp_size = case.atnagent_count // case.deployment_config.model_by_id[actual.model_id].atn_dp_size
                assert_sampler_replay(actual, run.observer_outdir, tp_size=tp_size)
        results[settings] = {result.model_id: result for result in run.results}
        durations.append(
            {
                "graph_settings": TypeAdapter(SglangGraphSettings).dump_python(settings, mode="json"),
                "daemon_startup_seconds": run.daemon_startup_seconds,
                "duration_seconds": run.duration_seconds,
                "observer_outdir": str(run.observer_outdir),
            }
        )
        write_json(root / SGLANG_DURATION_ARTIFACT_FILENAME, durations)
        assert_serving_evidence(case, run)

    stage_reports: list[JsonValue] = []
    end_to_end_reports: list[JsonValue] = []
    checks: list[tuple[DistributionComparison, float]] = []
    for settings, outputs in results.items():
        if settings == eager_settings:
            continue
        for model_id, actual in outputs.items():
            end_to_end = compare_distributions(
                results[eager_settings][model_id], actual, prefix_lengths=actual.prefix_lengths
            )
            end_to_end_reports.append(
                {
                    "reference": TypeAdapter(SglangGraphSettings).dump_python(eager_settings, mode="json"),
                    "candidate": TypeAdapter(SglangGraphSettings).dump_python(settings, mode="json"),
                    "model_id": str(model_id),
                    **TypeAdapter(DistributionComparison).dump_python(end_to_end, mode="json"),
                    "prefill_passed": end_to_end.kl_values[0] <= PREFILL_DISTRIBUTION_KL_LIMIT,
                    "decode_passed": all(value <= DECODE_DISTRIBUTION_KL_LIMIT for value in end_to_end.kl_values[1:]),
                }
            )
            for phase in ("prefill", "decode"):
                if phase == "prefill":
                    reference_settings = eager_settings
                    comparison = DistributionComparison(end_to_end.prefix_lengths[:1], end_to_end.kl_values[:1])
                    limit = PREFILL_DISTRIBUTION_KL_LIMIT
                elif settings.decode_backend == "disabled":
                    continue
                else:
                    reference_settings = replace(settings, decode_backend="disabled")
                    if reference_settings == eager_settings:
                        comparison = DistributionComparison(end_to_end.prefix_lengths[1:], end_to_end.kl_values[1:])
                    else:
                        comparison = compare_distributions(
                            results[reference_settings][model_id], actual, prefix_lengths=actual.prefix_lengths[1:]
                        )
                    limit = DECODE_DISTRIBUTION_KL_LIMIT
                checks.append((comparison, limit))
                stage_reports.append(
                    {
                        "phase": phase,
                        "reference": TypeAdapter(SglangGraphSettings).dump_python(reference_settings, mode="json"),
                        "candidate": TypeAdapter(SglangGraphSettings).dump_python(settings, mode="json"),
                        "model_id": str(model_id),
                        **TypeAdapter(DistributionComparison).dump_python(comparison, mode="json"),
                        "limit": limit,
                        "passed": all(value <= limit for value in comparison.kl_values),
                    }
                )
    write_json(
        root / "numerical.json",
        {
            "configured_decode_policy": "matched_arithmetic",
            "attempts": [
                {
                    "model_id": str(model_id),
                    "graph_settings": TypeAdapter(SglangGraphSettings).dump_python(settings, mode="json"),
                    "reported_attention_backends": {
                        "prefill": result.reported_prefill_attention_backend,
                        "decode": result.reported_decode_attention_backend,
                    },
                }
                for settings, outputs in results.items()
                for model_id, result in outputs.items()
            ],
            "prefill_kl_limit": PREFILL_DISTRIBUTION_KL_LIMIT,
            "decode_kl_limit": DECODE_DISTRIBUTION_KL_LIMIT,
            "stage_comparisons": stage_reports,
            "end_to_end_comparisons": end_to_end_reports,
        },
    )
    for comparison, limit in checks:
        comparison.assert_within(limit)


def assert_sampler_replay(result: SglangServerResult, directory: Path, *, tp_size: int) -> None:
    """Require actual selection coverage on every rank of the routed TP group."""

    evidence = [
        item
        for path in directory.glob("xkit.sampler-replay.*.jsonl")
        for line in path.read_bytes().splitlines()
        for item in (TypeAdapter(SamplerReplayEvidence).validate_json(line),)
        if item.model_id == result.model_id and item.request_id == result.request_id and item.prescribed_id is not None
    ]
    ranks = {item.rank for item in evidence}
    if len(ranks) != tp_size:
        raise AssertionError(f"sampler replay did not cover the routed TP group: {result.request_id}")
    for rank in ranks:
        rows = [item for item in evidence if item.rank == rank]
        if tuple(item.prefix_length for item in rows) != result.prefix_lengths:
            raise AssertionError(f"sampler replay has missing or repeated prefixes on rank {rank}")
        for index, item in enumerate(rows):
            allowed = ("extend", "mixed") if index == 0 else ("decode", "mixed")
            if item.prescribed_id != result.output_ids[index] or item.forward_mode not in allowed:
                raise AssertionError(f"sampler replay phase or token differs at prefix {item.prefix_length}")
