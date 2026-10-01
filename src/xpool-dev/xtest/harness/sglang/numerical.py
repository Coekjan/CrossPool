"""Independent per-model numerical evidence for installed real FFN execution."""

from __future__ import annotations

import os
import time
from itertools import pairwise
from pathlib import Path

import pytest
import safetensors.torch
import torch
from _pytest.mark.structures import ParameterSet

from xkit import ResourceRequirements
from xkit.config import assemble_config
from xkit.network import TcpEndpointReservation, TcpPortSpace
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.ffn import FfnLayerSpec, MoeFfnSpec
from xpool.native.ffn import DpRowLayout, ForwardMode, OutputRequirement
from xpool.runtime.ffnagent import architecture
from xpool.runtime.transport import InstanceRankTransportProfile
from xtest.harness.native.ffn.protocol import FfnInstanceSpec, FfnInvocationSpec
from xtest.harness.native.ffn.topology import ffn_cluster_environment, materialize_ffn_cluster_launch, run_ffn_topology
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eFfnNumericalCase
from xtest.harness.sglang.reference.ffn import SglangFfnReferenceCase, SglangFfnReferenceRunner
from xtest.harness.support.ffn import (
    FFN_NUMERICAL_ARTIFACT_FILENAME,
    FFN_OUTPUT_ATOL,
    FFN_OUTPUT_MAXIMUM_SCALED_ERROR_LIMIT,
    FFN_OUTPUT_NORMALIZED_RMSE_LIMIT,
    FFN_OUTPUT_RTOL,
    FFN_ROUTING_ATOL,
    FFN_ROUTING_RTOL,
    FfnNumericalEvidence,
    FfnNumericalSampleEvidence,
    compare_ffn_outputs,
    compare_ffn_routing,
    generate_ffn_hidden_states,
    read_ffn_routing_records,
)
from xtest.harness.support.wait import remaining_seconds

FFN_NUMERICAL_HARNESS_RESERVE_SECONDS = 120.0


def requirements_of(case: E2eFfnNumericalCase) -> ResourceRequirements:
    """Declare the peak production/reference lease and the case's local checkpoint."""

    return ResourceRequirements(case.production_required_gpu_count, True, True, (case.model_id,))


def case_parameter(case: E2eFfnNumericalCase) -> ParameterSet:
    """Retain native parameter IDs and case-owned scheduling/deadline marks."""

    return pytest.param(
        case,
        marks=(
            pytest.mark.timeout(case.timeout_seconds),
            pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
        ),
    )


def run_numerical_case(
    case: E2eFfnNumericalCase,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Compare installed true-TP2 FFN outputs with isolated original SGLang."""

    if case.timeout_seconds <= FFN_NUMERICAL_HARNESS_RESERVE_SECONDS:
        raise ValueError("FFN numerical timeout must exceed its exceptional cleanup reserve")
    deadline = time.monotonic() + case.timeout_seconds - FFN_NUMERICAL_HARNESS_RESERVE_SECONDS
    production_workdir = tmp_path / "production"
    config = assemble_config(
        (case.model_id,),
        deployment=case.deployment,
        runtime_config=e2e_base_config.path,
        overrides={
            "scheduler": {
                "ffn_policy": "fifo",
                "ffn_random_seed": None,
            },
        },
        env={
            **os.environ,
            **ffn_cluster_environment(
                config_path=production_workdir / "xpool.toml", observer_outdir=production_workdir / "observers"
            ),
        },
    )
    model = config.model_by_id[case.model_id]
    ffn_tp_size = model.ffn_tp_size if model.ffn_tp_size is not None else len(config.ffn.devices)
    model_path = config.model_path_of(case.model_id)
    model_spec = architecture.load(model_id=case.model_id, model_path=model_path)
    assert representative_layer_ids(model_spec.layers) == case.layer_ids

    hidden_states = generate_ffn_hidden_states(
        hidden_size=model_spec.hidden_size,
        seed=case.input_matrix.seed,
        row_counts=case.input_matrix.row_counts,
    )
    reference_cases = tuple(
        SglangFfnReferenceCase(
            case_id=sample_id(layer_id, row_count),
            layer_id=layer_id,
            hidden_states=values.clone(),
        )
        for layer_id in case.layer_ids
        for row_count, values in zip(case.input_matrix.row_counts, hidden_states, strict=True)
    )
    reference_results = SglangFfnReferenceRunner(
        workdir=tmp_path / "reference",
        timeout_seconds=remaining_seconds(deadline, "FFN numerical Reference"),
    ).run(
        model_path=model_path,
        tensor_parallel_size=ffn_tp_size,
        cases=reference_cases,
    )

    endpoint = TcpEndpointReservation.reserve(config.daemon.host, port_space=TcpPortSpace.local())
    launch = materialize_ffn_cluster_launch(
        config=config,
        daemon_port=endpoint.port,
        workdir=production_workdir,
    )
    exchange_directory = production_workdir / "exchange"
    exchange_directory.mkdir()
    input_paths: dict[int, Path] = {}
    for row_count, values in zip(case.input_matrix.row_counts, hidden_states, strict=True):
        input_path = exchange_directory / f"input-rows-{row_count}.safetensors"
        input_path.write_bytes(safetensors.torch.save({"hidden_states": values}))
        input_paths[row_count] = input_path

    layer_ordinal_by_id = {layer.layer_id: ordinal for ordinal, layer in enumerate(model_spec.layers)}
    output_paths: dict[str, Path] = {}
    invocations: list[FfnInvocationSpec] = []
    for layer_id in case.layer_ids:
        for row_count in case.input_matrix.row_counts:
            case_id = sample_id(layer_id, row_count)
            output_path = exchange_directory / f"output-{case_id}.safetensors"
            output_paths[case_id] = output_path
            invocations.append(
                FfnInvocationSpec(
                    case_id=case_id,
                    input_path=input_paths[row_count],
                    output_path=output_path,
                    layer_ordinal=layer_ordinal_by_id[layer_id],
                    forward_mode=ForwardMode.DECODE if row_count == 1 else ForwardMode.PREFILL,
                    output_requirement=OutputRequirement.PER_RANK_COMPLETE,
                    dp_row_layout=DpRowLayout.NONE,
                    dp_rank_payload_rows=None,
                )
            )
    ffn_profile = InstanceFfnProfile(
        payload_dtype=torch.bfloat16,
        hidden_size=model_spec.hidden_size,
        layers=tuple(InstanceFfnLayerProfile(layer_id=layer.layer_id, kind=layer.kind) for layer in model_spec.layers),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=case.input_matrix.row_counts[-1],
        group_sum_complete_admitted=False,
    )
    instance_spec = FfnInstanceSpec(
        environment=dict(launch.environment),
        model_id=case.model_id,
        rank=0,
        ffn_tp_size=ffn_tp_size,
        ffn_profile=ffn_profile,
        transport=InstanceRankTransportProfile(
            hidden_size=model_spec.hidden_size,
            payload_row_capacity=case.input_matrix.row_counts[-1],
            atn_tp_rank=0,
            atn_tp_size=config.atn_tp_size_of(case.model_id),
            atn_dp_rank=0,
            atn_dp_size=model.atn_dp_size,
        ),
        invocations=tuple(invocations),
    )
    (ready,) = run_ffn_topology(
        launch=launch,
        endpoint=endpoint,
        instance_specs=(instance_spec,),
        workdir=production_workdir,
        timeout_seconds=remaining_seconds(deadline, "FFN numerical Production"),
    )

    observed_routing = read_ffn_routing_records(production_workdir / "observers", ready.generation)
    samples: list[FfnNumericalSampleEvidence] = []
    reference_by_id = {result.case_id: result for result in reference_results}
    for invocation_sequence, invocation in enumerate(invocations, start=1):
        reference = reference_by_id[invocation.case_id]
        actual_tensors = safetensors.torch.load_file(output_paths[invocation.case_id], device="cpu")
        if set(actual_tensors) != {"hidden_states"}:
            raise AssertionError(f"FFN production output has invalid tensor keys: {sorted(actual_tensors)}")
        output_comparison = compare_ffn_outputs(
            reference.output,
            actual_tensors["hidden_states"],
            atol=FFN_OUTPUT_ATOL,
            rtol=FFN_OUTPUT_RTOL,
            normalized_rmse_limit=FFN_OUTPUT_NORMALIZED_RMSE_LIMIT,
            maximum_scaled_error_limit=FFN_OUTPUT_MAXIMUM_SCALED_ERROR_LIMIT,
        )
        layer = model_spec.layers[invocation.layer_ordinal]
        routing_comparison = None
        if isinstance(layer, MoeFfnSpec):
            if reference.routing is None:
                raise AssertionError("MoE Reference omitted Routing evidence")
            routing_key = (ready.instance_index, invocation_sequence, invocation.layer_ordinal)
            try:
                actual_ids, actual_weights = observed_routing.pop(routing_key)
            except KeyError as error:
                raise AssertionError(f"production omitted FFN Routing record {routing_key}") from error
            routing_comparison = compare_ffn_routing(
                reference.routing.topk_ids,
                reference.routing.topk_weights,
                actual_ids,
                actual_weights,
                expert_count=layer.routed_expert_count + layer.shared_expert_count,
                atol=FFN_ROUTING_ATOL,
                rtol=FFN_ROUTING_RTOL,
            )
        elif reference.routing is not None:
            raise AssertionError("Dense Reference unexpectedly returned Routing evidence")
        samples.append(
            FfnNumericalSampleEvidence(
                layer_id=layer.layer_id,
                row_count=reference.output.shape[0],
                output=output_comparison,
                routing=routing_comparison,
            )
        )
    if observed_routing:
        raise AssertionError(f"production returned unexpected FFN Routing records: {tuple(observed_routing)}")
    evidence = FfnNumericalEvidence(
        model_id=case.model_id,
        model_spec_digest=model_spec.digest(),
        generation=ready.generation,
        samples=tuple(samples),
    )
    evidence.write((task_artifact_dir or tmp_path) / FFN_NUMERICAL_ARTIFACT_FILENAME)
    assert all(sample.output.passed for sample in samples)
    assert all(sample.routing is None or sample.routing.passed for sample in samples)


def representative_layer_ids(layers: tuple[FfnLayerSpec, ...]) -> tuple[int, ...]:
    """Return the accepted first, kind-transition, and final Layer IDs."""

    if not layers:
        raise ValueError("FFN representative coverage requires typed nonempty layers")
    selected = [layers[0].layer_id]
    selected.extend(layer.layer_id for previous, layer in pairwise(layers) if layer.kind != previous.kind)
    selected.append(layers[-1].layer_id)
    return tuple(dict.fromkeys(selected))


def sample_id(layer_id: int, row_count: int) -> str:
    """Return one deterministic Reference and Production sample identity."""

    return f"layer-{layer_id}-rows-{row_count}"
