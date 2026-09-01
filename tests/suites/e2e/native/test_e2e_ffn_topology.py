"""Installed real-FFN topology and delivery qualification."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import pytest
import safetensors.torch
import torch
from _pytest.mark.structures import ParameterSet

from tests.harness.native.ffn.protocol import FfnInstanceReady, FfnInstanceSpec, FfnInvocationSpec
from tests.harness.native.ffn.topology import (
    FfnLiveTopologyObservation,
    materialize_ffn_cluster_launch,
    run_ffn_topology,
)
from tests.harness.runner.gpu import GpuPool
from tests.harness.runner.network import TcpEndpointReservation, TcpPortSpace
from tests.harness.sglang.manifest import E2E_MANIFEST_PATH, E2eFfnTopologyCase, E2eManifest, E2eModel
from tests.harness.sglang.reference.ffn import SglangFfnReferenceCase, SglangFfnReferenceRunner
from tests.harness.support.ffn import (
    FFN_OUTPUT_ATOL,
    FFN_OUTPUT_MAXIMUM_SCALED_ERROR_LIMIT,
    FFN_OUTPUT_NORMALIZED_RMSE_LIMIT,
    FFN_OUTPUT_RTOL,
    FFN_TOPOLOGY_ARTIFACT_FILENAME,
    FfnTopologyEvidence,
    FfnTopologyMpsServerEvidence,
    FfnTopologyOutputEvidence,
    FfnTopologyProcessEvidence,
    compare_ffn_outputs,
    generate_ffn_hidden_states,
)
from tests.harness.support.native.observer import (
    object_dict,
    object_list,
    read_fabric_observer_snapshots,
    read_json_object,
)
from tests.harness.support.wait import remaining_seconds
from xpool.config import ModelConfig, XpoolConfig
from xpool.fabric import FabricGenerationId, FifoSchedulerPolicy, InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.ffn import FfnModelSpec
from xpool.native.ffn import DpRowLayout, ForwardMode, OutputRequirement
from xpool.runtime.ffnagent import architecture
from xpool.runtime.transport import InstanceRankTransportProfile

pytest_plugins = ("tests.harness.support.config",)

MANIFEST = E2eManifest.load(E2E_MANIFEST_PATH)
FFN_TOPOLOGY_INPUT_SEED = 17


def case_parameter(case: E2eFfnTopologyCase) -> ParameterSet:
    """Attach manifest-derived resources to one topology world."""

    model_ids = tuple(MANIFEST.model(instance.model).model_id for instance in case.instances)
    return pytest.param(
        case,
        id=case.id,
        marks=(
            pytest.mark.requires_cuda(min_devices=case.required_gpu_count),
            pytest.mark.requires_config,
            pytest.mark.requires_mps,
            *(pytest.mark.requires_model_weights(model_id) for model_id in model_ids),
            pytest.mark.timeout(case.timeout_seconds),
            pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
        ),
    )


@pytest.mark.parametrize("case", tuple(case_parameter(case) for case in MANIFEST.ffn_topology_cases))
def test_e2e_ffn_topology(
    case: E2eFfnTopologyCase,
    e2e_base_config: XpoolConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove real placement, delivery, scheduling, MPS, and numerical output."""

    deadline = time.monotonic() + case.timeout_seconds
    models = tuple(MANIFEST.model(instance.model) for instance in case.instances)
    task_config = e2e_base_config.model_copy(update={"models": [ModelConfig(id=model.model_id) for model in models]})
    model_specs = tuple(
        architecture.load(model_id=model.model_id, model_path=task_config.model_path_of(model.model_id))
        for model in models
    )
    for instance, model_spec in zip(case.instances, model_specs, strict=True):
        if instance.layer_ordinal >= len(model_spec.layers):
            raise AssertionError(f"topology layer ordinal {instance.layer_ordinal} is outside {model_spec.model_id}")

    hidden_states = tuple(
        generate_ffn_hidden_states(
            hidden_size=model_specs[request.instance_index].hidden_size,
            seed=FFN_TOPOLOGY_INPUT_SEED + request_ordinal,
            row_counts=(sum(request.dp_rank_payload_rows),),
        )[0]
        for request_ordinal, request in enumerate(case.requests)
    )
    references = run_references(
        case,
        models=models,
        model_specs=model_specs,
        hidden_states=hidden_states,
        base_config=task_config,
        workdir=tmp_path / "reference",
        deadline=deadline,
    )

    endpoint = TcpEndpointReservation.reserve(task_config.daemon.host, port_space=TcpPortSpace.local())
    artifact_root = task_artifact_dir or tmp_path
    production_workdir = artifact_root / "production"
    launch = materialize_ffn_cluster_launch(
        base_config=task_config,
        model_tp_sizes=tuple(
            (model.model_id, instance.ffn_tp_size) for model, instance in zip(models, case.instances, strict=True)
        ),
        atnagent_count=case.atnagent_count,
        ffnagent_count=case.ffnagent_count,
        executor_lane_count=case.executor_lane_count,
        scheduler=FifoSchedulerPolicy(),
        daemon_port=endpoint.port,
        workdir=production_workdir,
    )
    exchange_directory = production_workdir / "exchange"
    exchange_directory.mkdir()
    input_paths = []
    for request_ordinal, values in enumerate(hidden_states):
        path = exchange_directory / f"input-request-{request_ordinal}.safetensors"
        path.write_bytes(safetensors.torch.save({"hidden_states": values}))
        input_paths.append(path)

    instance_specs, spec_instance_indices, output_paths = materialize_instance_specs(
        case,
        model_specs=model_specs,
        environment=dict(launch.environment),
        input_paths=tuple(input_paths),
        exchange_directory=exchange_directory,
    )
    live_observations: list[FfnLiveTopologyObservation] = []
    ready = run_ffn_topology(
        launch=launch,
        endpoint=endpoint,
        instance_specs=instance_specs,
        workdir=production_workdir,
        timeout_seconds=remaining_seconds(deadline, "FFN topology Production"),
        observation_sink=live_observations.append,
    )
    if len(live_observations) != 1:
        raise AssertionError("topology harness omitted its live process observation")
    generation = ready[0].generation
    actual_instance_indices = resolve_instance_indices(case, ready, spec_instance_indices)
    observer_outdir = production_workdir / "observers"
    execution_groups = resolve_execution_groups(case, ready, spec_instance_indices)
    validate_graph_snapshots(
        observer_outdir,
        generation=generation,
        case=case,
        ready=ready,
    )
    validate_trace(
        observer_outdir,
        case=case,
        actual_instance_indices=actual_instance_indices,
        execution_groups=execution_groups,
    )
    outputs = compare_outputs(case, references, output_paths)
    processes, mps_servers = platform_evidence(live_observations[0])
    evidence = FfnTopologyEvidence(
        case_id=case.id,
        generation=generation,
        processes=processes,
        mps_servers=mps_servers,
        outputs=outputs,
    )
    evidence.write(artifact_root / FFN_TOPOLOGY_ARTIFACT_FILENAME)
    assert all(output.comparison.passed for output in outputs)


def run_references(
    case: E2eFfnTopologyCase,
    *,
    models: tuple[E2eModel, ...],
    model_specs: tuple[FfnModelSpec, ...],
    hidden_states: tuple[torch.Tensor, ...],
    base_config: XpoolConfig,
    workdir: Path,
    deadline: float,
) -> dict[int, torch.Tensor]:
    """Evaluate each topology request with its original model FFN."""

    results: dict[int, torch.Tensor] = {}
    for instance_index, (instance, model, model_spec) in enumerate(
        zip(case.instances, models, model_specs, strict=True)
    ):
        request_ordinals = tuple(
            ordinal for ordinal, request in enumerate(case.requests) if request.instance_index == instance_index
        )
        reference_cases = tuple(
            SglangFfnReferenceCase(
                case_id=f"{case.id}-request-{ordinal}",
                layer_id=model_spec.layers[instance.layer_ordinal].layer_id,
                hidden_states=hidden_states[ordinal].clone(),
            )
            for ordinal in request_ordinals
        )
        reference_results = SglangFfnReferenceRunner(
            workdir=workdir / f"instance-{instance_index}",
            timeout_seconds=remaining_seconds(deadline, "FFN topology Reference"),
        ).run(
            model_path=base_config.model_path_of(model.model_id),
            tensor_parallel_size=instance.ffn_tp_size,
            cases=reference_cases,
        )
        for ordinal, result in zip(request_ordinals, reference_results, strict=True):
            results[ordinal] = result.output
    if set(results) != set(range(len(case.requests))):
        raise AssertionError("topology Reference omitted a request")
    return results


def materialize_instance_specs(
    case: E2eFfnTopologyCase,
    *,
    model_specs: tuple[FfnModelSpec, ...],
    environment: dict[str, str],
    input_paths: tuple[Path, ...],
    exchange_directory: Path,
) -> tuple[tuple[FfnInstanceSpec, ...], tuple[int, ...], dict[tuple[int, int], Path]]:
    """Build every rank-local Instance contract from one manifest case."""

    specs = []
    spec_instance_indices = []
    output_paths: dict[tuple[int, int], Path] = {}
    for instance_index, (instance, model_spec) in enumerate(zip(case.instances, model_specs, strict=True)):
        request_ordinals = tuple(
            ordinal for ordinal, request in enumerate(case.requests) if request.instance_index == instance_index
        )
        row_capacities = tuple(sum(case.requests[ordinal].dp_rank_payload_rows) for ordinal in request_ordinals)
        decode_capacities = tuple(
            rows
            for ordinal, rows in zip(request_ordinals, row_capacities, strict=True)
            if case.requests[ordinal].forward_mode == "decode"
        )
        prefill_capacities = tuple(
            rows
            for ordinal, rows in zip(request_ordinals, row_capacities, strict=True)
            if case.requests[ordinal].forward_mode == "prefill"
        )
        profile = InstanceFfnProfile(
            model_config_digest=model_spec.model_config_digest,
            payload_dtype=torch.bfloat16,
            hidden_size=model_spec.hidden_size,
            layers=tuple(
                InstanceFfnLayerProfile(layer_id=layer.layer_id, kind=layer.kind) for layer in model_spec.layers
            ),
            decode_payload_row_capacity=max(decode_capacities, default=1),
            prefill_payload_row_capacity=max(prefill_capacities, default=1),
            group_sum_complete_admitted=any(
                case.requests[ordinal].output_requirement == "group_sum_complete" for ordinal in request_ordinals
            ),
        )
        rank_count = instance.atn_tp_size * instance.atn_dp_size
        for rank in range(rank_count):
            dp_rank = rank // instance.atn_tp_size
            invocations = []
            for ordinal in request_ordinals:
                request = case.requests[ordinal]
                output_path = exchange_directory / f"output-request-{ordinal}-rank-{rank}.safetensors"
                output_paths[ordinal, rank] = output_path
                invocations.append(
                    FfnInvocationSpec(
                        case_id=f"{case.id}-request-{ordinal}",
                        input_path=input_paths[ordinal],
                        output_path=output_path,
                        layer_ordinal=instance.layer_ordinal,
                        forward_mode=(
                            ForwardMode.IDLE
                            if request.dp_rank_payload_rows[dp_rank] == 0
                            else ForwardMode[request.forward_mode.upper()]
                        ),
                        output_requirement=OutputRequirement[request.output_requirement.upper()],
                        dp_row_layout=(DpRowLayout.NONE if instance.atn_dp_size == 1 else DpRowLayout.PACKED_BY_RANK),
                        dp_rank_payload_rows=(None if instance.atn_dp_size == 1 else request.dp_rank_payload_rows),
                    )
                )
            specs.append(
                FfnInstanceSpec(
                    environment=environment.copy(),
                    instance_id=model_spec.model_id,
                    rank=rank,
                    ffn_tp_size=instance.ffn_tp_size,
                    ffn_profile=profile,
                    transport=InstanceRankTransportProfile(
                        hidden_size=model_spec.hidden_size,
                        payload_row_capacity=max(row_capacities),
                        atn_tp_rank=rank % instance.atn_tp_size,
                        atn_tp_size=instance.atn_tp_size,
                        atn_dp_rank=dp_rank,
                        atn_dp_size=instance.atn_dp_size,
                    ),
                    invocations=tuple(invocations),
                )
            )
            spec_instance_indices.append(instance_index)
    return tuple(specs), tuple(spec_instance_indices), output_paths


def resolve_instance_indices(
    case: E2eFfnTopologyCase,
    ready: tuple[FfnInstanceReady, ...],
    spec_instance_indices: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve daemon-assigned Instance indexes from aligned rank readiness."""

    resolved = []
    for instance_index in range(len(case.instances)):
        values = {
            entry.instance_index
            for entry, owner in zip(ready, spec_instance_indices, strict=True)
            if owner == instance_index
        }
        if len(values) != 1:
            raise AssertionError(f"topology Instance {instance_index} observed inconsistent indexes {values}")
        resolved.append(values.pop())
    if len(resolved) != len(set(resolved)):
        raise AssertionError("topology Instances received duplicate daemon indexes")
    return tuple(resolved)


def resolve_execution_groups(
    case: E2eFfnTopologyCase,
    ready: tuple[FfnInstanceReady, ...],
    spec_instance_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Resolve Layer Execution Groups from the admitted production Plan."""

    groups = []
    for instance_index, instance in enumerate(case.instances):
        values = {
            entry.layer_execution_groups[instance.layer_ordinal]
            for entry, owner in zip(ready, spec_instance_indices, strict=True)
            if owner == instance_index
        }
        if len(values) != 1:
            raise AssertionError(f"topology Instance {instance_index} observed inconsistent execution groups {values}")
        group = values.pop()
        if len(group) != instance.ffn_tp_size or len(group) != len(set(group)):
            raise AssertionError(
                f"topology Instance {instance_index} received an invalid Layer Execution Group {group}"
            )
        if any(index < 0 or index >= case.ffnagent_count for index in group):
            raise AssertionError(f"topology Instance {instance_index} received an out-of-range FfnAgent index")
        groups.append(group)
    return tuple(groups)


def validate_graph_snapshots(
    outdir: Path,
    *,
    generation: FabricGenerationId,
    case: E2eFfnTopologyCase,
    ready: tuple[FfnInstanceReady, ...],
) -> None:
    """Require actual Primary Graph and Lane Graph evidence from every FfnAgent."""

    paths = sorted(outdir.glob(f"xpool.graph-observer.{generation.format()}.*.json"))
    if len(paths) != case.ffnagent_count:
        raise AssertionError("topology Graph Observer omitted an FfnAgent snapshot")
    assigned_ffnagent_indices = {
        ffnagent_index
        for entry in ready
        for execution_group in entry.layer_execution_groups
        for ffnagent_index in execution_group
    }
    lane_counts = set()
    for path in paths:
        snapshot = read_json_object(path)
        if snapshot["generation"] != {"high": generation.high, "low": generation.low}:
            raise AssertionError("topology Graph Observer generation mismatch")
        lane_graphs = object_list(snapshot, "lane_graphs")
        lane_counts.add(len(lane_graphs))
        primary_graphs = object_list(snapshot, "primary_graphs")
        ffnagent_index = cast(int, snapshot["pe"]) - case.atnagent_count
        if ffnagent_index in assigned_ffnagent_indices and not primary_graphs:
            raise AssertionError("topology Graph Observer omitted an assigned FfnAgent Primary Graph")
        if any(cast(int, graph["binding_site_count"]) <= 0 for graph in primary_graphs):
            raise AssertionError("topology Graph Observer omitted Primary Graph binding-site evidence")
    if lane_counts != {case.executor_lane_count}:
        raise AssertionError(f"topology Graph Observer lane counts mismatch: {lane_counts}")


def validate_trace(
    outdir: Path,
    *,
    case: E2eFfnTopologyCase,
    actual_instance_indices: tuple[int, ...],
    execution_groups: tuple[tuple[int, ...], ...],
) -> None:
    """Require exact request membership, delivery, rows, and lease evidence."""

    snapshots = read_fabric_observer_snapshots(outdir)
    if len(snapshots) != case.atnagent_count + case.ffnagent_count:
        raise AssertionError("topology Fabric Observer omitted a PE snapshot")
    records = tuple(
        (cast(int, snapshot["pe"]), record) for snapshot in snapshots for record in object_list(snapshot, "records")
    )
    if any(snapshot["dropped"] != 0 for snapshot in snapshots):
        raise AssertionError("topology Fabric Observer dropped records")
    expected_identities: set[tuple[object, ...]] = set()
    sequence_by_instance: dict[int, int] = defaultdict(int)
    for request in case.requests:
        manifest_index = request.instance_index
        sequence_by_instance[manifest_index] += 1
        invocation_sequence = sequence_by_instance[manifest_index]
        actual_instance_index = actual_instance_indices[manifest_index]
        instance = case.instances[manifest_index]
        output_count = instance.atn_tp_size * instance.atn_dp_size
        payload_rows = sum(request.dp_rank_payload_rows)
        delivery = (
            "replicated_complete"
            if request.output_requirement == "per_rank_complete"
            else "direct_partial"
            if instance.ffn_tp_size <= output_count
            else "single_complete"
        )
        matching = [
            (pe, record)
            for pe, record in records
            if record["instance_index"] == actual_instance_index
            and record["invocation_sequence"] == invocation_sequence
            and record["layer_ordinal"] == instance.layer_ordinal
        ]
        atn_records = [(pe, record) for pe, record in matching if record["kind"] == "atnagent"]
        coordinator_records = [(pe, record) for pe, record in matching if record["kind"] == "coordinator"]
        ffn_records = [(pe, record) for pe, record in matching if record["kind"] == "ffnagent"]
        expected_atn_pes = set(range(output_count))
        expected_ffn_pes = {case.atnagent_count + index for index in execution_groups[manifest_index]}
        if (
            {pe for pe, _ in atn_records} != expected_atn_pes
            or {pe for pe, _ in coordinator_records} != {case.atnagent_count}
            or {pe for pe, _ in ffn_records} != expected_ffn_pes
        ):
            raise AssertionError("topology Fabric trace membership disagrees with the installed groups")
        for pe, record in atn_records:
            facts = object_dict(record, "facts")
            dp_rank = pe // instance.atn_tp_size
            expected_mode = "idle" if request.dp_rank_payload_rows[dp_rank] == 0 else request.forward_mode
            if (
                facts["forward_mode"] != expected_mode
                or facts["dp_rank_payload_rows"] != request.dp_rank_payload_rows[dp_rank]
            ):
                raise AssertionError("topology AtnAgent trace mode or local rows mismatch")
        for _, record in ffn_records:
            if object_dict(record, "facts")["delivery"] != delivery:
                raise AssertionError("topology FfnAgent trace delivery mismatch")
        common = matching[0][1]
        common_facts = object_dict(common, "facts")
        if (
            cast(int, common_facts["executor_lane_index"]) < 0
            or cast(int, common_facts["executor_lease_sequence"]) <= 0
        ):
            raise AssertionError("topology trace omitted its Executor lease")
        expected_layout = "none" if instance.atn_dp_size == 1 else "packed_by_rank"
        for pe, record in matching:
            facts = object_dict(record, "facts")
            observed_layout = record["dp_row_layout"]
            required_layout = expected_layout if record["kind"] == "atnagent" else None
            if (
                record["payload_rows"] != payload_rows
                or record["output_requirement"] != request.output_requirement
                or observed_layout != required_layout
                or facts["executor_lane_index"] != common_facts["executor_lane_index"]
                or facts["executor_lease_sequence"] != common_facts["executor_lease_sequence"]
            ):
                raise AssertionError("topology trace facts disagree within one invocation")
            expected_identities.add((record["kind"], pe, actual_instance_index, invocation_sequence))
    actual_identities = {
        (record["kind"], pe, record["instance_index"], record["invocation_sequence"]) for pe, record in records
    }
    if actual_identities != expected_identities:
        raise AssertionError("topology Fabric Observer returned missing or unexpected records")


def compare_outputs(
    case: E2eFfnTopologyCase,
    references: dict[int, torch.Tensor],
    output_paths: dict[tuple[int, int], Path],
) -> tuple[FfnTopologyOutputEvidence, ...]:
    """Compare complete per-rank or canonical rank-summed production outputs."""

    evidence = []
    for request_ordinal, request in enumerate(case.requests):
        instance = case.instances[request.instance_index]
        output_count = instance.atn_tp_size * instance.atn_dp_size
        actual = tuple(read_output(output_paths[request_ordinal, rank]) for rank in range(output_count))
        if request.output_requirement == "per_rank_complete":
            for rank, values in enumerate(actual):
                evidence.append(output_evidence(request_ordinal, (rank,), references[request_ordinal], values))
        else:
            combined = torch.stack(tuple(values.float() for values in actual)).sum(dim=0).to(torch.bfloat16)
            evidence.append(
                output_evidence(request_ordinal, tuple(range(output_count)), references[request_ordinal], combined)
            )
    return tuple(evidence)


def read_output(path: Path) -> torch.Tensor:
    """Read one exclusive production rank output."""

    tensors = safetensors.torch.load_file(path, device="cpu")
    if set(tensors) != {"hidden_states"}:
        raise AssertionError(f"topology output has invalid tensor keys: {sorted(tensors)}")
    return tensors["hidden_states"]


def output_evidence(
    request_ordinal: int,
    output_ranks: tuple[int, ...],
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> FfnTopologyOutputEvidence:
    """Construct one stored complete-output comparison."""

    return FfnTopologyOutputEvidence(
        request_ordinal=request_ordinal,
        output_ranks=output_ranks,
        comparison=compare_ffn_outputs(
            reference,
            actual,
            atol=FFN_OUTPUT_ATOL,
            rtol=FFN_OUTPUT_RTOL,
            normalized_rmse_limit=FFN_OUTPUT_NORMALIZED_RMSE_LIMIT,
            maximum_scaled_error_limit=FFN_OUTPUT_MAXIMUM_SCALED_ERROR_LIMIT,
        ),
    )


def platform_evidence(
    observation: FfnLiveTopologyObservation,
) -> tuple[
    tuple[FfnTopologyProcessEvidence, ...],
    tuple[FfnTopologyMpsServerEvidence, ...],
]:
    """Join supervised process, visible GPU, and live MPS observations."""

    pool = GpuPool.from_environment()
    try:
        visible_uuids = pool.uuids
    finally:
        pool.close()
    process_name_by_id = {process.process_id: process.name for process in observation.processes}
    process_devices = {process.name: process.cuda_device for process in observation.processes}
    process_uuids = {name: visible_uuids[device] for name, device in process_devices.items()}
    processes = tuple(
        FfnTopologyProcessEvidence(
            name=process.name,
            process_id=process.process_id,
            cuda_device=process.cuda_device,
            gpu_uuid=process_uuids[process.name],
        )
        for process in observation.processes
    )
    servers = tuple(
        FfnTopologyMpsServerEvidence(
            process_id=server.process_id,
            client_process_names=tuple(
                sorted(process_name_by_id[process_id] for process_id in server.client_process_ids)
            ),
            active_thread_percentage=server.active_thread_percentage,
        )
        for server in observation.mps_servers
    )
    return processes, servers
