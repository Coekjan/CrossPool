from __future__ import annotations

import multiprocessing
from pathlib import Path

import torch

from tests.harness.native.fabric.instance import run_fabric_instance
from tests.harness.native.fabric.participant import run_fabric_participant
from tests.harness.native.fabric.protocol import (
    FABRIC_TIMEOUT_SECONDS,
    FabricArenasPublished,
    FabricInstanceCommand,
    FabricInstanceReady,
    FabricInstanceResult,
    FabricInstanceRunning,
    FabricInstanceSpec,
    FabricParticipantCommand,
    FabricParticipantDrained,
    FabricParticipantQuiesced,
    FabricParticipantReady,
    FabricParticipantSpec,
    FabricTopologyReport,
)
from tests.harness.runner.child import PythonChildProcess
from xpool.fabric import FabricUid
from xpool.native import RuntimeRole
from xpool.native.ffn import ForwardMode, LayerKind, OutputRequirement


def run_fabric_topology(
    uid: FabricUid,
    *,
    workdir: Path,
    atnagent_count: int,
    ffnagent_count: int,
    executor_lane_count: int,
    forward_modes: tuple[ForwardMode, ...],
    layer_kind: LayerKind,
    quiesce_after_first_request: bool = False,
    repetition_count: int = 3,
    pre_admission_rejection: bool = False,
    execution_tp_size: int | None = None,
    execution_layer_count: int = 1,
    decode_payload_row_capacity: int = 4,
    prefill_payload_row_capacity: int = 4,
    payload_dtype: torch.dtype = torch.bfloat16,
    payload_rows: tuple[int, ...] | None = None,
    layer_ordinals: tuple[int, ...] = (0,),
    output_requirement: OutputRequirement = OutputRequirement.PER_RANK_COMPLETE,
) -> FabricTopologyReport:
    """Run concurrent requests and return complete native Fabric evidence."""

    if not forward_modes or repetition_count <= 0:
        raise ValueError("Fabric modes and repetition count must be non-empty and positive")
    selected_payload_rows = (4,) * len(forward_modes) if payload_rows is None else payload_rows
    if len(selected_payload_rows) != len(forward_modes):
        raise ValueError("Fabric qualification payload rows must match the configured instances")
    for mode, rows in zip(forward_modes, selected_payload_rows, strict=True):
        capacity = (
            prefill_payload_row_capacity
            if mode is ForwardMode.PREFILL
            else decode_payload_row_capacity
            if mode is ForwardMode.DECODE
            else 0
        )
        if not 0 < rows <= capacity:
            raise ValueError("Fabric qualification payload rows must fit the selected forward mode")
    if (
        execution_layer_count <= 0
        or not layer_ordinals
        or any(not 0 <= ordinal < execution_layer_count for ordinal in layer_ordinals)
    ):
        raise ValueError("Fabric qualification layer sequence is outside the execution Plan")
    selected_tp_size = ffnagent_count if execution_tp_size is None else execution_tp_size
    if not 1 <= selected_tp_size <= ffnagent_count:
        raise ValueError("qualification FFN TP size must fit the FfnAgent Fleet")
    participant_specs = tuple(
        FabricParticipantSpec(
            role=RuntimeRole.ATNAGENT,
            device=index,
            uid=uid.value,
            pe=index,
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            execution_tp_size=selected_tp_size,
            execution_layer_count=execution_layer_count,
            decode_payload_row_capacity=decode_payload_row_capacity,
            prefill_payload_row_capacity=prefill_payload_row_capacity,
            payload_dtype=payload_dtype,
            executor_lane_count=executor_lane_count,
            forward_modes=forward_modes,
            layer_kind=layer_kind,
        )
        for index in range(atnagent_count)
    ) + tuple(
        FabricParticipantSpec(
            role=RuntimeRole.FFNAGENT,
            device=atnagent_count + index,
            uid=uid.value,
            pe=atnagent_count + index,
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            execution_tp_size=selected_tp_size,
            execution_layer_count=execution_layer_count,
            decode_payload_row_capacity=decode_payload_row_capacity,
            prefill_payload_row_capacity=prefill_payload_row_capacity,
            payload_dtype=payload_dtype,
            executor_lane_count=executor_lane_count,
            forward_modes=forward_modes,
            layer_kind=layer_kind,
        )
        for index in range(ffnagent_count)
    )
    workdir.mkdir(parents=True, exist_ok=False)
    log_directory = workdir
    participants: list[PythonChildProcess] = []
    instances: list[PythonChildProcess] = []
    try:
        for spec in participant_specs:
            participants.append(
                PythonChildProcess.start(
                    f"{spec.role.name.lower()}-{spec.pe}",
                    run_fabric_participant,
                    spec,
                    log_path=log_directory / f"participant-{spec.pe}.log",
                )
            )
        for spec, process in zip(participant_specs, participants, strict=True):
            if spec.role is RuntimeRole.FFNAGENT:
                process.receive(FabricParticipantReady, timeout_seconds=FABRIC_TIMEOUT_SECONDS)

        published_arenas: list[tuple[int, int, str]] = []
        for atnagent_pe, process in enumerate(participants[:atnagent_count]):
            published = process.receive(
                FabricArenasPublished,
                timeout_seconds=FABRIC_TIMEOUT_SECONDS,
            )
            if len(published.arenas) != len(forward_modes):
                raise RuntimeError(f"AtnAgent PE {atnagent_pe} published invalid Transport arenas")
            published_arenas.extend(
                (atnagent_pe, instance_index, arena) for instance_index, arena in enumerate(published.arenas)
            )

        context = multiprocessing.get_context("spawn")
        start_barrier = context.Barrier(len(published_arenas))
        for atnagent_pe, instance_index, arena in published_arenas:
            spec = FabricInstanceSpec(
                device=atnagent_pe,
                atnagent_pe=atnagent_pe,
                arena=arena,
                forward_mode=forward_modes[instance_index],
                ffnagent_count=ffnagent_count,
                execution_tp_size=selected_tp_size,
                atnagent_count=atnagent_count,
                output_requirement=output_requirement,
                instance_index=instance_index,
                payload_rows=selected_payload_rows[instance_index],
                payload_dtype=payload_dtype,
                layer_ordinals=layer_ordinals,
                payload_offset=instance_index * 100,
                repetition_count=100 if quiesce_after_first_request else repetition_count,
                stop_on_command=quiesce_after_first_request,
                start_barrier=start_barrier,
                layer_kind=layer_kind,
                pre_admission_rejection=pre_admission_rejection,
            )
            instances.append(
                PythonChildProcess.start(
                    f"instance-{atnagent_pe}-{instance_index}",
                    run_fabric_instance,
                    spec,
                    log_path=log_directory / f"instance-{atnagent_pe}-{instance_index}.log",
                )
            )

        for instance in instances:
            instance.receive(FabricInstanceReady, timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        for instance in instances:
            instance.send(FabricInstanceCommand.RUN)
        for instance in instances:
            instance.receive(FabricInstanceRunning, timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        if quiesce_after_first_request:
            for instance in instances:
                instance.send(FabricInstanceCommand.STOP)

        instance_results = tuple(
            instance.receive(FabricInstanceResult, timeout_seconds=FABRIC_TIMEOUT_SECONDS) for instance in instances
        )
        for instance in instances:
            instance.wait(timeout_seconds=FABRIC_TIMEOUT_SECONDS)

        for participant in participants:
            participant.send(FabricParticipantCommand.QUIESCE)
        for participant in participants:
            participant.receive(FabricParticipantQuiesced, timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        for participant in participants:
            participant.send(FabricParticipantCommand.DRAIN)
        participant_reports = tuple(
            participant.receive(FabricParticipantDrained, timeout_seconds=FABRIC_TIMEOUT_SECONDS).report
            for participant in participants
        )
        for participant in participants:
            participant.wait(timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        return FabricTopologyReport(
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            executor_lane_count=executor_lane_count,
            forward_modes=forward_modes,
            instances=instance_results,
            participants=participant_reports,
        )
    finally:
        PythonChildProcess.terminate_all(tuple((*participants, *instances)))
        for process in (*participants, *instances):
            process.close()
