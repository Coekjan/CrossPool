"""Typed subprocess harness for native multi-PE Fabric tests."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from tests.harness.native.fabric.instance import run_fabric_instance
from tests.harness.native.fabric.participant import run_fabric_participant
from tests.harness.native.fabric.protocol import (
    FABRIC_LOOPBACK_TIMEOUT_SECONDS,
    FabricArenasPublished,
    FabricInstanceCommand,
    FabricInstanceReady,
    FabricInstanceResult,
    FabricInstanceRunning,
    FabricInstanceSpec,
    FabricParticipantActivationRejected,
    FabricParticipantCommand,
    FabricParticipantDrained,
    FabricParticipantQuiesced,
    FabricParticipantReady,
    FabricParticipantSpec,
    FabricTopologyReport,
)
from tests.harness.runner.child import PythonChildProcess
from xpool.abi import TensorDType, XPoolForwardMode
from xpool.fabric import FabricUid
from xpool.runtime import RuntimeRole


def run_fabric_topology(
    uid: FabricUid,
    *,
    workdir: Path,
    atnagent_count: int,
    ffnagent_count: int,
    executor_count: int,
    forward_modes: tuple[XPoolForwardMode, ...],
    dtype: TensorDType,
    quiesce_after_first_request: bool = False,
    repetition_count: int = 3,
    loopback_enabled: bool = True,
    expect_activation_rejection: bool = False,
) -> FabricTopologyReport:
    """Run concurrent requests and return complete native Fabric evidence."""

    if not forward_modes or repetition_count <= 0:
        raise ValueError("Fabric loopback modes and repetition count must be non-empty and positive")
    participant_specs = tuple(
        FabricParticipantSpec(
            role=RuntimeRole.ATNAGENT,
            device=index,
            uid=uid.value,
            pe=index,
            dtype=dtype,
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            executor_count=executor_count,
            forward_modes=forward_modes,
            loopback_enabled=loopback_enabled,
            expect_activation_rejection=False,
        )
        for index in range(atnagent_count)
    ) + tuple(
        FabricParticipantSpec(
            role=RuntimeRole.FFNAGENT,
            device=atnagent_count + index,
            uid=uid.value,
            pe=atnagent_count + index,
            dtype=dtype,
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            executor_count=executor_count,
            forward_modes=forward_modes,
            loopback_enabled=loopback_enabled,
            expect_activation_rejection=expect_activation_rejection,
        )
        for index in range(ffnagent_count)
    )
    workdir.mkdir(parents=True, exist_ok=False)
    log_directory = workdir
    participants: list[PythonChildProcess] = []
    instances: list[PythonChildProcess] = []
    activation_rejections: list[FabricParticipantActivationRejected] = []
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
                if expect_activation_rejection:
                    activation_rejections.append(
                        process.receive(
                            FabricParticipantActivationRejected,
                            timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS,
                        )
                    )
                else:
                    process.receive(FabricParticipantReady, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)

        published_arenas: list[tuple[int, int, str]] = []
        for atnagent_pe, process in enumerate(participants[:atnagent_count]):
            published = process.receive(
                FabricArenasPublished,
                timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS,
            )
            if len(published.arenas) != len(forward_modes):
                raise RuntimeError(f"AtnAgent PE {atnagent_pe} published invalid Transport arenas")
            published_arenas.extend(
                (atnagent_pe, instance_index, arena) for instance_index, arena in enumerate(published.arenas)
            )

        if not expect_activation_rejection:
            context = multiprocessing.get_context("spawn")
            start_barrier = context.Barrier(len(published_arenas))
            for atnagent_pe, instance_index, arena in published_arenas:
                spec = FabricInstanceSpec(
                    device=atnagent_pe,
                    atnagent_pe=atnagent_pe,
                    arena=arena,
                    forward_mode=forward_modes[instance_index],
                    dtype=dtype,
                    instance_index=instance_index,
                    payload_offset=instance_index * 100,
                    repetition_count=100 if quiesce_after_first_request else repetition_count,
                    stop_on_command=quiesce_after_first_request,
                    start_barrier=start_barrier,
                    loopback_enabled=loopback_enabled,
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
            instance.receive(FabricInstanceReady, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
        for instance in instances:
            instance.send(FabricInstanceCommand.RUN)
        for instance in instances:
            instance.receive(FabricInstanceRunning, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
        if quiesce_after_first_request:
            for instance in instances:
                instance.send(FabricInstanceCommand.STOP)

        instance_results = tuple(
            instance.receive(FabricInstanceResult, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
            for instance in instances
        )
        for instance in instances:
            instance.wait(timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)

        for participant in participants:
            participant.send(FabricParticipantCommand.QUIESCE)
        for participant in participants:
            participant.receive(FabricParticipantQuiesced, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
        for participant in participants:
            participant.send(FabricParticipantCommand.DRAIN)
        participant_reports = tuple(
            participant.receive(FabricParticipantDrained, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS).report
            for participant in participants
        )
        for participant in participants:
            participant.wait(timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
        return FabricTopologyReport(
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            executor_count=executor_count,
            forward_modes=forward_modes,
            instances=instance_results,
            participants=participant_reports,
            activation_rejections=tuple(activation_rejections),
        )
    finally:
        PythonChildProcess.terminate_all(tuple((*participants, *instances)))
        for process in (*participants, *instances):
            process.close()
