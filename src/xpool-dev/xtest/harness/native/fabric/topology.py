from __future__ import annotations

import logging
import multiprocessing
import signal
import time
from contextlib import ExitStack
from pathlib import Path
from types import FrameType

import torch

from xkit.child import PythonChildProcess
from xkit.task import TaskCancelled, get_task_root
from xpool.fabric import FabricUid
from xpool.native import RuntimeRole
from xpool.native.ffn import ForwardMode, LayerKind, OutputRequirement
from xpool.utils.device import visible_uuids
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S, MpsEndpoint, MpsScope
from xpool.utils.procs import ProcUniqId
from xpool.utils.sighandler import defer_signal_exceptions, sighandle
from xtest.harness.native.fabric.instance import run_fabric_instance
from xtest.harness.native.fabric.participant import run_fabric_participant
from xtest.harness.native.fabric.protocol import (
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
    FabricParticipantReport,
    FabricParticipantSpec,
    FabricTopologyReport,
)
from xtest.harness.support.wait import remaining_seconds

logger = logging.getLogger(__name__)


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
    visibility = visible_uuids()
    if atnagent_count + ffnagent_count > len(visibility):
        raise ValueError("Fabric topology exceeds the available device view")
    scope = MpsScope(MpsEndpoint(visibility[:atnagent_count]))
    attention_environment = scope.endpoint.environment()
    attention_environment["CUDA_VISIBLE_DEVICES"] = ",".join(visibility)
    ffn_environment = {
        "CUDA_VISIBLE_DEVICES": ",".join(visibility),
        "CUDA_MPS_PIPE_DIRECTORY": "",
    }
    participant_specs = tuple(
        FabricParticipantSpec(
            role=RuntimeRole.ATNAGENT,
            device=index,
            environment=attention_environment,
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
            environment=ffn_environment,
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
    root = get_task_root()
    task_scope = None
    participants_ready = False
    participant_reports: tuple[FabricParticipantReport, ...] = ()
    cleanup_deadline: float | None = None
    cancelled: int | None = None
    handlers = ExitStack()

    def record_cancellation(signum: int, frame: FrameType | None) -> None:
        nonlocal cleanup_deadline, cancelled
        if cancelled is None:
            cancelled = signum
        if root is not None:
            root.consume_cancellation()
        if cleanup_deadline is None:
            cleanup_deadline = (
                root.cleanup_deadline
                if root is not None and root.cleanup_deadline is not None
                else time.monotonic() + MPS_CLEANUP_TIMEOUT_S
            )

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers.enter_context(sighandle(signum, record_cancellation))
        if root is not None:
            with defer_signal_exceptions():
                task_scope = root.register_scope()
            root.activate()
        scope.start()
        if cancelled is not None:
            raise TaskCancelled(f"Fabric topology cancelled by signal {cancelled}")
        for spec in participant_specs:
            with defer_signal_exceptions():
                participants.append(
                    PythonChildProcess(
                        f"{spec.role.name.lower()}-{spec.pe}",
                        run_fabric_participant,
                        spec,
                        log_path=log_directory / f"participant-{spec.pe}.log",
                    )
                )
            participants[-1].start()
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

        participants_ready = True
        if cancelled is not None:
            raise TaskCancelled(f"Fabric topology cancelled by signal {cancelled}")
        context = multiprocessing.get_context("spawn")
        start_barrier = context.Barrier(len(published_arenas))
        for atnagent_pe, instance_index, arena in published_arenas:
            spec = FabricInstanceSpec(
                device=atnagent_pe,
                environment=attention_environment,
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
            with defer_signal_exceptions():
                instances.append(
                    PythonChildProcess(
                        f"instance-{atnagent_pe}-{instance_index}",
                        run_fabric_instance,
                        spec,
                        log_path=log_directory / f"instance-{atnagent_pe}-{instance_index}.log",
                    )
                )
            instances[-1].start()

        for instance in instances:
            instance.receive(FabricInstanceReady, timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        if cancelled is not None:
            raise TaskCancelled(f"Fabric topology cancelled by signal {cancelled}")
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

        if cancelled is not None:
            raise TaskCancelled(f"Fabric topology cancelled by signal {cancelled}")
    finally:
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
        try:
            if any(process.process.is_alive() for process in participants):
                if not participants_ready:
                    raise RuntimeError("Fabric startup did not publish a complete participant world")
                live_instances = [
                    ProcUniqId(process.process.pid)
                    for process in instances
                    if process.process.pid is not None and process.process.is_alive()
                ]
                # Fixed Instance handles are consumers, not collective PEs.
                # Confirm all their contexts before issuing any host signal.
                for identity in live_instances:
                    scope.terminate_client(identity, deadline=cleanup_deadline)
                for identity in live_instances:
                    identity.send_signal(signal.SIGKILL)
                for process in instances:
                    if process.process.pid is not None:
                        process.process.join(max(0.0, cleanup_deadline - time.monotonic()))
                    if process.process.is_alive():
                        raise TimeoutError("Fabric Instance retirement is unconfirmed")
                for participant in participants:
                    participant.send(FabricParticipantCommand.QUIESCE)
                for participant in participants:
                    participant.receive(
                        FabricParticipantQuiesced,
                        timeout_seconds=remaining_seconds(cleanup_deadline, "Fabric quiesce"),
                    )
                for participant in participants:
                    participant.send(FabricParticipantCommand.DRAIN)
                participant_reports = tuple(
                    participant.receive(
                        FabricParticipantDrained,
                        timeout_seconds=remaining_seconds(cleanup_deadline, "Fabric drain"),
                    ).report
                    for participant in participants
                )
                for participant in participants:
                    participant.wait(timeout_seconds=remaining_seconds(cleanup_deadline, "Fabric participant exit"))
            for process in (*participants, *instances):
                process.close()
            scope.stop(deadline=cleanup_deadline)
            if task_scope is not None:
                task_scope.complete()
        except BaseException as error:
            logger.error("Fabric topology cleanup unconfirmed; retaining owner and MPS: %s", error)
            while True:
                time.sleep(1.0)
        handlers.close()

    return FabricTopologyReport(
        atnagent_count=atnagent_count,
        ffnagent_count=ffnagent_count,
        executor_lane_count=executor_lane_count,
        forward_modes=forward_modes,
        instances=instance_results,
        participants=participant_reports,
    )
