"""Typed subprocess harness for native multi-PE Fabric tests."""

from __future__ import annotations

import multiprocessing
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Barrier
from pathlib import Path

import torch

import xpool.native
from tests.harness.process import SpawnedProcess
from xpool.abi import FfnResultCode, TensorDType, XPoolForwardMode
from xpool.cext import ensure_native_loaded
from xpool.config import DebugConfig
from xpool.fabric import FabricUid
from xpool.runtime import RuntimeRole

FABRIC_LOOPBACK_TIMEOUT_SECONDS = 120.0


class FabricParticipantCommand(StrEnum):
    """Lifecycle commands accepted by one Fabric participant."""

    QUIESCE = "quiesce"
    DRAIN = "drain"


class FabricBootstrapCommand(StrEnum):
    """Lifecycle commands accepted by the daemon-role UID bootstrap child."""

    STOP = "stop"


class FabricInstanceCommand(StrEnum):
    """Synchronization commands accepted by one Fabric Instance."""

    RUN = "run"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class FabricUidCreated:
    """Daemon child response carrying one opaque NVSHMEM UID."""

    value: str


@dataclass(frozen=True, slots=True)
class FabricBootstrapStopped:
    """Acknowledgement that the UID bootstrap child may be reaped."""


@dataclass(frozen=True, slots=True)
class FabricParticipantSpec:
    """Complete startup configuration for one Fabric PE."""

    role: RuntimeRole
    device: int
    uid: str
    pe: int
    dtype: TensorDType
    atnagent_count: int
    ffnagent_count: int
    executor_count: int
    forward_modes: tuple[XPoolForwardMode, ...]
    loopback_enabled: bool
    expect_activation_rejection: bool


@dataclass(frozen=True, slots=True)
class FabricParticipantReady:
    """FfnAgent acknowledgement that its coordinator is active."""


@dataclass(frozen=True, slots=True)
class FabricParticipantActivationRejected:
    """Recoverable FfnAgent activation rejection observed before launch."""

    pe: int
    message: str


@dataclass(frozen=True, slots=True)
class FabricArenasPublished:
    """AtnAgent acknowledgement carrying its Instance arena handles."""

    arenas: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FabricParticipantQuiesced:
    """Participant acknowledgement that ingress is closed."""


@dataclass(frozen=True, slots=True)
class FabricAtnAgentTrace:
    """One completed AtnAgent-side Fabric invocation timeline."""

    model_index: int
    invocation_sequence: int
    executor_index: int | None
    forward_mode: XPoolForwardMode | None
    prepared_ns: int
    acknowledged_ns: int


@dataclass(frozen=True, slots=True)
class FabricCoordinatorTrace:
    """One Coordinator scheduling and Executor-lease timeline."""

    model_index: int
    invocation_sequence: int
    executor_index: int | None
    ready_ticket: int | None
    enqueued_ns: int
    scheduled_ns: int
    released_ns: int


@dataclass(frozen=True, slots=True)
class FabricExecutionTrace:
    """One FfnAgent execution timeline for a distributed Executor."""

    model_index: int
    invocation_sequence: int
    executor_index: int | None
    observed_ns: int
    started_ns: int
    completed_ns: int
    published_ns: int


type FabricTrace = FabricAtnAgentTrace | FabricCoordinatorTrace | FabricExecutionTrace


@dataclass(frozen=True, slots=True)
class FabricFailureEvidence:
    """One PE's host-observed canonical Fabric failure state."""

    claim: int
    publication: int
    result_code: FfnResultCode
    origin_pe: int
    model_index: int
    invocation_sequence: int
    layer_ordinal: int


@dataclass(frozen=True, slots=True)
class FabricParticipantReport:
    """Lifecycle and local trace evidence returned by one Fabric participant."""

    role: RuntimeRole
    pe: int
    sequence: int
    dropped: int
    records: tuple[FabricTrace, ...]
    failure: FabricFailureEvidence | None


@dataclass(frozen=True, slots=True)
class FabricParticipantDrained:
    """Participant acknowledgement carrying post-drain trace evidence."""

    report: FabricParticipantReport


@dataclass(frozen=True, slots=True)
class FabricInstanceSpec:
    """Complete startup configuration for one Fabric Instance client."""

    device: int
    atnagent_pe: int
    arena: str
    forward_mode: XPoolForwardMode
    dtype: TensorDType
    instance_index: int
    payload_offset: int
    repetition_count: int
    stop_on_command: bool
    start_barrier: Barrier
    loopback_enabled: bool


@dataclass(frozen=True, slots=True)
class FabricInstanceReady:
    """Instance acknowledgement that it reached the shared start gate."""


@dataclass(frozen=True, slots=True)
class FabricInstanceRunning:
    """Instance acknowledgement that its first request was submitted."""


@dataclass(frozen=True, slots=True)
class FabricInstanceResult:
    """Numerical output and completion facts returned by one Instance."""

    atnagent_pe: int
    instance_index: int
    forward_mode: XPoolForwardMode
    repetitions_completed: int
    result_code: FfnResultCode
    actual: tuple[float, ...]
    expected: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FabricTopologyReport:
    """Aggregate evidence from one complete native Fabric topology run."""

    atnagent_count: int
    ffnagent_count: int
    executor_count: int
    forward_modes: tuple[XPoolForwardMode, ...]
    instances: tuple[FabricInstanceResult, ...]
    participants: tuple[FabricParticipantReport, ...]
    activation_rejections: tuple[FabricParticipantActivationRejected, ...]


def create_fabric_uid_child(connection: Connection, spec: None) -> None:
    """Initialize a daemon-role child and publish one native Fabric UID."""

    ensure_native_loaded()
    xpool.native.initialize(int(RuntimeRole.DAEMON))
    connection.send(FabricUidCreated(xpool.native.fabric.create_uid()))


def run_fabric_bootstrap(connection: Connection, spec: None) -> None:
    """Create a UID and keep its socket bootstrap owner alive."""

    ensure_native_loaded()
    xpool.native.initialize(int(RuntimeRole.DAEMON))
    connection.send(FabricUidCreated(xpool.native.fabric.create_uid()))
    command = connection.recv()
    if command is not FabricBootstrapCommand.STOP:
        raise RuntimeError(f"Fabric bootstrap expected STOP, received {command!r}")
    connection.send(FabricBootstrapStopped())


def create_fabric_uid() -> FabricUid:
    """Create one native NVSHMEM UID in a fresh daemon-role child."""

    with tempfile.TemporaryDirectory(prefix="xpool-fabric-uid-") as directory:
        process = SpawnedProcess.start(
            "fabric-uid",
            create_fabric_uid_child,
            None,
            log_path=Path(directory) / "uid.log",
        )
        try:
            created = process.receive(FabricUidCreated, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
            process.wait(timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
            return FabricUid(value=created.value)
        finally:
            if process.process.is_alive():
                SpawnedProcess.terminate_all((process,))
            process.close()


@contextmanager
def fabric_bootstrap() -> Generator[FabricUid, None, None]:
    """Keep the daemon-role owner of one UID alive for a Fabric generation."""

    with tempfile.TemporaryDirectory(prefix="xpool-fabric-bootstrap-") as directory:
        process = SpawnedProcess.start(
            "fabric-bootstrap",
            run_fabric_bootstrap,
            None,
            log_path=Path(directory) / "bootstrap.log",
        )
        try:
            created = process.receive(FabricUidCreated, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
            yield FabricUid(value=created.value)
            process.send(FabricBootstrapCommand.STOP)
            process.receive(FabricBootstrapStopped, timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
            process.wait(timeout_seconds=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
        finally:
            if process.process.is_alive():
                SpawnedProcess.terminate_all((process,))
            process.close()


def fabric_debug_options(*, loopback_enabled: bool = True) -> str:
    """Return native debug options for traced FfnAgent loopback."""

    config = DebugConfig.model_validate(
        {
            "loopback": {
                "enable": loopback_enabled,
                "site": "ffnagent" if loopback_enabled else None,
            },
            "fabric_observer": {
                "enable": True,
                "outdir": Path.cwd(),
            },
        }
    )
    return config.model_dump_json(
        include={
            "loopback": {"enable", "site"},
            "transport_observer": {"enable", "trace_capacity"},
            "fabric_observer": {"enable", "trace_capacity"},
        }
    )


def torch_dtype(dtype: TensorDType) -> torch.dtype:
    """Return the Torch dtype corresponding to one stable xpool dtype."""

    match dtype:
        case TensorDType.BF16:
            return torch.bfloat16
        case TensorDType.FP16:
            return torch.float16
        case TensorDType.FP32:
            return torch.float32


def fabric_trace_report(
    role: RuntimeRole,
    snapshot: xpool.native.FabricTraceSnapshot,
    failure: xpool.native.FabricFailure | None,
) -> FabricParticipantReport:
    """Copy one bound native snapshot into picklable typed trace facts."""

    validate_fabric_trace_event_families(snapshot)
    records: list[FabricTrace] = []
    for record in snapshot.records:
        key = record.key
        if record.kind == xpool.native.FabricTraceKind.ATNAGENT:
            event = xpool.native.AtnAgentTraceEvent
            records.append(
                FabricAtnAgentTrace(
                    model_index=key.model_index,
                    invocation_sequence=key.invocation_sequence,
                    executor_index=record.executor_index,
                    forward_mode=None if record.forward_mode is None else XPoolForwardMode(record.forward_mode),
                    prepared_ns=record.timestamp(event.SUBMISSION_PREPARED),
                    acknowledged_ns=record.timestamp(event.ACKNOWLEDGEMENT_PUBLISHED),
                )
            )
        elif record.kind == xpool.native.FabricTraceKind.COORDINATOR:
            event = xpool.native.CoordinatorTraceEvent
            records.append(
                FabricCoordinatorTrace(
                    model_index=key.model_index,
                    invocation_sequence=key.invocation_sequence,
                    executor_index=record.executor_index,
                    ready_ticket=record.ready_ticket,
                    enqueued_ns=record.timestamp(event.ENQUEUED),
                    scheduled_ns=record.timestamp(event.SCHEDULED),
                    released_ns=record.timestamp(event.SCHEDULER_RELEASED),
                )
            )
        elif record.kind == xpool.native.FabricTraceKind.EXECUTION:
            event = xpool.native.ExecutionTraceEvent
            records.append(
                FabricExecutionTrace(
                    model_index=key.model_index,
                    invocation_sequence=key.invocation_sequence,
                    executor_index=record.executor_index,
                    observed_ns=record.timestamp(event.INVOCATION_OBSERVED),
                    started_ns=record.timestamp(event.EXECUTION_STARTED),
                    completed_ns=record.timestamp(event.EXECUTION_COMPLETED),
                    published_ns=record.timestamp(event.COMPLETION_PUBLISHED),
                )
            )
        else:
            raise RuntimeError(f"native Fabric returned unknown trace kind {record.kind!r}")
    return FabricParticipantReport(
        role=role,
        pe=snapshot.pe,
        sequence=snapshot.sequence,
        dropped=snapshot.dropped,
        records=tuple(records),
        failure=(
            None
            if failure is None
            else FabricFailureEvidence(
                claim=failure.claim,
                publication=failure.publication,
                result_code=FfnResultCode(failure.payload.result_code),
                origin_pe=failure.payload.origin_pe,
                model_index=failure.payload.key.model_index,
                invocation_sequence=failure.payload.key.invocation_sequence,
                layer_ordinal=failure.payload.layer_ordinal,
            )
        ),
    )


def validate_fabric_trace_event_families(snapshot: xpool.native.FabricTraceSnapshot) -> None:
    """Prove mismatched bound event families fail recoverably for real records."""

    families = (
        (xpool.native.FabricTraceKind.ATNAGENT, xpool.native.AtnAgentTraceEvent.SUBMISSION_PREPARED),
        (xpool.native.FabricTraceKind.COORDINATOR, xpool.native.CoordinatorTraceEvent.ENQUEUED),
        (xpool.native.FabricTraceKind.EXECUTION, xpool.native.ExecutionTraceEvent.INVOCATION_OBSERVED),
    )
    for record in snapshot.records:
        for kind, event in families:
            if record.kind == kind:
                continue
            for operation in (record.recorded, record.timestamp):
                try:
                    operation(event)
                except ValueError:
                    continue
                raise AssertionError(f"Fabric trace accepted {event!r} for {record.kind!r}")


def run_fabric_participant(connection: Connection, spec: FabricParticipantSpec) -> None:
    """Join one Fabric PE and serve typed lifecycle commands."""

    ensure_native_loaded()
    torch.cuda.set_device(spec.device)
    xpool.native.initialize(
        int(spec.role),
        spec.device,
        fabric_debug_options(loopback_enabled=spec.loopback_enabled),
    )
    xpool.native.fabric.join(
        xpool.native.FabricJoinMetadata(
            uid=spec.uid,
            pe=spec.pe,
            atnagent_count=spec.atnagent_count,
            ffnagent_count=spec.ffnagent_count,
            executor_count=spec.executor_count,
            scheduler_policy=xpool.native.FfnSchedulerPolicy.fifo(),
            models=[
                xpool.native.FabricModelMetadata(
                    max_decode_rows=4,
                    max_prefill_rows=4,
                    dtype=int(spec.dtype),
                    hidden_size=8,
                    atn_tp_size=spec.atnagent_count,
                    atn_dp_size=1,
                    layers=[xpool.native.FabricLayerMetadata(layer_id=0, kind=1)],
                )
                for _ in spec.forward_modes
            ],
        )
    )
    arenas: list[str] = []
    if spec.role is RuntimeRole.FFNAGENT:
        try:
            xpool.native.fabric.activate()
        except RuntimeError as error:
            if not spec.expect_activation_rejection:
                raise
            connection.send(FabricParticipantActivationRejected(spec.pe, str(error)))
        else:
            if spec.expect_activation_rejection:
                raise RuntimeError("FfnAgent activation unexpectedly accepted an over-capacity Resident")
            connection.send(FabricParticipantReady())
    else:
        for instance_index in range(len(spec.forward_modes)):
            arena = xpool.native.transport.create_arena(
                instance_index,
                0,
                4,
                8,
                int(spec.dtype),
                spec.pe,
                spec.atnagent_count,
                0,
                1,
            )
            arenas.append(str(arena))
        xpool.native.transport.activate()
        connection.send(FabricArenasPublished(tuple(arenas)))

    command = connection.recv()
    if command is not FabricParticipantCommand.QUIESCE:
        raise RuntimeError(f"Fabric participant expected QUIESCE, received {command!r}")
    if not spec.expect_activation_rejection:
        xpool.native.fabric.check_health()
    if arenas:
        xpool.native.transport.check_health()
        xpool.native.transport.drain_async()
        deadline = time.monotonic() + 60.0
        while xpool.native.transport.drain_pending():
            if time.monotonic() >= deadline:
                raise RuntimeError("Transport Resident drain timed out")
            time.sleep(0.01)
        xpool.native.transport.drain_async()
        if xpool.native.transport.drain_pending():
            raise RuntimeError("completed Transport Resident drain became pending again")
    connection.send(FabricParticipantQuiesced())

    command = connection.recv()
    if command is not FabricParticipantCommand.DRAIN:
        raise RuntimeError(f"Fabric participant expected DRAIN, received {command!r}")
    xpool.native.fabric.drain_async()
    deadline = time.monotonic() + 60.0
    while xpool.native.fabric.drain_pending():
        if time.monotonic() >= deadline:
            raise RuntimeError("Fabric drain timed out")
        time.sleep(0.01)
    xpool.native.fabric.drain_async()
    if xpool.native.fabric.drain_pending():
        raise RuntimeError("completed Fabric drain became pending again")
    snapshot = xpool.native.fabric.read_trace()
    if snapshot is None:
        raise RuntimeError("Fabric trace observation was enabled but returned no snapshot")
    report = fabric_trace_report(spec.role, snapshot, xpool.native.fabric.failure())
    xpool.native.fabric.shutdown()
    for operation in (
        xpool.native.fabric.drain_async,
        xpool.native.fabric.drain_pending,
    ):
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Fabric drain operation was accepted after finalization")
    if arenas:
        xpool.native.transport.destroy_arenas(arenas)
    connection.send(FabricParticipantDrained(report))


def run_fabric_instance(connection: Connection, spec: FabricInstanceSpec) -> None:
    """Attach one Instance and return repeated FFN loopback outputs."""

    ensure_native_loaded()
    torch.cuda.set_device(spec.device)
    xpool.native.initialize(
        int(RuntimeRole.INSTANCE),
        spec.device,
        fabric_debug_options(loopback_enabled=spec.loopback_enabled),
    )
    dtype = torch_dtype(spec.dtype)
    hidden_states = (torch.arange(32, device=spec.device, dtype=dtype) + spec.payload_offset).reshape(4, 8)
    xpool.native.transport.attach_arena(spec.instance_index, 0, spec.arena)
    try:
        x_values = hidden_states[..., 0::2]
        y_values = hidden_states[..., 1::2]
        expected = torch.empty_like(hidden_states)
        expected[..., 0::2] = (x_values - y_values) / (2.0**0.5)
        expected[..., 1::2] = (x_values + y_values) / (2.0**0.5)
        connection.send(FabricInstanceReady())
        command = connection.recv()
        if command is not FabricInstanceCommand.RUN:
            raise RuntimeError(f"Fabric Instance expected RUN, received {command!r}")
        repetitions_completed = 0
        result_code = FfnResultCode.OK
        output = torch.empty_like(hidden_states)
        for repetition in range(spec.repetition_count):
            spec.start_barrier.wait(timeout=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
            output = torch.ops.xpool.ffn_shim(
                hidden_states,
                None,
                0,
                int(spec.forward_mode),
                1,
                0,
            )
            repetitions_completed += 1
            if repetition == 0:
                connection.send(FabricInstanceRunning())
            torch.cuda.synchronize(spec.device)
            result_code = FfnResultCode(xpool.native.transport.read_generation_failure())
            if result_code is not FfnResultCode.OK:
                break
            if spec.stop_on_command:
                command = connection.recv()
                if command is not FabricInstanceCommand.STOP:
                    raise RuntimeError(f"Fabric Instance expected STOP, received {command!r}")
                break
        connection.send(
            FabricInstanceResult(
                atnagent_pe=spec.atnagent_pe,
                instance_index=spec.instance_index,
                forward_mode=spec.forward_mode,
                repetitions_completed=repetitions_completed,
                result_code=result_code,
                actual=tuple(output.float().cpu().flatten().tolist()),
                expected=tuple(expected.float().cpu().flatten().tolist()),
            )
        )
    finally:
        xpool.native.transport.detach_arena()


def run_fabric_topology(
    uid: FabricUid,
    *,
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
    with tempfile.TemporaryDirectory(prefix="xpool-fabric-topology-") as directory:
        log_directory = Path(directory)
        participants = [
            SpawnedProcess.start(
                f"{spec.role.name.lower()}-{spec.pe}",
                run_fabric_participant,
                spec,
                log_path=log_directory / f"participant-{spec.pe}.log",
            )
            for spec in participant_specs
        ]
        instances: list[SpawnedProcess] = []
        activation_rejections: list[FabricParticipantActivationRejected] = []
        try:
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
                        SpawnedProcess.start(
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
            SpawnedProcess.terminate_all(tuple((*participants, *instances)))
            for process in (*participants, *instances):
                process.close()
