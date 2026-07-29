"""Typed subprocess harness for native multi-PE Fabric tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.synchronize import Barrier

from xpool.abi import FfnResultCode, TensorDType, XPoolForwardMode
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
