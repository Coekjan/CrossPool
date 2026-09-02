from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.synchronize import Barrier

import torch

import xpool.native
from xpool.native import RuntimeRole
from xpool.native.ffn import ForwardMode, LayerKind, OutputRequirement, ResultCode

FABRIC_TIMEOUT_SECONDS = 120.0


class FabricParticipantCommand(StrEnum):
    """Lifecycle commands accepted by one Fabric participant."""

    QUIESCE = "quiesce"
    DRAIN = "drain"


class FabricBootstrapCommand(StrEnum):
    """Lifecycle commands accepted by the daemon-role UID bootstrap child."""

    STOP = "stop"


class FabricInstanceCommand(StrEnum):
    """Synchronization commands accepted by one Fabric Instance rank."""

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
    atnagent_count: int
    ffnagent_count: int
    execution_tp_size: int
    execution_layer_count: int
    decode_payload_row_capacity: int
    prefill_payload_row_capacity: int
    payload_dtype: torch.dtype
    executor_lane_count: int
    forward_modes: tuple[ForwardMode, ...]
    layer_kind: LayerKind


@dataclass(frozen=True, slots=True)
class FabricParticipantReady:
    """FfnAgent acknowledgement that its coordinator is active."""


@dataclass(frozen=True, slots=True)
class FabricArenasPublished:
    """AtnAgent acknowledgement carrying its Instance-rank arena handles."""

    arenas: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FabricParticipantQuiesced:
    """Participant acknowledgement that ingress is closed."""


@dataclass(frozen=True, slots=True)
class FabricAtnAgentTrace:
    """One completed AtnAgent-side Fabric invocation timeline."""

    instance_index: int
    invocation_sequence: int
    layer_ordinal: int
    payload_rows: int
    executor_lane_index: int | None
    executor_lease_sequence: int | None
    forward_mode: ForwardMode | None
    submission_prepared_ns: int
    output_acknowledgement_published_ns: int


@dataclass(frozen=True, slots=True)
class FabricCoordinatorTrace:
    """One Coordinator scheduling and Executor-lease timeline."""

    instance_index: int
    invocation_sequence: int
    layer_ordinal: int
    payload_rows: int
    executor_lane_index: int | None
    executor_lease_sequence: int | None
    ready_ticket: int | None
    enqueued_ns: int
    scheduled_ns: int
    lane_released_ns: int


@dataclass(frozen=True, slots=True)
class FabricFfnAgentTrace:
    """One FfnAgent execution timeline for a distributed Executor Lane."""

    instance_index: int
    invocation_sequence: int
    layer_ordinal: int
    payload_rows: int
    executor_lane_index: int | None
    executor_lease_sequence: int | None
    payload_row_capacity: int | None
    delivery: xpool.native.fabric.DeliveryVariant | None
    lane_execution_observed_ns: int
    compute_started_ns: int
    compute_completed_ns: int
    completion_published_ns: int


type FabricTrace = FabricAtnAgentTrace | FabricCoordinatorTrace | FabricFfnAgentTrace


@dataclass(frozen=True, slots=True)
class FabricGraphSnapshotEvidence:
    """Picklable actual Primary Graph and Lane Graph snapshot evidence."""

    primary_graph_binding_site_counts: tuple[int, ...]
    lane_compute_branch_counts: tuple[int, ...]
    lane_delivery_branch_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FabricRoutingRecordEvidence:
    """Bit-preserving semantic Routing evidence for one invocation."""

    instance_index: int
    invocation_sequence: int
    layer_ordinal: int
    row_count: int
    effective_topk: int
    topk_ids_bytes: bytes
    topk_weights_bytes: bytes


@dataclass(frozen=True, slots=True)
class FabricRoutingSnapshotEvidence:
    """One drained FfnAgent Routing Observer snapshot."""

    sequence: int
    dropped: int
    records: tuple[FabricRoutingRecordEvidence, ...]


@dataclass(frozen=True, slots=True)
class FabricFailureEvidence:
    """One PE's host-observed canonical Fabric failure state."""

    result_code: ResultCode
    origin_pe: int
    instance_index: int
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
    graph_snapshot: FabricGraphSnapshotEvidence | None
    routing: FabricRoutingSnapshotEvidence | None


@dataclass(frozen=True, slots=True)
class FabricParticipantDrained:
    """Participant acknowledgement carrying post-drain trace evidence."""

    report: FabricParticipantReport


@dataclass(frozen=True, slots=True)
class FabricInstanceSpec:
    """Complete startup configuration for one Fabric Instance-rank client."""

    device: int
    atnagent_pe: int
    arena: str
    forward_mode: ForwardMode
    ffnagent_count: int
    execution_tp_size: int
    atnagent_count: int
    output_requirement: OutputRequirement
    instance_index: int
    payload_rows: int
    payload_dtype: torch.dtype
    layer_ordinals: tuple[int, ...]
    payload_offset: int
    repetition_count: int
    stop_on_command: bool
    start_barrier: Barrier
    layer_kind: LayerKind
    pre_admission_rejection: bool


@dataclass(frozen=True, slots=True)
class FabricInstanceReady:
    """Instance-rank acknowledgement that it reached the shared start gate."""


@dataclass(frozen=True, slots=True)
class FabricInstanceRunning:
    """Instance-rank acknowledgement that its first request was submitted."""


@dataclass(frozen=True, slots=True)
class FabricInstanceResult:
    """Numerical output and completion facts returned by one Instance rank."""

    atnagent_pe: int
    instance_index: int
    forward_mode: ForwardMode
    repetitions_completed: int
    result_code: ResultCode
    actual: tuple[float, ...]
    expected: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class FabricTopologyReport:
    """Aggregate evidence from one complete native Fabric topology run."""

    atnagent_count: int
    ffnagent_count: int
    executor_lane_count: int
    forward_modes: tuple[ForwardMode, ...]
    instances: tuple[FabricInstanceResult, ...]
    participants: tuple[FabricParticipantReport, ...]
