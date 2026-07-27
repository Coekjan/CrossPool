"""Daemon control-plane wire models shared by service clients and route handlers."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xpool.abi import FfnResultCode
from xpool.fabric import FabricGeneration, FabricGenerationPhase, FabricParticipantPhase, FfnWorkload
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.errors import XpoolDaemonError, XpoolDaemonErrorKind
from xpool.transport import TransportArenaHandle


class WireModel(BaseModel):
    """Base class for daemon control-plane wire models."""

    model_config = ConfigDict(extra="forbid")


class XpoolDaemonErrorDetail(WireModel):
    """Daemon-domain error detail returned through HTTP error responses."""

    kind: XpoolDaemonErrorKind = Field(description="General daemon-domain error kind.")
    message: str = Field(description="Human-readable daemon-domain error detail.")

    @classmethod
    def from_error(cls, error: XpoolDaemonError) -> XpoolDaemonErrorDetail:
        """Build HTTP error detail from a daemon-domain exception."""

        return cls(kind=error.kind, message=error.message)


class AtnAgentTransportArenaBinding(WireModel):
    """Association between an instance id and one AtnAgent-owned arena."""

    instance_id: str = Field(description="Instance id whose rank can attach this arena.")
    rank: int = Field(ge=0, description="Rank-local process index within the instance.")
    handle: TransportArenaHandle = Field(
        description="CUDA IPC transport arena handle owned by the publishing AtnAgent."
    )


class ProcessRef(WireModel):
    """Common process identity fields for registration views and owner proofs."""

    pid: int = Field(ge=1, description="Host process id of the registering process.")
    abi_version: int = Field(ge=1, description="xpool native ABI version used by the registering process.")


class AtnAgentTransportArenaUpsertRequest(WireModel):
    """Request body for upserting an AtnAgent's transport arenas."""

    publisher: ProcessRef = Field(description="Process identity for the publishing AtnAgent.")
    bindings: list[AtnAgentTransportArenaBinding] = Field(
        description="Transport arena bindings to merge into the AtnAgent's published arena set."
    )


type ControlPlaneWarningKind = Literal[
    "stale_atnagent",
    "stale_ffnagent",
    "stale_instance",
    "quiescing_atnagent",
]


class ControlPlaneWarning(WireModel):
    """Device-scoped warning returned during daemon heartbeat."""

    kind: ControlPlaneWarningKind = Field(description="General control-plane warning kind.")
    cuda_device: int = Field(ge=0, description="CUDA device whose local control-plane state produced this warning.")
    message: str = Field(description="Human-readable warning detail.")


class FabricInvocationFailure(WireModel):
    """Canonical failure of one distributed Fabric invocation."""

    result_code: FfnResultCode = Field(description="Canonical device result code.")
    origin_pe: int = Field(ge=0, description="PE that first published the canonical failure.")
    model_index: int = Field(ge=0, description="Model index of the failed invocation.")
    invocation_sequence: int = Field(ge=1, description="Generation-local model invocation sequence.")
    layer_ordinal: int = Field(ge=0, description="Canonical layer ordinal of the failed invocation.")


class FabricOwnerFailureReason(StrEnum):
    """Daemon-observed reason that a retained generation owner was lost.

    Attributes:
        EXITED: The retained process identity no longer exists.
        STALE: The owner stopped renewing its registration before the timeout.
        REPLACED: A different process claimed the owner's logical placement.
    """

    EXITED = "exited"
    STALE = "stale"
    REPLACED = "replaced"


class FabricPeOwner(WireModel):
    """Logical AtnAgent or FfnAgent owner of one Fabric PE."""

    role: Literal["atnagent", "ffnagent"] = Field(description="Agent role owning the Fabric PE.")
    pe: int = Field(ge=0, description="Fabric PE owned by the process.")
    cuda_device: int = Field(ge=0, description="CUDA device assigned to the PE.")


class FabricInstanceOwner(WireModel):
    """Logical Instance-rank owner retained by one Fabric generation."""

    role: Literal["instance"] = Field(default="instance", description="Instance role owning the registered rank.")
    instance_id: str = Field(description="Configured model instance id.")
    rank: int = Field(ge=0, description="Instance rank within the attention world.")
    cuda_device: int = Field(ge=0, description="CUDA device assigned to the Instance rank.")


class FabricOwnerFailure(WireModel):
    """Daemon-observed loss of one retained generation owner."""

    owner: FabricPeOwner | FabricInstanceOwner = Field(description="Logical owner that was lost.")
    reason: FabricOwnerFailureReason = Field(description="Observed owner-loss reason.")


class FabricProtocolFailure(WireModel):
    """Diagnostic control-protocol or lifecycle invariant failure."""

    message: str = Field(description="Human-readable diagnostic; not a stable machine code.")


class FabricParticipantReport(WireModel):
    """One Agent's acknowledged local Fabric lifecycle report."""

    owner: ProcessRef = Field(description="Exact process identity submitting this report.")
    generation: FabricGeneration = Field(description="Fabric generation observed by the participant.")
    pe: int = Field(ge=0, description="Deterministic Fabric PE owned by this Agent.")
    phase: FabricParticipantPhase = Field(description="Daemon-acknowledged current local lifecycle phase.")
    plan_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Semantic Fabric plan digest activated by this participant.",
    )
    invocation_failure: FabricInvocationFailure | None = Field(
        default=None, description="Canonical invocation failure observed by this PE."
    )
    protocol_failure: FabricProtocolFailure | None = Field(
        default=None, description="Local protocol failure observed by this PE."
    )


class HeartbeatResponse(WireModel):
    """Heartbeat response carrying warnings and authoritative Fabric state."""

    warnings: list[ControlPlaneWarning] = Field(description="Current device-scoped daemon warnings.")
    generation: FabricGeneration | None = Field(default=None, description="Active Fabric generation, if one exists.")
    fabric_phase: FabricGenerationPhase | None = Field(
        default=None,
        description="Daemon-authoritative Fabric phase, or null before generation creation.",
    )
    fabric_invocation_failure: FabricInvocationFailure | None = Field(
        default=None, description="Canonical invocation failure, if published."
    )
    fabric_owner_failure: FabricOwnerFailure | None = Field(
        default=None, description="First daemon-observed owner failure."
    )
    fabric_protocol_failure: FabricProtocolFailure | None = Field(
        default=None, description="First control-protocol failure."
    )


class FabricQuiesceRequest(WireModel):
    """Authenticated request to stop admission for one Fabric generation."""

    owner: ProcessRef = Field(description="Current Agent participant requesting quiesce.")
    generation: FabricGeneration = Field(description="Retained generation to quiesce.")


class AtnAgentRegistration(ProcessRef):
    """AtnAgent registration payload and list-entry view."""

    cuda_device: int = Field(ge=0, description="CUDA device index owned by the AtnAgent.")


class FfnAgentRegistration(ProcessRef):
    """FfnAgent registration payload and list-entry view."""

    cuda_device: int = Field(ge=0, description="CUDA device index owned by the FfnAgent.")


class InstanceRankRef(ProcessRef):
    """Reference to one instance-rank host process."""

    instance_id: str = Field(description="Instance id from the resolved xpool config.")
    rank: int = Field(ge=0, description="Rank-local process index within the instance.")


class AtnAgentTransportLeaseQuiesceResponse(WireModel):
    """Result of closing one AtnAgent generation's lease admission."""

    in_use: list[InstanceRankRef] = Field(
        description="Instance ranks that still hold fresh acquired arena leases.",
    )


class InstanceRegistration(InstanceRankRef):
    """Instance-rank registration payload and list-entry view."""

    transport: InstanceTransportAttributes = Field(
        description="Transport attributes declared by this instance rank at registration time.",
    )
    workload: FfnWorkload = Field(
        description="Resolved rank-independent model workload agreed by every instance rank.",
    )


class InstanceInitializedPublication(WireModel):
    """Owner-bound publication that one SGLang rank completed graph capture."""

    owner: ProcessRef = Field(description="Process identity owning the instance-rank registration.")
    generation: FabricGeneration = Field(description="Executable fabric generation observed by the rank.")
    plan_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Executable fabric plan digest observed by the rank."
    )


class ReadinessStatus(StrEnum):
    """Process registration status reported by the daemon readiness endpoint.

    Attributes:
        ONLINE: The configured process slot has a live registration.
        STALE: The configured process slot has a live registration that missed the heartbeat watermark.
        OFFLINE: The configured process slot has no live registration.
    """

    ONLINE = "online"
    STALE = "stale"
    OFFLINE = "offline"


class ReadinessEntry(WireModel):
    """Common readiness fields for one configured process slot."""

    pid: int | None = Field(default=None, ge=1, description="Registered process pid, or null when offline.")
    status: ReadinessStatus = Field(description="Live registration status for this process slot.")


class ReadinessAtnAgent(ReadinessEntry):
    """One configured AtnAgent slot reported by readiness."""

    cuda_device: int = Field(ge=0, description="CUDA device index owned by this AtnAgent.")


class ReadinessFfnAgent(ReadinessEntry):
    """One configured FfnAgent slot reported by readiness."""

    cuda_device: int = Field(ge=0, description="CUDA device index owned by this FfnAgent.")


class ReadinessInstance(ReadinessEntry):
    """One configured instance rank slot reported by the readiness endpoint."""

    instance_id: str = Field(description="Instance id from the resolved xpool config.")
    cuda_device: int = Field(ge=0, description="Attention CUDA device assigned to this instance rank.")
    rank: int = Field(ge=0, description="Rank-local process index within the instance.")


class ReadinessSnapshot(WireModel):
    """Daemon readiness response."""

    ready: bool = Field(description="Whether the complete configured xpool system is ready.")
    generation: FabricGeneration | None = Field(description="Active Fabric generation, or null before creation.")
    fabric_phase: FabricGenerationPhase | None = Field(
        description="Daemon-authoritative Fabric generation phase, or null before creation."
    )
    fabric_invocation_failure: FabricInvocationFailure | None = Field(
        description="Canonical invocation failure, or null when absent."
    )
    fabric_owner_failure: FabricOwnerFailure | None = Field(
        description="Daemon-observed owner failure, or null when absent."
    )
    fabric_protocol_failure: FabricProtocolFailure | None = Field(
        description="Control-protocol failure, or null when absent."
    )
    transport_ready: bool = Field(description="Whether every configured Transport publication accepts leases.")
    instances_initialized: bool = Field(description="Whether every Instance rank completed initialization.")
    mps_status: ReadinessStatus = Field(
        description="CUDA MPS controller status; only online and offline are produced.",
    )
    cuda_devices: tuple[int, ...] = Field(description="Configured CUDA devices managed by xpool.")
    atnagents: list[ReadinessAtnAgent] = Field(description="AtnAgent readiness entries ordered by rank.")
    ffnagents: list[ReadinessFfnAgent] = Field(description="FfnAgent readiness entries ordered by rank.")
    instances: list[ReadinessInstance] = Field(
        description="Per-instance rank readiness entries ordered by instance and rank.",
    )
