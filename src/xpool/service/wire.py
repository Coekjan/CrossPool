"""Daemon control-plane wire models shared by service clients and route handlers."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xpool import ffn
from xpool.fabric import FabricGenerationId, FabricGenerationPhase, FabricParticipantPhase, InstanceFfnProfile
from xpool.native.ffn import ResultCode
from xpool.runtime.transport import InstanceRankTransportProfile
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
    abi_version: int = Field(ge=1, description="CrossPool native ABI version used by the registering process.")


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

    result_code: ResultCode = Field(description="Canonical device result code.")
    origin_pe: int = Field(ge=0, description="PE that first published the canonical failure.")
    instance_index: int = Field(ge=0, description="Instance index of the failed invocation.")
    invocation_sequence: int = Field(ge=1, description="Instance-local invocation sequence.")
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


class FabricPeOwnerFailure(WireModel):
    """Daemon-observed loss of one AtnAgent or FfnAgent PE owner."""

    role: Literal["atnagent", "ffnagent"] = Field(description="Agent role owning the Fabric PE.")
    pe: int = Field(ge=0, description="Fabric PE owned by the process.")
    reason: FabricOwnerFailureReason = Field(description="Reason this Fabric PE owner was lost.")


class FabricInstanceRankOwnerFailure(WireModel):
    """Daemon-observed loss of one Instance-rank owner."""

    role: Literal["instance"] = Field(default="instance", description="Instance role owning the registered rank.")
    instance_id: str = Field(description="Configured model instance id.")
    rank: int = Field(ge=0, description="Instance rank within the attention world.")
    reason: FabricOwnerFailureReason = Field(description="Reason this Instance-rank owner was lost.")


type FabricOwnerFailure = Annotated[
    FabricPeOwnerFailure | FabricInstanceRankOwnerFailure,
    Field(discriminator="role"),
]


class FabricParticipantReport(WireModel):
    """One Agent's acknowledged local Fabric lifecycle report."""

    owner: ProcessRef = Field(description="Exact process identity submitting this report.")
    generation: FabricGenerationId = Field(description="Fabric generation observed by the participant.")
    pe: int = Field(ge=0, description="Deterministic Fabric PE owned by this Agent.")
    phase: FabricParticipantPhase = Field(description="Daemon-acknowledged current local lifecycle phase.")
    invocation_failure: FabricInvocationFailure | None = Field(
        default=None, description="Canonical invocation failure observed by this PE."
    )
    control_failure: str | None = Field(
        default=None,
        min_length=1,
        description="First local control/lifecycle diagnostic observed by this PE.",
    )


class HeartbeatResponse(WireModel):
    """Heartbeat response carrying warnings and authoritative Fabric state."""

    warnings: list[ControlPlaneWarning] = Field(description="Current device-scoped daemon warnings.")
    generation: FabricGenerationId | None = Field(default=None, description="Active Fabric generation, if one exists.")
    fabric_phase: FabricGenerationPhase | None = Field(
        default=None,
        description="Daemon-authoritative Fabric phase, or null before generation creation.",
    )


class FabricQuiesceRequest(WireModel):
    """Authenticated request to stop admission for one Fabric generation."""

    owner: ProcessRef = Field(description="Current Agent participant requesting quiesce.")
    generation: FabricGenerationId = Field(description="Retained generation to quiesce.")


class AtnAgentRegistration(ProcessRef):
    """AtnAgent registration payload and list-entry view."""

    cuda_device: int = Field(ge=0, description="CUDA device index owned by the AtnAgent.")


class FfnAgentRegistration(ProcessRef):
    """FfnAgent registration payload and list-entry view."""

    cuda_device: int = Field(ge=0, description="CUDA device index owned by the FfnAgent.")
    cuda_total_memory_bytes: int = Field(ge=1, description="Total bytes reported by the owned CUDA device.")
    cuda_free_memory_bytes: int = Field(ge=1, description="Free bytes reported by the owned CUDA device.")
    model_specs: tuple[ffn.FfnModelSpec, ...] = Field(
        min_length=1,
        description="Generation-independent FFN Model Specs loaded by this FfnAgent.",
    )

    @model_validator(mode="after")
    def validate_memory_and_specs(self) -> FfnAgentRegistration:
        """Require one legal memory observation and unique ordered Model IDs."""

        if self.cuda_free_memory_bytes > self.cuda_total_memory_bytes:
            raise ValueError("FfnAgent free CUDA memory exceeds total CUDA memory")
        model_ids = tuple(spec.model_id for spec in self.model_specs)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("FfnAgent Model Specs must have unique Model IDs")
        return self


class InstanceRankRef(ProcessRef):
    """Reference to one instance-rank host process."""

    instance_id: str = Field(description="Instance id from the resolved CrossPool config.")
    rank: int = Field(ge=0, description="Rank-local process index within the instance.")


class AtnAgentTransportLeaseQuiesceResponse(WireModel):
    """Result of closing one AtnAgent generation's lease admission."""

    in_use: list[InstanceRankRef] = Field(
        description="Instance ranks that still hold fresh acquired arena leases.",
    )


class InstanceRankRegistration(InstanceRankRef):
    """Instance-rank registration payload and list-entry view."""

    transport: InstanceRankTransportProfile = Field(
        description="Transport attributes declared by this instance rank at registration time.",
    )
    ffn_profile: InstanceFfnProfile = Field(
        description="Resolved rank-independent FFN profile agreed by every instance rank.",
    )


class ServingListener(WireModel):
    """Public HTTP listener declared by one Instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(min_length=1, description="Host bound by the Instance serving process.")
    port: int = Field(ge=1, le=65535, description="TCP port bound by the Instance serving process.")


class InstanceRankInitializedPublication(WireModel):
    """Owner-bound publication that one Instance Rank completed scheduler construction."""

    owner: ProcessRef = Field(description="Process identity owning the instance-rank registration.")
    generation: FabricGenerationId = Field(description="Executable fabric generation observed by the rank.")
    serving_listener: ServingListener = Field(description="Public HTTP listener shared by every rank in the Instance.")


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


class ReadinessInstanceRank(ReadinessEntry):
    """One configured instance rank slot reported by the readiness endpoint."""

    instance_id: str = Field(description="Instance id from the resolved CrossPool config.")
    cuda_device: int = Field(ge=0, description="Attention CUDA device assigned to this instance rank.")
    rank: int = Field(ge=0, description="Rank-local process index within the instance.")


class ReadinessSnapshot(WireModel):
    """Daemon readiness response."""

    ready: bool = Field(description="Whether the complete configured CrossPool system is ready.")
    generation: FabricGenerationId | None = Field(description="Active Fabric generation, or null before creation.")
    fabric_phase: FabricGenerationPhase | None = Field(
        description="Daemon-authoritative Fabric generation phase, or null before creation."
    )
    fabric_invocation_failure: FabricInvocationFailure | None = Field(
        description="Canonical invocation failure, or null when absent."
    )
    fabric_owner_failure: FabricOwnerFailure | None = Field(
        description="Daemon-observed owner failure, or null when absent."
    )
    fabric_control_failure: str | None = Field(
        description="First control/lifecycle diagnostic, or null when absent.",
    )
    transport_ready: bool = Field(description="Whether every configured Transport publication accepts leases.")
    instances_initialized: bool = Field(description="Whether every Instance rank completed initialization.")
    mps_status: ReadinessStatus = Field(
        description="CUDA MPS controller status; only online and offline are produced.",
    )
    cuda_devices: tuple[int, ...] = Field(description="Configured CUDA devices managed by CrossPool.")
    atnagents: list[ReadinessAtnAgent] = Field(description="AtnAgent readiness entries ordered by rank.")
    ffnagents: list[ReadinessFfnAgent] = Field(description="FfnAgent readiness entries ordered by rank.")
    instances: list[ReadinessInstanceRank] = Field(
        description="Per-instance rank readiness entries ordered by instance and rank.",
    )
