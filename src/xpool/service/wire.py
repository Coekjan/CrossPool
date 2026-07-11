"""Daemon control-plane wire models shared by service clients and route handlers."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xpool.abi import TRANSPORT_ARENA_HANDLE_HEX_LENGTH, TransportArenaHandle
from xpool.config import DeviceRole
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.errors import XpoolDaemonError, XpoolDaemonErrorKind


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


class TransportArenaHandleRecord(WireModel):
    """CUDA IPC transport arena handle record brokered by the daemon."""

    handle: str = Field(
        pattern=rf"^[0-9a-f]{{{TRANSPORT_ARENA_HANDLE_HEX_LENGTH}}}$",
        description="Lowercase hex CUDA IPC transport arena handle.",
    )

    @classmethod
    def from_handle(cls, handle: TransportArenaHandle) -> TransportArenaHandleRecord:
        """Build a daemon wire record from a runtime transport arena handle."""

        return cls(handle=handle.handle)

    def to_handle(self) -> TransportArenaHandle:
        """Return the runtime transport arena handle represented by this record."""

        return TransportArenaHandle(handle=self.handle)


class DevagentTransportArenaBinding(WireModel):
    """Association between an instance id and one devagent-owned transport arena."""

    instance_id: str = Field(description="Instance id whose rank can attach this arena.")
    rank: int = Field(ge=0, description="Rank-local process index within the instance.")
    handle: TransportArenaHandleRecord = Field(
        description="CUDA IPC transport arena handle owned by the publishing devagent."
    )


class ProcessRef(WireModel):
    """Common process identity fields for registration views and owner proofs."""

    pid: int = Field(ge=1, description="Host process id of the registering process.")
    abi_version: int = Field(ge=1, description="xpool descriptor ABI version used by the registering process.")


class DevagentTransportArenaUpsertRequest(WireModel):
    """Request body for upserting a devagent's transport arenas."""

    publisher: ProcessRef = Field(description="Process identity for the publishing devagent.")
    bindings: list[DevagentTransportArenaBinding] = Field(
        description="Transport arena bindings to merge into the devagent's published arena set."
    )


type ControlPlaneWarningKind = Literal[
    "stale_devagent",
    "stale_instance",
    "terminating_devagent",
]


class ControlPlaneWarning(WireModel):
    """Device-scoped warning returned during daemon heartbeat."""

    kind: ControlPlaneWarningKind = Field(description="General control-plane warning kind.")
    cuda_device: int = Field(ge=0, description="CUDA device whose local control-plane state produced this warning.")
    message: str = Field(description="Human-readable warning detail.")


class ProcessHeartbeat(ProcessRef):
    """Heartbeat payload for one registered runtime process."""


class HeartbeatResponse(WireModel):
    """Heartbeat response carrying daemon-observed global warnings."""

    warnings: list[ControlPlaneWarning] = Field(description="Current device-scoped daemon warnings.")


class DevagentRegistration(ProcessRef):
    """Devagent registration payload and list-entry view."""

    cuda_device: int = Field(ge=0, description="CUDA device index owned by the devagent.")


class InstanceRankRef(ProcessRef):
    """Reference to one instance-rank host process."""

    instance_id: str = Field(description="Instance id from the resolved xpool config.")
    rank: int = Field(ge=0, description="Rank-local process index within the instance.")


class DevagentTransportArenaDrainResponse(WireModel):
    """Result of a devagent transport arena drain attempt."""

    in_use: list[InstanceRankRef] = Field(
        description="Instance ranks that still hold fresh acquired arena leases.",
    )


class InstanceRegistration(InstanceRankRef):
    """Instance-rank registration payload and list-entry view."""

    transport: InstanceTransportAttributes = Field(
        description="Transport attributes declared by this instance rank at registration time.",
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


class ReadinessScope(StrEnum):
    """Participant scope selected by the daemon readiness endpoint.

    Attributes:
        ATN: Require attention devagents, instance ranks, and matching arena
            publications.
        FFN: Require configured FFN devagents.
    """

    ATN = "atn"
    FFN = "ffn"


class ReadinessEntry(WireModel):
    """Common readiness fields for one configured process slot."""

    pid: int | None = Field(default=None, ge=1, description="Registered process pid, or null when offline.")
    status: ReadinessStatus = Field(description="Live registration status for this process slot.")


class ReadinessDevagent(ReadinessEntry):
    """One configured devagent slot reported by the readiness endpoint."""

    cuda_device: int = Field(ge=0, description="CUDA device index owned by this devagent.")
    role: DeviceRole = Field(description="Configured role hosted by this devagent.")


class ReadinessInstance(ReadinessEntry):
    """One configured instance rank slot reported by the readiness endpoint."""

    instance_id: str = Field(description="Instance id from the resolved xpool config.")
    cuda_device: int = Field(ge=0, description="Attention CUDA device assigned to this instance rank.")
    rank: int = Field(ge=0, description="Rank-local process index within the instance.")


class ReadinessSnapshot(WireModel):
    """Daemon readiness response."""

    ready: bool = Field(description="Whether every requested readiness scope currently passes.")
    mps_status: ReadinessStatus = Field(
        description="CUDA MPS controller status; only online and offline are produced.",
    )
    scopes: dict[ReadinessScope, bool] = Field(
        description="Readiness result for each requested participant scope.",
    )
    cuda_devices: tuple[int, ...] = Field(description="Configured CUDA devices managed by xpool.")
    devagents: list[ReadinessDevagent] = Field(description="Per-devagent readiness entries ordered by CUDA device.")
    instances: list[ReadinessInstance] = Field(
        description="Per-instance rank readiness entries ordered by instance and rank.",
    )
