"""Typed startup contracts for the generation-scoped FFN fabric."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xpool.abi import TensorDType
from xpool.config import FfnSchedulingPolicy

__all__ = [
    "FABRIC_UID_HEX_LENGTH",
    "FabricGeneration",
    "FabricGenerationPhase",
    "FabricModelPlan",
    "FabricParticipantPhase",
    "FabricPePlacement",
    "FabricPlan",
    "FabricRole",
    "FabricUid",
    "FfnLayerKind",
    "FfnLayerSpec",
    "FfnSchedulerPlan",
    "FfnWorkload",
    "FifoSchedulerPlan",
    "RandomSchedulerPlan",
]

FABRIC_UID_HEX_LENGTH = 256
"""Character count of a lowercase hexadecimal NVSHMEM unique id."""

UINT64_MAX = 2**64 - 1


class FabricGenerationPhase(StrEnum):
    """Daemon-authoritative lifecycle of one Fabric generation.

    Attributes:
        JOINING: Plan installed while participant joins and Instance
            initialization are incomplete; ordinary execution is unavailable.
        EXECUTABLE: Every required owner is ready and new invocation and
            Transport lease admission is allowed.
        QUIESCING: New admission is closed while existing work converges.
        DRAINING: Participants are retiring resident work.
        FINALIZING: Drained participants are releasing collective resources.
        ABORTING: Fail-stop path selected after owner or protocol failure.
        STOPPED: Retained terminal generation awaiting owner retirement.
    """

    JOINING = "joining"
    EXECUTABLE = "executable"
    QUIESCING = "quiescing"
    DRAINING = "draining"
    FINALIZING = "finalizing"
    ABORTING = "aborting"
    STOPPED = "stopped"

    def allows(self, successor: FabricGenerationPhase) -> bool:
        """Return whether ``successor`` is one exact lifecycle edge."""

        normal_successor = {
            FabricGenerationPhase.JOINING: FabricGenerationPhase.EXECUTABLE,
            FabricGenerationPhase.EXECUTABLE: FabricGenerationPhase.QUIESCING,
            FabricGenerationPhase.QUIESCING: FabricGenerationPhase.DRAINING,
            FabricGenerationPhase.DRAINING: FabricGenerationPhase.FINALIZING,
            FabricGenerationPhase.FINALIZING: FabricGenerationPhase.STOPPED,
            FabricGenerationPhase.ABORTING: FabricGenerationPhase.STOPPED,
        }
        if successor is FabricGenerationPhase.ABORTING:
            return self not in {FabricGenerationPhase.ABORTING, FabricGenerationPhase.STOPPED}
        return normal_successor.get(self) is successor


class FabricParticipantPhase(StrEnum):
    """Daemon-acknowledged local lifecycle of one Fabric PE.

    Attributes:
        JOINING: Participant has started joining the retained generation.
        JOINED: Local NVSHMEM and arena state are initialized.
        ACTIVE: Required resident execution has started, or the AtnAgent PE is
            ready to submit through Fabric.
        QUIESCED: Participant accepted the generation admission closure.
        DRAINING: Local resident and publication work is retiring.
        DRAINED: Local device work has completed and trace is readable.
        FINALIZED: Participant released collective resources and NVSHMEM state.
    """

    JOINING = "joining"
    JOINED = "joined"
    ACTIVE = "active"
    QUIESCED = "quiesced"
    DRAINING = "draining"
    DRAINED = "drained"
    FINALIZED = "finalized"

    def allows(self, successor: FabricParticipantPhase) -> bool:
        """Return whether ``successor`` is the next exact participant edge."""

        return {
            FabricParticipantPhase.JOINING: FabricParticipantPhase.JOINED,
            FabricParticipantPhase.JOINED: FabricParticipantPhase.ACTIVE,
            FabricParticipantPhase.ACTIVE: FabricParticipantPhase.QUIESCED,
            FabricParticipantPhase.QUIESCED: FabricParticipantPhase.DRAINING,
            FabricParticipantPhase.DRAINING: FabricParticipantPhase.DRAINED,
            FabricParticipantPhase.DRAINED: FabricParticipantPhase.FINALIZED,
        }.get(self) is successor


@dataclass(frozen=True, slots=True)
class FabricGeneration:
    """Random identity of one daemon-authoritative Fabric world.

    Attributes:
        high: Most-significant 64 generation bits.
        low: Least-significant 64 generation bits.
    """

    high: int
    low: int

    def __post_init__(self) -> None:
        """Validate both unsigned 64-bit generation words."""

        if not 0 <= self.high < 2**64 or not 0 <= self.low < 2**64:
            raise ValueError("xpool fabric generation words must be unsigned 64-bit integers")
        if self.high == 0 and self.low == 0:
            raise ValueError("xpool fabric generation must be nonzero")

    @classmethod
    def create(cls) -> FabricGeneration:
        """Create a cryptographically random nonzero generation identity."""

        value = 0
        while value == 0:
            value = secrets.randbits(128)
        return cls(high=value >> 64, low=value & (2**64 - 1))

    @classmethod
    def parse(cls, value: str) -> FabricGeneration:
        """Parse exactly 32 lowercase hexadecimal characters."""

        if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("xpool fabric generation must contain 32 lowercase hex characters")
        return cls(high=int(value[:16], 16), low=int(value[16:], 16))

    def format(self) -> str:
        """Format this generation as 32 lowercase hexadecimal characters."""

        return f"{self.high:016x}{self.low:016x}"


class FfnLayerKind(IntEnum):
    """Structural FFN layer kinds stored in a Fabric workload.

    Attributes:
        DENSE: Dense feed-forward network implementation.
        SPARSE: Sparse mixture-of-experts implementation.
    """

    DENSE = 1
    SPARSE = 2


@dataclass(frozen=True, slots=True)
class FabricUid:
    """Opaque NVSHMEM unique id used to bootstrap one fabric world.

    Attributes:
        value: Lowercase hexadecimal encoding of the 128-byte NVSHMEM UID.
    """

    value: str

    def __post_init__(self) -> None:
        """Validate the UID at Python Fabric boundaries."""

        if len(self.value) != FABRIC_UID_HEX_LENGTH:
            raise ValueError(f"fabric UID must contain {FABRIC_UID_HEX_LENGTH} lowercase hex characters")
        if any(character not in "0123456789abcdef" for character in self.value):
            raise ValueError("fabric UID must contain only lowercase hex characters")


class FabricModel(BaseModel):
    """Immutable strict base for fabric contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FfnLayerSpec(FabricModel):
    """One ordered decoder FFN layer in an instance workload."""

    layer_id: int = Field(ge=0, description="Model decoder layer identifier consumed by the shim.")
    kind: FfnLayerKind = Field(description="Structural FFN implementation kind.")


class FfnWorkload(FabricModel):
    """Resolved rank-independent FFN workload for one configured model."""

    model_config_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="SHA-256 digest of model architecture facts used by both adapters."
    )
    dtype: TensorDType = Field(description="Hidden-state dtype carried by A2F and F2A payloads.")
    hidden_size: int = Field(ge=1, description="Hidden-state columns in every FFN invocation.")
    layers: tuple[FfnLayerSpec, ...] = Field(
        min_length=1,
        description="FFN layers in decoder execution order; tuple position is the canonical ordinal.",
    )
    max_decode_rows: int = Field(ge=1, description="Maximum physical rows for eager or full-graph decode.")
    max_prefill_rows: int = Field(ge=1, description="Maximum physical rows for eager or piecewise-graph prefill.")

    @model_validator(mode="after")
    def validate_layers(self) -> FfnWorkload:
        """Validate ordered layer identity.

        Returns:
            Validated workload contract.

        Raises:
            ValueError: If layer order or identities are inconsistent.
        """

        if len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise ValueError("FFN layer ids must be unique")
        return self


class FabricRole(StrEnum):
    """Agent role represented in the deterministic Fabric PE map."""

    ATNAGENT = "atnagent"
    FFNAGENT = "ffnagent"


class FabricPePlacement(FabricModel):
    """One deterministic NVSHMEM PE placement in a fabric plan."""

    pe: int = Field(ge=0, description="NVSHMEM PE index used only by native fabric APIs.")
    role: FabricRole = Field(description="xpool Agent role that owns the PE.")
    cuda_device: int = Field(ge=0, description="Host CUDA device index owned by the Agent.")

    @classmethod
    def validate_order(cls, placements: tuple[FabricPePlacement, ...]) -> None:
        """Validate the canonical ATN prefix and FFN suffix placement order.

        Args:
            placements: Complete ordered PE placement table.

        Raises:
            ValueError: If PEs or role regions are not contiguous, or either
                role is absent.
        """

        if tuple(placement.pe for placement in placements) != tuple(range(len(placements))):
            raise ValueError("fabric PEs must be contiguous from zero")
        atnagent_count = sum(placement.role is FabricRole.ATNAGENT for placement in placements)
        ffnagent_count = len(placements) - atnagent_count
        if atnagent_count == 0 or ffnagent_count == 0:
            raise ValueError("fabric placements require both AtnAgent and FfnAgent PEs")
        expected_roles = (FabricRole.ATNAGENT,) * atnagent_count + (FabricRole.FFNAGENT,) * ffnagent_count
        if tuple(placement.role for placement in placements) != expected_roles:
            raise ValueError("fabric placements must use an AtnAgent prefix and FfnAgent suffix")


class FifoSchedulerPlan(FabricModel):
    """Immutable FIFO scheduler policy for one Fabric generation."""

    policy: Literal[FfnSchedulingPolicy.FIFO] = FfnSchedulingPolicy.FIFO


class RandomSchedulerPlan(FabricModel):
    """Immutable deterministic random scheduler policy for one generation."""

    policy: Literal[FfnSchedulingPolicy.RANDOM] = FfnSchedulingPolicy.RANDOM
    seed: int = Field(ge=1, le=UINT64_MAX, description="Nonzero uint64 scheduler seed.")


type FfnSchedulerPlan = Annotated[
    FifoSchedulerPlan | RandomSchedulerPlan,
    Field(discriminator="policy"),
]


class FabricModelPlan(FabricModel):
    """One workload and its complete attention TP-by-DP topology."""

    workload: FfnWorkload = Field(description="Canonical rank-independent model workload.")
    atn_tp_size: int = Field(ge=1, description="Attention tensor-parallel size for this model.")
    atn_dp_size: int = Field(ge=1, description="Attention data-parallel size for this model.")


class FabricPlan(FabricModel):
    """Complete semantic execution plan for one fabric generation."""

    generation: FabricGeneration = Field(description="Random identity for this fabric initialization.")
    uid: FabricUid = Field(description="Opaque NVSHMEM unique id shared by every joining participant.")
    pe_placements: tuple[FabricPePlacement, ...] = Field(min_length=2, description="Complete ordered PE map.")
    executor_count: int = Field(ge=1, description="Number of independent distributed FFN executors.")
    scheduler: FfnSchedulerPlan = Field(description="Tagged native executor admission policy.")
    models: tuple[FabricModelPlan, ...] = Field(
        min_length=1,
        description="Canonical model plans in config instance order.",
    )

    @model_validator(mode="after")
    def validate_identity_order(self) -> FabricPlan:
        """Require canonical PE and model ordering.

        Returns:
            Validated fabric plan.

        Raises:
            ValueError: If PE or model identities are duplicated or unordered.
        """

        FabricPePlacement.validate_order(self.pe_placements)
        atnagent_count = sum(placement.role is FabricRole.ATNAGENT for placement in self.pe_placements)
        if any(model.atn_tp_size * model.atn_dp_size != atnagent_count for model in self.models):
            raise ValueError("every fabric model topology must cover the complete AtnAgent PE set")
        if any(model.atn_tp_size > 1 and model.atn_dp_size > 1 for model in self.models):
            raise ValueError("fabric does not support combined attention TP-by-DP")
        return self

    def digest(self) -> str:
        """Return the canonical SHA-256 plan digest excluding generation.

        Returns:
            Lowercase hexadecimal digest shared by every fabric participant.
        """

        payload = self.model_dump(mode="json", exclude={"generation", "uid"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
