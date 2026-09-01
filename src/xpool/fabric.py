"""Typed startup contracts for the generation-scoped FFN fabric."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_serializer, field_validator, model_validator

from xpool.config import FfnSchedulingPolicy
from xpool.native.ffn import LayerKind

__all__ = [
    "FABRIC_UID_HEX_LENGTH",
    "DenseFfnLayerPlan",
    "FabricGenerationId",
    "FabricGenerationPhase",
    "FabricInstancePlan",
    "FabricParticipantPhase",
    "FabricPePlacement",
    "FabricPlan",
    "FabricRole",
    "FabricUid",
    "FfnLayerPlan",
    "FfnModelPlan",
    "FfnSchedulerPolicy",
    "FifoSchedulerPolicy",
    "InstanceFfnLayerProfile",
    "InstanceFfnProfile",
    "InstanceRankTopology",
    "MoeFfnLayerPlan",
    "RandomSchedulerPolicy",
]

FABRIC_UID_HEX_LENGTH = 256
"""Character count of a lowercase hexadecimal NVSHMEM unique id."""

UINT64_MAX = 2**64 - 1


class FabricGenerationPhase(StrEnum):
    """Daemon-authoritative lifecycle of one Fabric generation."""

    PREPARING_JOIN = "preparing_join"
    JOINING = "joining"
    PREPARING_EXECUTION = "preparing_execution"
    ACTIVATING = "activating"
    EXECUTABLE = "executable"
    QUIESCING = "quiescing"
    DRAINING = "draining"
    FINALIZING = "finalizing"
    ABORTING = "aborting"
    STOPPED = "stopped"

    def allows(self, successor: FabricGenerationPhase) -> bool:
        """Return whether ``successor`` is one exact lifecycle edge."""

        normal_successor = {
            FabricGenerationPhase.PREPARING_JOIN: FabricGenerationPhase.JOINING,
            FabricGenerationPhase.JOINING: FabricGenerationPhase.PREPARING_EXECUTION,
            FabricGenerationPhase.PREPARING_EXECUTION: FabricGenerationPhase.ACTIVATING,
            FabricGenerationPhase.ACTIVATING: FabricGenerationPhase.EXECUTABLE,
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
    """Daemon-acknowledged local lifecycle of one Fabric PE."""

    JOIN_READY = "join_ready"
    JOINING = "joining"
    JOINED = "joined"
    EXECUTION_READY = "execution_ready"
    ACTIVE = "active"
    QUIESCED = "quiesced"
    DRAINING = "draining"
    DRAINED = "drained"
    FINALIZED = "finalized"

    def allows(self, successor: FabricParticipantPhase) -> bool:
        """Return whether ``successor`` is one exact participant edge."""

        return {
            FabricParticipantPhase.JOIN_READY: FabricParticipantPhase.JOINING,
            FabricParticipantPhase.JOINING: FabricParticipantPhase.JOINED,
            FabricParticipantPhase.JOINED: FabricParticipantPhase.EXECUTION_READY,
            FabricParticipantPhase.EXECUTION_READY: FabricParticipantPhase.ACTIVE,
            FabricParticipantPhase.ACTIVE: FabricParticipantPhase.QUIESCED,
            FabricParticipantPhase.QUIESCED: FabricParticipantPhase.DRAINING,
            FabricParticipantPhase.DRAINING: FabricParticipantPhase.DRAINED,
            FabricParticipantPhase.DRAINED: FabricParticipantPhase.FINALIZED,
        }.get(self) is successor


@dataclass(frozen=True, slots=True)
class FabricGenerationId:
    """Random identity of one daemon-authoritative Fabric world."""

    high: int
    low: int

    def __post_init__(self) -> None:
        """Validate both unsigned 64-bit generation words."""

        if not 0 <= self.high <= UINT64_MAX or not 0 <= self.low <= UINT64_MAX:
            raise ValueError("xpool fabric generation words must be unsigned 64-bit integers")
        if self.high == 0 and self.low == 0:
            raise ValueError("xpool fabric generation must be nonzero")

    @classmethod
    def create(cls) -> FabricGenerationId:
        """Create a cryptographically random nonzero generation identity."""

        value = 0
        while value == 0:
            value = secrets.randbits(128)
        return cls(high=value >> 64, low=value & UINT64_MAX)

    @classmethod
    def parse(cls, value: str) -> FabricGenerationId:
        """Parse exactly 32 lowercase hexadecimal characters."""

        if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("xpool fabric generation must contain 32 lowercase hex characters")
        return cls(high=int(value[:16], 16), low=int(value[16:], 16))

    def format(self) -> str:
        """Format this generation as 32 lowercase hexadecimal characters."""

        return f"{self.high:016x}{self.low:016x}"


@dataclass(frozen=True, slots=True)
class FabricUid:
    """Opaque NVSHMEM unique id used to bootstrap one Fabric world."""

    value: str

    def __post_init__(self) -> None:
        """Validate the UID at Python Fabric boundaries."""

        if len(self.value) != FABRIC_UID_HEX_LENGTH:
            raise ValueError(f"fabric UID must contain {FABRIC_UID_HEX_LENGTH} lowercase hex characters")
        if any(character not in "0123456789abcdef" for character in self.value):
            raise ValueError("fabric UID must contain only lowercase hex characters")


class FabricModel(BaseModel):
    """Immutable strict base for Fabric contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InstanceFfnLayerProfile(FabricModel):
    """One ordered decoder FFN layer exposed by an Instance."""

    layer_id: int = Field(ge=0, description="Model-local decoder layer identity.")
    kind: LayerKind = Field(description="FFN structure used by this layer.")


class InstanceFfnProfile(FabricModel):
    """Intrinsic rank-independent FFN profile for one Instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_config_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the source model config.",
    )
    payload_dtype: Annotated[torch.dtype, WithJsonSchema({"type": "string"})] = Field(
        description="Hidden-state dtype exchanged through the Fabric."
    )
    hidden_size: int = Field(ge=1, description="Hidden-state width in elements.")
    layers: tuple[InstanceFfnLayerProfile, ...] = Field(
        min_length=1,
        description="Ordered decoder FFN layers exposed by this Instance.",
    )
    decode_payload_row_capacity: int = Field(
        ge=1,
        description="Largest decode hidden-state row count admitted by this instance.",
    )
    prefill_payload_row_capacity: int = Field(
        ge=1,
        description="Largest prefill hidden-state row count admitted by this instance.",
    )
    group_sum_complete_admitted: bool = Field(
        description="Whether the instance accepts group-summed complete FFN output.",
    )

    @field_validator("payload_dtype", mode="before")
    @classmethod
    def validate_payload_dtype(cls, value: object) -> torch.dtype:
        """Restore one Torch dtype from its canonical control-plane name."""

        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            payload_dtype = getattr(torch, value, None)
            if isinstance(payload_dtype, torch.dtype) and str(payload_dtype) == f"torch.{value}":
                return payload_dtype
        raise ValueError("payload dtype must be a torch.dtype or its canonical unqualified Torch name")

    @field_serializer("payload_dtype", when_used="json")
    def serialize_payload_dtype(self, value: torch.dtype) -> str:
        """Encode one Torch dtype with its canonical control-plane name."""

        return str(value).removeprefix("torch.")

    @model_validator(mode="after")
    def validate_layers(self) -> InstanceFfnProfile:
        """Require unique Layer IDs."""

        if len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise ValueError("FFN layer ids must be unique")
        return self


class DenseFfnLayerPlan(FabricModel):
    """Generation-static realization of one Dense FFN layer."""

    kind: Literal[LayerKind.DENSE] = Field(  # ty: ignore[invalid-type-form]
        default=LayerKind.DENSE,
        description="Dense layer discriminator.",
    )
    ffnagent_indices: tuple[int, ...] = Field(
        min_length=1,
        description="Ordered FfnAgent indices forming this layer's TP group.",
    )
    local_intermediate_size: int = Field(
        ge=1,
        description="Intermediate width owned by each FfnAgent TP rank.",
    )

    @model_validator(mode="after")
    def validate_ffnagent_indices(self) -> DenseFfnLayerPlan:
        """Require one unique FfnAgent per TP rank."""

        if len(set(self.ffnagent_indices)) != len(self.ffnagent_indices):
            raise ValueError("FFN layer execution group must not repeat an FfnAgent")
        if any(index < 0 for index in self.ffnagent_indices):
            raise ValueError("FfnAgent indices must be nonnegative")
        return self


class MoeFfnLayerPlan(FabricModel):
    """Generation-static realization of one MoE FFN layer."""

    kind: Literal[LayerKind.MOE] = Field(  # ty: ignore[invalid-type-form]
        default=LayerKind.MOE,
        description="MoE layer discriminator.",
    )
    ffnagent_indices: tuple[int, ...] = Field(
        min_length=1,
        description="Ordered FfnAgent indices forming this layer's TP group.",
    )
    local_intermediate_size: int = Field(
        ge=1,
        description="Intermediate width owned by each FfnAgent TP rank and Expert.",
    )
    effective_topk: int = Field(ge=1, description="Total routed and always-selected Expert slots per row.")

    @model_validator(mode="after")
    def validate_realization(self) -> MoeFfnLayerPlan:
        """Require unique TP members and valid final routing width."""

        if len(set(self.ffnagent_indices)) != len(self.ffnagent_indices):
            raise ValueError("FFN layer execution group must not repeat an FfnAgent")
        if any(index < 0 for index in self.ffnagent_indices):
            raise ValueError("FfnAgent indices must be nonnegative")
        return self


type FfnLayerPlan = Annotated[
    DenseFfnLayerPlan | MoeFfnLayerPlan,
    Field(discriminator="kind"),
]


class FfnModelPlan(FabricModel):
    """Placed FFN realization paired with one Instance Plan."""

    model_spec_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the generation-independent FFN Model Spec.",
    )
    layers: tuple[FfnLayerPlan, ...] = Field(
        min_length=1,
        description="Ordered placed FFN layer realizations.",
    )

    @property
    def tp_size(self) -> int:
        """Return the common Layer Execution Group width."""

        return len(self.layers[0].ffnagent_indices)

    @model_validator(mode="after")
    def validate_layer_groups(self) -> FfnModelPlan:
        """Require one common TP width across all layers."""

        if any(len(layer.ffnagent_indices) != self.tp_size for layer in self.layers):
            raise ValueError("all FFN Model Plan layers must use one TP width")
        return self


class InstanceRankTopology(FabricModel):
    """Attention rank topology and concrete AtnAgent membership."""

    atn_tp_size: int = Field(ge=1, description="Attention tensor-parallel world size.")
    atn_dp_size: int = Field(ge=1, description="Attention data-parallel world size.")
    atnagent_indices: tuple[int, ...] = Field(
        min_length=1,
        description="TP-fastest ordered AtnAgent membership.",
    )

    @model_validator(mode="after")
    def validate_membership(self) -> InstanceRankTopology:
        """Require exact TP-fastest topology membership."""

        if self.atn_tp_size > 1 and self.atn_dp_size > 1:
            raise ValueError("fabric does not support combined attention TP-by-DP")
        if len(self.atnagent_indices) != self.atn_tp_size * self.atn_dp_size:
            raise ValueError("AtnAgent membership does not match attention topology")
        if len(set(self.atnagent_indices)) != len(self.atnagent_indices):
            raise ValueError("AtnAgent membership must not contain duplicates")
        if any(index < 0 for index in self.atnagent_indices):
            raise ValueError("AtnAgent indices must be nonnegative")
        return self


class FabricInstancePlan(FabricModel):
    """One Instance FFN profile and its attention topology."""

    instance_id: str = Field(min_length=1, description="Configured model instance identity.")
    ffn_profile: InstanceFfnProfile = Field(description="Rank-independent FFN profile for this instance.")
    instance_rank_topology: InstanceRankTopology = Field(
        description="Attention rank topology and AtnAgent membership for this instance.",
    )


class FabricRole(StrEnum):
    """Agent role represented in the deterministic Fabric PE map."""

    ATNAGENT = "atnagent"
    FFNAGENT = "ffnagent"


class FabricPePlacement(FabricModel):
    """One Fabric PE placement; tuple position is the PE."""

    role: FabricRole = Field(description="Agent role assigned to this Fabric PE.")
    cuda_device: int = Field(ge=0, description="Process-visible CUDA device assigned to this Fabric PE.")

    @classmethod
    def validate_order(cls, placements: tuple[FabricPePlacement, ...]) -> tuple[int, int]:
        """Validate the canonical AtnAgent prefix and FfnAgent suffix."""

        atnagent_count = sum(placement.role is FabricRole.ATNAGENT for placement in placements)
        ffnagent_count = len(placements) - atnagent_count
        if atnagent_count == 0 or ffnagent_count == 0:
            raise ValueError("fabric placements require both AtnAgent and FfnAgent PEs")
        expected = (FabricRole.ATNAGENT,) * atnagent_count + (FabricRole.FFNAGENT,) * ffnagent_count
        if tuple(placement.role for placement in placements) != expected:
            raise ValueError("fabric placements must use an AtnAgent prefix and FfnAgent suffix")
        return atnagent_count, ffnagent_count


class FifoSchedulerPolicy(FabricModel):
    """Immutable FIFO scheduler policy for one Fabric generation."""

    policy: Literal[FfnSchedulingPolicy.FIFO] = Field(
        default=FfnSchedulingPolicy.FIFO,
        description="FIFO scheduler discriminator.",
    )


class RandomSchedulerPolicy(FabricModel):
    """Immutable deterministic Random scheduler policy for one Generation."""

    policy: Literal[FfnSchedulingPolicy.RANDOM] = Field(
        default=FfnSchedulingPolicy.RANDOM,
        description="Deterministic Random scheduler discriminator.",
    )
    seed: int = Field(ge=1, le=UINT64_MAX, description="Generation-static nonzero scheduler seed.")


type FfnSchedulerPolicy = Annotated[
    FifoSchedulerPolicy | RandomSchedulerPolicy,
    Field(discriminator="policy"),
]


class FabricPlan(FabricModel):
    """Complete semantic execution plan for one Fabric Generation."""

    generation: FabricGenerationId = Field(description="Daemon-authoritative generation identity.")
    uid: FabricUid = Field(description="NVSHMEM unique id used to bootstrap the generation.")
    pe_placements: tuple[FabricPePlacement, ...] = Field(
        min_length=2,
        description="Canonical AtnAgent-prefix and FfnAgent-suffix PE map.",
    )
    executor_lane_count: int = Field(ge=1, description="Executor lanes allocated per FfnAgent.")
    scheduler: FfnSchedulerPolicy = Field(description="Generation-static FFN scheduling policy.")
    model_plans: tuple[FfnModelPlan, ...] = Field(
        min_length=1,
        description="Placed FFN Model Plans co-indexed with Instance Plans.",
    )
    instance_plans: tuple[FabricInstancePlan, ...] = Field(
        min_length=1,
        description="Instance Plans co-indexed with FFN Model Plans.",
    )

    @model_validator(mode="after")
    def validate_plan(self) -> FabricPlan:
        """Require exact co-indexing, topology, and placed Layer membership."""

        atnagent_count, ffnagent_count = FabricPePlacement.validate_order(self.pe_placements)
        if len(self.model_plans) != len(self.instance_plans):
            raise ValueError("Model Plans and Instance Plans must be co-indexed")
        instance_ids = tuple(plan.instance_id for plan in self.instance_plans)
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("Instance IDs must be unique")

        for model_plan, instance_plan in zip(self.model_plans, self.instance_plans, strict=True):
            profile = instance_plan.ffn_profile
            topology = instance_plan.instance_rank_topology
            if any(index >= atnagent_count for index in topology.atnagent_indices):
                raise ValueError("Instance topology contains an out-of-range AtnAgent index")
            if any(index >= ffnagent_count for layer in model_plan.layers for index in layer.ffnagent_indices):
                raise ValueError("FFN Layer Execution Group contains an out-of-range FfnAgent index")
            if len(model_plan.layers) != len(profile.layers):
                raise ValueError("Model Plan and Instance FFN Profile layer counts differ")
            if any(
                planned.kind is not declared.kind
                for planned, declared in zip(model_plan.layers, profile.layers, strict=True)
            ):
                raise ValueError("Model Plan and Instance FFN Profile layer kinds differ")
        return self
