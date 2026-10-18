"""Generation-independent FFN model semantics."""

from __future__ import annotations

import hashlib
import json
import math
from enum import IntEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import xpool.native
from xpool.model import ModelId

__all__ = [
    "ActivationKind",
    "DenseFfnSpec",
    "FfnLayerSpec",
    "FfnModelSpec",
    "GatedFfnCheckpointKeys",
    "MoeFfnCheckpointKeys",
    "MoeFfnSpec",
    "Mxfp4MoeCheckpointKeys",
]


class ActivationKind(IntEnum):
    """Gated FFN activation semantics."""

    SILU = 1
    SWIGLU_OAI = 2


class FfnModel(BaseModel):
    """Strict immutable base for generation-independent FFN values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GatedFfnCheckpointKeys(FfnModel):
    """Checkpoint keys for one gated FFN projection triplet."""

    gate_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the gate projection.")
    up_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the up projection.")
    down_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the down projection.")


class MoeFfnCheckpointKeys(FfnModel):
    """Checkpoint keys for one MoE layer and its ordered Experts."""

    router_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the Router weight.")
    router_correction_bias_key: (
        Annotated[
            str,
            Field(strict=True, min_length=1, description="Safetensors key for the optional Router correction bias."),
        ]
        | None
    )
    routed_experts: tuple[GatedFfnCheckpointKeys, ...] = Field(
        min_length=1,
        description="Routed Expert projection keys in Expert-id order.",
    )
    shared_expert: GatedFfnCheckpointKeys | None = Field(
        description="Optional wide shared-Expert projection keys.",
    )


class Mxfp4MoeCheckpointKeys(FfnModel):
    """Fused MXFP4 Expert matrices with explicit scales and additive biases."""

    format: Literal["mxfp4"] = "mxfp4"
    expert_count: int = Field(strict=True, ge=1, description="Number of packed Experts.")
    router_weight_key: str = Field(min_length=1, description="Unquantized Router projection key.")
    router_bias_key: str = Field(min_length=1, description="Additive Router bias key.")
    gate_up_blocks_key: str = Field(min_length=1, description="Interleaved gate/up packed E2M1 key.")
    gate_up_scales_key: str = Field(min_length=1, description="Gate/up UE8M0 block-scale key.")
    gate_up_bias_key: str = Field(min_length=1, description="Interleaved gate/up bias key.")
    down_blocks_key: str = Field(min_length=1, description="Down packed E2M1 key.")
    down_scales_key: str = Field(min_length=1, description="Down UE8M0 block-scale key.")
    down_bias_key: str = Field(min_length=1, description="Down additive bias key.")

    @property
    def router_correction_bias_key(self) -> None:
        """MXFP4 additive Router bias is distinct from selection correction."""

        return None


class DenseFfnSpec(FfnModel):
    """Intrinsic semantics and checkpoint identity of one gated Dense layer."""

    kind: Literal[xpool.native.ffn.LayerKind.DENSE] = Field(  # ty: ignore[invalid-type-form]
        description="Dense layer discriminator."
    )
    layer_id: int = Field(strict=True, ge=0, description="Model-local decoder layer identity.")
    intermediate_size: int = Field(strict=True, ge=1, description="Full Dense intermediate width.")
    checkpoint: GatedFfnCheckpointKeys = Field(description="Checkpoint projections consumed by this layer.")


class MoeFfnSpec(FfnModel):
    """Intrinsic semantics and checkpoint identity of one gated MoE layer."""

    kind: Literal[xpool.native.ffn.LayerKind.MOE] = Field(  # ty: ignore[invalid-type-form]
        description="MoE layer discriminator."
    )
    layer_id: int = Field(strict=True, ge=0, description="Model-local decoder layer identity.")
    expert_intermediate_size: int = Field(strict=True, ge=1, description="Intermediate width of one Expert.")
    shared_expert_count: int = Field(strict=True, ge=0, description="Logical always-selected shared Expert count.")
    routed_topk: int = Field(strict=True, ge=1, description="Routed Expert count selected per hidden-state row.")
    renormalize: bool = Field(strict=True, description="Whether selected routed weights are normalized to sum to one.")
    routed_scaling_factor: float = Field(gt=0, description="Scale applied to selected routed weights.")
    checkpoint: MoeFfnCheckpointKeys | Mxfp4MoeCheckpointKeys = Field(
        description="Router and Expert checkpoint keys consumed by this layer."
    )

    @field_validator("routed_scaling_factor", mode="before")
    @classmethod
    def validate_routed_scaling_factor(cls, value: object) -> object:
        """Reject Boolean and non-finite routing scales before coercion."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("routed_scaling_factor must be a finite positive JSON number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive")
        return value

    @model_validator(mode="after")
    def validate_routing(self) -> MoeFfnSpec:
        """Validate Expert cardinality, grouping, and optional resources."""

        routed_expert_count = self.routed_expert_count
        if self.routed_topk > routed_expert_count:
            raise ValueError("routed_topk cannot exceed routed Expert count")
        if isinstance(self.checkpoint, Mxfp4MoeCheckpointKeys):
            if self.shared_expert_count:
                raise ValueError("MXFP4 Experts do not admit shared Experts")
        elif (self.shared_expert_count == 0) != (self.checkpoint.shared_expert is None):
            raise ValueError("shared Expert count and checkpoint triplet disagree")
        return self

    @property
    def routed_expert_count(self) -> int:
        """Return the number of routed Experts from tuple identity."""

        if isinstance(self.checkpoint, Mxfp4MoeCheckpointKeys):
            return self.checkpoint.expert_count
        return len(self.checkpoint.routed_experts)


type FfnLayerSpec = Annotated[
    DenseFfnSpec | MoeFfnSpec,
    Field(discriminator="kind"),
]


class FfnModelSpec(FfnModel):
    """Complete model-source-intrinsic FFN semantics."""

    model_id: ModelId = Field(description="Canonical configured model identity.")
    architecture_name: str = Field(strict=True, min_length=1, description="Selected FFN Model Adapter identity.")
    hidden_size: int = Field(strict=True, ge=1, description="Model hidden-state width in elements.")
    activation: ActivationKind = Field(description="Gated activation shared by all model FFN layers.")
    activation_alpha: float = Field(default=1.0, gt=0, allow_inf_nan=False, description="Gated sigmoid slope.")
    activation_limit: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Activation clamp limit."
    )
    layers: tuple[FfnLayerSpec, ...] = Field(
        min_length=1,
        description="Ordered decoder FFN layer specifications.",
    )

    @model_validator(mode="after")
    def validate_model_semantics(self) -> FfnModelSpec:
        """Require unique layer identities and checkpoint keys."""

        if self.activation is ActivationKind.SILU:
            if self.activation_alpha != 1.0 or self.activation_limit is not None:
                raise ValueError("SiLU requires default activation parameters")
            if any(
                isinstance(layer, MoeFfnSpec) and isinstance(layer.checkpoint, Mxfp4MoeCheckpointKeys)
                for layer in self.layers
            ):
                raise ValueError("MXFP4 Experts require OpenAI SwiGLU")
        elif self.activation_limit is None or any(
            not isinstance(layer, MoeFfnSpec) or not isinstance(layer.checkpoint, Mxfp4MoeCheckpointKeys)
            for layer in self.layers
        ):
            raise ValueError("OpenAI SwiGLU requires MXFP4 MoE layers and a clamp limit")
        layer_ids = tuple(layer.layer_id for layer in self.layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("FFN Model Spec layer ids must be unique")
        keys = tuple(key for layer in self.layers for key in checkpoint_keys_for_layer(layer))
        if len(set(keys)) != len(keys):
            raise ValueError("FFN Model Spec checkpoint keys must be globally unique")
        return self

    def digest(self) -> str:
        """Return the canonical SHA-256 identity of every stored Spec field."""

        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def checkpoint_keys_for_layer(layer: FfnLayerSpec) -> tuple[str, ...]:
    """Return one layer's checkpoint keys in canonical semantic order."""

    if isinstance(layer, DenseFfnSpec):
        return (
            layer.checkpoint.gate_weight_key,
            layer.checkpoint.up_weight_key,
            layer.checkpoint.down_weight_key,
        )

    if isinstance(layer.checkpoint, Mxfp4MoeCheckpointKeys):
        checkpoint = layer.checkpoint
        return (
            checkpoint.router_weight_key,
            checkpoint.router_bias_key,
            checkpoint.gate_up_blocks_key,
            checkpoint.gate_up_scales_key,
            checkpoint.gate_up_bias_key,
            checkpoint.down_blocks_key,
            checkpoint.down_scales_key,
            checkpoint.down_bias_key,
        )
    keys = [layer.checkpoint.router_weight_key]
    if layer.checkpoint.router_correction_bias_key is not None:
        keys.append(layer.checkpoint.router_correction_bias_key)
    experts = list(layer.checkpoint.routed_experts)
    if layer.checkpoint.shared_expert is not None:
        experts.append(layer.checkpoint.shared_expert)
    for expert in experts:
        keys.extend((expert.gate_weight_key, expert.up_weight_key, expert.down_weight_key))
    return tuple(keys)
