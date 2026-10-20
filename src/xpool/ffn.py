"""Generation-independent FFN model semantics."""

from __future__ import annotations

import hashlib
import json
import math
from enum import IntEnum
from typing import Annotated, Literal

import torch
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
    "RouterAlgorithm",
    "RouterPrecision",
]


class ActivationKind(IntEnum):
    """Gated FFN activation semantics."""

    SILU = 1


class RouterAlgorithm(IntEnum):
    """Activated scoring rule, independent of projection precision."""

    SOFTMAX = 1
    SIGMOID = 2


class RouterPrecision(IntEnum):
    """Tensor dtypes used by Router projection inputs, weights and logits."""

    PAYLOAD = 1
    FP32 = 2
    FP32_LOGITS = 3

    def weight_dtype(self, *, payload_dtype: torch.dtype) -> torch.dtype:
        """Resolve retained Router weights for an admitted payload dtype."""

        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Router requires BF16 or FP16 payloads")
        return torch.float32 if self is RouterPrecision.FP32 else payload_dtype


class FfnModel(BaseModel):
    """Strict immutable base for generation-independent FFN values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GatedFfnCheckpointKeys(FfnModel):
    """Checkpoint keys for one gated FFN projection triplet."""

    gate_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the gate projection.")
    up_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the up projection.")
    down_weight_key: str = Field(strict=True, min_length=1, description="Safetensors key for the down projection.")

    def tensor_keys(self) -> tuple[str, ...]:
        """Return checkpoint keys in gate, up, down order."""

        return self.gate_weight_key, self.up_weight_key, self.down_weight_key


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

    def tensor_keys(self) -> tuple[str, ...]:
        """Return Router, optional bias and ordered routed/shared Expert keys."""

        keys = [self.router_weight_key]
        if self.router_correction_bias_key is not None:
            keys.append(self.router_correction_bias_key)
        for expert in self.routed_experts:
            keys.extend(expert.tensor_keys())
        if self.shared_expert is not None:
            keys.extend(self.shared_expert.tensor_keys())
        return tuple(keys)


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
    router_algorithm: RouterAlgorithm = Field(description="Router activated scoring rule selected during compilation.")
    router_precision: RouterPrecision = Field(description="Router projection precision selected during compilation.")
    expert_intermediate_size: int = Field(strict=True, ge=1, description="Intermediate width of one Expert.")
    shared_expert_count: int = Field(strict=True, ge=0, description="Logical always-selected shared Expert count.")
    routed_topk: int = Field(strict=True, ge=1, description="Routed Expert count selected per hidden-state row.")
    renormalize: bool = Field(strict=True, description="Whether selected routed weights are normalized to sum to one.")
    routed_scaling_factor: float = Field(gt=0, description="Scale applied to selected routed weights.")
    checkpoint: MoeFfnCheckpointKeys = Field(description="Router and Expert checkpoint keys consumed by this layer.")

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
        if (self.shared_expert_count == 0) != (self.checkpoint.shared_expert is None):
            raise ValueError("shared Expert count and checkpoint triplet disagree")
        match self.router_algorithm, self.router_precision:
            case RouterAlgorithm.SOFTMAX, RouterPrecision.PAYLOAD | RouterPrecision.FP32_LOGITS:
                pass
            case RouterAlgorithm.SIGMOID, RouterPrecision.FP32:
                if not self.renormalize:
                    raise ValueError("Sigmoid routing requires normalization")
            case _:
                raise ValueError("unsupported Router algorithm and precision combination")
        return self

    @property
    def routed_expert_count(self) -> int:
        """Return the number of routed Experts from tuple identity."""

        return len(self.checkpoint.routed_experts)


type FfnLayerSpec = Annotated[
    DenseFfnSpec | MoeFfnSpec,
    Field(discriminator="kind"),
]


class FfnModelSpec(FfnModel):
    """Complete model-source-intrinsic FFN semantics."""

    model_id: ModelId = Field(description="Canonical configured model identity.")
    hidden_size: int = Field(strict=True, ge=1, description="Model hidden-state width in elements.")
    activation: ActivationKind = Field(description="Gated activation shared by all model FFN layers.")
    layers: tuple[FfnLayerSpec, ...] = Field(
        min_length=1,
        description="Ordered decoder FFN layer specifications.",
    )

    @model_validator(mode="after")
    def validate_model_semantics(self) -> FfnModelSpec:
        """Require unique layer identities and checkpoint keys."""

        layer_ids = tuple(layer.layer_id for layer in self.layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("FFN Model Spec layer ids must be unique")
        keys = tuple(key for layer in self.layers for key in layer.checkpoint.tensor_keys())
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
