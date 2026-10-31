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
from xpool.quantization import QuantizationSpec, TensorSlice

__all__ = [
    "ActivationKind",
    "ActivationSpec",
    "DenseFfnSpec",
    "FfnLayerSpec",
    "FfnModelSpec",
    "GateUpOrder",
    "GatedProjectionSpec",
    "MoeFfnSpec",
    "PerExpertSource",
    "ProjectionSpec",
    "ReluActivation",
    "RoutedExpertSource",
    "RouterAlgorithm",
    "RouterProjectionPrecision",
    "SiluActivation",
    "StackedExpertSource",
    "TopKWeightRounding",
    "UnaryProjectionSpec",
    "ValuesSourceLayout",
]


class ActivationKind(IntEnum):
    """Dense FFN activation semantics."""

    SILU = 1
    RELU = 2


class GateUpOrder(IntEnum):
    """Physical ordering of the gate and up projection rows."""

    CONCATENATED = 1
    INTERLEAVED = 2

    def row_slices(self, *, axis: int, width: int, start: int, length: int) -> tuple[TensorSlice, TensorSlice]:
        """Select Gate then Up from prepared geometry, counting semantic rows.

        Width is the intermediate width of this source or destination, including
        destination padding. The preparation owner establishes interval bounds.
        """

        match self:
            case GateUpOrder.CONCATENATED:
                return TensorSlice(axis, start, length), TensorSlice(axis, width + start, length)
            case GateUpOrder.INTERLEAVED:
                return TensorSlice(axis, 2 * start, length, step=2), TensorSlice(axis, 2 * start + 1, length, step=2)


class ValuesSourceLayout(IntEnum):
    """Checkpoint axes preceding the scheme's canonical matrix view."""

    MATRIX = 1
    SCALE_BLOCKED = 2


class RouterAlgorithm(IntEnum):
    """Activated scoring rule, independent of projection precision."""

    SOFTMAX = 1
    SIGMOID = 2


class RouterProjectionPrecision(IntEnum):
    """Tensor dtypes used by Router projection inputs, weights and logits."""

    PAYLOAD = 1
    FP32 = 2
    FP32_LOGITS = 3

    def weight_dtype(self, *, payload_dtype: torch.dtype) -> torch.dtype:
        """Resolve retained Router weights for an admitted payload dtype."""

        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Router requires BF16 or FP16 payloads")
        return torch.float32 if self is RouterProjectionPrecision.FP32 else payload_dtype


class TopKWeightRounding(IntEnum):
    """Additional rounding of normalized routed-Expert mixing coefficients.

    ``NONE`` preserves computed FP32 values. ``LOGITS_DTYPE`` rounds to the
    projection logits' element type before widening into the FP32 carrier.
    Shared Expert coefficients are appended after this rounding boundary.
    """

    NONE = 1
    LOGITS_DTYPE = 2


class FfnModel(BaseModel):
    """Strict immutable base for generation-independent FFN values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SiluActivation(FfnModel):
    """Combine biased gate/up projections with parameterized gated SiLU.

    Optional clipping bounds gate above and up on both sides before computing
    ``g * sigmoid(alpha * g) * (u + up_offset)``. Ordinary gated SiLU uses the
    defaults; model adapters declare their own slope, clipping and Up offset.
    """

    kind: Literal[ActivationKind.SILU] = ActivationKind.SILU
    alpha: float = Field(default=1.0, strict=True, gt=0, allow_inf_nan=False)
    limit: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)
    up_offset: float = Field(default=0.0, strict=True, allow_inf_nan=False)


class ReluActivation(FfnModel):
    """Apply an elementwise ReLU to the biased fc1 projection result.

    This is the unary Dense FFN activation: one input projection, one
    nonlinearity and one output projection. It consumes no gate/up pair.
    """

    kind: Literal[ActivationKind.RELU] = ActivationKind.RELU


type ActivationSpec = SiluActivation | ReluActivation


class ProjectionSpec(FfnModel):
    """Weight encoding and physical checkpoint keys for ``X @ W.T + bias``."""

    quantization: QuantizationSpec
    values_key: str = Field(strict=True, min_length=1)
    scales_key: str | None = Field(default=None, strict=True, min_length=1)
    bias_key: str | None = Field(default=None, strict=True, min_length=1)
    values_layout: ValuesSourceLayout = ValuesSourceLayout.MATRIX

    @model_validator(mode="after")
    def validate_values_layout(self) -> ProjectionSpec:
        """Require the declared scale groups used by a blocked values source."""

        if self.values_layout is ValuesSourceLayout.SCALE_BLOCKED and self.scales_key is None:
            raise ValueError("scale-blocked source values require a scales key")
        return self

    def tensor_keys(self) -> tuple[str, ...]:
        """Return values, optional scales and optional additive-bias keys."""

        return tuple(key for key in (self.values_key, self.scales_key, self.bias_key) if key is not None)


class GatedProjectionSpec(FfnModel):
    """Separate gate, up and down projection sources for one gated FFN."""

    gate: ProjectionSpec
    up: ProjectionSpec
    down: ProjectionSpec

    def tensor_keys(self) -> tuple[str, ...]:
        """Return each projection's physical keys in gate, up, down order."""

        return self.gate.tensor_keys() + self.up.tensor_keys() + self.down.tensor_keys()


class UnaryProjectionSpec(FfnModel):
    """Separate input and output projection sources for one unary FFN.

    The input projection is executed as ``fc1`` and the output projection as
    ``fc2``. Both projections carry their own optional additive bias.
    """

    kind: Literal["unary"] = "unary"
    fc1: ProjectionSpec
    fc2: ProjectionSpec

    def tensor_keys(self) -> tuple[str, ...]:
        """Return each projection's physical keys in fc1, fc2 order."""

        return self.fc1.tensor_keys() + self.fc2.tensor_keys()


class PerExpertSource(FfnModel):
    """Independent projection keys in routed Expert-id order."""

    kind: Literal["per_expert"] = "per_expert"
    experts: tuple[GatedProjectionSpec, ...] = Field(min_length=1)

    def tensor_keys(self) -> tuple[str, ...]:
        """Return each Expert's physical keys in semantic Expert order."""

        return tuple(key for expert in self.experts for key in expert.tensor_keys())


class StackedExpertSource(FfnModel):
    """Expert-major fused gate/up and down sources, each counted once."""

    kind: Literal["stacked"] = "stacked"
    gate_up: ProjectionSpec
    down: ProjectionSpec
    gate_up_order: GateUpOrder = GateUpOrder.CONCATENATED

    def tensor_keys(self) -> tuple[str, ...]:
        """Return fused gate/up keys followed by down keys."""

        return self.gate_up.tensor_keys() + self.down.tensor_keys()


type RoutedExpertSource = Annotated[PerExpertSource | StackedExpertSource, Field(discriminator="kind")]


class DenseFfnSpec(FfnModel):
    """Intrinsic semantics and checkpoint identity of one Dense layer.

    Gated layers declare gate/up/down sources. Unary layers declare fc1/fc2
    sources and are the biased unary Dense form.
    """

    kind: Literal[xpool.native.ffn.LayerKind.DENSE] = Field(  # ty: ignore[invalid-type-form]
        description="Dense layer discriminator."
    )
    layer_id: int = Field(strict=True, ge=0, description="Model-local decoder layer identity.")
    intermediate_size: int = Field(strict=True, ge=1, description="Full Dense intermediate width.")
    projections: GatedProjectionSpec | UnaryProjectionSpec = Field(
        description="Checkpoint projections consumed by this layer."
    )

    def tensor_keys(self) -> tuple[str, ...]:
        """Return all physical checkpoint keys owned by this layer."""

        return self.projections.tensor_keys()


class MoeFfnSpec(FfnModel):
    """Intrinsic semantics and checkpoint identity of one gated MoE layer."""

    kind: Literal[xpool.native.ffn.LayerKind.MOE] = Field(  # ty: ignore[invalid-type-form]
        description="MoE layer discriminator."
    )
    layer_id: int = Field(strict=True, ge=0, description="Model-local decoder layer identity.")
    router_algorithm: RouterAlgorithm = Field(description="Router activated scoring rule selected during compilation.")
    router_projection_precision: RouterProjectionPrecision = Field(
        description="Router projection precision selected during compilation."
    )
    topk_weight_rounding: TopKWeightRounding = Field(
        default=TopKWeightRounding.NONE,
        description="Additional rounding of normalized routed-Expert coefficients before publication.",
    )
    expert_intermediate_size: int = Field(strict=True, ge=1, description="Intermediate width of one Expert.")
    routed_expert_count: int = Field(strict=True, ge=1, description="Semantic number of routed Experts.")
    shared_expert_count: int = Field(strict=True, ge=0, description="Logical always-selected shared Expert count.")
    routed_topk: int = Field(strict=True, ge=1, description="Routed Expert count selected per hidden-state row.")
    renormalize: bool = Field(strict=True, description="Whether selected routed weights are normalized to sum to one.")
    routed_scaling_factor: float = Field(gt=0, description="Scale applied to selected routed weights.")
    router: ProjectionSpec
    routed_experts: RoutedExpertSource
    shared_expert: GatedProjectionSpec | None
    router_correction_bias_key: str | None = Field(default=None, strict=True, min_length=1)

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
        if isinstance(self.routed_experts, PerExpertSource) and len(self.routed_experts.experts) != routed_expert_count:
            raise ValueError("per-Expert source count and routed Expert count disagree")
        if self.routed_topk > routed_expert_count:
            raise ValueError("routed_topk cannot exceed routed Expert count")
        if (self.shared_expert_count == 0) != (self.shared_expert is None):
            raise ValueError("shared Expert count and projection sources disagree")
        match self.router_algorithm, self.router_projection_precision:
            case RouterAlgorithm.SOFTMAX, RouterProjectionPrecision.PAYLOAD | RouterProjectionPrecision.FP32_LOGITS:
                pass
            case RouterAlgorithm.SIGMOID, RouterProjectionPrecision.FP32:
                if not self.renormalize:
                    raise ValueError("Sigmoid routing requires normalization")
            case _:
                raise ValueError("unsupported Router algorithm and precision combination")
        return self

    def expert_projection_sources(self) -> tuple[tuple[ProjectionSpec, ...], tuple[ProjectionSpec, ...]]:
        """Return gate/up and down source groups, including wide Shared Experts.

        Groups enumerate physical declarations rather than repeat stacked keys
        for every logical Expert. No Tensor or layout is constructed.
        """

        source = self.routed_experts
        if isinstance(source, StackedExpertSource):
            gate_up = (source.gate_up,)
            down = (source.down,)
        else:
            gate_up = tuple(projection for expert in source.experts for projection in (expert.gate, expert.up))
            down = tuple(expert.down for expert in source.experts)
        if self.shared_expert is not None:
            gate_up += (self.shared_expert.gate, self.shared_expert.up)
            down += (self.shared_expert.down,)
        return gate_up, down

    def tensor_keys(self) -> tuple[str, ...]:
        """Return Router, correction, routed and wide Shared Expert keys."""

        keys = self.router.tensor_keys()
        if self.router_correction_bias_key is not None:
            keys += (self.router_correction_bias_key,)
        keys += self.routed_experts.tensor_keys()
        if self.shared_expert is not None:
            keys += self.shared_expert.tensor_keys()
        return keys


type FfnLayerSpec = Annotated[
    DenseFfnSpec | MoeFfnSpec,
    Field(discriminator="kind"),
]


class FfnModelSpec(FfnModel):
    """Complete model-source-intrinsic FFN semantics."""

    model_id: ModelId = Field(description="Canonical configured model identity.")
    hidden_size: int = Field(strict=True, ge=1, description="Model hidden-state width in elements.")
    activation: ActivationSpec = Field(description="Activation shared by all model FFN layers.")
    layers: tuple[FfnLayerSpec, ...] = Field(
        min_length=1,
        description="Ordered decoder FFN layer specifications.",
    )
    checkpoint_namespace: str | None = Field(
        default=None,
        strict=True,
        description=(
            "Layer-relative checkpoint namespace template containing exactly one {layer_id} placeholder. "
            "None selects the default model.layers.{layer_id}.mlp FFN namespace."
        ),
    )

    @model_validator(mode="after")
    def validate_model_semantics(self) -> FfnModelSpec:
        """Require unique layer identities, checkpoint keys and layer semantics."""

        layer_ids = tuple(layer.layer_id for layer in self.layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("FFN Model Spec layer ids must be unique")
        keys = tuple(key for layer in self.layers for key in layer.tensor_keys())
        if len(set(keys)) != len(keys):
            raise ValueError("FFN Model Spec checkpoint keys must be globally unique")
        namespace = self.checkpoint_namespace
        if namespace is not None:
            if not namespace or namespace.count("{layer_id}") != 1:
                raise ValueError("checkpoint_namespace must contain exactly one {layer_id} placeholder")
            if "{" in namespace.replace("{layer_id}", "") or "}" in namespace.replace("{layer_id}", ""):
                raise ValueError("checkpoint_namespace must not contain other placeholders")
        unary = isinstance(self.activation, ReluActivation)
        for layer in self.layers:
            if isinstance(layer, DenseFfnSpec):
                if isinstance(layer.projections, UnaryProjectionSpec) != unary:
                    raise ValueError("Dense FFN projections must match the model activation")
            elif unary:
                raise ValueError("ReluActivation is only supported by unary Dense FFNs")
        return self

    def digest(self) -> str:
        """Return the canonical SHA-256 identity of every stored Spec field."""

        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()
