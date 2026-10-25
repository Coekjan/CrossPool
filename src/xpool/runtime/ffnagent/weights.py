"""Canonical FfnAgent-owned true-TP weight tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import xpool.native
from xpool import ffn
from xpool.quantization import TensorLayout


def validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    dimensions: int,
) -> None:
    """Validate intrinsic placement and storage facts for one retained weight."""

    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be CUDA-resident")
    if tensor.dtype is not dtype:
        raise ValueError(f"{name} must use {dtype}")
    if tensor.ndim != dimensions or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} must have {dimensions} positive dimensions")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def validate_weight_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    """Validate intrinsic storage ownership, independent of the prepared layout."""

    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be CUDA-resident")
    if not tensor.is_contiguous() or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} must have contiguous positive dimensions")
    storage = tensor.untyped_storage()
    if tensor.storage_offset() != 0 or tensor.data_ptr() != storage.data_ptr() or tensor.nbytes != storage.nbytes():
        raise ValueError(f"{name} must own its exact CUDA storage")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


@dataclass(frozen=True, slots=True)
class ProjectionLayout:
    """Final physical allocation order and neutral initialization for a projection.

    Values, optional scales and optional additive bias are independent storages.
    The selected operation determines the bias dtype; the encoding determines
    value/scale layouts. This description retains no Tensor or device state.
    """

    values: TensorLayout
    scales: TensorLayout | None
    bias: TensorLayout | None

    def allocate(self, device: torch.device) -> ProjectionWeights:
        """Allocate uninitialized values, scales and bias in their declared order.

        The loading owner fills every source and neutral padding region before
        execution. These storages stay alive under the layer's weight owner.
        """

        return ProjectionWeights(
            values=torch.empty(self.values.shape, dtype=self.values.dtype, device=device),
            scales=None
            if self.scales is None
            else torch.empty(self.scales.shape, dtype=self.scales.dtype, device=device),
            bias=None if self.bias is None else torch.empty(self.bias.shape, dtype=self.bias.dtype, device=device),
        )

    def zero_probe(self, device: torch.device) -> ProjectionWeights:
        """Allocate shape-equivalent, encoding-correct zero resources for capture."""

        return ProjectionWeights(
            values=torch.full(self.values.shape, self.values.zero_value, dtype=self.values.dtype, device=device),
            scales=None
            if self.scales is None
            else torch.full(
                self.scales.shape,
                self.scales.zero_value,
                dtype=self.scales.dtype,
                device=device,
            ),
            bias=None
            if self.bias is None
            else torch.full(
                self.bias.shape,
                self.bias.zero_value,
                dtype=self.bias.dtype,
                device=device,
            ),
        )

    def allocation_bytes(self) -> tuple[int, ...]:
        """Return positive Tensor byte requests in the actual allocation order."""

        return tuple(
            math.prod(layout.shape) * layout.dtype.itemsize
            for layout in (self.values, self.scales, self.bias)
            if layout is not None
        )


@dataclass(frozen=True, slots=True)
class ProjectionWeights:
    """Independently owned projection storages borrowed by native binding views."""

    values: torch.Tensor
    scales: torch.Tensor | None
    bias: torch.Tensor | None

    def __post_init__(self) -> None:
        validate_weight_tensor(self.values, name="projection values")
        for name, tensor in (("scales", self.scales), ("bias", self.bias)):
            if tensor is None:
                continue
            validate_weight_tensor(tensor, name=f"projection {name}")
            if tensor.device != self.values.device:
                raise ValueError("projection resources must share one device")

    def validate(self, layout: ProjectionLayout) -> None:
        """Require exact resource presence, shape and dtype against preparation."""

        for name, tensor, expected in (
            ("values", self.values, layout.values),
            ("scales", self.scales, layout.scales),
            ("bias", self.bias, layout.bias),
        ):
            if (tensor is None) != (expected is None):
                raise ValueError(f"projection {name} presence disagrees with its prepared layout")
            if tensor is not None and expected is not None:
                if tuple(tensor.shape) != expected.shape or tensor.dtype != expected.dtype:
                    raise ValueError(f"projection {name} shape or dtype disagrees with its prepared layout")

    def binding_view(self) -> xpool.native.ffnagent.ProjectionBindingView:
        """Borrow addresses and byte extents while this owner retains the Tensors."""

        return xpool.native.ffnagent.ProjectionBindingView(values=self.values, scales=self.scales, bias=self.bias)


@dataclass(frozen=True, slots=True)
class ExpertLayout:
    """Capacity-independent physical resources for the two Expert projections."""

    gate_up: ProjectionLayout
    down: ProjectionLayout

    @classmethod
    def from_layer(
        cls,
        layer: ffn.MoeFfnSpec,
        *,
        hidden_size: int,
        local_intermediate_size: int,
        payload_dtype: torch.dtype,
        bias_dtype: torch.dtype,
    ) -> ExpertLayout:
        """Compose encoded Expert geometry with operation-owned bias storage.

        Sources in each group must share an encoding and its scale presence.
        This pure construction owns no Tensor and performs no device query.
        """

        gate_up_sources, down_sources = layer.expert_projection_sources()
        expert_count = layer.routed_expert_count + layer.shared_expert_count
        projections = []
        for sources, shape, bias_shape in (
            (
                gate_up_sources,
                (expert_count, 2 * local_intermediate_size, hidden_size),
                (expert_count, 2 * local_intermediate_size),
            ),
            (down_sources, (expert_count, hidden_size, local_intermediate_size), (expert_count, hidden_size)),
        ):
            scheme = sources[0].quantization
            if any(source.quantization != scheme for source in sources):
                raise ValueError("Expert projection group requires a homogeneous encoding")
            encoded = scheme.weight_layout(logical_shape=shape, dtype=payload_dtype)
            if any((source.scales_key is not None) != (encoded.scales is not None) for source in sources):
                raise ValueError("Expert projection scale keys disagree with the encoding")
            projections.append(
                ProjectionLayout(
                    values=encoded.values,
                    scales=encoded.scales,
                    bias=TensorLayout(bias_shape, bias_dtype, 0)
                    if any(source.bias_key is not None for source in sources)
                    else None,
                )
            )
        return cls(gate_up=projections[0], down=projections[1])


@dataclass(frozen=True, slots=True)
class DenseFfnWeights:
    """Retained gate/up and down resources for one local Dense shard."""

    gate_up: ProjectionWeights
    down: ProjectionWeights

    @classmethod
    def allocate(
        cls,
        *,
        gate_up: ProjectionLayout,
        down: ProjectionLayout,
        device: torch.device,
    ) -> DenseFfnWeights:
        """Allocate neutral resources in the prepared projection order.

        Neutral initialization also covers missing source biases and padding.
        Loading overwrites the valid source intervals without another weight copy.
        """

        return cls(gate_up=gate_up.zero_probe(device), down=down.zero_probe(device))

    def __post_init__(self) -> None:
        if self.gate_up.values.ndim != 2 or self.down.values.ndim != 2:
            raise ValueError("Dense projections require matrix values")
        if self.down.values.device != self.gate_up.values.device:
            raise ValueError("Dense projections must share one device")

    def binding_view(self) -> xpool.native.ffnagent.DenseBindingView:
        """Borrow complete projection addresses while this owner retains storage."""

        return xpool.native.ffnagent.DenseBindingView(
            gate_up=self.gate_up.binding_view(),
            down=self.down.binding_view(),
        )


@dataclass(frozen=True, slots=True)
class MoeRouterWeights:
    """Retained Router projection and separate selection-only correction."""

    projection: ProjectionWeights
    correction_bias: torch.Tensor | None

    def __post_init__(self) -> None:
        if self.projection.values.ndim != 2:
            raise ValueError("Router projection requires matrix values")
        if self.correction_bias is not None:
            validate_weight_tensor(self.correction_bias, name="router correction bias")
            if self.correction_bias.dtype is not torch.float32 or self.correction_bias.ndim != 1:
                raise ValueError("Router correction bias requires a one-dimensional FP32 Tensor")
            if self.correction_bias.shape != self.projection.values.shape[:1]:
                raise ValueError("Router correction bias cardinality disagrees with its projection")
            if self.correction_bias.device != self.projection.values.device:
                raise ValueError("Router resources must share one device")


@dataclass(frozen=True, slots=True)
class MoeFfnWeights:
    """Retained Expert projections and optional TP Router ownership."""

    gate_up: ProjectionWeights
    down: ProjectionWeights
    router: MoeRouterWeights | None

    @classmethod
    def allocate(
        cls,
        *,
        gate_up: ProjectionLayout,
        down: ProjectionLayout,
        router: ProjectionLayout | None,
        correction_bias: TensorLayout | None,
        device: torch.device,
    ) -> MoeFfnWeights:
        """Allocate neutral Expert and Router storages in prepared ledger order."""

        if router is None and correction_bias is not None:
            raise ValueError("Router correction requires an owned Router projection")
        gate_up_weights = gate_up.zero_probe(device)
        down_weights = down.zero_probe(device)
        router_weights = None
        if router is not None:
            projection = router.zero_probe(device)
            correction = (
                torch.full(
                    correction_bias.shape, correction_bias.zero_value, dtype=correction_bias.dtype, device=device
                )
                if correction_bias is not None
                else None
            )
            router_weights = MoeRouterWeights(projection=projection, correction_bias=correction)
        return cls(gate_up=gate_up_weights, down=down_weights, router=router_weights)

    def __post_init__(self) -> None:
        if self.gate_up.values.ndim != 3 or self.down.values.ndim != 3:
            raise ValueError("Expert projections require Expert-major matrix values")
        if self.gate_up.values.shape[0] != self.down.values.shape[0]:
            raise ValueError("Expert projections must have matching cardinality")
        if self.down.values.device != self.gate_up.values.device:
            raise ValueError("Expert projections must share one device")
        if self.router is not None and self.router.projection.values.device != self.gate_up.values.device:
            raise ValueError("Expert and Router resources must share one device")

    def binding_view(self) -> xpool.native.ffnagent.MoeBindingView:
        """Borrow complete Expert and Router addresses without transferring owners."""

        return xpool.native.ffnagent.MoeBindingView(
            gate_up=self.gate_up.binding_view(),
            down=self.down.binding_view(),
            router=None
            if self.router is None
            else xpool.native.ffnagent.MoeRouterBindingView(
                projection=self.router.projection.binding_view(),
                correction_bias=self.router.correction_bias,
            ),
        )


type FfnLayerWeights = DenseFfnWeights | MoeFfnWeights
