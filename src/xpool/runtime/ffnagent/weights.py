"""Canonical FfnAgent-owned true-TP weight tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import xpool.native
from xpool import ffn
from xpool.quantization import QuantizationSpec, TensorLayout


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

    quantization: QuantizationSpec
    values: TensorLayout
    scales: TensorLayout | None
    bias: TensorLayout | None
    gate_up_order: ffn.GateUpOrder | None

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

        return tuple(layout.nbytes for layout in (self.values, self.scales, self.bias) if layout is not None)


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
class DenseFfnWeights:
    """Retained input and output resources for one local Dense shard."""

    i_proj: ProjectionWeights
    o_proj: ProjectionWeights

    @classmethod
    def allocate(
        cls,
        *,
        i_proj: ProjectionLayout,
        o_proj: ProjectionLayout,
        device: torch.device,
    ) -> DenseFfnWeights:
        """Allocate neutral resources in the prepared projection order.

        Neutral initialization also covers missing source biases and padding.
        Loading overwrites the valid source intervals without another weight copy.
        """

        return cls(i_proj=i_proj.zero_probe(device), o_proj=o_proj.zero_probe(device))

    def __post_init__(self) -> None:
        if self.i_proj.values.ndim != 2 or self.o_proj.values.ndim != 2:
            raise ValueError("Dense projections require matrix values")
        if self.o_proj.values.device != self.i_proj.values.device:
            raise ValueError("Dense projections must share one device")

    def binding_view(self) -> xpool.native.ffnagent.DenseBindingView:
        """Borrow complete projection addresses while this owner retains storage."""

        return xpool.native.ffnagent.DenseBindingView(
            i_proj=self.i_proj.binding_view(),
            o_proj=self.o_proj.binding_view(),
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

    i_proj: ProjectionWeights
    o_proj: ProjectionWeights
    router: MoeRouterWeights | None

    @classmethod
    def allocate(
        cls,
        *,
        i_proj: ProjectionLayout,
        o_proj: ProjectionLayout,
        router: ProjectionLayout | None,
        correction_bias: TensorLayout | None,
        device: torch.device,
    ) -> MoeFfnWeights:
        """Allocate neutral Expert and Router storages in prepared ledger order."""

        if router is None and correction_bias is not None:
            raise ValueError("Router correction requires an owned Router projection")
        i_proj_weights = i_proj.zero_probe(device)
        o_proj_weights = o_proj.zero_probe(device)
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
        return cls(i_proj=i_proj_weights, o_proj=o_proj_weights, router=router_weights)

    def __post_init__(self) -> None:
        if self.i_proj.values.ndim != 3 or self.o_proj.values.ndim != 3:
            raise ValueError("Expert projections require Expert-major matrix values")
        if self.i_proj.values.shape[0] != self.o_proj.values.shape[0]:
            raise ValueError("Expert projections must have matching cardinality")
        if self.o_proj.values.device != self.i_proj.values.device:
            raise ValueError("Expert projections must share one device")
        if self.router is not None and self.router.projection.values.device != self.i_proj.values.device:
            raise ValueError("Expert and Router resources must share one device")

    def binding_view(self) -> xpool.native.ffnagent.MoeBindingView:
        """Borrow complete Expert and Router addresses without transferring owners."""

        return xpool.native.ffnagent.MoeBindingView(
            i_proj=self.i_proj.binding_view(),
            o_proj=self.o_proj.binding_view(),
            router=None
            if self.router is None
            else xpool.native.ffnagent.MoeRouterBindingView(
                projection=self.router.projection.binding_view(),
                correction_bias=self.router.correction_bias,
            ),
        )


type FfnLayerWeights = DenseFfnWeights | MoeFfnWeights
