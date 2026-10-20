"""Canonical FfnAgent-owned true-TP weight tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import xpool.native
from xpool import ffn


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
    dtype: torch.dtype,
    dimensions: int,
) -> None:
    """Validate one exact, independently owned canonical weight allocation."""

    validate_tensor(tensor, name=name, dtype=dtype, dimensions=dimensions)
    storage = tensor.untyped_storage()
    if tensor.storage_offset() != 0 or tensor.data_ptr() != storage.data_ptr() or tensor.nbytes != storage.nbytes():
        raise ValueError(f"{name} must own its exact CUDA storage")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


@dataclass(frozen=True, slots=True)
class DenseFfnWeights:
    """One canonical local gated-Dense shard.

    Attributes:
        gate_up_weight: Gate-then-up payload tensor shaped ``[2 * I_r, H]``.
        down_weight: Down-projection payload tensor shaped ``[H, I_r]``.
    """

    gate_up_weight: torch.Tensor
    down_weight: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        hidden_size: int,
        local_intermediate_size: int,
        payload_dtype: torch.dtype,
        device: torch.device,
    ) -> DenseFfnWeights:
        """Own empty canonical shards, ready for checkpoint or synthetic filling."""

        return cls(
            gate_up_weight=torch.empty((2 * local_intermediate_size, hidden_size), dtype=payload_dtype, device=device),
            down_weight=torch.empty((hidden_size, local_intermediate_size), dtype=payload_dtype, device=device),
        )

    def binding_projection(self) -> xpool.native.ffnagent.DenseBindingResourceProjection:
        """Expose retained addresses without transferring Tensor ownership."""

        return xpool.native.ffnagent.DenseBindingResourceProjection(
            gate_up_weight=self.gate_up_weight,
            down_weight=self.down_weight,
        )

    def zeros_like(self) -> DenseFfnWeights:
        """Allocate zero-valued weights with the same shape, dtype and device."""

        return DenseFfnWeights(
            gate_up_weight=torch.zeros_like(self.gate_up_weight),
            down_weight=torch.zeros_like(self.down_weight),
        )

    def __post_init__(self) -> None:
        """Validate canonical Dense ranks, dimensions, storage, and device."""

        payload_dtype = self.gate_up_weight.dtype
        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Dense weights require BF16 or FP16 payloads")
        validate_weight_tensor(self.gate_up_weight, name="gate_up_weight", dtype=payload_dtype, dimensions=2)
        validate_weight_tensor(self.down_weight, name="down_weight", dtype=payload_dtype, dimensions=2)
        local_intermediate_twice, hidden_size = self.gate_up_weight.shape
        if local_intermediate_twice % 2 != 0:
            raise ValueError("gate_up_weight first dimension must be even")
        local_intermediate_size = local_intermediate_twice // 2
        if self.down_weight.shape != (hidden_size, local_intermediate_size):
            raise ValueError("Dense down_weight dimensions disagree with gate_up_weight")
        if self.down_weight.device != self.gate_up_weight.device:
            raise ValueError("Dense weights must share one device")


@dataclass(frozen=True, slots=True)
class MoeRouterWeights:
    """Router-owner-only canonical MoE weights.

    Attributes:
        weight: Routed-Expert Router tensor shaped ``[E_r, H]`` with model-owned precision.
        correction_bias: Optional corrected-routing FP32 tensor shaped
            ``[E_r]``.
    """

    weight: torch.Tensor
    correction_bias: torch.Tensor | None

    def __post_init__(self) -> None:
        """Validate Router weight and optional correction bias."""

        router_dtype = self.weight.dtype
        if router_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Router weights require BF16, FP16, or FP32")
        validate_weight_tensor(self.weight, name="router weight", dtype=router_dtype, dimensions=2)
        if self.correction_bias is None:
            return
        validate_weight_tensor(
            self.correction_bias,
            name="router correction bias",
            dtype=torch.float32,
            dimensions=1,
        )
        if self.correction_bias.shape[0] != self.weight.shape[0]:
            raise ValueError("Router correction bias cardinality disagrees with Router weight")
        if self.correction_bias.device != self.weight.device:
            raise ValueError("Router weights must share one device")


@dataclass(frozen=True, slots=True)
class MoeFfnWeights:
    """One canonical local MoE Expert shard and optional Router ownership.

    Attributes:
        expert_gate_up_weight: Routed-then-shared gate/up payload tensor shaped
            ``[E, 2 * I_r, H]``.
        expert_down_weight: Routed-then-shared down payload tensor shaped
            ``[E, H, I_r]``.
        router: Router weights on TP rank zero, otherwise ``None``.
    """

    expert_gate_up_weight: torch.Tensor
    expert_down_weight: torch.Tensor
    router: MoeRouterWeights | None

    @classmethod
    def allocate(
        cls,
        layer: ffn.MoeFfnSpec,
        *,
        hidden_size: int,
        local_intermediate_size: int,
        payload_dtype: torch.dtype,
        router_weight_dtype: torch.dtype | None,
        device: torch.device,
    ) -> MoeFfnWeights:
        """Own empty Expert shards and any Router tensors in production order.

        A missing Router dtype selects a non-owner TP shard. The caller selects
        precision from the declared execution policy and fills the empty tensors.
        """

        expert_count = layer.routed_expert_count + layer.shared_expert_count
        expert_gate_up_weight = torch.empty(
            (expert_count, 2 * local_intermediate_size, hidden_size), dtype=payload_dtype, device=device
        )
        expert_down_weight = torch.empty(
            (expert_count, hidden_size, local_intermediate_size), dtype=payload_dtype, device=device
        )
        router = None
        if router_weight_dtype is not None:
            router_weight = torch.empty(
                (layer.routed_expert_count, hidden_size), dtype=router_weight_dtype, device=device
            )
            correction_bias = (
                torch.empty((layer.routed_expert_count,), dtype=torch.float32, device=device)
                if layer.checkpoint.router_correction_bias_key is not None
                else None
            )
            router = MoeRouterWeights(weight=router_weight, correction_bias=correction_bias)
        return cls(
            expert_gate_up_weight=expert_gate_up_weight,
            expert_down_weight=expert_down_weight,
            router=router,
        )

    def binding_projection(self) -> xpool.native.ffnagent.MoeBindingResourceProjection:
        """Expose Expert and optional Router addresses, retaining their owners."""

        router = self.router
        return xpool.native.ffnagent.MoeBindingResourceProjection(
            expert_gate_up_weight=self.expert_gate_up_weight,
            expert_down_weight=self.expert_down_weight,
            router=(
                None
                if router is None
                else xpool.native.ffnagent.MoeRouterBindingResourceProjection(
                    weight=router.weight,
                    correction_bias=router.correction_bias,
                )
            ),
        )

    def zeros_like(self) -> MoeFfnWeights:
        """Allocate zero-valued shape-equivalent weights with matching ownership."""

        router = self.router
        return MoeFfnWeights(
            expert_gate_up_weight=torch.zeros_like(self.expert_gate_up_weight),
            expert_down_weight=torch.zeros_like(self.expert_down_weight),
            router=(
                None
                if router is None
                else MoeRouterWeights(
                    weight=torch.zeros_like(router.weight),
                    correction_bias=None
                    if router.correction_bias is None
                    else torch.zeros_like(router.correction_bias),
                )
            ),
        )

    def __post_init__(self) -> None:
        """Validate canonical Expert ranks, dimensions, storage, and device."""

        payload_dtype = self.expert_gate_up_weight.dtype
        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("MoE Expert weights require BF16 or FP16 payloads")
        validate_weight_tensor(
            self.expert_gate_up_weight,
            name="expert_gate_up_weight",
            dtype=payload_dtype,
            dimensions=3,
        )
        validate_weight_tensor(
            self.expert_down_weight,
            name="expert_down_weight",
            dtype=payload_dtype,
            dimensions=3,
        )
        expert_count, local_intermediate_twice, hidden_size = self.expert_gate_up_weight.shape
        if local_intermediate_twice % 2 != 0:
            raise ValueError("expert_gate_up_weight intermediate dimension must be even")
        expected_down_shape = (expert_count, hidden_size, local_intermediate_twice // 2)
        if self.expert_down_weight.shape != expected_down_shape:
            raise ValueError("MoE expert_down_weight dimensions disagree with expert_gate_up_weight")
        if self.expert_down_weight.device != self.expert_gate_up_weight.device:
            raise ValueError("MoE Expert weights must share one device")
        if self.router is not None:
            if self.router.weight.device != self.expert_gate_up_weight.device:
                raise ValueError("MoE Expert and Router weights must share one device")
            if self.router.weight.shape[1] != hidden_size or self.router.weight.shape[0] > expert_count:
                raise ValueError("MoE Router dimensions disagree with Expert weights")


type FfnLayerWeights = DenseFfnWeights | MoeFfnWeights
