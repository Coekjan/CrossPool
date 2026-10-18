"""Canonical FfnAgent-owned true-TP weight tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
    bias: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate Router weight and optional correction bias."""

        router_dtype = self.weight.dtype
        if router_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Router weights require BF16, FP16, or FP32")
        validate_weight_tensor(self.weight, name="router weight", dtype=router_dtype, dimensions=2)
        if self.bias is not None:
            validate_weight_tensor(self.bias, name="router bias", dtype=router_dtype, dimensions=1)
            if self.bias.shape != (self.weight.shape[0],) or self.bias.device != self.weight.device:
                raise ValueError("Router bias dimensions or device disagree with weight")
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


@dataclass(frozen=True, slots=True)
class Mxfp4MoeFfnWeights:
    """Canonical CUDA MXFP4 shard; matrices stay packed, with 32-value scales.

    Gate/up rows remain interleaved. Only TP rank zero retains down bias and
    Router resources. Each tensor owns its exact independent Device storage.
    """

    gate_up_packed: torch.Tensor
    gate_up_scales: torch.Tensor
    down_packed: torch.Tensor
    down_scales: torch.Tensor
    gate_up_bias: torch.Tensor
    down_bias: torch.Tensor | None
    router: MoeRouterWeights | None

    def __post_init__(self) -> None:
        """Validate packed layout and block alignment before capture."""

        for name in ("gate_up_packed", "gate_up_scales", "down_packed", "down_scales"):
            validate_weight_tensor(getattr(self, name), name=name, dtype=torch.uint8, dimensions=3)
        e, twice_i, half_h = self.gate_up_packed.shape
        h, i = half_h * 2, twice_i // 2
        if twice_i % 2 or h % 32 or i % 32:
            raise ValueError("MXFP4 dimensions must contain complete 32-value blocks")
        if (
            self.gate_up_scales.shape != (e, 2 * i, h // 32)
            or self.down_packed.shape != (e, h, i // 2)
            or self.down_scales.shape != (e, h, i // 32)
        ):
            raise ValueError("MXFP4 packed weights and scales disagree")
        validate_weight_tensor(self.gate_up_bias, name="gate/up bias", dtype=torch.bfloat16, dimensions=2)
        if self.gate_up_bias.shape != (e, 2 * i):
            raise ValueError("MXFP4 gate/up bias dimensions disagree")
        if self.down_bias is not None:
            validate_weight_tensor(self.down_bias, name="down bias", dtype=torch.bfloat16, dimensions=2)
            if self.down_bias.shape != (e, h):
                raise ValueError("MXFP4 down bias dimensions disagree")
        tensors = [self.gate_up_scales, self.down_packed, self.down_scales, self.gate_up_bias]
        if self.down_bias is not None:
            tensors.append(self.down_bias)
        if self.router is not None:
            if self.router.weight.shape != (e, h) or self.router.bias is None:
                raise ValueError("MXFP4 Router dimensions or additive bias disagree")
            tensors.append(self.router.weight)
        if any(t.device != self.gate_up_packed.device for t in tensors):
            raise ValueError("MXFP4 resources must share one Device")


type FfnLayerWeights = DenseFfnWeights | MoeFfnWeights | Mxfp4MoeFfnWeights
