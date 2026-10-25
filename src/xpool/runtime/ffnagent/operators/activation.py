"""Gated SiLU launches and fused descriptors for caller-owned FFN resources."""

from __future__ import annotations

import torch
import triton
from triton import language
from triton_kernels import matmul
from triton_kernels.specialize import FnSpecs
from triton_kernels.swiglu import swiglu_fn

from xpool import ffn


@triton.jit
def compute_gated_silu(gate, up, alpha, limit, up_offset):
    gate = gate.to(language.float32)
    up = up.to(language.float32)
    if limit is not None:
        gate = language.minimum(gate, limit)
        up = language.clamp(up, -limit, limit)
    return (gate / (1 + language.exp(-alpha * gate))) * (up + up_offset)


@triton.jit
def gated_silu_kernel(
    inputs,
    output,
    ELEMENTS: language.constexpr,
    WIDTH: language.constexpr,
    INTERLEAVED: language.constexpr,
    ALPHA: language.constexpr,
    LIMIT: language.constexpr,
    UP_OFFSET: language.constexpr,
    BLOCK: language.constexpr,
):
    index = language.program_id(0) * BLOCK + language.arange(0, BLOCK)
    row = index // WIDTH
    column = index % WIDTH
    if INTERLEAVED:
        gate_offset = row * 2 * WIDTH + 2 * column
        up_offset = gate_offset + 1
    else:
        gate_offset = row * 2 * WIDTH + column
        up_offset = gate_offset + WIDTH
    gate = language.load(inputs + gate_offset, index < ELEMENTS, other=0)
    up = language.load(inputs + up_offset, index < ELEMENTS, other=0)
    value = compute_gated_silu(gate, up, ALPHA, LIMIT, UP_OFFSET)
    language.store(output + index, value, index < ELEMENTS)


@triton.jit
def fused_gated_silu(inputs, alpha, limit, up_offset):
    gate, up = language.split(language.reshape(inputs, (inputs.shape[0], inputs.shape[1] // 2, 2)))
    return compute_gated_silu(gate, up, alpha, limit, up_offset)


def apply_into(
    spec: ffn.ActivationSpec,
    *,
    inputs: torch.Tensor,
    output: torch.Tensor,
    order: ffn.GateUpOrder,
) -> None:
    """Write gated SiLU into the caller's existing activation buffer.

    The operation owner supplies checked contiguous payload resources shaped
    ``[rows, 2 * width]`` and ``[rows, width]`` on the same device, with the
    existing 16-byte row alignment. No Tensor is allocated. Projection rounding
    has already occurred at the input boundary.
    """

    if order is ffn.GateUpOrder.CONCATENATED and spec.alpha == 1.0 and spec.limit is None and spec.up_offset == 0.0:
        # Import selects a device; preparation must remain free of device work.
        import sgl_kernel

        if sgl_kernel.silu_and_mul(inputs, out=output) is not output:
            raise RuntimeError("sgl_kernel.silu_and_mul did not preserve caller-owned output")
        return
    gated_silu_kernel.run(
        inputs,
        output,
        output.numel(),
        output.shape[1],
        order is ffn.GateUpOrder.INTERLEAVED,
        spec.alpha,
        spec.limit,
        spec.up_offset,
        BLOCK=256,
        grid=(triton.cdiv(output.numel(), 256),),
        warmup=False,
    )


def fused(spec: ffn.ActivationSpec) -> matmul.FusedActivation:
    """Describe activation before the packed GEMM's payload conversion.

    The upstream function keeps its qualified arithmetic for a unit Up offset.
    Other offsets use the common formula with the same interleaved row layout.
    """

    if spec.up_offset == 1.0:
        return matmul.FusedActivation(
            FnSpecs(swiglu_fn.__name__, swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (spec.alpha, spec.limit),
        )
    return matmul.FusedActivation(
        FnSpecs(fused_gated_silu.__name__, fused_gated_silu, ("alpha", "limit", "up_offset"), reduction_n=2),
        (spec.alpha, spec.limit, spec.up_offset),
    )
