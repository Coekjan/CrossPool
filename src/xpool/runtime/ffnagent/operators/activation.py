"""Activation launches and matrix epilogues for caller-owned FFN resources."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch
import triton
from triton import language
from triton.language.extra.cuda import libdevice
from triton_kernels import matmul
from triton_kernels.specialize import FnSpecs
from triton_kernels.swiglu import swiglu_fn

from xpool import ffn


@triton.jit
def compute_relu(value):
    # Preserve NaN payloads while matching ReLU's treatment of signed zero.
    if value.dtype == language.float32:
        bits = value.to(language.uint32, bitcast=True)
    else:
        bits = value.to(language.uint16, bitcast=True)
    result = language.where((value > 0) | (value != value), bits, 0).to(bits.dtype)
    return result.to(value.dtype, bitcast=True)


@triton.jit
def compute_silu(value, alpha):
    value = value.to(language.float32)
    return value / (1 + language.exp(-alpha * value))


@triton.jit
def compute_gelu(value, APPROXIMATE: language.constexpr):
    value = value.to(language.float32)
    if APPROXIMATE:
        return 0.5 * value * (1 + libdevice.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)))
    return 0.5 * value * (1 + libdevice.erf(value * 0.7071067811865476))


@triton.jit
def compute_gated(gate, up, FUNCTION: language.constexpr, FUNCTION_ARGS, limit, up_offset):
    gate = gate.to(language.float32)
    up = up.to(language.float32)
    if limit is not None:
        gate = language.minimum(gate, limit)
        up = language.clamp(up, -limit, limit)
    return FUNCTION(gate, *FUNCTION_ARGS) * (up + up_offset)


@triton.jit
def activation_kernel(
    inputs,
    output,
    ELEMENTS: language.constexpr,
    WIDTH: language.constexpr,
    FUNCTION: language.constexpr,
    FUNCTION_ARGS,
    GATED: language.constexpr,
    INTERLEAVED: language.constexpr,
    LIMIT: language.constexpr,
    UP_OFFSET: language.constexpr,
    BLOCK: language.constexpr,
):
    index = language.program_id(0) * BLOCK + language.arange(0, BLOCK)
    if GATED:
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
        value = compute_gated(gate, up, FUNCTION, FUNCTION_ARGS, LIMIT, UP_OFFSET)
    else:
        value = language.load(inputs + index, index < ELEMENTS, other=0)
        value = FUNCTION(value, *FUNCTION_ARGS)
    language.store(output + index, value, index < ELEMENTS)


@dataclass(frozen=True, slots=True)
class Activation:
    """A layer's activation and optional Gate/Up combination."""

    function: ffn.ActivationSpec
    gate: ffn.GateTransform | None

    @property
    def supports_inplace(self) -> bool:
        """Return whether the standalone operation may reuse its input."""

        return self.gate is None

    def elementwise(self) -> tuple[triton.runtime.jit.JITFunction, tuple[float | language.constexpr, ...]]:
        """Resolve the shared scalar expression and its Triton arguments."""

        match self.function:
            case ffn.ReluActivation():
                return compute_relu, ()
            case ffn.SiluActivation(alpha=alpha):
                return compute_silu, (alpha,)
            case ffn.GeluActivation(approximate=approximate):
                return compute_gelu, (language.constexpr(approximate == "tanh"),)

    @cache
    def fused(
        self,
        *,
        order: ffn.GateUpOrder | None,
        projection_round_dtype: torch.dtype | None,
    ) -> matmul.FusedActivation | None:
        """Describe an epilogue, preserving the pre-activation rounding boundary.

        Gated fusion requires adjacent Gate/Up rows. None for the rounding dtype
        retains accumulator precision. No Tensor or layer resource is captured.
        """

        gate = self.gate
        if gate is not None and order is not ffn.GateUpOrder.INTERLEAVED:
            return None
        if (
            isinstance(self.function, ffn.SiluActivation)
            and gate is not None
            and gate.up_offset == 1.0
            and projection_round_dtype is None
        ):
            return matmul.FusedActivation(
                FnSpecs("swiglu_fn", swiglu_fn, ("alpha", "limit"), reduction_n=2),
                (self.function.alpha, gate.limit),
            )
        function, arguments = self.elementwise()
        round_dtype = language.constexpr(
            {
                None: None,
                torch.bfloat16: language.bfloat16,
                torch.float16: language.float16,
                torch.float32: language.float32,
            }[projection_round_dtype]
        )
        gated = language.constexpr(gate is not None)

        def epilogue(inputs, function_args, limit, up_offset):
            if round_dtype is not None:
                inputs = inputs.to(round_dtype).to(language.float32)
            if gated:
                gate, up = language.split(language.reshape(inputs, (inputs.shape[0], inputs.shape[1] // 2, 2)))
                return compute_gated(gate, up, function, function_args, limit, up_offset)
            else:
                return function(inputs, *function_args)

        name = f"{function.__name__}_{'gated' if gate is not None else 'plain'}_{projection_round_dtype}"
        name = name.replace(".", "_")
        epilogue.__name__ = name
        epilogue.__qualname__ = name
        return matmul.FusedActivation(
            FnSpecs(
                name,
                triton.jit(epilogue),
                ("function_args", "limit", "up_offset"),
                reduction_n=2 if gate is not None else 1,
            ),
            (
                arguments,
                gate.limit if gate is not None else None,
                gate.up_offset if gate is not None else 0.0,
            ),
        )

    def apply_into(
        self,
        *,
        inputs: torch.Tensor,
        output: torch.Tensor,
        order: ffn.GateUpOrder | None,
    ) -> None:
        """Write a standalone activation into contiguous FP16/BF16 storage.

        Inputs already include projection rounding. Ungated input/output may
        alias; gated input has twice the output width and requires distinct
        output storage. The owner supplies the prepared physical row order.
        """

        gate = self.gate
        if gate is not None and order is None:
            raise ValueError("gated activation requires a Gate/Up order")
        function, arguments = self.elementwise()
        activation_kernel.run(
            inputs,
            output,
            output.numel(),
            output.shape[-1],
            function,
            arguments,
            gate is not None,
            order is ffn.GateUpOrder.INTERLEAVED,
            gate.limit if gate is not None else None,
            gate.up_offset if gate is not None else 0.0,
            BLOCK=256,
            grid=(triton.cdiv(output.numel(), 256),),
            warmup=False,
        )
