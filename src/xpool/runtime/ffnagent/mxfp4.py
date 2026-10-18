"""SM80 W4A16 Expert kernels with tile-local MXFP4 decoding."""

from __future__ import annotations

import triton
import triton.language as language


@triton.jit
def decode_mxfp4_tile(packed, scales, expert, columns, reduction, N: language.constexpr, K: language.constexpr):
    """Decode one E2M1/UE8M0 tile directly into a BF16 dot operand."""

    valid = (columns[None, :] < N) & (reduction[:, None] < K)
    byte = language.load(
        packed + expert * N * (K // 2) + columns[None, :] * (K // 2) + reduction[:, None] // 2, mask=valid, other=0
    )
    nibble = (byte >> ((reduction[:, None] % 2) * 4)) & 15
    magnitude = nibble & 7
    value = language.where(
        magnitude < 4,
        magnitude.to(language.float32) * 0.5,
        language.where(magnitude < 6, 2.0, 4.0) * language.where((magnitude & 1) != 0, 1.5, 1.0),
    )
    value = language.where((nibble & 8) != 0, -value, value)
    scale = language.load(
        scales + expert * N * (K // 32) + columns[None, :] * (K // 32) + reduction[:, None] // 32, mask=valid, other=127
    )
    return (value * language.exp2(scale.to(language.float32) - 127.0)).to(language.bfloat16)


@triton.jit
def mxfp4_gemm_kernel(
    inputs,
    packed,
    scales,
    bias,
    output,
    sorted_ids,
    block_experts,
    padded_count,
    probabilities,
    R: language.constexpr,
    N: language.constexpr,
    K: language.constexpr,
    E: language.constexpr,
    TOPK: language.constexpr,
    DOWN: language.constexpr,
    HAS_BIAS: language.constexpr,
):
    """Run a fixed-capacity routed GEMM without expanding resident weights."""

    block = language.program_id(0)
    if block * 16 >= language.load(padded_count):
        return
    expert = language.load(block_experts + block)
    if expert < 0 or expert >= E:
        return
    routes = language.load(sorted_ids + block * 16 + language.arange(0, 16))
    columns = language.program_id(1) * 64 + language.arange(0, 64)
    accumulator = language.full((16, 64), 0, language.float32)
    for start in range(language.cdiv(K, 32)):
        reduction = start * 32 + language.arange(0, 32)
        rows = routes if DOWN else routes // TOPK
        operand = language.load(
            inputs + rows[:, None] * K + reduction[None, :],
            mask=(routes[:, None] < R) & (reduction[None, :] < K),
            other=0,
        )
        weight = decode_mxfp4_tile(packed, scales, expert, columns, reduction, N, K)
        accumulator = language.dot(operand, weight, accumulator)
    if HAS_BIAS:
        accumulator += language.load(bias + expert * N + columns, mask=columns < N, other=0)[None, :].to(
            language.float32
        )
    if DOWN:
        accumulator *= language.load(probabilities + routes, mask=routes < R, other=0)[:, None]
    language.store(
        output + routes[:, None] * N + columns[None, :],
        accumulator,
        mask=(routes[:, None] < R) & (columns[None, :] < N),
    )


@triton.jit
def oai_activation_kernel(
    gate_up,
    activated,
    expert_ids,
    R: language.constexpr,
    WIDTH: language.constexpr,
    ALPHA: language.constexpr,
    LIMIT: language.constexpr,
):
    """Apply interleaved asymmetric-clamp SwiGLU in caller-owned storage."""

    route = language.program_id(0)
    columns = language.program_id(1) * 256 + language.arange(0, 256)
    valid = (columns < WIDTH) & (language.load(expert_ids + route) >= 0)
    gate = language.load(gate_up + route * 2 * WIDTH + 2 * columns, mask=valid, other=0).to(language.float32)
    up = language.load(gate_up + route * 2 * WIDTH + 2 * columns + 1, mask=valid, other=0).to(language.float32)
    gate = language.minimum(gate, LIMIT)
    up = language.minimum(language.maximum(up, -LIMIT), LIMIT)
    value = gate / (1.0 + language.exp(-ALPHA * gate)) * (up + 1.0)
    language.store(activated + route * WIDTH + columns, value, mask=columns < WIDTH)


@triton.jit
def combine_routes_kernel(
    route_outputs, expert_ids, output, C: language.constexpr, H: language.constexpr, TOPK: language.constexpr
):
    """Fold valid per-route BF16 results in route order using FP32."""

    row = language.program_id(0)
    columns = language.program_id(1) * 256 + language.arange(0, 256)
    accumulator = language.full((256,), 0, language.float32)
    for slot in range(TOPK):
        route = row * TOPK + slot
        valid = (columns < H) & (language.load(expert_ids + route) >= 0)
        accumulator += language.load(route_outputs + route * H + columns, mask=valid, other=0).to(language.float32)
    language.store(output + row * H + columns, accumulator, mask=columns < H)
