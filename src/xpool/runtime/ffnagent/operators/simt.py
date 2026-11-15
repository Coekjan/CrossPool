"""Caller-backed IEEE matrix primitives and their shared projection epilogue."""

from typing import no_type_check

import triton
import triton.experimental.gluon.language
from triton import language
from triton.experimental import gluon
from triton.experimental.gluon.language.nvidia.ampere import async_copy


@triton.jit
def row_projection_kernel(
    inputs,
    weights,
    bias,
    output,
    K: language.constexpr,
    N: language.constexpr,
    OUT_N: language.constexpr,
    BK: language.constexpr,
    BN: language.constexpr,
    REDUCTION: language.constexpr,
    FUNCTION: language.constexpr,
    ARGUMENTS,
):
    row = language.program_id(0)
    column = language.program_id(1) * BN + language.arange(0, BN)
    offsets = language.arange(0, BK)
    accumulator = language.full((BN, BK), 0, language.float32)
    for start in range(triton.cdiv(K, BK)):
        k = start * BK + offsets
        lhs = language.load(inputs + row * K + k, k < K, other=0).to(language.float32)
        rhs = language.load(
            weights + column[:, None] * K + k[None, :], (column[:, None] < N) & (k[None, :] < K), other=0
        ).to(language.float32)
        accumulator += lhs[None, :] * rhs
    value = language.sum(accumulator, 1)
    if bias is not None:
        value += language.load(bias + column, column < N, other=0).to(language.float32)
    value = language.reshape(value, (1, BN))
    if FUNCTION is not None:
        value = FUNCTION(value, *ARGUMENTS)
    out_column = language.program_id(1) * (BN // REDUCTION) + language.arange(0, BN // REDUCTION)
    language.store(output + row * OUT_N + out_column[None, :], value, out_column[None, :] < OUT_N)


@triton.jit
def reduce_projection_kernel(
    partials,
    bias,
    output,
    M: language.constexpr,
    N: language.constexpr,
    SPLITS: language.constexpr,
    PADDED_SPLITS: language.constexpr,
    REDUCTION: language.constexpr,
    FUNCTION: language.constexpr,
    ARGUMENTS,
    BLOCK: language.constexpr,
):
    split = language.arange(0, PADDED_SPLITS)
    index = language.program_id(0) * BLOCK + language.arange(0, BLOCK)
    values = language.load(
        partials + split[:, None] * M * N + index[None, :],
        (split[:, None] < SPLITS) & (index[None, :] < M * N),
        other=0,
    )
    value = language.sum(values, 0)
    if bias is not None:
        value += language.load(bias + index % N, index < M * N, other=0).to(language.float32)
    value = language.reshape(value, (1, BLOCK))
    if FUNCTION is not None:
        value = FUNCTION(value, *ARGUMENTS)
    output_index = language.program_id(0) * (BLOCK // REDUCTION) + language.arange(0, BLOCK // REDUCTION)
    language.store(output + output_index[None, :], value, output_index[None, :] < M * N // REDUCTION)


# Gluon interprets compile-time layout annotations as DSL values.
@gluon.jit
@no_type_check
def shared_pipeline_kernel(
    inputs,
    weights,
    output,
    M: language.constexpr,
    N: language.constexpr,
    K: language.constexpr,
    BM: language.constexpr,
    BN: language.constexpr,
    BK: language.constexpr,
    SPLITS: language.constexpr,
):
    RM: language.constexpr = 2
    RN: language.constexpr = 2
    WM: language.constexpr = 2
    WN: language.constexpr = 2
    UNROLL: language.constexpr = 32
    SWIZZLE: language.constexpr = 8
    TM: language.constexpr = 4
    ORDER_M: language.constexpr = False
    PER_PHASE: language.constexpr = 2
    pm, pn, ps = gluon.language.program_id(0), gluon.language.program_id(1), gluon.language.program_id(2)
    layout: language.constexpr = gluon.language.BlockedLayout(
        [1, RM, RN], [1, TM, 32 // TM], [1, WM, WN], [1, 2, 0] if ORDER_M else [2, 1, 0]
    )
    KL: language.constexpr = min(8, BK // 4)
    load_layout: language.constexpr = gluon.language.BlockedLayout([1, 4], [32 // KL, KL], [WM * WN, 1], [1, 0])
    rows = pm * BM + gluon.language.arange(0, BM, layout=gluon.language.SliceLayout(1, load_layout))
    columns = pn * BN + gluon.language.arange(0, BN, layout=gluon.language.SliceLayout(1, load_layout))
    offsets = gluon.language.arange(0, BK, layout=gluon.language.SliceLayout(0, load_layout))
    extent: language.constexpr = triton.cdiv(K, SPLITS * BK) * BK
    shared_layout: language.constexpr = gluon.language.SwizzledSharedLayout(4, PER_PHASE, SWIZZLE, [1, 0])
    shared_x = gluon.language.allocate_shared_memory(gluon.language.float32, [2, BM, BK], shared_layout)
    shared_w = gluon.language.allocate_shared_memory(gluon.language.float32, [2, BN, BK], shared_layout)
    accumulator = gluon.language.full((1, BM, BN), 0, gluon.language.float32, layout)
    k = ps * extent + offsets
    async_copy.async_copy_global_to_shared(
        shared_x.index(0), inputs + rows[:, None] * K + k[None, :], mask=(rows[:, None] < M) & (k[None, :] < K)
    )
    async_copy.async_copy_global_to_shared(
        shared_w.index(0), weights + columns[:, None] * K + k[None, :], mask=(columns[:, None] < N) & (k[None, :] < K)
    )
    async_copy.commit_group()
    for block in range(extent // BK):
        current = block % 2
        if block + 1 < extent // BK:
            k = ps * extent + (block + 1) * BK + offsets
            async_copy.async_copy_global_to_shared(
                shared_x.index((block + 1) % 2),
                inputs + rows[:, None] * K + k[None, :],
                mask=(rows[:, None] < M) & (k[None, :] < K),
            )
            async_copy.async_copy_global_to_shared(
                shared_w.index((block + 1) % 2),
                weights + columns[:, None] * K + k[None, :],
                mask=(columns[:, None] < N) & (k[None, :] < K),
            )
            async_copy.commit_group()
            async_copy.wait_group(1)
        else:
            async_copy.wait_group(0)
        gluon.language.barrier()
        for chunk in range(BK // UNROLL):
            for step in gluon.language.static_range(UNROLL):
                kk = chunk * UNROLL + step
                ix = gluon.language.full((1, BM), kk, gluon.language.int32, gluon.language.SliceLayout(2, layout))
                iw = gluon.language.full((1, BN), kk, gluon.language.int32, gluon.language.SliceLayout(1, layout))
                a = gluon.language.convert_layout(
                    shared_x.index(current).gather(ix.T, axis=1).T, gluon.language.SliceLayout(2, layout)
                )
                b = gluon.language.convert_layout(
                    shared_w.index(current).gather(iw.T, axis=1).T, gluon.language.SliceLayout(1, layout)
                )
                accumulator = gluon.language.fma(a[:, :, None], b[:, None, :], accumulator)
        gluon.language.barrier()
    result_layout: language.constexpr = gluon.language.SliceLayout(0, layout)
    rr = pm * BM + gluon.language.arange(0, BM, layout=gluon.language.SliceLayout(1, result_layout))
    cc = pn * BN + gluon.language.arange(0, BN, layout=gluon.language.SliceLayout(0, result_layout))
    gluon.language.store(
        output + ps * M * N + rr[:, None] * N + cc[None, :],
        gluon.language.sum(accumulator, 0),
        (rr[:, None] < M) & (cc[None, :] < N),
    )


# Gluon interprets compile-time layout annotations as DSL values.
@gluon.jit
@no_type_check
def microtile_kernel(
    inputs,
    weights,
    output,
    M: language.constexpr,
    N: language.constexpr,
    K: language.constexpr,
    BM: language.constexpr,
    BN: language.constexpr,
    BK: language.constexpr,
    SPLITS: language.constexpr,
):
    LOAD_VEC: language.constexpr = 4
    pm, pn, ps = gluon.language.program_id(0), gluon.language.program_id(1), gluon.language.program_id(2)
    layout: language.constexpr = gluon.language.DistributedLinearLayout(
        reg_bases=[[0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 2, 0], [0, 0, 32], [0, 0, 64], [0, 64, 0]],
        lane_bases=[[0, 0, 4], [0, 0, 8], [0, 4, 0], [0, 8, 0], [0, 16, 0]],
        warp_bases=[[0, 32, 0], [0, 0, 16]],
        block_bases=[],
        shape=[1, 128, 128],
    )
    KL: language.constexpr = min(8, BK // LOAD_VEC)
    copy_layout: language.constexpr = gluon.language.BlockedLayout([1, LOAD_VEC], [32 // KL, KL], [4, 1], [1, 0])
    rows = pm * BM + gluon.language.arange(0, BM, layout=gluon.language.SliceLayout(1, copy_layout))
    columns = pn * BN + gluon.language.arange(0, BN, layout=gluon.language.SliceLayout(1, copy_layout))
    offsets = gluon.language.arange(0, BK, layout=gluon.language.SliceLayout(0, copy_layout))
    x_layout: language.constexpr = gluon.language.PaddedSharedLayout.with_identity_for([[BM, 4]], [BK, BM], [1, 0])
    w_layout: language.constexpr = gluon.language.PaddedSharedLayout.with_identity_for([[BN, 4]], [BK, BN], [1, 0])
    sx = gluon.language.allocate_shared_memory(gluon.language.float32, [BK, BM], x_layout)
    sw = gluon.language.allocate_shared_memory(gluon.language.float32, [BK, BN], w_layout)
    extent: language.constexpr = triton.cdiv(K, SPLITS * BK) * BK
    k = ps * extent + offsets
    current_x = gluon.language.load(
        inputs + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), other=0
    )
    current_w = gluon.language.load(
        weights + columns[:, None] * K + k[None, :], (columns[:, None] < N) & (k[None, :] < K), other=0
    )
    accumulator = gluon.language.full((1, BM, BN), 0, gluon.language.float32, layout)
    for block in range(extent // BK):
        sx.store(current_x.T)
        sw.store(current_w.T)
        gluon.language.barrier()
        if block + 1 < extent // BK:
            next_k = ps * extent + (block + 1) * BK + offsets
            next_x = gluon.language.load(
                inputs + rows[:, None] * K + next_k[None, :], (rows[:, None] < M) & (next_k[None, :] < K), other=0
            )
            next_w = gluon.language.load(
                weights + columns[:, None] * K + next_k[None, :],
                (columns[:, None] < N) & (next_k[None, :] < K),
                other=0,
            )
        else:
            next_x = current_x
            next_w = current_w
        for kk in gluon.language.static_range(BK):
            a = sx.slice(kk, 1).load(gluon.language.SliceLayout(2, layout))
            b = sw.slice(kk, 1).load(gluon.language.SliceLayout(1, layout))
            accumulator = gluon.language.fma(a[:, :, None], b[:, None, :], accumulator)
        gluon.language.barrier()
        current_x = next_x
        current_w = next_w
    result_layout: language.constexpr = gluon.language.SliceLayout(0, layout)
    rr = pm * BM + gluon.language.arange(0, BM, layout=gluon.language.SliceLayout(1, result_layout))
    cc = pn * BN + gluon.language.arange(0, BN, layout=gluon.language.SliceLayout(0, result_layout))
    gluon.language.store(
        output + ps * M * N + rr[:, None] * N + cc[None, :],
        gluon.language.sum(accumulator, 0),
        (rr[:, None] < M) & (cc[None, :] < N),
    )
