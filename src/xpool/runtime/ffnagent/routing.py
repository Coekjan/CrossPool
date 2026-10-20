"""Router projection, selection and final Routing Metadata computation."""

from __future__ import annotations

import math
import typing

import torch
import triton
from triton import language

from xpool import ffn
from xpool.runtime.ffnagent import execution, weights

ROUTING_FINALIZATION_BLOCK_SIZE = 256


@torch.compile(fullgraph=True)
def copy_router_input(source: torch.Tensor, target: torch.Tensor) -> None:
    """Convert into caller-owned FP32 storage before the Router GEMM.

    Pre-capture warmup compiles this operation. Inductor avoids the opaque
    empty-object parameters in ATen's copy kernel that conflict with the
    declared-address-only Graph parameterization contract.
    """

    target.copy_(source)


@triton.jit
def router_topk_kernel(
    logits_pointer,
    correction_bias_pointer,
    routed_ids_pointer,
    routed_weights_pointer,
    EXPERT_COUNT: language.constexpr,
    TOPK: language.constexpr,
    EXPERT_BLOCK_SIZE: language.constexpr,
    TOPK_BLOCK_SIZE: language.constexpr,
    SOFTMAX: language.constexpr,
    CORRECTION_BIAS_PRESENT: language.constexpr,
    RENORMALIZE: language.constexpr,
):
    """Select corrected routes and retain uncorrected activated weights.

    Sigmoid preserves its FP32 stored-selection boundary and higher-ID ties;
    Softmax breaks ties by lower ID. Expert padding is never selected.
    """

    row = language.program_id(0)
    experts = language.arange(0, EXPERT_BLOCK_SIZE)
    valid = experts < EXPERT_COUNT
    logits_offsets = row * EXPERT_COUNT + experts
    logits = language.load(logits_pointer + logits_offsets, mask=valid, other=-float("inf")).to(language.float32)
    if SOFTMAX:
        exponentials = language.exp(logits - language.max(logits, axis=0))
        scores = exponentials / language.sum(exponentials, axis=0)
    else:
        scores = language.sigmoid(logits)
    selection = scores
    if CORRECTION_BIAS_PRESENT:
        selection += language.load(correction_bias_pointer + experts, mask=valid, other=0.0)
    if not SOFTMAX:
        language.store(logits_pointer + logits_offsets, selection, mask=valid)
        language.debug_barrier()
        selection = language.load(logits_pointer + logits_offsets, mask=valid, other=-float("inf"))
    selection = language.where(valid, selection, -float("inf"))

    normalizer = language.full((), 0.0, language.float32)
    for slot in range(TOPK):
        maximum = language.max(selection, axis=0)
        if SOFTMAX:
            expert = language.min(language.where(valid & (selection == maximum), experts, EXPERT_COUNT), axis=0)
        else:
            expert = language.max(language.where(valid & (selection == maximum), experts, -1), axis=0)
        if SOFTMAX and not CORRECTION_BIAS_PRESENT:
            weight = maximum
        else:
            weight = language.sum(language.where(experts == expert, scores, 0.0), axis=0)
        language.store(routed_ids_pointer + row * TOPK + slot, expert)
        language.store(routed_weights_pointer + row * TOPK + slot, weight)
        normalizer += weight
        selection = language.where(experts == expert, -float("inf"), selection)

    if RENORMALIZE:
        # Slot loads can use different threads from the selection-loop stores.
        language.debug_barrier()
        slots = language.arange(0, TOPK_BLOCK_SIZE)
        output_offsets = row * TOPK + slots
        selected_weights = language.load(routed_weights_pointer + output_offsets, mask=slots < TOPK, other=0.0)
        denominator = language.where(normalizer > 0.0, normalizer, 1.0)
        language.store(routed_weights_pointer + output_offsets, selected_weights / denominator, mask=slots < TOPK)


def compute_routed_topk(
    algorithm: ffn.RouterAlgorithm,
    precision: ffn.RouterPrecision,
    *,
    hidden_states: torch.Tensor,
    router_weights: weights.MoeRouterWeights,
    workspace: torch.Tensor,
    routed_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    renormalize: bool,
) -> None:
    """Write qualified routed-only IDs and weights into caller-owned storage.

    Precision selects projection dtypes and caller-owned workspace. Optional
    correction changes selection only. Capture warmup compiles FP32 conversion.
    """

    payload_dtype = hidden_states.dtype
    expected_dtype = precision.weight_dtype(payload_dtype=payload_dtype)
    weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(workspace, name="Router workspace", dtype=torch.uint8, dimensions=1)
    weights.validate_tensor(routed_ids, name="routed Expert ids", dtype=torch.int32, dimensions=2)
    weights.validate_tensor(routed_weights, name="routed Expert weights", dtype=torch.float32, dimensions=2)
    row_capacity, hidden_size = hidden_states.shape
    routed_expert_count, router_hidden_size = router_weights.weight.shape
    routed_topk = routed_ids.shape[1]
    if (
        router_hidden_size != hidden_size
        or router_weights.weight.dtype is not expected_dtype
        or routed_ids.shape != routed_weights.shape
        or routed_ids.shape[0] != row_capacity
    ):
        raise ValueError("Router tensors disagree with the execution geometry")
    correction_bias = router_weights.correction_bias
    expected_bytes = execution.router_workspace_bytes(
        precision,
        payload_dtype=payload_dtype,
        payload_row_capacity=row_capacity,
        hidden_size=hidden_size,
        routed_expert_count=routed_expert_count,
        routed_topk=routed_topk,
    )
    if workspace.numel() != expected_bytes:
        raise ValueError(f"Router workspace must contain exactly {expected_bytes} bytes")
    tensors = (router_weights.weight, workspace, routed_ids, routed_weights)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("Router tensors must share one device")
    if correction_bias is not None and correction_bias.device != hidden_states.device:
        raise ValueError("Router correction bias must share the input device")

    match precision:
        case ffn.RouterPrecision.PAYLOAD:
            logits = workspace.view(payload_dtype).view(row_capacity, routed_expert_count)
            torch.mm(hidden_states, router_weights.weight.t(), out=logits)
        case ffn.RouterPrecision.FP32_LOGITS:
            logits = workspace.view(torch.float32).view(row_capacity, routed_expert_count)
            torch.mm(hidden_states, router_weights.weight.t(), out=logits, out_dtype=torch.float32)
        case ffn.RouterPrecision.FP32:
            scratch = workspace.view(torch.float32)
            input_elements = row_capacity * hidden_size
            router_input = scratch[:input_elements].view(row_capacity, hidden_size)
            logits = scratch[input_elements:].view(row_capacity, routed_expert_count)
            copy_router_input(hidden_states, router_input)
            torch.mm(router_input, router_weights.weight.t(), out=logits)
    # Triton launch metadata exceeds the constexpr-annotated signature.
    typing.cast(typing.Callable[..., object], router_topk_kernel[(row_capacity,)])(
        logits,
        correction_bias,
        routed_ids,
        routed_weights,
        EXPERT_COUNT=routed_expert_count,
        TOPK=routed_topk,
        EXPERT_BLOCK_SIZE=triton.next_power_of_2(routed_expert_count),
        TOPK_BLOCK_SIZE=triton.next_power_of_2(routed_topk),
        SOFTMAX=algorithm is ffn.RouterAlgorithm.SOFTMAX,
        CORRECTION_BIAS_PRESENT=correction_bias is not None,
        RENORMALIZE=renormalize,
        num_warps=1,
    )


@triton.jit
def finalize_moe_routing_kernel(
    routed_ids_pointer,
    routed_weights_pointer,
    final_ids_pointer,
    final_weights_pointer,
    payload_rows_pointer,
    ROW_CAPACITY: language.constexpr,
    ROUTED_TOPK: language.constexpr,
    EFFECTIVE_TOPK: language.constexpr,
    ROUTED_EXPERT_COUNT: language.constexpr,
    ROUTED_SCALING_FACTOR: language.constexpr,
    BLOCK_SIZE: language.constexpr,
):
    """Write routed-plus-shared live rows and an invalid Capacity tail."""

    offsets = language.program_id(0) * BLOCK_SIZE + language.arange(0, BLOCK_SIZE)
    element_count = ROW_CAPACITY * EFFECTIVE_TOPK
    valid = offsets < element_count
    rows = offsets // EFFECTIVE_TOPK
    slots = offsets % EFFECTIVE_TOPK
    live = rows < language.load(payload_rows_pointer)
    routed = slots < ROUTED_TOPK
    routed_offsets = rows * ROUTED_TOPK + slots
    routed_ids = language.load(
        routed_ids_pointer + routed_offsets,
        mask=valid & live & routed,
        other=-1,
    )
    routed_weights = language.load(
        routed_weights_pointer + routed_offsets,
        mask=valid & live & routed,
        other=0.0,
    )
    expert_ids = language.where(routed, routed_ids, ROUTED_EXPERT_COUNT + slots - ROUTED_TOPK)
    expert_weights = language.where(routed, routed_weights, 1.0 / ROUTED_SCALING_FACTOR)
    language.store(final_ids_pointer + offsets, language.where(live, expert_ids, -1), mask=valid)
    language.store(final_weights_pointer + offsets, language.where(live, expert_weights, 0.0), mask=valid)


def finalize_moe_routing(
    *,
    routed_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    final_ids: torch.Tensor,
    final_weights: torch.Tensor,
    payload_rows: torch.Tensor,
    routed_expert_count: int,
    shared_expert_count: int,
    routed_scaling_factor: float,
) -> None:
    """Finalize fixed-Capacity semantic Routing Metadata in one kernel.

    Args:
        routed_ids: Contiguous CUDA int32 Router output shaped
            ``[C, routed_topk]``.
        routed_weights: Contiguous CUDA FP32 Router output with the same shape.
        final_ids: Caller-owned contiguous CUDA int32 destination shaped
            ``[C, effective_topk]``.
        final_weights: Caller-owned contiguous CUDA FP32 destination with the
            same shape as ``final_ids``.
        payload_rows: One-element contiguous CUDA int64 Tensor containing the
            positive live row count.
        routed_expert_count: Positive routed Expert cardinality.
        shared_expert_count: Nonnegative always-selected Shared Expert count.
        routed_scaling_factor: Finite positive scale used to derive reciprocal
            Shared Expert carrier weights.

    Raises:
        ValueError: If shapes, dtypes, devices, counts, or the warmup live-row
            value disagree with the fixed Signature.

    Side Effects:
        Writes complete fixed-Capacity ID and weight destinations on the
        current CUDA stream. It allocates no Tensor or auxiliary workspace.
    """

    weights.validate_tensor(routed_ids, name="routed Expert ids", dtype=torch.int32, dimensions=2)
    weights.validate_tensor(routed_weights, name="routed Expert weights", dtype=torch.float32, dimensions=2)
    weights.validate_tensor(final_ids, name="final Expert ids", dtype=torch.int32, dimensions=2)
    weights.validate_tensor(final_weights, name="final Expert weights", dtype=torch.float32, dimensions=2)
    weights.validate_tensor(payload_rows, name="payload rows", dtype=torch.int64, dimensions=1)
    row_capacity, routed_topk = routed_ids.shape
    effective_topk = routed_topk + shared_expert_count
    if routed_weights.shape != routed_ids.shape:
        raise ValueError("routed Expert id and weight dimensions disagree")
    if final_ids.shape != (row_capacity, effective_topk) or final_weights.shape != final_ids.shape:
        raise ValueError("final Routing Metadata dimensions disagree")
    if payload_rows.numel() != 1:
        raise ValueError("payload rows must contain exactly one element")
    if routed_expert_count < routed_topk or shared_expert_count < 0:
        raise ValueError("Routing Metadata Expert counts are inconsistent")
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise ValueError("routed scaling factor must be finite and positive")
    tensors = (routed_ids, routed_weights, final_ids, final_weights, payload_rows)
    if any(tensor.device != routed_ids.device for tensor in tensors):
        raise ValueError("Routing Metadata tensors must share one device")
    if not torch.cuda.is_current_stream_capturing():
        live_rows = int(payload_rows.item())
        if live_rows <= 0 or live_rows > row_capacity:
            raise ValueError("payload rows must be positive and no greater than Capacity")

    element_count = row_capacity * effective_topk
    grid = (triton.cdiv(element_count, ROUTING_FINALIZATION_BLOCK_SIZE),)
    finalize_moe_routing_kernel[grid](
        routed_ids,
        routed_weights,
        final_ids,
        final_weights,
        payload_rows,
        ROW_CAPACITY=typing.cast(language.constexpr, row_capacity),
        ROUTED_TOPK=typing.cast(language.constexpr, routed_topk),
        EFFECTIVE_TOPK=typing.cast(language.constexpr, effective_topk),
        ROUTED_EXPERT_COUNT=typing.cast(language.constexpr, routed_expert_count),
        ROUTED_SCALING_FACTOR=typing.cast(language.constexpr, routed_scaling_factor),
        BLOCK_SIZE=typing.cast(language.constexpr, ROUTING_FINALIZATION_BLOCK_SIZE),
    )
