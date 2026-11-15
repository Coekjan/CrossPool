"""Router projection, selection and final Routing Metadata computation."""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass

import torch
import triton
from triton import language

from xpool import ffn
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import linear

ROUTING_FINALIZATION_BLOCK_SIZE = 256


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
    ROUND_TO_LOGITS_DTYPE: language.constexpr,
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

    if RENORMALIZE or ROUND_TO_LOGITS_DTYPE:
        # Slot loads can use different threads from the selection-loop stores.
        language.debug_barrier()
        slots = language.arange(0, TOPK_BLOCK_SIZE)
        output_offsets = row * TOPK + slots
        selected_weights = language.load(routed_weights_pointer + output_offsets, mask=slots < TOPK, other=0.0)
        if RENORMALIZE:
            denominator = language.where(normalizer > 0.0, normalizer, 1.0)
            selected_weights = selected_weights / denominator
        if ROUND_TO_LOGITS_DTYPE:
            selected_weights = selected_weights.to(logits_pointer.dtype.element_ty).to(language.float32)
        language.store(routed_weights_pointer + output_offsets, selected_weights, mask=slots < TOPK)


@dataclass(frozen=True, slots=True)
class RouterExecutionPlan:
    """Router matrix, precision conversion and route-output storage."""

    signature: execution.MoeRouterExecutionSignature
    projection: linear.GemmPlan
    workspace: execution.WorkspaceLayout

    @classmethod
    def prepare(
        cls,
        signature: execution.MoeRouterExecutionSignature,
        *,
        input_layout: TensorLayout,
        routed_topk: int,
    ) -> RouterExecutionPlan:
        """Compose exact Router views from the immutable mathematical declaration."""

        rows, hidden = input_layout.shape
        input_dtype = signature.projection_precision.weight_dtype(payload_dtype=input_layout.dtype)
        logits_dtype = (
            input_layout.dtype
            if signature.projection_precision is ffn.RouterProjectionPrecision.PAYLOAD
            else torch.float32
        )
        projection = linear.GemmPlan.prepare(
            input_layout=TensorLayout((rows, hidden), input_dtype, 0),
            weight_layout=signature.projection,
            result_layout=TensorLayout((rows, signature.routed_expert_count), logits_dtype, 0),
            activation=None,
        )
        converted = (
            (torch.float32, (rows, hidden))
            if signature.projection_precision is ffn.RouterProjectionPrecision.FP32
            else (torch.uint8, (0,))
        )
        workspace = execution.WorkspaceLayout.from_regions(
            (
                (projection.output_layout.dtype, projection.output_layout.shape),
                converted,
                (torch.int32, (rows, routed_topk)),
                (torch.float32, (rows, routed_topk)),
                (torch.uint8, (projection.workspace.nbytes,)),
            )
        )
        return cls(
            signature=signature,
            projection=projection,
            workspace=workspace,
        )


class Router:
    """Capture-time projection and TopK over prepared caller-owned storage."""

    def __init__(self, plan: RouterExecutionPlan, workspace: torch.Tensor) -> None:
        views = plan.workspace.views(workspace)
        self.plan = plan
        self.logits, self.converted_input, self.ids, self.weights = views[:4]
        self.projection = linear.Gemm(
            plan.projection,
            views[-1],
        )

    def run_into(self, *, hidden_states: torch.Tensor, router_weights: weights.MoeRouterWeights) -> None:
        """Write routed-only IDs and FP32 carrier weights at the declared rounding boundary."""

        signature = self.plan.signature
        router_input = hidden_states
        if signature.projection_precision is ffn.RouterProjectionPrecision.FP32:
            router_input = self.converted_input
            linear.copy_tensor_kernel.run(
                hidden_states,
                router_input,
                hidden_states.numel(),
                BLOCK=256,
                grid=(triton.cdiv(hidden_states.numel(), 256),),
                warmup=False,
            )
        self.projection.run_into(router_input, router_weights.projection, output=self.logits)
        rows = self.logits.shape[0]
        topk = self.ids.shape[1]
        # Triton launch metadata exceeds the constexpr-annotated signature.
        typing.cast(typing.Callable[..., object], router_topk_kernel[(rows,)])(
            self.logits,
            router_weights.correction_bias,
            self.ids,
            self.weights,
            EXPERT_COUNT=signature.routed_expert_count,
            TOPK=topk,
            EXPERT_BLOCK_SIZE=triton.next_power_of_2(signature.routed_expert_count),
            TOPK_BLOCK_SIZE=triton.next_power_of_2(topk),
            SOFTMAX=signature.algorithm is ffn.RouterAlgorithm.SOFTMAX,
            CORRECTION_BIAS_PRESENT=router_weights.correction_bias is not None,
            RENORMALIZE=signature.renormalize,
            ROUND_TO_LOGITS_DTYPE=signature.weight_rounding is ffn.TopKWeightRounding.LOGITS_DTYPE,
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
