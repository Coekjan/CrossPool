"""One prepared Expert route, matrix and reduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
from triton import language
from triton_kernels import reduce
from triton_kernels.specialize import FnSpecs
from triton_kernels.tensor_details import bitmatrix, ragged_tensor
from triton_kernels.tensor_details.bitmatrix_details import sum_bitmatrix_rows

from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import linear
from xpool.utils import align_up

# NVIDIA metadata rows in the pinned Triton-Kernels representation.
RAGGED_METADATA_BLOCK_SIZES_LOG2 = (4, 5, 6, 7)


@triton.jit
def compute_route_order_kernel(
    sorted_to_original,
    original_to_sorted,
    route_ids,
    token_count,
    partial_column_sums,
    partial_token_stride,
    partial_expert_stride,
    column_offsets,
    TOPK: language.constexpr,
    TOKENS_PER_BLOCK: language.constexpr,
    BLOCK_SIZE: language.constexpr,
):
    # Triton bitmatrix stage two: power-of-two padding preserves the original route stride.
    # https://github.com/triton-lang/triton/blob/7c1a7ca5dc89fd1b6fc7008436cad04f0dea1b1e/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py
    ACTUAL_SIZE: language.constexpr = TOKENS_PER_BLOCK * TOPK
    language.static_assert(BLOCK_SIZE <= 32768)
    block = language.program_id(0)
    local = language.arange(0, BLOCK_SIZE)
    original = block * ACTUAL_SIZE + local
    valid = (local < ACTUAL_SIZE) & (original < token_count * TOPK)
    expert = language.load(route_ids + original, valid, other=-1).to(language.uint32)
    pairs = language.sort((expert << 16) | local.to(language.uint32), 0)
    expert = pairs >> 16
    original = block * ACTUAL_SIZE + (pairs & 0xFFFF)
    valid = expert != 0xFFFF
    inclusive = language.associative_scan((pairs & 0xFFFF0000) | 1, 0, bitmatrix._keyed_add)
    within_expert = (inclusive - 1) & 0xFFFF
    sorted_index = language.load(
        partial_column_sums + block * partial_token_stride + expert * partial_expert_stride, valid, other=0
    )
    sorted_index += language.load(column_offsets + expert, valid, other=0)
    sorted_index += within_expert
    language.store(original_to_sorted + original, sorted_index, valid)
    language.store(sorted_to_original + sorted_index, original, valid)


@triton.jit
def scale_route_sum(total, routed_scaling_factor):
    return total * routed_scaling_factor


@triton.jit
def routes_to_bitmatrix_kernel(
    ids,
    bits,
    TOPK: language.constexpr,
    EXPERTS: language.constexpr,
    ROW_STRIDE: language.constexpr,
    WORD_STRIDE: language.constexpr,
    SLOT_BLOCK: language.constexpr,
):
    """Encode the already selected, unique Expert IDs; -1 rows remain empty."""

    row = language.program_id(0)
    word = language.program_id(1)
    slots = language.arange(0, SLOT_BLOCK)
    expert = language.load(ids + row * TOPK + slots, slots < TOPK, other=-1)
    selected = (expert >= 0) & (expert < EXPERTS) & (expert // 32 == word)
    mask = language.where(selected, language.full((), 1, language.uint32) << (expert % 32), 0)
    language.store(bits + row * ROW_STRIDE + word * WORD_STRIDE, language.sum(mask, 0))


@triton.jit
def grouped_route_locations_kernel(
    ordered,
    route_weights,
    gather,
    scatter,
    gammas,
    ROUTES: language.constexpr,
    TOPK: language.constexpr,
    BLOCK: language.constexpr,
):
    """Project upstream sorted route ordinals into gather/scatter and FP32 weights."""

    index = language.program_id(0) * BLOCK + language.arange(0, BLOCK)
    route = language.load(ordered + index, index < ROUTES, other=-1)
    valid = (index < ROUTES) & (route >= 0)
    probability = language.load(route_weights + route, valid, other=0.0)
    language.store(gather + index, language.where(valid, route // TOPK, 0), index < ROUTES)
    language.store(scatter + index, language.where(valid, route, 0), index < ROUTES)
    language.store(gammas + index, probability, index < ROUTES)


@dataclass(frozen=True, slots=True)
class ExpertExecutionPlan:
    """Grouped matrices, routing metadata and exact shared intermediate storage."""

    signature: execution.MoeFfnExecutionSignature
    i_proj: linear.GemmPlan
    o_proj: linear.GemmPlan
    workspace: execution.WorkspaceLayout

    @classmethod
    def prepare(cls, signature: execution.MoeFfnExecutionSignature) -> ExpertExecutionPlan:
        """Prepare the complete pinned metadata format independently of launch tuning."""

        capacity, experts, topk = signature.payload_row_capacity, signature.expert_count, signature.effective_topk
        minimum_block = 1 << RAGGED_METADATA_BLOCK_SIZES_LOG2[0]
        if experts >= 65536 or topk > 1024:
            raise ValueError("routing sorter requires fewer than 65536 Experts and at most 1024 routes per row")
        if triton.cdiv(capacity, minimum_block) > 65536:
            raise ValueError("ragged schedule exceeds its 16-bit per-Expert block index")
        routes = capacity * topk
        activated = TensorLayout((routes, signature.local_intermediate_size), signature.payload_dtype, 0)
        i_proj = linear.GemmPlan.prepare(
            input_layout=TensorLayout(signature.payload_shape, signature.payload_dtype, 0),
            weight_layout=signature.i_proj,
            result_layout=activated,
            activation=signature.activation,
        )
        o_proj = linear.GemmPlan.prepare(
            input_layout=activated,
            weight_layout=signature.o_proj,
            result_layout=TensorLayout((routes, signature.hidden_size), signature.payload_dtype, 0),
            activation=None,
        )
        schedule_columns = ragged_tensor.RaggedTensorMetadata.n_blocks(experts, routes, minimum_block)
        block_count = len(RAGGED_METADATA_BLOCK_SIZES_LOG2)
        projected_bytes = 0 if i_proj.fused_activation is not None else i_proj.output_layout.nbytes
        overlap_bytes = max(projected_bytes, o_proj.output_layout.nbytes)
        workspace = execution.WorkspaceLayout.from_regions(
            (
                (torch.uint32, (triton.cdiv(experts, 32), align_up(capacity, 32))),
                (torch.int32, (align_up(experts, 128),)),
                (torch.int32, (align_up(experts, 32), triton.cdiv(capacity, 128) * 4)),
                (torch.int32, (experts,)),
                (torch.int32, (2 * routes,)),
                (torch.int32, (routes,)),
                (torch.int32, (routes,)),
                (torch.float32, (routes,)),
                (torch.int32, (block_count + 1, align_up(experts + 1, 512))),
                (torch.int32, (block_count, align_up(schedule_columns, 512))),
                (activated.dtype, activated.shape),
                (torch.uint8, (overlap_bytes,)),
                (torch.uint8, (i_proj.workspace.nbytes,)),
                (torch.uint8, (o_proj.workspace.nbytes,)),
            )
        )
        return cls(
            signature=signature,
            i_proj=i_proj,
            o_proj=o_proj,
            workspace=workspace,
        )


class Experts:
    """Capture-time grouped FFN borrowing prepared views and current weights."""

    def __init__(self, plan: ExpertExecutionPlan, workspace: torch.Tensor) -> None:
        views = plan.workspace.views(workspace)
        signature = plan.signature
        self.plan = plan
        self.route_reduction = reduce.PostprocessFn(
            specs=FnSpecs("scale_route_sum", scale_route_sum, ("routed_scaling_factor",)),
            fn_args=(plan.signature.routed_scaling_factor,),
        )
        self.bitmatrix = views[0].transpose(0, 1)[: signature.payload_row_capacity]
        self.histogram = views[1]
        self.partials = views[2].transpose(0, 1)[: triton.cdiv(signature.payload_row_capacity, 32)]
        self.column_offsets = views[3]
        self.ordered, self.gather, self.scatter, self.gammas = views[4:8]
        self.offsets, self.schedule, self.activated = views[8:11]
        overlap = views[11].view(signature.payload_dtype)
        self.projected = (
            overlap[: plan.i_proj.output_layout.shape[0] * plan.i_proj.output_layout.shape[1]].view(
                plan.i_proj.output_layout.shape
            )
            if plan.i_proj.fused_activation is None
            else self.activated
        )
        self.routes = overlap[: signature.payload_row_capacity * signature.effective_topk * signature.hidden_size].view(
            signature.payload_row_capacity * signature.effective_topk, signature.hidden_size
        )
        self.metadata = ragged_tensor.RaggedTensorMetadata(
            self.histogram[: signature.expert_count],
            self.offsets[0, : signature.expert_count + 1],
            self.offsets[1:, : signature.expert_count + 1],
            self.schedule,
        )
        self.i_proj = linear.Gemm(plan.i_proj, views[-2])
        self.o_proj = linear.Gemm(plan.o_proj, views[-1])

    def run_into(
        self,
        *,
        hidden_states: torch.Tensor,
        i_proj: weights.ProjectionWeights,
        o_proj: weights.ProjectionWeights,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write a grouped Partial from final routes, retaining invalid Capacity tails."""

        signature = self.plan.signature
        capacity, experts, topk = signature.payload_row_capacity, signature.expert_count, signature.effective_topk
        routes_to_bitmatrix_kernel.run(
            topk_ids,
            self.bitmatrix,
            topk,
            experts,
            self.bitmatrix.stride(0),
            self.bitmatrix.stride(1),
            triton.next_power_of_2(topk),
            grid=(capacity, triton.cdiv(experts, 32)),
            warmup=False,
        )
        self.histogram.zero_()
        sum_bitmatrix_rows._sum_bitmatrix_rows.run(
            self.bitmatrix,
            capacity,
            self.bitmatrix.stride(0),
            self.bitmatrix.stride(1),
            self.histogram,
            self.partials,
            self.partials.stride(0),
            self.partials.stride(1),
            self.partials.shape[1],
            BLOCK_M=32,
            BLOCK_MM=128,
            num_warps=8,
            grid=(triton.cdiv(capacity, 128), triton.cdiv(experts, 32)),
            warmup=False,
        )
        route_count = capacity * topk
        bitmatrix._bitmatrix_metadata_compute_stage1.run(
            self.ordered,
            2 * route_count,
            -1,
            1024,
            self.histogram,
            self.column_offsets,
            experts,
            self.partials,
            self.partials.shape[0],
            self.partials.stride(0),
            self.partials.stride(1),
            BLOCK_M=512,
            BLOCK_N=512,
            grid=(triton.cdiv(2 * route_count, 1024) + experts + 1,),
            warmup=False,
        )
        compute_route_order_kernel.run(
            self.ordered[:route_count],
            self.ordered[route_count:],
            topk_ids,
            capacity,
            self.partials,
            self.partials.stride(0),
            self.partials.stride(1),
            self.column_offsets,
            TOPK=topk,
            TOKENS_PER_BLOCK=32,
            BLOCK_SIZE=triton.next_power_of_2(32 * topk),
            grid=(triton.cdiv(capacity, 32),),
            warmup=False,
        )
        grouped_route_locations_kernel.run(
            self.ordered,
            topk_weights,
            self.gather,
            self.scatter,
            self.gammas,
            route_count,
            topk,
            256,
            grid=(triton.cdiv(route_count, 256),),
            warmup=False,
        )
        block_count = len(RAGGED_METADATA_BLOCK_SIZES_LOG2)
        ragged_tensor._ragged_tensor_metadata_memset.run(
            self.histogram,
            experts,
            self.offsets,
            self.offsets.stride(0),
            self.schedule,
            RAGGED_METADATA_BLOCK_SIZES_LOG2[0],
            SIZES=block_count,
            BLOCK=512,
            num_warps=4,
            grid=(block_count + 1 + self.schedule.numel() // 512,),
            warmup=False,
        )
        ragged_tensor._ragged_tensor_metadata_compute.run(
            self.histogram,
            self.offsets[1:],
            self.offsets.stride(0),
            self.schedule,
            self.schedule.stride(0),
            RAGGED_METADATA_BLOCK_SIZES_LOG2[0],
            SIZES=block_count,
            BLOCK=512,
            num_warps=4,
            grid=(block_count * experts,),
            warmup=False,
        )
        self.i_proj.run_into(hidden_states, i_proj, output=self.projected, metadata=self.metadata, gather=self.gather)
        if self.plan.i_proj.fused_activation is None:
            signature.activation.apply_into(
                inputs=self.projected,
                output=self.activated,
                order=self.plan.i_proj.weight_layout.gate_up_order,
            )
        self.routes.zero_()
        self.o_proj.run_into(
            self.activated,
            o_proj,
            output=self.routes,
            metadata=self.metadata,
            scatter=self.scatter,
            gammas=self.gammas,
        )
        reduce.reduce_forward(
            self.routes.view(capacity, topk, signature.hidden_size),
            dim=1,
            y=output,
            y_has_mx=False,
            postprocess_fn1=self.route_reduction,
        )
