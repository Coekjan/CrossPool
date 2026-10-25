"""Packed Expert execution with caller-owned routing and matmul storage."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Generator
from typing import ClassVar

import torch
import triton
from triton import language
from triton_kernels import matmul, tensor
from triton_kernels.matmul_details import opt_flags
from triton_kernels.tensor_details import bitmatrix, dtype, ragged_tensor
from triton_kernels.tensor_details.bitmatrix_details import sum_bitmatrix_rows

from xpool import ffn
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import activation
from xpool.utils import align_up


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


class TritonKernelsExperts:
    """Strided MXFP4 grouped matmuls with FP32 bias and declared gated SiLU.

    The object borrows its workspace and retains no layer-weight addresses.
    Existing pre-capture warmup resolves upstream launch flags against actual
    resources; later captures reuse them. Native replay performs no Python
    selection. Temporary dependency bindings are confined to serial startup.
    """

    required_gate_up_order: ClassVar[ffn.GateUpOrder] = ffn.GateUpOrder.INTERLEAVED
    bias_dtype: ClassVar[torch.dtype] = torch.float32

    @staticmethod
    def workspace_layout(
        signature: execution.MoeFfnExecutionSignature,
    ) -> tuple[tuple[tuple[torch.dtype, tuple[int, ...]], ...], tuple[int, ...], int]:
        """Account complete padded Expert metadata and intermediates."""

        capacity, experts, topk = signature.payload_row_capacity, signature.expert_count, signature.effective_topk
        if experts >= 65536 or topk > 1024:
            raise ValueError("packed routing sorter requires fewer than 65536 Experts and at most 1024 routes per row")
        if triton.cdiv(capacity, min(execution.QUALIFIED_MOE_BLOCK_SIZE_M_VALUES)) > 65536:
            raise ValueError("packed ragged schedule exceeds its 16-bit per-Expert block index")
        routes = capacity * topk
        block_sizes = execution.QUALIFIED_MOE_BLOCK_SIZE_M_VALUES
        maximum_blocks = ragged_tensor.RaggedTensorMetadata.n_blocks(experts, routes, min(block_sizes))
        regions = [
            (torch.uint32, (triton.cdiv(experts, 32), align_up(capacity, 32))),
            (torch.int32, (align_up(experts, 128),)),
            (torch.int32, (align_up(experts, 32), triton.cdiv(capacity, 128) * 4)),
            (torch.int32, (experts,)),
            (torch.int32, (2 * routes,)),
            (torch.int32, (routes,)),
            (torch.int32, (routes,)),
            (torch.float32, (routes,)),
            (torch.int32, (len(block_sizes) + 1, align_up(experts + 1, 512))),
            (torch.int32, (len(block_sizes), align_up(maximum_blocks, 512))),
            (signature.payload_dtype, (routes, signature.local_intermediate_size)),
            (signature.payload_dtype, (routes, signature.hidden_size)),
        ]
        offsets = []
        cursor = 0
        for storage_dtype, shape in regions:
            cursor = align_up(cursor, execution.OPERATOR_ALIGNMENT_BYTES)
            offsets.append(cursor)
            cursor += math.prod(shape) * storage_dtype.itemsize
        return tuple(regions), tuple(offsets), cursor

    def __init__(self, signature: execution.MoeFfnExecutionSignature, workspace: torch.Tensor) -> None:
        regions, offsets, total = self.workspace_layout(signature)
        weights.validate_tensor(workspace, name="packed Expert workspace", dtype=torch.uint8, dimensions=1)
        if workspace.numel() < total:
            raise ValueError("packed Expert workspace is smaller than its declared extent")
        views = tuple(
            workspace[offset : offset + math.prod(shape) * dtype.itemsize].view(dtype).view(shape)
            for offset, (dtype, shape) in zip(offsets, regions, strict=True)
        )
        self.signature = signature
        self.workspace = workspace
        self.bitmatrix = views[0].transpose(0, 1)[: signature.payload_row_capacity]
        self.histogram = views[1]
        self.partials = views[2].transpose(0, 1)[: triton.cdiv(signature.payload_row_capacity, 32)]
        self.column_offsets = views[3]
        self.ordered = views[4]
        self.gather = views[5]
        self.scatter = views[6]
        self.gammas = views[7]
        self.offsets = views[8]
        self.schedule = views[9]
        self.intermediate = views[10]
        self.routes = views[11]
        self.metadata = ragged_tensor.RaggedTensorMetadata(
            self.histogram[: signature.expert_count],
            self.offsets[0, : signature.expert_count + 1],
            self.offsets[1:, : signature.expert_count + 1],
            self.schedule[
                :,
                : ragged_tensor.RaggedTensorMetadata.n_blocks(
                    signature.expert_count,
                    signature.payload_row_capacity * signature.effective_topk,
                    min(execution.QUALIFIED_MOE_BLOCK_SIZE_M_VALUES),
                ),
            ],
        )
        self.activation = activation.fused(signature.activation)
        self.configurations: dict[tuple[int, int], opt_flags.OptFlags] = {}

    @contextlib.contextmanager
    def configuration(self) -> Generator[None, None, None]:
        """Cache real upstream tuning during warmup and restore borrowed bindings."""

        previous_selector = matmul.make_opt_flags
        previous_constraints = opt_flags._opt_flags_constraints
        previous_flags = opt_flags._opt_flags

        def select(
            out_dtype: dtype.FloatType | dtype.IntegerType,
            lhs_dtype: dtype.FloatType | dtype.IntegerType,
            rhs_dtype: dtype.FloatType | dtype.IntegerType,
            precision_config: matmul.PrecisionConfig,
            batch_size: int,
            m: int,
            n: int,
            k: int,
            ragged_metadata: ragged_tensor.RaggedTensorMetadata | None,
            can_use_persistent_tma: bool,
            can_use_split_k: bool,
            epilogue_effective_itemsize: float | None,
            x_transpose: bool,
            has_y_acc_in: bool,
            block_k: int | None,
        ) -> opt_flags.OptFlags:
            key = (n, k)
            selected = self.configurations.get(key)
            if selected is None:
                selected = previous_selector(
                    out_dtype,
                    lhs_dtype,
                    rhs_dtype,
                    precision_config,
                    batch_size,
                    m,
                    n,
                    k,
                    ragged_metadata,
                    can_use_persistent_tma,
                    can_use_split_k,
                    epilogue_effective_itemsize,
                    x_transpose,
                    has_y_acc_in,
                    block_k,
                )
                if selected.block_m not in execution.QUALIFIED_MOE_BLOCK_SIZE_M_VALUES:
                    raise ValueError("packed matmul selected an unsupported ragged block size")
                self.configurations[key] = selected
            return selected

        opt_flags._opt_flags_constraints = {**previous_constraints, "is_persistent": False, "split_k": 1}
        opt_flags._opt_flags = None
        setattr(matmul, "make_opt_flags", select)
        try:
            yield
        finally:
            setattr(matmul, "make_opt_flags", previous_selector)
            opt_flags._opt_flags_constraints = previous_constraints
            opt_flags._opt_flags = previous_flags

    def run(
        self,
        *,
        hidden_states: torch.Tensor,
        gate_up: weights.ProjectionWeights,
        down: weights.ProjectionWeights,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write a packed Expert Partial using final routes and retained resources.

        Every route row contains distinct valid Expert IDs or an invalid -1
        tail. All output and scratch storage is caller-owned. Scale and bias
        addresses are supplied afresh for both Primary and Control capture.
        """

        signature = self.signature
        capacity, experts, topk = signature.payload_row_capacity, signature.expert_count, signature.effective_topk
        if gate_up.scales is None or down.scales is None:
            raise ValueError("packed Expert projections require scales")
        for name, projection, layout in (
            ("gate/up", gate_up, signature.gate_up),
            ("down", down, signature.down),
        ):
            projection.validate(layout)
            if projection.values.device != self.workspace.device:
                raise ValueError(f"packed Expert {name} must share the workspace device")
        for name, value, expected_dtype, shape in (
            ("hidden states", hidden_states, signature.payload_dtype, (capacity, signature.hidden_size)),
            ("routes", topk_ids, torch.int32, (capacity, topk)),
            ("route weights", topk_weights, torch.float32, (capacity, topk)),
            ("output", output, signature.payload_dtype, (capacity, signature.hidden_size)),
        ):
            weights.validate_tensor(value, name=name, dtype=expected_dtype, dimensions=2)
            if tuple(value.shape) != shape or value.device != self.workspace.device:
                raise ValueError(f"packed Expert {name} disagrees with its execution layout")
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
        bitmatrix._bitmatrix_metadata_compute_stage2.run(
            self.ordered[:route_count],
            self.ordered[route_count:],
            topk_ids,
            capacity,
            self.partials,
            self.partials.stride(0),
            self.partials.stride(1),
            self.column_offsets,
            TOKS_PER_ROW=topk,
            BLOCK_PER_TOK=32,
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
        block_count = len(execution.QUALIFIED_MOE_BLOCK_SIZE_M_VALUES)
        ragged_tensor._ragged_tensor_metadata_memset.run(
            self.histogram,
            experts,
            self.offsets,
            self.offsets.stride(0),
            self.schedule,
            4,
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
            4,
            SIZES=block_count,
            BLOCK=512,
            num_warps=4,
            grid=(block_count * experts,),
            warmup=False,
        )
        with self.configuration():
            matmul.matmul(
                hidden_states,
                tensor.wrap_torch_tensor(gate_up.values.transpose(-1, -2), dtype=tensor.FP4),
                gate_up.bias,
                a_ragged_metadata=self.metadata,
                gather_indx=self.gather,
                precision_config=matmul.PrecisionConfig(b_mx_scale=gate_up.scales.transpose(-1, -2), allow_tf32=False),
                c=self.intermediate,
                fused_activation=self.activation,
            )
            self.routes.zero_()
            matmul.matmul(
                self.intermediate,
                tensor.wrap_torch_tensor(down.values.transpose(-1, -2), dtype=tensor.FP4),
                down.bias,
                a_ragged_metadata=self.metadata,
                scatter_indx=self.scatter,
                gammas=self.gammas,
                precision_config=matmul.PrecisionConfig(b_mx_scale=down.scales.transpose(-1, -2), allow_tf32=False),
                c=self.routes,
            )
        # Loading sgl_kernel selects a device; keep it after role/device setup.
        import sgl_kernel

        sgl_kernel.moe_sum_reduce(
            self.routes.view(capacity, topk, signature.hidden_size),
            output,
            signature.routed_scaling_factor,
        )
