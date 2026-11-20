"""Prepared ordinary and grouped projections into caller-owned output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import triton
from triton import language
from triton.runtime.jit import JITFunction
from triton_kernels import matmul, tensor
from triton_kernels.matmul_details import opt_flags
from triton_kernels.tensor_details import ragged_tensor

from xpool.quantization import Mxfp4, TensorLayout, Unquantized
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import simt
from xpool.runtime.ffnagent.operators.activation import Activation


@triton.jit
def copy_tensor_kernel(source, target, ELEMENTS: language.constexpr, BLOCK: language.constexpr):
    index = language.program_id(0) * BLOCK + language.arange(0, BLOCK)
    value = language.load(source + index, index < ELEMENTS, other=0)
    language.store(target + index, value, index < ELEMENTS)


@dataclass(frozen=True, slots=True)
class GemmPlan:
    """Address-free operand, arithmetic and local scratch contract."""

    input_layout: TensorLayout
    weight_layout: weights.ProjectionLayout
    output_layout: TensorLayout
    fused_activation: matmul.FusedActivation | None
    workspace: execution.WorkspaceLayout
    tile_shape: tuple[int, int, int] | None
    split_k: int = 1

    @classmethod
    def prepare(
        cls,
        *,
        input_layout: TensorLayout,
        weight_layout: weights.ProjectionLayout,
        result_layout: TensorLayout,
        activation: Activation | None,
    ) -> GemmPlan:
        """Prepare exact stores and preserve the declared activation rounding."""

        if len(input_layout.shape) != 2 or len(result_layout.shape) != 2:
            raise ValueError("GEMM input and result require two-dimensional layouts")
        if len(weight_layout.values.shape) not in (2, 3):
            raise ValueError("GEMM weights require ordinary or grouped matrices")
        fused = None
        if activation is not None:
            match weight_layout.quantization:
                case Unquantized():
                    projection_round_dtype = result_layout.dtype
                case Mxfp4():
                    projection_round_dtype = None
            fused = activation.fused(
                order=weight_layout.gate_up_order,
                projection_round_dtype=projection_round_dtype,
            )
        width = result_layout.shape[1]
        projected_width = width * (2 if activation is not None and activation.gate is not None else 1)
        logical_shape = (*weight_layout.values.shape[:-2], projected_width, input_layout.shape[1])
        encoded = weight_layout.quantization.weight_layout(logical_shape=logical_shape, dtype=input_layout.dtype)
        if weight_layout.values != encoded.values or weight_layout.scales != encoded.scales:
            raise ValueError("GEMM weight representation disagrees with its logical geometry")
        if activation is not None and activation.gate is not None and weight_layout.gate_up_order is None:
            raise ValueError("gated GEMM preparation requires physical Gate/Up order")
        output_width = width if fused is not None else projected_width
        rows, hidden = input_layout.shape
        tile_shape = None
        split_k = 1
        if isinstance(weight_layout.quantization, Unquantized) and len(weight_layout.values.shape) == 2:
            if rows == 1 or (projected_width <= 128 and rows <= 256):
                tile_shape = (1, 4, min(2048, triton.next_power_of_2(hidden)))
            elif input_layout.dtype is torch.float32 and weight_layout.values.dtype is torch.float32:
                tile_shape = (128, 64, 32) if projected_width <= 128 else (128, 128, 8)
                split_k = min(8 if projected_width <= 128 else 4, triton.cdiv(hidden, tile_shape[2]))
        regions = []
        if tile_shape is not None and tile_shape[0] != 1:
            regions.append((torch.float32, (split_k, rows, projected_width)))
        return cls(
            input_layout=input_layout,
            weight_layout=weight_layout,
            output_layout=TensorLayout((result_layout.shape[0], output_width), result_layout.dtype, 0),
            fused_activation=fused,
            workspace=execution.WorkspaceLayout.from_regions(tuple(regions)),
            tile_shape=tile_shape,
            split_k=split_k,
        )

    def launch_constraints(self, *, expert_count: int | None = None) -> dict[str, int]:
        """Apply measured tile and warp constraints before the library selects a launch."""

        if expert_count is not None:
            # Small grouped tiles otherwise use one or two warps, which regresses Expert replay.
            return {"num_warps": 4} if self.output_layout.shape[0] // expert_count <= 16 else {}
        rows, width = self.input_layout.shape[0], self.weight_layout.values.shape[-2]
        if width <= 128 and rows > 256:
            return {"block_m": 32, "num_warps": 4}
        if width > 128 and rows == 32:
            return {"block_m": 16, "num_warps": 4}
        return {}

    def split_k_workspace_bytes(self, sm_count: int, *, expert_count: int | None = None) -> int:
        """Size the bundled FP32 partials for ordinary or gather execution."""

        if self.tile_shape is not None or self.weight_layout.scales is not None:
            return 0
        width, reduction = self.weight_layout.values.shape[-2:]
        if expert_count is None:
            rows = self.input_layout.shape[0]
            block_m = self.launch_constraints().get("block_m", 128)
        else:
            rows = self.output_layout.shape[0]
            rows_per_expert = max(1, rows // expert_count)
            block_m = (
                min(64, triton.next_power_of_2(2 * rows_per_expert))
                if rows_per_expert <= 64
                else min(128, triton.next_power_of_2(rows_per_expert))
            )
            block_m = max(16, block_m)
        tiles = triton.cdiv(rows, block_m) * triton.cdiv(width, 256)
        splits = min(sm_count // tiles, triton.cdiv(reduction, 64) // 4)
        return splits * rows * width * torch.float32.itemsize if splits > 1 else 0


class Gemm:
    """Capture-time matrix operation borrowing its plan and scratch.

    Each launch supplies the current layer's values, scales and bias over
    the prepared layouts.
    """

    def __init__(self, plan: GemmPlan, workspace: torch.Tensor) -> None:
        if workspace.device.type != "cuda":
            raise ValueError("GEMM scratch must reside on its execution device")
        views = plan.workspace.views(workspace)
        self.plan = plan
        self.partials = views[0] if views else None

    def operands(self, projection: weights.ProjectionWeights) -> tuple[tensor.Tensor, matmul.PrecisionConfig]:
        """Borrow the current layer's logical weight view and precision settings."""

        precision = matmul.PrecisionConfig(out_dtype=self.plan.output_layout.dtype, allow_tf32=False)
        match self.plan.weight_layout.quantization:
            case Unquantized():
                rhs = tensor.wrap_torch_tensor(projection.values.transpose(-1, -2))
            case Mxfp4():
                rhs = tensor.wrap_torch_tensor(projection.values.transpose(-1, -2), dtype=tensor.FP4)
                scales = projection.scales
                if scales is None:
                    raise ValueError("MXFP4 projection requires its declared scales")
                precision.b_mx_scale = scales.transpose(-1, -2)
        return rhs, precision

    def run_into(
        self,
        inputs: torch.Tensor,
        projection: weights.ProjectionWeights,
        *,
        output: torch.Tensor,
        metadata: ragged_tensor.RaggedTensorMetadata | None = None,
        gather: torch.Tensor | None = None,
        scatter: torch.Tensor | None = None,
        gammas: torch.Tensor | None = None,
    ) -> None:
        """Write the prepared projection using current layer addresses."""

        tile = self.plan.tile_shape
        if tile is not None:
            rows, hidden = self.plan.input_layout.shape
            width = self.plan.weight_layout.values.shape[-2]
            fused = self.plan.fused_activation
            function = fused.specs.fn if fused is not None else None
            arguments = fused.fn_args if fused is not None else ()
            reduction = fused.specs.reduction_n if fused is not None else 1
            block_m, block_n, block_k = tile
            if block_m == 1:
                simt.row_projection_kernel.run(
                    inputs,
                    projection.values,
                    projection.bias,
                    output,
                    hidden,
                    width,
                    self.plan.output_layout.shape[1],
                    block_k,
                    block_n,
                    reduction,
                    function,
                    arguments,
                    grid=(rows, triton.cdiv(width, block_n)),
                    num_warps=4,
                    warmup=False,
                )
            else:
                # Gluon's decorator annotation also includes its decorator-factory return.
                kernel = cast(JITFunction, simt.shared_pipeline_kernel if block_n == 64 else simt.microtile_kernel)
                kernel.run(
                    inputs,
                    projection.values,
                    self.partials,
                    rows,
                    width,
                    hidden,
                    block_m,
                    block_n,
                    block_k,
                    self.plan.split_k,
                    grid=(triton.cdiv(rows, block_m), triton.cdiv(width, block_n), self.plan.split_k),
                    num_warps=4,
                    num_stages=1,
                    warmup=False,
                )
                simt.reduce_projection_kernel.run(
                    self.partials,
                    projection.bias,
                    output,
                    rows,
                    width,
                    self.plan.split_k,
                    triton.next_power_of_2(self.plan.split_k),
                    reduction,
                    function,
                    arguments,
                    BLOCK=256,
                    grid=(triton.cdiv(rows * width, 256),),
                    num_warps=4,
                    warmup=False,
                )
            return
        rhs, precision = self.operands(projection)
        constraints = self.plan.launch_constraints(expert_count=None if metadata is None else metadata.n_slices)
        if constraints:
            opt_flags.update_opt_flags_constraints(constraints)
        try:
            matmul.matmul(
                inputs,
                rhs,
                projection.bias,
                a_ragged_metadata=metadata,
                gather_indx=gather,
                scatter_indx=scatter,
                gammas=gammas,
                precision_config=precision,
                fused_activation=self.plan.fused_activation,
                c=output,
            )
        finally:
            if constraints:
                opt_flags.reset_opt_flags_constraints()
