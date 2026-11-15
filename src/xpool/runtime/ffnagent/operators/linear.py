"""Prepared ordinary and grouped projections into caller-owned output."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from dataclasses import dataclass, replace
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
        elif tile_shape is None and weight_layout.bias is not None and weight_layout.bias.dtype is not torch.float32:
            regions.append((torch.float32, weight_layout.bias.shape))
        return cls(
            input_layout=input_layout,
            weight_layout=weight_layout,
            output_layout=TensorLayout((result_layout.shape[0], output_width), result_layout.dtype, 0),
            fused_activation=fused,
            workspace=execution.WorkspaceLayout.from_regions(tuple(regions)),
            tile_shape=tile_shape,
            split_k=split_k,
        )


class Gemm:
    """Capture-time matrix operation borrowing its plan and scratch.

    Registry validation establishes layer layouts before capture. Warmup binds
    launch flags; each invocation supplies current layer values, scales and
    bias without retaining their addresses.
    """

    def __init__(self, plan: GemmPlan, workspace: torch.Tensor) -> None:
        if workspace.device.type != "cuda":
            raise ValueError("GEMM scratch must reside on its execution device")
        views = plan.workspace.views(workspace)
        self.plan = plan
        self.bias = views[0] if views and plan.tile_shape is None else None
        self.partials = views[0] if views and plan.tile_shape is not None else None
        self.flags: opt_flags.OptFlags | None = None

    def prepare_launch(
        self,
        inputs: torch.Tensor,
        projection: weights.ProjectionWeights,
        *,
        metadata: ragged_tensor.RaggedTensorMetadata | None = None,
        gather: torch.Tensor | None = None,
        scatter: torch.Tensor | None = None,
    ) -> None:
        """Bind flags during uncaptured warmup from the actual operand geometry."""

        rhs, precision = self.operands(projection)
        lhs = tensor.wrap_torch_tensor(inputs)
        opt_flags.reset_opt_flags()
        opt_flags.reset_opt_flags_constraints()
        opt_flags.update_opt_flags_constraints({"is_persistent": False, "split_k": 1})
        flags = opt_flags.make_opt_flags(
            out_dtype=tensor.torch_dtype_to_dtype(self.plan.output_layout.dtype),
            lhs_dtype=lhs.dtype,
            rhs_dtype=rhs.dtype,
            precision_config=precision,
            batch_size=self.plan.weight_layout.values.shape[0] if metadata is None and rhs.ndim == 3 else 1,
            m=gather.shape[0] if gather is not None else inputs.shape[-2],
            n=self.plan.weight_layout.values.shape[-2],
            k=inputs.shape[-1],
            ragged_metadata=metadata,
            can_use_persistent_tma=False,
            can_use_split_k=scatter is None and precision.b_mx_scale is None,
            epilogue_effective_itemsize=False,
            x_transpose=inputs.stride(-1) != 1,
            has_y_acc_in=False,
            block_k=None,
        )
        if metadata is None and lhs.dtype.bitwidth == rhs.dtype.bitwidth == 16:
            # Bound column work per CTA while exposing enough tiles to the device.
            width = self.plan.weight_layout.values.shape[-2]
            block_m = min(flags.block_m, 32) if width <= 128 else flags.block_m
            tiles = triton.cdiv(inputs.shape[-2], block_m)
            multiprocessors = torch.cuda.get_device_properties(inputs.device).multi_processor_count
            block_n = min(flags.block_n, max(16, triton.next_power_of_2(triton.cdiv(width * tiles, multiprocessors))))
            if self.plan.output_layout.dtype is torch.float32 and inputs.shape[-2] > 256:
                block_m, block_n = 64, 64
            flags = replace(flags, block_m=block_m, block_n=block_n, num_warps=max(flags.num_warps, 4))
        elif metadata is not None and lhs.dtype.bitwidth == 16 and rhs.dtype.bitwidth in (4, 16):
            # The bundled one-warp ragged selection regresses measured floating and MXFP4 execution.
            flags = replace(flags, num_warps=max(flags.num_warps, 4))
        self.flags = flags

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

    @contextlib.contextmanager
    def configuration(self) -> Generator[None, None, None]:
        """Install this GEMM's flags within the serial FFN capture scope."""

        if self.flags is None:
            raise RuntimeError("GEMM launch configuration requires uncaptured warmup")
        opt_flags.reset_opt_flags_constraints()
        opt_flags.reset_opt_flags()
        try:
            opt_flags.set_opt_flags(self.flags)
            yield
        finally:
            opt_flags.reset_opt_flags()
            opt_flags.reset_opt_flags_constraints()
            opt_flags.update_opt_flags_constraints({"is_persistent": False, "split_k": 1})

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
        if self.flags is None:
            self.prepare_launch(inputs, projection, metadata=metadata, gather=gather, scatter=scatter)
        rhs, precision = self.operands(projection)
        bias = projection.bias
        if bias is not None and self.bias is not None:
            # The bundled non-persistent GEMM merges bias loads with an FP32 zero branch.
            copy_tensor_kernel.run(
                bias,
                self.bias,
                bias.numel(),
                BLOCK=256,
                grid=(triton.cdiv(bias.numel(), 256),),
                warmup=False,
            )
            bias = self.bias
        with self.configuration():
            matmul.matmul(
                inputs,
                rhs,
                bias,
                a_ragged_metadata=metadata,
                gather_indx=gather,
                scatter_indx=scatter,
                gammas=gammas,
                precision_config=precision,
                fused_activation=self.plan.fused_activation,
                c=output,
            )
