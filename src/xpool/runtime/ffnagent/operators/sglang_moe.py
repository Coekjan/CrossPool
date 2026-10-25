"""Concrete FfnAgent-owned FFN operator composition."""

from __future__ import annotations

import contextlib
import math
import types
from collections.abc import Generator
from typing import ClassVar

import torch
from triton import language

from xpool import ffn
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import activation


@contextlib.contextmanager
def sglang_moe_config_selection() -> Generator[None, None, None]:
    """Supply the pinned MoE selector's sole startup execution value.

    Side Effects:
        Temporarily replaces the selector module's bound
        ``get_exec`` function and restores the exact previous
        function before returning or propagating an exception. The context is
        single-owner startup state and is not safe for concurrent use.
    """

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    previous = fused_moe_triton_config.get_exec
    setattr(
        fused_moe_triton_config,
        "get_exec",
        lambda: types.SimpleNamespace(
            deterministic=types.SimpleNamespace(enable_deterministic_inference=False),
        ),
    )
    try:
        yield
    finally:
        setattr(fused_moe_triton_config, "get_exec", previous)


def copy_moe_kernel_config(config: object, *, name: str) -> dict[str, int]:
    """Copy one selected private-launcher configuration into plain values."""

    if not isinstance(config, dict):
        raise RuntimeError(f"{name} MoE kernel configuration must be a dict")
    copied: dict[str, int] = {}
    for key, value in config.items():
        if key == "USE_TMA":
            continue
        if not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"{name} MoE kernel configuration contains an invalid entry")
        copied[key] = value
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"):
        if key not in copied:
            raise RuntimeError(f"{name} MoE kernel configuration is missing {key}")
    return copied


def select_moe_kernel_configs(
    *,
    signature: execution.MoeFfnExecutionSignature,
) -> tuple[dict[str, int], dict[str, int] | None]:
    """Select immutable pinned W13 and optional W2 launch configurations.

    Args:
        signature: Fixed geometry used for the pinned configuration tables.

    Returns:
        Plain copied W13 configuration and an optional independently selected
        W2 configuration. TMA is unconditionally disabled.

    Raises:
        ValueError: If Capacity or route width is inconsistent with weights.
        RuntimeError: If the pinned selector returns an invalid mapping.

    Side Effects:
        Imports and consults pinned SGLang startup configuration while a
        dedicated execution context is installed. The original context getter
        is restored before return.
    """

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    with sglang_moe_config_selection():
        selected, down = fused_moe_triton_config.try_get_optimal_moe_config(
            (signature.expert_count, 2 * signature.local_intermediate_size, signature.hidden_size),
            (signature.expert_count, signature.hidden_size, signature.local_intermediate_size),
            signature.effective_topk,
            None,
            signature.payload_row_capacity,
            is_marlin=False,
            block_shape=None,
            per_channel_quant=False,
            return_down_config=True,
        )
    w13_config = copy_moe_kernel_config(selected, name="W13")
    down_config = down[0]
    w2_config = None if down_config is None else copy_moe_kernel_config(down_config, name="W2")
    block_size_m = w13_config["BLOCK_SIZE_M"]
    if block_size_m not in execution.QUALIFIED_MOE_BLOCK_SIZE_M_VALUES:
        raise RuntimeError(f"W13 BLOCK_SIZE_M {block_size_m} is outside the qualified domain")
    if w2_config is not None and w2_config["BLOCK_SIZE_M"] != block_size_m:
        raise RuntimeError("W13 and W2 BLOCK_SIZE_M must match one alignment")
    return w13_config, w2_config


class SglangTritonExperts:
    """Pinned SGLang grouped execution over borrowed capture workspace."""

    required_gate_up_order: ClassVar[ffn.GateUpOrder] = ffn.GateUpOrder.CONCATENATED
    bias_dtype: ClassVar[torch.dtype] = torch.float32

    @staticmethod
    def workspace_layout(
        signature: execution.MoeFfnExecutionSignature,
    ) -> tuple[tuple[tuple[torch.dtype, tuple[int, ...]], ...], tuple[int, ...], int]:
        """Cover every admitted configuration with the existing overlapping layout."""

        return max(
            (
                execution.moe_workspace_layout(signature, block_size_m=value)
                for value in execution.QUALIFIED_MOE_BLOCK_SIZE_M_VALUES
            ),
            key=lambda layout: layout[2],
        )

    def __init__(self, signature: execution.MoeFfnExecutionSignature, workspace: torch.Tensor) -> None:
        weights.validate_tensor(workspace, name="floating Expert workspace", dtype=torch.uint8, dimensions=1)
        if workspace.numel() < self.workspace_layout(signature)[2]:
            raise ValueError("floating Expert workspace is smaller than its declared extent")
        self.signature = signature
        self.workspace = workspace
        self.w13_config, self.w2_config = select_moe_kernel_configs(signature=signature)
        regions, offsets, _ = execution.moe_workspace_layout(
            signature,
            block_size_m=self.w13_config["BLOCK_SIZE_M"],
        )
        views = tuple(
            workspace[offset : offset + math.prod(shape) * dtype.itemsize].view(dtype).view(shape)
            for offset, (dtype, shape) in zip(offsets, regions, strict=True)
        )
        self.sorted_token_ids, self.expert_ids, self.num_tokens_post_padded, self.cumsum_buffer = views[:4]
        self.activated = views[4]
        overlap = views[5].view(signature.payload_dtype)
        routes = signature.payload_row_capacity * signature.effective_topk
        self.gate_up = overlap[: routes * 2 * signature.local_intermediate_size].view(
            routes, 2 * signature.local_intermediate_size
        )
        self.route_outputs = overlap[: routes * signature.hidden_size].view(
            signature.payload_row_capacity, signature.effective_topk, signature.hidden_size
        )

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
        """Write one Partial with final routing metadata and caller-owned storage."""

        signature = self.signature
        if tuple(hidden_states.shape) != (signature.payload_row_capacity, signature.hidden_size):
            raise ValueError("floating Expert input disagrees with its execution layout")
        for projection, layout in (
            (gate_up, signature.gate_up),
            (down, signature.down),
        ):
            projection.validate(layout)
        payload_dtype = signature.payload_dtype
        weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
        weights.validate_tensor(topk_ids, name="topk ids", dtype=torch.int32, dimensions=2)
        weights.validate_tensor(topk_weights, name="topk weights", dtype=torch.float32, dimensions=2)
        weights.validate_tensor(output, name="MoE Partial output", dtype=payload_dtype, dimensions=2)

        row_capacity = signature.payload_row_capacity
        expert_count = signature.expert_count
        if topk_ids.shape != topk_weights.shape or topk_ids.shape != (row_capacity, signature.effective_topk):
            raise ValueError("MoE TopK id and weight dimensions disagree")
        effective_topk = topk_ids.shape[1]
        if output.shape != hidden_states.shape:
            raise ValueError("MoE Partial output dimensions disagree")

        w13 = self.w13_config
        w2 = w13 if self.w2_config is None else self.w2_config
        block_size_m = w13["BLOCK_SIZE_M"]

        tensors = (
            hidden_states,
            gate_up.values,
            down.values,
            topk_ids,
            topk_weights,
            self.sorted_token_ids,
            self.expert_ids,
            self.num_tokens_post_padded,
            self.cumsum_buffer,
            self.gate_up,
            self.activated,
            self.route_outputs,
            output,
        )
        if any(tensor.device != hidden_states.device for tensor in tensors):
            raise ValueError("MoE Expert tensors must share one device")

        output_dtype = language.bfloat16 if payload_dtype is torch.bfloat16 else language.float16
        # Loading sgl_kernel selects a device; pure preparation must not import it.
        import sgl_kernel
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel

        # The pinned extension writes the three caller-owned alignment buffers;
        # the final flag routes invalid Capacity padding to the sentinel Expert.
        sgl_kernel.moe_align_block_size(
            topk_ids,
            expert_count + 1,
            block_size_m,
            self.sorted_token_ids,
            self.expert_ids,
            self.num_tokens_post_padded,
            self.cumsum_buffer,
            True,
        )
        # Pinned SGLang exposes this launcher positionally: W13 consumes hidden
        # states and semantic routes, then writes the caller-owned Gate/Up tensor.
        invoke_fused_moe_kernel(
            hidden_states,
            gate_up.values,
            gate_up.bias,
            self.gate_up,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            self.sorted_token_ids,
            self.expert_ids,
            self.num_tokens_post_padded,
            False,
            effective_topk,
            w13,
            output_dtype,
            False,
            False,
            False,
            False,
            False,
            filter_expert=True,
        )
        activation.apply_into(
            signature.activation,
            inputs=self.gate_up,
            output=self.activated,
            order=self.required_gate_up_order,
        )
        # W2 reuses the same aligned route metadata and writes one output per route;
        # the following combine is the only row-level reduction.
        invoke_fused_moe_kernel(
            self.activated,
            down.values,
            down.bias,
            self.route_outputs,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            self.sorted_token_ids,
            self.expert_ids,
            self.num_tokens_post_padded,
            True,
            1,
            w2,
            output_dtype,
            False,
            False,
            False,
            False,
            False,
            filter_expert=True,
        )
        sgl_kernel.moe_sum_reduce(self.route_outputs, output, signature.routed_scaling_factor)
