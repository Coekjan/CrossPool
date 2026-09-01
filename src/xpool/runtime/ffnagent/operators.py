"""Concrete FfnAgent-owned FFN operator composition."""

from __future__ import annotations

import contextlib
import math
import types
import typing
from collections.abc import Generator

import torch
import triton
from triton import language

from xpool.ffn import ActivationKind
from xpool.runtime.ffnagent import execution, weights

ROUTING_FINALIZATION_BLOCK_SIZE = 256


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


@contextlib.contextmanager
def sglang_moe_config_selection() -> Generator[None, None, None]:
    """Supply the pinned MoE selector's sole startup ServerArgs value.

    Side Effects:
        Temporarily replaces the selector module's bound
        ``get_global_server_args`` function and restores the exact previous
        function before returning or propagating an exception. The context is
        single-owner startup state and is not safe for concurrent use.
    """

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    previous = fused_moe_triton_config.get_global_server_args
    setattr(
        fused_moe_triton_config,
        "get_global_server_args",
        lambda: types.SimpleNamespace(enable_deterministic_inference=False),
    )
    try:
        yield
    finally:
        setattr(fused_moe_triton_config, "get_global_server_args", previous)


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
    layer_weights: weights.MoeFfnWeights,
    row_capacity: int,
    effective_topk: int,
) -> tuple[dict[str, int], dict[str, int] | None]:
    """Select immutable pinned W13 and optional W2 launch configurations.

    Args:
        layer_weights: Canonical local Expert weights whose shapes select the
            pinned configuration tables.
        row_capacity: Positive fixed Graph row Capacity used as selector ``M``.
        effective_topk: Positive routed-plus-shared route width.

    Returns:
        Plain copied W13 configuration and an optional independently selected
        W2 configuration. TMA is unconditionally disabled.

    Raises:
        ValueError: If Capacity or route width is inconsistent with weights.
        RuntimeError: If the pinned selector returns an invalid mapping.

    Side Effects:
        Imports and consults pinned SGLang startup configuration while the
        dedicated scoped ServerArgs view is installed. No process-global
        SGLang ServerArgs object remains after return.
    """

    expert_count = layer_weights.expert_gate_up_weight.shape[0]
    if row_capacity <= 0 or effective_topk <= 0 or effective_topk > expert_count:
        raise ValueError("MoE kernel selection dimensions are inconsistent")

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    with sglang_moe_config_selection():
        selected, down = fused_moe_triton_config.try_get_optimal_moe_config(
            layer_weights.expert_gate_up_weight.shape,
            layer_weights.expert_down_weight.shape,
            effective_topk,
            None,
            row_capacity,
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
        raise ValueError("Routing Metadata tensors must share one CUDA device")
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


def compute_moe_partial(
    *,
    hidden_states: torch.Tensor,
    layer_weights: weights.MoeFfnWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    gate_up: torch.Tensor,
    activated: torch.Tensor,
    route_outputs: torch.Tensor,
    output: torch.Tensor,
    w13_config: dict[str, int],
    w2_config: dict[str, int] | None,
    activation: ActivationKind,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Compute one fixed-Capacity rank-local MoE Partial.

    Args:
        hidden_states: Caller-owned contiguous payload input shaped ``[C, H]``.
        layer_weights: Canonical local routed-then-shared Expert weight shards.
        topk_ids: Final semantic contiguous int32 routes shaped ``[C, K]``.
        topk_weights: Final semantic contiguous FP32 route weights shaped
            ``[C, K]``.
        sorted_token_ids: Caller-owned int32 alignment output.
        expert_ids: Caller-owned int32 aligned-block Expert IDs.
        num_tokens_post_padded: Caller-owned one-element int32 aligned count.
        cumsum_buffer: Caller-owned int32 alignment scratch.
        gate_up: Caller-owned payload W13 output shaped ``[C * K, 2 * I_r]``.
        activated: Caller-owned payload activation shaped ``[C * K, I_r]``.
        route_outputs: Caller-owned payload W2 output shaped ``[C, K, H]``.
            Its storage may alias ``gate_up`` because their lifetimes do not
            overlap.
        output: Caller-owned contiguous payload Partial shaped ``[C, H]``.
        w13_config: Frozen private-launcher Gate/Up configuration.
        w2_config: Optional frozen Down configuration; ``None`` reuses W13.
        activation: Startup-selected gated activation semantic.
        routed_scaling_factor: Finite positive post-combine scale.

    Returns:
        The exact ``output`` object after every Capacity row is evaluated.

    Raises:
        ValueError: If tensors, configurations, activation, or geometry do not
            match the fixed Expert implementation.
        ImportError: If a pinned operator dependency cannot load.

    Side Effects:
        Writes all caller-owned workspace and output tensors on the current
        CUDA stream. It allocates no Torch output or persistent tensor.
    """

    if activation is not ActivationKind.SILU:
        raise ValueError(f"unsupported MoE activation {activation!r}")
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise ValueError("MoE routed scaling factor must be finite and positive")
    payload_dtype = hidden_states.dtype
    if payload_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MoE execution requires BF16 or FP16 payloads")
    weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(topk_ids, name="topk ids", dtype=torch.int32, dimensions=2)
    weights.validate_tensor(topk_weights, name="topk weights", dtype=torch.float32, dimensions=2)
    weights.validate_tensor(sorted_token_ids, name="sorted token ids", dtype=torch.int32, dimensions=1)
    weights.validate_tensor(expert_ids, name="aligned Expert ids", dtype=torch.int32, dimensions=1)
    weights.validate_tensor(
        num_tokens_post_padded,
        name="post-padding token count",
        dtype=torch.int32,
        dimensions=1,
    )
    weights.validate_tensor(cumsum_buffer, name="alignment cumsum", dtype=torch.int32, dimensions=1)
    weights.validate_tensor(gate_up, name="Expert gate/up output", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(activated, name="Expert activation output", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(route_outputs, name="Expert route outputs", dtype=payload_dtype, dimensions=3)
    weights.validate_tensor(output, name="MoE Partial output", dtype=payload_dtype, dimensions=2)

    row_capacity, hidden_size = hidden_states.shape
    expert_count, local_intermediate_twice, weight_hidden_size = layer_weights.expert_gate_up_weight.shape
    local_intermediate_size = local_intermediate_twice // 2
    if weight_hidden_size != hidden_size or layer_weights.expert_gate_up_weight.dtype is not payload_dtype:
        raise ValueError("MoE input and Expert hidden dimensions disagree")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != row_capacity:
        raise ValueError("MoE TopK id and weight dimensions disagree")
    effective_topk = topk_ids.shape[1]
    route_count = row_capacity * effective_topk
    if gate_up.shape != (route_count, local_intermediate_twice):
        raise ValueError("MoE Gate/Up workspace dimensions disagree")
    if activated.shape != (route_count, local_intermediate_size):
        raise ValueError("MoE activation workspace dimensions disagree")
    if route_outputs.shape != (row_capacity, effective_topk, hidden_size):
        raise ValueError("MoE route-output workspace dimensions disagree")
    if output.shape != hidden_states.shape:
        raise ValueError("MoE Partial output dimensions disagree")

    w13 = copy_moe_kernel_config(w13_config, name="W13")
    w2 = w13 if w2_config is None else copy_moe_kernel_config(w2_config, name="W2")
    block_size_m = w13["BLOCK_SIZE_M"]
    if w2["BLOCK_SIZE_M"] != block_size_m:
        raise ValueError("W13 and W2 BLOCK_SIZE_M must match one alignment")
    maximum_padded, expert_block_count, cumsum_count = execution.moe_alignment_workspace_shapes(
        row_capacity=row_capacity,
        effective_topk=effective_topk,
        expert_count=expert_count,
        block_size_m=block_size_m,
    )
    if sorted_token_ids.numel() != maximum_padded:
        raise ValueError("sorted-token workspace has the wrong element count")
    if expert_ids.numel() != expert_block_count:
        raise ValueError("aligned Expert-id workspace has the wrong element count")
    if num_tokens_post_padded.numel() != 1 or cumsum_buffer.numel() != cumsum_count:
        raise ValueError("MoE alignment scalar or cumsum workspace has the wrong element count")

    tensors = (
        hidden_states,
        layer_weights.expert_gate_up_weight,
        layer_weights.expert_down_weight,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        gate_up,
        activated,
        route_outputs,
        output,
    )
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("MoE Expert tensors must share one CUDA device")

    import sgl_kernel
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_kernels

    output_dtype = language.bfloat16 if payload_dtype is torch.bfloat16 else language.float16
    # The pinned extension writes the three caller-owned alignment buffers;
    # the final flag routes invalid Capacity padding to the sentinel Expert.
    sgl_kernel.moe_align_block_size(
        topk_ids,
        expert_count + 1,
        block_size_m,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,
    )
    # Pinned SGLang exposes this launcher positionally: W13 consumes hidden
    # states and semantic routes, then writes the caller-owned Gate/Up tensor.
    fused_moe_triton_kernels.invoke_fused_moe_kernel(
        hidden_states,
        layer_weights.expert_gate_up_weight,
        None,
        gate_up,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
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
    returned_activation = sgl_kernel.silu_and_mul(gate_up, out=activated)
    if returned_activation is not activated:
        raise RuntimeError("sgl_kernel.silu_and_mul did not preserve caller-owned output")
    # W2 reuses the same aligned route metadata and writes one output per route;
    # the following combine is the only row-level reduction.
    fused_moe_triton_kernels.invoke_fused_moe_kernel(
        activated,
        layer_weights.expert_down_weight,
        None,
        route_outputs,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
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
    sgl_kernel.moe_sum_reduce(route_outputs, output, routed_scaling_factor)
    return output


def compute_dense_partial(
    *,
    hidden_states: torch.Tensor,
    layer_weights: weights.DenseFfnWeights,
    workspace: torch.Tensor,
    output: torch.Tensor,
    activation: ActivationKind,
) -> torch.Tensor:
    """Compute one fixed-Capacity rank-local gated Dense Partial.

    Args:
        hidden_states: Caller-owned contiguous payload input shaped ``[C, H]``.
        layer_weights: Canonical local true-TP projection weights.
        workspace: Caller-owned contiguous CUDA byte tensor with the exact
        result of :func:`execution.dense_workspace_bytes`.
        output: Caller-owned contiguous payload destination shaped ``[C, H]``.
        activation: Startup-selected gated activation semantic.

    Returns:
        The exact ``output`` object after all Capacity rows are evaluated.

    Raises:
        ValueError: If activation, shapes, dtype, device, contiguity, or
            alignment do not match the fixed Dense implementation.
        ImportError: If the direct ``sgl_kernel`` dependency cannot load.

    Side Effects:
        Writes ``workspace`` and ``output`` on the current CUDA stream. It
        allocates no Torch output or persistent auxiliary tensor.
    """

    if activation is not ActivationKind.SILU:
        raise ValueError(f"unsupported Dense activation {activation!r}")
    payload_dtype = hidden_states.dtype
    if payload_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Dense execution requires BF16 or FP16 payloads")
    weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(output, name="Dense Partial output", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(workspace, name="Dense workspace", dtype=torch.uint8, dimensions=1)

    row_capacity, hidden_size = hidden_states.shape
    local_intermediate_twice, weight_hidden_size = layer_weights.gate_up_weight.shape
    local_intermediate_size = local_intermediate_twice // 2
    if (
        weight_hidden_size != hidden_size
        or layer_weights.gate_up_weight.dtype is not payload_dtype
        or output.shape != hidden_states.shape
    ):
        raise ValueError("Dense input, output, and weight dimensions disagree")
    expected_workspace_bytes = execution.dense_workspace_bytes(
        payload_dtype=payload_dtype,
        row_capacity=row_capacity,
        local_intermediate_size=local_intermediate_size,
    )
    if workspace.numel() != expected_workspace_bytes:
        raise ValueError(f"Dense workspace must contain exactly {expected_workspace_bytes} bytes")
    device = hidden_states.device
    if output.device != device or workspace.device != device or layer_weights.gate_up_weight.device != device:
        raise ValueError("Dense input, output, workspace, and weights must share one CUDA device")
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("Dense Partial output", output),
        ("Dense workspace", workspace),
        ("gate/up weight", layer_weights.gate_up_weight),
        ("down weight", layer_weights.down_weight),
    ):
        if tensor.data_ptr() % execution.OPERATOR_ALIGNMENT_BYTES != 0:
            raise ValueError(f"{name} must be 16-byte aligned")
    if local_intermediate_twice * payload_dtype.itemsize % execution.OPERATOR_ALIGNMENT_BYTES != 0:
        raise ValueError("Dense gate/up rows must be 16-byte aligned")

    workspace_values = workspace.view(payload_dtype)
    gate_up_elements = row_capacity * local_intermediate_twice
    gate_up = workspace_values[:gate_up_elements].view(row_capacity, local_intermediate_twice)
    activated_up = workspace_values[gate_up_elements:].view(row_capacity, local_intermediate_size)

    import sgl_kernel

    torch.mm(hidden_states, layer_weights.gate_up_weight.t(), out=gate_up)
    returned_output = sgl_kernel.silu_and_mul(gate_up, out=activated_up)
    if returned_output is not activated_up:
        raise RuntimeError("sgl_kernel.silu_and_mul did not preserve caller-owned output")
    torch.mm(activated_up, layer_weights.down_weight.t(), out=output)
    return output
