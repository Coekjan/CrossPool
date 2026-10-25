"""Gated Dense execution into caller-owned storage."""

from __future__ import annotations

import torch

from xpool.ffn import ActivationSpec, GateUpOrder
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import linear
from xpool.runtime.ffnagent.operators.activation import apply_into


def compute_dense_partial(
    *,
    hidden_states: torch.Tensor,
    layer_weights: weights.DenseFfnWeights,
    workspace: torch.Tensor,
    output: torch.Tensor,
    activation: ActivationSpec,
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

    payload_dtype = hidden_states.dtype
    if payload_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Dense execution requires BF16 or FP16 payloads")
    weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(output, name="Dense Partial output", dtype=payload_dtype, dimensions=2)
    weights.validate_tensor(workspace, name="Dense workspace", dtype=torch.uint8, dimensions=1)

    row_capacity, hidden_size = hidden_states.shape
    local_intermediate_twice, weight_hidden_size = layer_weights.gate_up.values.shape
    local_intermediate_size = local_intermediate_twice // 2
    if (
        weight_hidden_size != hidden_size
        or layer_weights.gate_up.values.dtype is not payload_dtype
        or output.shape != hidden_states.shape
    ):
        raise ValueError("Dense input, output, and weight dimensions disagree")
    if layer_weights.gate_up.scales is not None or layer_weights.down.scales is not None:
        raise ValueError("Dense floating execution does not support encoded scales")
    expected_workspace_bytes = execution.dense_workspace_bytes(
        payload_dtype=payload_dtype,
        row_capacity=row_capacity,
        local_intermediate_size=local_intermediate_size,
    )
    if workspace.numel() != expected_workspace_bytes:
        raise ValueError(f"Dense workspace must contain exactly {expected_workspace_bytes} bytes")
    device = hidden_states.device
    if output.device != device or workspace.device != device or layer_weights.gate_up.values.device != device:
        raise ValueError("Dense input, output, workspace, and weights must share one device")
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("Dense Partial output", output),
        ("Dense workspace", workspace),
        ("gate/up weight", layer_weights.gate_up.values),
        ("down weight", layer_weights.down.values),
    ):
        if tensor.data_ptr() % execution.OPERATOR_ALIGNMENT_BYTES != 0:
            raise ValueError(f"{name} must be 16-byte aligned")
    if local_intermediate_twice * payload_dtype.itemsize % execution.OPERATOR_ALIGNMENT_BYTES != 0:
        raise ValueError("Dense gate/up rows must be 16-byte aligned")

    workspace_values = workspace.view(payload_dtype)
    gate_up_elements = row_capacity * local_intermediate_twice
    gate_up = workspace_values[:gate_up_elements].view(row_capacity, local_intermediate_twice)
    activated_up = workspace_values[gate_up_elements:].view(row_capacity, local_intermediate_size)

    linear.compute_projection(
        linear.LinearOperationKind.TORCH,
        hidden_states=hidden_states,
        projection=layer_weights.gate_up,
        output=gate_up,
    )
    apply_into(activation, inputs=gate_up, output=activated_up, order=GateUpOrder.CONCATENATED)
    linear.compute_projection(
        linear.LinearOperationKind.TORCH,
        hidden_states=activated_up,
        projection=layer_weights.down,
        output=output,
    )
    return output
