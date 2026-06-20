"""Compile-friendly Torch operators for xpool."""

from __future__ import annotations

import torch


@torch.library.custom_op("xpool::ffn_shim_graph", mutates_args=())
def ffn_shim(
    hidden_states: torch.Tensor,
    instance_index: int,
    model_index: int,
    layer_id: int,
    forward_mode: int,
) -> torch.Tensor:
    """Dispatch the production FFN shim through a compile-friendly op.

    Args:
        hidden_states: FFN input tensor with shape ``[num_tokens, hidden_size]``.
        instance_index: Integer serving-instance id from the xpool config order.
        model_index: Integer model id from the xpool config order.
        layer_id: Decoder layer id.
        forward_mode: Integer xpool ABI forward mode.

    Returns:
        Fake tensor metadata during tracing. Real execution will return the
        production native FFN output after the transport path is implemented.

    Raises:
        RuntimeError: On real execution while the production native
            ``torch.ops.xpool.ffn_shim`` remains intentionally unimplemented.

    Side Effects:
        Dispatches to ``torch.ops.xpool.ffn_shim`` at runtime. During
        torch.compile tracing, the registered fake implementation preserves the
        symbolic token dimension so SGLang PCG can reuse one compiled graph
        across capture buckets.
    """

    return torch.ops.xpool.ffn_shim(hidden_states, instance_index, model_index, layer_id, forward_mode)


@torch.library.custom_op("xpool::ffn_shim_loopback_graph", mutates_args=())
def ffn_shim_loopback(
    hidden_states: torch.Tensor,
    instance_index: int,
    model_index: int,
    layer_id: int,
    forward_mode: int,
) -> torch.Tensor:
    """Call the debug loopback native FFN shim through a compile-friendly op.

    Args:
        hidden_states: FFN input tensor with shape ``[num_tokens, hidden_size]``.
        instance_index: Integer serving-instance id from the xpool config order.
        model_index: Integer model id from the xpool config order.
        layer_id: Decoder layer id.
        forward_mode: Integer xpool ABI forward mode.

    Returns:
        Tensor returned by the native loopback shim.

    Side Effects:
        Dispatches to ``torch.ops.xpool.ffn_shim_loopback`` at runtime. During
        torch.compile tracing, the registered fake implementation preserves the
        symbolic token dimension so SGLang PCG can reuse one compiled graph
        across capture buckets.
    """

    return torch.ops.xpool.ffn_shim_loopback(hidden_states, instance_index, model_index, layer_id, forward_mode)


def fake_ffn_shim_output(
    hidden_states: torch.Tensor,
    *metadata: int,
) -> torch.Tensor:
    """Return fake tensor metadata shared by FFN shim graph wrappers.

    Args:
        hidden_states: Fake or Meta hidden-state tensor.
        *metadata: Integer ABI metadata ignored by fake dispatch.

    Returns:
        Fake tensor with the same symbolic shape, dtype, and device as
        ``hidden_states``.
    """

    return torch.empty_like(hidden_states)


ffn_shim.register_fake(fake_ffn_shim_output)
ffn_shim_loopback.register_fake(fake_ffn_shim_output)
