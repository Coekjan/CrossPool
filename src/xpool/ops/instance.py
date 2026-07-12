"""Python facade for instance-owned native FFN shim operators."""

from __future__ import annotations

import torch

from xpool.abi import FfnRequestMetadata, FfnResultErrorCode, TransportArenaHandle

__all__ = ["attach_transport_arena", "detach_transport_arena", "ffn_shim", "transport_error_snapshot"]


@torch.library.custom_op("xpool::instance.ffn_shim_wrapper", mutates_args=())
def ffn_shim_graph(
    hidden_states: torch.Tensor,
    global_num_tokens_gpu: torch.Tensor | None,
    instance_index: int,
    rank: int,
    layer_id: int,
    forward_mode: int,
    collective_policy: int,
    dp_padding_mode: int,
    global_dp_buffer_len: int,
    atn_tp_rank: int,
    atn_tp_size: int,
    atn_dp_rank: int,
    atn_dp_size: int,
) -> torch.Tensor:
    """Dispatch one FFN request through the compile-friendly native graph op.

    Args:
        hidden_states: Contiguous CUDA tensor shaped ``[num_tokens, hidden_size]``.
        global_num_tokens_gpu: Optional contiguous CUDA int32 tensor containing
            one token count per attention DP rank.
        instance_index: Integer instance index from config declaration order.
        rank: Rank-local process index within the instance.
        layer_id: Decoder layer whose FFN executor consumes the request.
        forward_mode: Native :class:`xpool.abi.XPoolForwardMode` value.
        collective_policy: Native :class:`xpool.abi.FfnCollectivePolicy` value.
        dp_padding_mode: Native :class:`xpool.abi.DpPaddingMode` value.
        global_dp_buffer_len: Global padded or packed DP token capacity.
        atn_tp_rank: Attention tensor-parallel rank.
        atn_tp_size: Attention tensor-parallel world size.
        atn_dp_rank: Attention data-parallel rank.
        atn_dp_size: Attention data-parallel world size.

    Returns:
        CUDA tensor with the same shape, dtype, and device as ``hidden_states``.

    Raises:
        RuntimeError: If no matching arena is attached, tensor metadata exceeds
            arena capacity, or the native transport contract is invalid.

    Side Effects:
        Enqueues stream-ordered native transport work and is safe to capture in
        full or piecewise CUDA graphs. It does not synchronize the host.
    """

    return torch.ops.xpool.instance.ffn_shim(
        hidden_states,
        global_num_tokens_gpu,
        instance_index,
        rank,
        layer_id,
        forward_mode,
        collective_policy,
        dp_padding_mode,
        global_dp_buffer_len,
        atn_tp_rank,
        atn_tp_size,
        atn_dp_rank,
        atn_dp_size,
    )


def fake_ffn_shim_output(
    hidden_states: torch.Tensor,
    *args: object,
) -> torch.Tensor:
    """Return fake FFN shim output for graph tracing."""

    return torch.empty_like(hidden_states)


ffn_shim_graph.register_fake(fake_ffn_shim_output)


def attach_transport_arena(instance_index: int, rank: int, handle: TransportArenaHandle) -> None:
    """Attach a daemon-brokered native transport arena handle to an instance rank.

    Args:
        instance_index: Integer instance index from config declaration order.
        rank: Rank-local process index within the instance.
        handle: CUDA IPC arena handle acquired from the daemon.

    Raises:
        RuntimeError: If the handle is invalid or the key already owns a
            different arena. Reattaching the same handle is idempotent.

    Side Effects:
        Opens CUDA IPC storage and installs process-local native arena state.
    """

    torch.ops.xpool.instance.attach_transport_arena(
        instance_index,
        rank,
        handle.handle,
    )


def detach_transport_arena(instance_index: int, rank: int) -> None:
    """Detach one native transport arena from an instance rank.

    Args:
        instance_index: Integer instance index from config declaration order.
        rank: Rank-local process index within the instance.

    Raises:
        RuntimeError: If in-flight work cannot be synchronized or IPC cleanup fails.

    Side Effects:
        Stops new launches, waits for recorded in-flight work, and closes the
        CUDA IPC mapping. Detaching an absent key is idempotent.
    """

    torch.ops.xpool.instance.detach_transport_arena(instance_index, rank)


def transport_error_snapshot(instance_index: int, rank: int) -> FfnResultErrorCode:
    """Return the sticky executor error for one attached transport arena.

    Args:
        instance_index: Integer instance index from xpool config order.
        rank: Rank-local process index within the instance.

    Returns:
        Sticky executor error, or ``OK`` when the arena remains healthy.

    Side Effects:
        Copies one device-resident error word to host memory.
    """

    return FfnResultErrorCode(torch.ops.xpool.instance.transport_error_snapshot(instance_index, rank))


def ffn_shim(
    hidden_states: torch.Tensor,
    global_num_tokens_gpu: torch.Tensor | None,
    request_metadata: FfnRequestMetadata,
    rank: int,
) -> torch.Tensor:
    """Dispatch the FFN shim through a compile-friendly op.

    Args:
        hidden_states: FFN input tensor with shape ``[num_tokens, hidden_size]``.
        global_num_tokens_gpu: Optional per-DP-rank token counts on device.
        request_metadata: Request metadata unpacked only at the private
            custom-op boundary.
        rank: Rank-local process index within the instance.

    Returns:
        Fake output during tracing. Real execution returns either the debug
        instance-loopback output or the daemon-brokered FFN output, depending on
        process-global debug config.

    Side Effects:
        Dispatches through the registered ``xpool::instance.ffn_shim_wrapper``
        custom op.
    """

    return ffn_shim_graph(
        hidden_states,
        global_num_tokens_gpu,
        request_metadata.instance_index,
        rank,
        request_metadata.layer_id,
        request_metadata.forward_mode,
        request_metadata.collective_policy,
        request_metadata.dp_padding_mode,
        request_metadata.global_dp_buffer_len,
        request_metadata.atn_tp_rank,
        request_metadata.atn_tp_size,
        request_metadata.atn_dp_rank,
        request_metadata.atn_dp_size,
    )
