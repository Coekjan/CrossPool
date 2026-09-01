"""Compile-visible xpool Tensor operators."""

from __future__ import annotations

import torch

import xpool.cext
from xpool.transport import FfnRequestMetadata

__all__ = ["ffn_shim"]

xpool.cext.ensure_native_loaded()


@torch.library.register_fake("xpool::ffn_shim")
def fake_ffn_shim_output(
    hidden_states: torch.Tensor, dp_rank_payload_rows: torch.Tensor | None, output: torch.Tensor, *args: object
) -> None:
    """Accept caller-owned fake output for graph tracing."""

    return None


def ffn_shim(
    hidden_states: torch.Tensor,
    dp_rank_payload_rows: torch.Tensor | None,
    request_metadata: FfnRequestMetadata,
) -> torch.Tensor:
    """Dispatch one FFN request through the native Tensor operator.

    Args:
        hidden_states: Contiguous CUDA tensor shaped
            ``[payload_rows, hidden_size]``.
        dp_rank_payload_rows: Optional contiguous CUDA int32 or int64 tensor
            containing one physical row span per attention DP rank. Native
            transport validates and stages these values as protocol uint32
            rows.
        request_metadata: Validated transport metadata for this invocation.

    Returns:
        Tensor with the same shape, dtype, and device as ``hidden_states``.

    Raises:
        RuntimeError: If no matching arena is attached, tensor metadata exceeds
            arena capacity, or the native transport contract is invalid.

    Side Effects:
        Enqueues stream-ordered transport work and remains safe to capture in
        full or piecewise CUDA graphs. It does not synchronize the host.
    """

    output = torch.empty_like(hidden_states)
    torch.ops.xpool.ffn_shim(
        hidden_states,
        dp_rank_payload_rows,
        output,
        request_metadata.layer_ordinal,
        int(request_metadata.forward_mode),
        int(request_metadata.output_requirement),
        int(request_metadata.dp_row_layout),
    )
    return output
