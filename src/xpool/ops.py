"""Compile-visible xpool Tensor operators."""

from __future__ import annotations

import torch

import xpool.cext
from xpool.transport import FfnRequestMetadata

__all__ = ["ffn_shim"]

xpool.cext.ensure_native_loaded()


@torch.library.register_fake("xpool::ffn_shim")
def fake_ffn_shim_output(hidden_states: torch.Tensor, *args: object) -> torch.Tensor:
    """Return shape-preserving fake output for graph tracing."""

    return torch.empty_like(hidden_states)


def ffn_shim(
    hidden_states: torch.Tensor,
    global_num_tokens_gpu: torch.Tensor | None,
    request_metadata: FfnRequestMetadata,
) -> torch.Tensor:
    """Dispatch one FFN request through the native Tensor operator.

    Args:
        hidden_states: Contiguous CUDA tensor shaped
            ``[num_tokens, hidden_size]``.
        global_num_tokens_gpu: Optional contiguous CUDA int32 or int64 tensor
            containing one token count per attention DP rank. Native transport
            validates and stages these values as protocol uint32 counts.
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

    return torch.ops.xpool.ffn_shim(
        hidden_states,
        global_num_tokens_gpu,
        request_metadata.layer_ordinal,
        int(request_metadata.forward_mode),
        int(request_metadata.result_handoff),
        int(request_metadata.dp_padding_mode),
    )
