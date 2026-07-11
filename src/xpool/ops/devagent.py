"""Python facade for devagent-owned native transport operators."""

from __future__ import annotations

from typing import cast

import torch

from xpool.abi import TransportArenaHandle, TransportTraceSnapshot


def create_transport_arena(
    cuda_device: int,
    max_tokens: int,
    hidden_size: int,
    element_size: int,
    atn_dp_size: int,
) -> TransportArenaHandle:
    """Create one native transport arena and return its handle.

    Args:
        cuda_device: CUDA device that owns the arena allocation.
        max_tokens: Maximum token rows supported by one slot.
        hidden_size: Hidden-state width supported by one slot.
        element_size: Hidden-state element width in bytes.
        atn_dp_size: Attention data-parallel world size.

    Returns:
        Arena handle for daemon publication and instance attachment.

    Raises:
        RuntimeError: If geometry is invalid, CUDA allocation fails, or the
            current process is not initialized as a devagent.

    Side Effects:
        Allocates devagent-owned CUDA IPC storage. The caller owns the returned
        arena until exactly one successful destroy transaction completes.
    """

    value = torch.ops.xpool.devagent.create_transport_arena(
        cuda_device,
        max_tokens,
        hidden_size,
        element_size,
        atn_dp_size,
    )
    return TransportArenaHandle(handle=cast(str, value))


def destroy_transport_arena(handle: TransportArenaHandle) -> TransportTraceSnapshot:
    """Stop, snapshot, and destroy one native transport arena.

    Args:
        handle: Opaque transport arena handle identifying the arena.

    Returns:
        Snapshot copied after the resident kernel exits. Its records are empty
        when transport observation was disabled.

    Raises:
        RuntimeError: If the handle is unknown or already destroyed, or if
            resident-kernel drain, observation copy, or CUDA release fails.

    Side Effects:
        Requests shutdown, synchronizes the resident kernel, copies enabled
        observer state to host memory, releases CUDA storage, and invalidates
        ``handle``.
    """

    sequence, dropped, records = torch.ops.xpool.devagent.destroy_transport_arena(handle.handle)
    return TransportTraceSnapshot.from_raw(int(sequence), int(dropped), records)


def launch_transport_kernel(handle: TransportArenaHandle) -> None:
    """Launch the persistent transport kernel for one native arena.

    Args:
        handle: Opaque transport arena handle identifying the arena.

    Raises:
        RuntimeError: If the handle is unknown, a kernel is already running, or
            CUDA launch setup fails.

    Side Effects:
        Launches one persistent resident kernel on the arena owner's CUDA
        device. The kernel remains live until arena destruction requests drain.
    """

    torch.ops.xpool.devagent.launch_transport_kernel(handle.handle)
