"""Python facade for xpool native Torch operators."""

from __future__ import annotations

import torch

from xpool.abi import DebugOption, RuntimeRole
from xpool.config import get_global_config
from xpool.ops import devagent, instance

__all__ = ["abi_version", "devagent", "init", "instance"]


def abi_version() -> int:
    """Return the native ABI version reported by the loaded extension."""

    return int(torch.ops.xpool.abi_version())


def init(cuda_device: int, role: RuntimeRole | int) -> None:
    """Run native one-time initialization for one CUDA device.

    Args:
        cuda_device: CUDA device to initialize in the current process.
        role: Native runtime role for this process.

    Side Effects:
        Reads the process-global xpool config to resolve debug options and
        calls the native ``xpool.init`` operator.
    """

    config = get_global_config()
    debug_options = DebugOption(0)
    if config.debug.shim_loopback.enable:
        debug_options |= DebugOption.SHIM_LOOPBACK
    if config.debug.transport_loopback.enable:
        debug_options |= DebugOption.TRANSPORT_LOOPBACK
    if config.debug.transport_observer.enable:
        debug_options |= DebugOption.TRANSPORT_OBSERVER
    torch.ops.xpool.init(cuda_device, int(role), int(debug_options))
