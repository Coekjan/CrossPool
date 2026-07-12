"""Python facade for xpool native Torch operators."""

from __future__ import annotations

import torch

from xpool.abi import DebugLoopbackSite, DebugOptions, RuntimeRole
from xpool.config import LoopbackSite, get_global_config
from xpool.ops import atnagent, instance

__all__ = ["abi_version", "atnagent", "init", "instance"]


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
    loopback_sites = {
        None: DebugLoopbackSite.NONE,
        LoopbackSite.INSTANCE: DebugLoopbackSite.INSTANCE,
        LoopbackSite.ATNAGENT: DebugLoopbackSite.ATNAGENT,
        LoopbackSite.FFNAGENT: DebugLoopbackSite.FFNAGENT,
    }
    debug_options = DebugOptions.create(
        loopback_site=loopback_sites[config.debug.loopback.site],
        transport_observer=config.debug.transport_observer.enable,
    )
    torch.ops.xpool.init(cuda_device, int(role), debug_options.raw)
