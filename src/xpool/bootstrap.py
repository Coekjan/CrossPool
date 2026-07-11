"""Process-wide xpool runtime bootstrap state."""

from __future__ import annotations

from threading import Lock

import xpool.ops
from xpool.abi import RuntimeRole
from xpool.cext import ensure_xpool_ops_loaded

__all__ = ["get_runtime_role", "init"]

runtime_role: RuntimeRole | None = None
runtime_cuda_device: int | None = None
runtime_lock = Lock()


def init(cuda_device: int, role: RuntimeRole) -> None:
    """Initialize one process-wide Python and native runtime role.

    Args:
        cuda_device: CUDA device owned by this process.
        role: Instance or devagent role assigned to this process.

    Raises:
        RuntimeError: If the process was already initialized with a different
            CUDA device or runtime role.
        NativeLoadError: If the native extension cannot be loaded or its ABI is
            incompatible.

    Side Effects:
        Loads the native extension, initializes native debug state, and records
        the process-wide role after native initialization succeeds. Repeated
        calls with identical arguments are idempotent.
    """

    global runtime_cuda_device, runtime_role
    with runtime_lock:
        if runtime_role is not None:
            if runtime_role != role or runtime_cuda_device != cuda_device:
                raise RuntimeError(
                    "xpool runtime is already initialized for "
                    f"CUDA device {runtime_cuda_device} with role {runtime_role.name}"
                )
            return
        ensure_xpool_ops_loaded()
        xpool.ops.init(cuda_device, role)
        runtime_cuda_device = cuda_device
        runtime_role = role


def get_runtime_role() -> RuntimeRole:
    """Return the initialized process-wide runtime role.

    Returns:
        Instance or devagent role installed by :func:`init`.

    Raises:
        RuntimeError: If bootstrap has not initialized this process.
    """

    if runtime_role is None:
        raise RuntimeError("xpool runtime role is unavailable before bootstrap initialization")
    return runtime_role
