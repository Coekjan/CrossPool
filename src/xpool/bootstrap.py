"""Process-wide xpool runtime bootstrap state."""

from __future__ import annotations

from threading import Lock

import xpool.cext
import xpool.native
from xpool.config import get_global_config
from xpool.runtime import RuntimeRole
from xpool.utils.procs import set_process_title

__all__ = ["get_runtime_role", "init"]

runtime_role: RuntimeRole | None = None
runtime_cuda_device: int | None = None
runtime_lock = Lock()


def init(cuda_device: int | None, role: RuntimeRole) -> None:
    """Initialize one process-wide Python and native runtime role.

    Args:
        cuda_device: CUDA device owned by a GPU runtime, or ``None`` for the
            host-only daemon.
        role: Daemon, instance, or agent role assigned to this process.

    Raises:
        RuntimeError: If the process was already initialized with a different
            CUDA device or runtime role.
        NativeLoadError: If the native extension cannot be loaded or its ABI is
            incompatible.

    Side Effects:
        Loads the native extension, initializes native debug state for GPU
        roles, installs the role-specific process title for the daemon and
        agents, and records the process-wide role after native initialization
        succeeds. Repeated calls with identical arguments are idempotent.
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
        if role is RuntimeRole.DAEMON and cuda_device is not None:
            raise RuntimeError("xpool daemon runtime must not own a CUDA device")
        xpool.cext.ensure_native_loaded()
        if role is RuntimeRole.DAEMON:
            xpool.native.initialize(role)
        else:
            if cuda_device is None:
                raise RuntimeError(f"xpool {role.name.lower()} runtime requires a CUDA device")
            debug_options = get_global_config().debug.model_dump_json(
                include={
                    "loopback": {"enable", "site"},
                    "transport_observer": {"enable", "trace_capacity"},
                    "fabric_observer": {"enable", "trace_capacity"},
                }
            )
            xpool.native.initialize(role, cuda_device, debug_options)
        match role:
            case RuntimeRole.DAEMON:
                set_process_title("xpool::daemon")
            case RuntimeRole.ATNAGENT:
                set_process_title("xpool::atnagent")
            case RuntimeRole.FFNAGENT:
                set_process_title("xpool::ffnagent")
            case RuntimeRole.INSTANCE:
                pass
        runtime_cuda_device = cuda_device
        runtime_role = role


def get_runtime_role() -> RuntimeRole:
    """Return the initialized process-wide runtime role.

    Returns:
        Daemon, instance, or agent role installed by :func:`init`.

    Raises:
        RuntimeError: If bootstrap has not initialized this process.
    """

    if runtime_role is None:
        raise RuntimeError("xpool runtime role is unavailable before bootstrap initialization")
    return runtime_role
