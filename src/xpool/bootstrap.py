"""Process-wide CrossPool runtime bootstrap state."""

from __future__ import annotations

from threading import Lock

import torch

import xpool.cext
import xpool.logging
import xpool.native
from xpool.config import get_global_config
from xpool.native import RuntimeRole
from xpool.utils.procs import set_process_title

__all__ = ["get_runtime_role", "init"]

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
        Loads the native extension, installs role-specific runtime logging,
        selects the process CUDA device for GPU roles, initializes native debug
        state, and installs the role-specific process title for the daemon and
        agents. Native identity initialization is idempotent; repeated calls
        may repeat harmless Python-side setup.
    """

    with runtime_lock:
        xpool.cext.ensure_native_loaded()
        config = get_global_config()
        xpool.logging.configure(role)
        debug_options = config.debug.native_options()
        xpool.native.initialize(role, cuda_device, debug_options)
        if role is not RuntimeRole.DAEMON:
            torch.cuda.set_device(cuda_device)
        match role:
            case RuntimeRole.DAEMON:
                set_process_title("xpool::daemon")
            case RuntimeRole.ATNAGENT:
                set_process_title("xpool::atnagent")
            case RuntimeRole.FFNAGENT:
                set_process_title("xpool::ffnagent")
            case RuntimeRole.INSTANCE:
                pass


def get_runtime_role() -> RuntimeRole:
    """Return the initialized process-wide runtime role.

    Returns:
        Daemon, instance, or agent role installed by :func:`init`.

    Raises:
        RuntimeError: If bootstrap has not initialized this process.
    """

    return xpool.native.runtime_role()
