"""Runtime participants and native shim integration helpers for xpool."""

from enum import IntEnum

__all__ = ["RuntimeRole"]


class RuntimeRole(IntEnum):
    """Native process role selected during xpool initialization.

    Attributes:
        DAEMON: Host-only control-plane process.
        INSTANCE: SGLang process that invokes the FFN shim.
        ATNAGENT: Process that owns CUDA IPC transport arenas.
        FFNAGENT: Process that owns Fabric FFN execution.
    """

    DAEMON = 1
    INSTANCE = 2
    ATNAGENT = 3
    FFNAGENT = 4
