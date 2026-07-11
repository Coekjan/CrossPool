"""Public devagent runtime factory."""

from xpool.config import DeviceRole, get_global_config
from xpool.runtime.devagent.atn import AtnDevagent
from xpool.runtime.devagent.common import Devagent, DevagentError
from xpool.runtime.devagent.ffn import FfnDevagent

__all__ = ["Devagent", "DevagentError", "create_devagent"]


def create_devagent(cuda_device: int) -> Devagent:
    """Create the configured role-specific devagent for one CUDA device.

    Args:
        cuda_device: CUDA device index assigned to the devagent process.

    Returns:
        ATN or FFN devagent selected by the process-global config.

    Raises:
        DevagentError: If the CUDA device is unknown or has an unsupported role.
    """

    config = get_global_config()
    devagent = config.devagent_by_cuda_device.get(cuda_device)
    if devagent is None:
        known = ", ".join(str(device) for device in sorted(config.devagent_by_cuda_device)) or "<none>"
        raise DevagentError(f"unknown devagent CUDA device {cuda_device}; configured CUDA devices: {known}")
    match devagent.role:
        case DeviceRole.ATN:
            return AtnDevagent(cuda_device=cuda_device, role=devagent.role)
        case DeviceRole.FFN:
            return FfnDevagent(cuda_device=cuda_device, role=devagent.role)
