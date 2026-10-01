"""SGLang CUDA placement derived from the configured attention devices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class SglangCudaPlacement:
    """Arithmetic CUDA placement accepted by SGLang's base-and-step arguments."""

    base_gpu_id: int
    gpu_id_step: int

    @classmethod
    def derive(cls, atn_cuda_devices: Sequence[int]) -> SglangCudaPlacement:
        """Resolve ordered devices, rejecting empty or non-arithmetic placement."""

        if not atn_cuda_devices:
            raise RuntimeError("xpool SGLang integration requires at least one attention CUDA device")
        gpu_id_step = atn_cuda_devices[1] - atn_cuda_devices[0] if len(atn_cuda_devices) > 1 else 1
        if any(left >= right for left, right in pairwise(atn_cuda_devices)):
            raise RuntimeError("xpool SGLang integration requires strictly increasing attention CUDA devices")
        if any((right - left) != gpu_id_step for left, right in pairwise(atn_cuda_devices)):
            raise RuntimeError(
                "xpool SGLang integration requires attention CUDA devices to match base_gpu_id + rank * gpu_id_step"
            )
        return cls(base_gpu_id=atn_cuda_devices[0], gpu_id_step=gpu_id_step)
