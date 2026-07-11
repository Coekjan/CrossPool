"""SGLang E2E environment derivation from validated test resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.harness.requirements import (
    RequirementUnavailable,
    ResolvedConfig,
    ResolvedModelWeights,
)
from tests.harness.sglang.offline_probe import MODEL_ID
from xpool.config import XpoolConfig
from xpool.integrations.sglang.topology import derive_parallel_policy, load_model_spec


@dataclass(frozen=True, slots=True)
class SglangTestEnvironment:
    """Validated config, model path, and CUDA placement for an E2E probe."""

    config_path: Path
    config: XpoolConfig
    model_path: Path
    base_gpu_id: int


def create_sglang_test_environment(
    *,
    resolved_config: ResolvedConfig,
    resolved_weights: ResolvedModelWeights,
    device_count: int,
) -> SglangTestEnvironment:
    """Derive contiguous SGLang TP placement from validated resources."""

    if resolved_weights.model_id != MODEL_ID:
        raise ValueError(f"expected weights for {MODEL_ID}, got {resolved_weights.model_id}")
    devices_per_probe = devices_per_sglang_probe(
        config=resolved_config.config,
        model_path=resolved_weights.path,
    )
    base_gpu_id = configured_probe_base_gpu_id(
        config=resolved_config.config,
        device_count=device_count,
        devices_per_probe=devices_per_probe,
    )
    return SglangTestEnvironment(
        config_path=resolved_config.path,
        config=resolved_config.config,
        model_path=resolved_weights.path,
        base_gpu_id=base_gpu_id,
    )


def devices_per_sglang_probe(*, config: XpoolConfig, model_path: Path) -> int:
    """Return the tensor-parallel device count required by one probe."""

    spec = load_model_spec(model_path, model_id=MODEL_ID)
    policy = derive_parallel_policy(
        spec,
        atn_device_count=len(config.devices.atn_cuda_devices),
        ffn_tp_size=len(config.devices.ffn_cuda_devices),
    )
    return policy.sglang_tp_size


def configured_probe_base_gpu_id(
    *,
    config: XpoolConfig,
    device_count: int,
    devices_per_probe: int,
) -> int:
    """Validate configured TP placement and return its first CUDA device."""

    if devices_per_probe <= 0:
        raise ValueError(f"devices_per_probe must be positive, got {devices_per_probe}")
    configured_devices = tuple(config.devices.atn_cuda_devices)
    if len(configured_devices) != devices_per_probe:
        raise RequirementUnavailable(
            "SGLang probe TP devices must match configured ATN devices: "
            f"probe requires {devices_per_probe}, configured {configured_devices}"
        )
    base_gpu_id = configured_devices[0]
    expected_devices = tuple(range(base_gpu_id, base_gpu_id + devices_per_probe))
    if configured_devices != expected_devices:
        raise RequirementUnavailable(
            f"SGLang Engine requires contiguous configured ATN devices, got {configured_devices}"
        )
    if expected_devices[-1] >= device_count:
        raise RequirementUnavailable(
            f"configured ATN devices {configured_devices} exceed {device_count} visible CUDA devices"
        )
    return base_gpu_id
