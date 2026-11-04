"""Persistent environment-qualified device-memory calibration Profiles."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field, fields

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from xpool.config import get_global_config
from xpool.native import ABI_VERSION
from xpool.utils import align_up
from xpool.utils.cuda import cuda_versions
from xpool.utils.files import atomic_write_text
from xpool.utils.integer import SINT64_MAX, ensure_sint64_nonnegative

MIB = 1 << 20


@dataclass(frozen=True, slots=True)
class DeviceMemoryFeatures:
    """Modeled post-baseline storage and execution features at one startup point."""

    tensor_storage_bytes: int
    tensor_storage_allocation_count: int
    dense_graph_capture_count: int
    moe_graph_capture_count: int
    executor_lane_count: int
    compute_branch_count: int
    dense_implementation_present: bool
    moe_implementation_present: bool
    joined_ffnagent: bool

    def __post_init__(self) -> None:
        ensure_sint64_nonnegative(
            self.tensor_storage_bytes,
            self.tensor_storage_allocation_count,
            self.dense_graph_capture_count,
            self.moe_graph_capture_count,
            self.executor_lane_count,
            self.compute_branch_count,
        )

    def calibration_values(self) -> tuple[int, ...]:
        """Normalize units and booleans in the stable coefficient fit order."""

        return (
            1,
            align_up(self.tensor_storage_bytes, MIB) // MIB,
            self.tensor_storage_allocation_count,
            self.dense_graph_capture_count,
            self.moe_graph_capture_count,
            self.executor_lane_count,
            self.compute_branch_count,
            int(self.dense_implementation_present),
            int(self.moe_implementation_present),
            int(self.joined_ffnagent),
        )


@dataclass(frozen=True, slots=True)
class FfnMemoryCalibrationCoefficients:
    """One immutable non-negative upper envelope over opaque FFN overhead."""

    base_bytes: int = field(metadata={"description": "Constant opaque device-memory bytes."})
    bytes_per_tensor_storage_mib: int = field(
        metadata={"description": "Opaque bytes per ceiling MiB of live post-baseline Tensor storage."}
    )
    bytes_per_tensor_storage_allocation: int = field(
        metadata={"description": "Opaque bytes per independently owned live post-baseline Tensor storage."}
    )
    bytes_per_dense_graph_capture: int = field(
        metadata={"description": "Opaque bytes per Capacity-specialized Dense Graph Capture."}
    )
    bytes_per_moe_graph_capture: int = field(
        metadata={"description": "Opaque bytes per Capacity-specialized MoE Graph Capture."}
    )
    bytes_per_executor_lane: int = field(metadata={"description": "Opaque bytes per installed Executor Lane."})
    bytes_per_compute_branch: int = field(
        metadata={"description": "Opaque bytes per installed Lane and Execution Signature branch."}
    )
    dense_implementation_bytes: int = field(
        metadata={"description": "Opaque bytes while constructing or retaining local Dense execution."}
    )
    moe_implementation_bytes: int = field(
        metadata={"description": "Opaque bytes while constructing or retaining local MoE execution."}
    )
    joined_ffnagent_bytes: int = field(
        metadata={"description": "Opaque bytes present after one FfnAgent joins Fabric."}
    )

    def __post_init__(self) -> None:
        """Reject negative or out-of-domain coefficients."""

        ensure_sint64_nonnegative(*self.values())

    def values(self) -> tuple[int, ...]:
        """Return coefficients in their stable fit tie-break order."""

        return tuple(getattr(self, coefficient.name) for coefficient in fields(self))

    def evaluate(self, features: DeviceMemoryFeatures) -> int:
        """Evaluate the exact signed-64-bit opaque-overhead expression."""

        result = sum(
            coefficient * value for coefficient, value in zip(self.values(), features.calibration_values(), strict=True)
        )
        ensure_sint64_nonnegative(result)
        return result


class MemoryCalibrationDevice(BaseModel):
    """One ordered FfnAgent device recorded by a calibration run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1, description="device marketing name.")
    compute_capability: tuple[int, int] = Field(description="CUDA major and minor compute capability.")
    total_memory_bytes: int = Field(ge=1, description="Total device memory in bytes.")
    uuid: str = Field(min_length=1, description="device UUID retained as non-matching provenance.")

    @model_validator(mode="after")
    def validate_compute_capability(self) -> MemoryCalibrationDevice:
        """Reject negative compute-capability components."""

        if any(value < 0 for value in self.compute_capability):
            raise ValueError("device compute capability must contain non-negative integers")
        return self


class MemoryCalibrationEnvironment(BaseModel):
    """Hardware and software compatibility facts for one calibration Profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    native_abi_version: int = Field(ge=1, description="CrossPool native ABI used for calibration.")
    ffnagent_devices: tuple[MemoryCalibrationDevice, ...] = Field(
        min_length=1,
        description="FfnAgent devices in configured FfnAgent index order.",
    )
    cuda_driver_version: int = Field(ge=1, description="CUDA driver API integer version.")
    cuda_runtime_version: int = Field(ge=1, description="CUDA runtime API integer version.")
    torch_version: str = Field(min_length=1, description="Installed Torch distribution version.")
    triton_version: str = Field(min_length=1, description="Installed Triton distribution version.")
    sglang_version: str = Field(min_length=1, description="Installed SGLang distribution version.")
    sglang_kernel_version: str = Field(min_length=1, description="Installed sglang-kernel distribution version.")
    nvshmem_version: str = Field(min_length=1, description="Installed NVIDIA NVSHMEM distribution version.")


class FfnMemoryCalibration(BaseModel):
    """FFN opaque-memory calibration for one exact deployment geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    atnagent_count: int = Field(ge=1, description="Configured AtnAgent count used by every calibration world.")
    executor_lane_count: int = Field(ge=1, description="Configured Executor Lane count used by every world.")
    minimum_held_out_headroom_bytes: int = Field(
        ge=0,
        le=SINT64_MAX,
        description="Minimum nonnegative held-out prediction headroom retained for diagnostics.",
    )
    coefficients: FfnMemoryCalibrationCoefficients = Field(description="Qualified FFN opaque-overhead envelope.")


class XpoolMemoryCalibrationProfile(BaseModel):
    """Complete environment-qualified CrossPool device-memory calibration input."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    environment: MemoryCalibrationEnvironment = Field(description="Profile compatibility evidence.")
    ffn: FfnMemoryCalibration = Field(description="Current FFN calibration payload.")


@dataclass(frozen=True, slots=True)
class DeviceMemoryEstimate:
    """Estimated retained and peak device bytes before operator margin."""

    retained_bytes: int = field(metadata={"description": "Estimated bytes retained after successful startup."})
    peak_bytes: int = field(metadata={"description": "Maximum estimated bytes across all startup stages."})

    def __post_init__(self) -> None:
        """Reject invalid byte counts."""

        ensure_sint64_nonnegative(self.retained_bytes, self.peak_bytes)
        if self.retained_bytes > self.peak_bytes:
            raise ValueError("retained device memory cannot exceed startup peak memory")


def validate_memory_calibration_profile(profile: XpoolMemoryCalibrationProfile) -> None:
    """Require one Profile to match the installed Host and configured geometry.

    Args:
        profile: Strict Profile parsed from the configured path.

    Raises:
        RuntimeError: If the Profile is incompatible or Host evidence cannot be
            observed.

    """

    config = get_global_config()
    driver_version, runtime_version = cuda_versions()
    environment = profile.environment
    expected = {
        "native ABI": ABI_VERSION,
        "FfnAgent count": len(config.ffn.devices),
        "AtnAgent count": len(config.atn.devices),
        "Executor Lane count": config.scheduler.ffn_concurrency,
        "CUDA driver": driver_version,
        "CUDA runtime": runtime_version,
        "Torch": importlib.metadata.version("torch"),
        "Triton": importlib.metadata.version("triton"),
        "SGLang": importlib.metadata.version("sglang"),
        "sglang-kernel": importlib.metadata.version("sglang-kernel"),
        "NVSHMEM": importlib.metadata.version("nvidia-nvshmem-cu13"),
    }
    actual = {
        "native ABI": environment.native_abi_version,
        "FfnAgent count": len(environment.ffnagent_devices),
        "AtnAgent count": profile.ffn.atnagent_count,
        "Executor Lane count": profile.ffn.executor_lane_count,
        "CUDA driver": environment.cuda_driver_version,
        "CUDA runtime": environment.cuda_runtime_version,
        "Torch": environment.torch_version,
        "Triton": environment.triton_version,
        "SGLang": environment.sglang_version,
        "sglang-kernel": environment.sglang_kernel_version,
        "NVSHMEM": environment.nvshmem_version,
    }
    mismatches = [
        f"{name}: expected {value!r}, found {actual[name]!r}"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if mismatches:
        raise RuntimeError("memory calibration Profile is incompatible: " + "; ".join(mismatches))


def load_memory_calibration_profile() -> XpoolMemoryCalibrationProfile | None:
    """Load and validate the configured memory calibration Profile.

    Returns:
        ``None`` when no Profile path is configured, otherwise the compatible
        strict Profile.

    Raises:
        RuntimeError: If the configured Profile cannot be read, parsed, or
            validated against the current Host and configuration.
    """

    path = get_global_config().ffn.device_memory_calibration
    if path is None:
        return None
    try:
        profile = XpoolMemoryCalibrationProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise RuntimeError(f"failed to load memory calibration Profile {path}: {error}") from error
    validate_memory_calibration_profile(profile)
    return profile


def write_memory_calibration_profile(profile: XpoolMemoryCalibrationProfile) -> None:
    """Atomically publish one complete Profile to the configured path.

    Args:
        profile: Complete held-out-qualified Profile to serialize.

    Raises:
        RuntimeError: If no output path is configured or publication fails.

    Side Effects:
        Writes one temporary sibling file and atomically replaces the configured
        destination after serialization completes.
    """

    path = get_global_config().ffn.device_memory_calibration
    if path is None:
        raise RuntimeError("ffn.device_memory_calibration is required to write a calibration Profile")
    try:
        atomic_write_text(path, profile.model_dump_json(indent=2) + "\n")
    except OSError as error:
        raise RuntimeError(f"failed to write memory calibration Profile {path}: {error}") from error
