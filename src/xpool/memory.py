"""Persistent environment-qualified device-memory calibration Profiles."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from xpool.config import get_global_config
from xpool.mps import probe_mps_controller
from xpool.native import ABI_VERSION
from xpool.utils import align_up

MIB = 1 << 20
SIGNED_INT64_MAX = 2**63 - 1


def ensure_nonnegative_int64(*values: int) -> None:
    """Reject Boolean, negative, or signed-64-bit-overflow values."""

    if any(type(value) is not int or not 0 <= value <= SIGNED_INT64_MAX for value in values):
        raise ValueError("memory calibration arithmetic requires non-negative signed 64-bit integers")


@dataclass(frozen=True, slots=True)
class FfnMemoryCalibrationCoefficients:
    """One immutable non-negative upper envelope over opaque FFN overhead."""

    base_bytes: int = field(metadata={"description": "Constant opaque device-memory bytes."})
    bytes_per_tensor_storage_mib: int = field(
        metadata={"description": "Opaque bytes per ceiling MiB of live Tensor storage."}
    )
    bytes_per_tensor_storage_allocation: int = field(
        metadata={"description": "Opaque bytes per independently owned live Tensor storage."}
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
        metadata={"description": "Opaque bytes present when the local runtime installs Dense execution."}
    )
    moe_implementation_bytes: int = field(
        metadata={"description": "Opaque bytes present when the local runtime installs MoE execution."}
    )
    joined_ffnagent_bytes: int = field(
        metadata={"description": "Opaque bytes present after one FfnAgent joins Fabric."}
    )

    def __post_init__(self) -> None:
        """Reject negative or out-of-domain coefficients."""

        ensure_nonnegative_int64(*self.values())

    def values(self) -> tuple[int, ...]:
        """Return coefficients in their stable fit tie-break order."""

        return (
            self.base_bytes,
            self.bytes_per_tensor_storage_mib,
            self.bytes_per_tensor_storage_allocation,
            self.bytes_per_dense_graph_capture,
            self.bytes_per_moe_graph_capture,
            self.bytes_per_executor_lane,
            self.bytes_per_compute_branch,
            self.dense_implementation_bytes,
            self.moe_implementation_bytes,
            self.joined_ffnagent_bytes,
        )

    def evaluate(
        self,
        *,
        tensor_storage_bytes: int,
        tensor_storage_allocation_count: int,
        dense_graph_capture_count: int,
        moe_graph_capture_count: int,
        executor_lane_count: int,
        compute_branch_count: int,
        dense_implementation_present: bool,
        moe_implementation_present: bool,
        joined_ffnagent: bool,
    ) -> int:
        """Evaluate the exact signed-64-bit opaque-overhead expression."""

        numeric = (
            tensor_storage_bytes,
            tensor_storage_allocation_count,
            dense_graph_capture_count,
            moe_graph_capture_count,
            executor_lane_count,
            compute_branch_count,
        )
        ensure_nonnegative_int64(*numeric)
        tensor_storage_mib = align_up(tensor_storage_bytes, MIB) // MIB
        features = (
            1,
            tensor_storage_mib,
            tensor_storage_allocation_count,
            dense_graph_capture_count,
            moe_graph_capture_count,
            executor_lane_count,
            compute_branch_count,
            int(dense_implementation_present),
            int(moe_implementation_present),
            int(joined_ffnagent),
        )
        result = sum(coefficient * value for coefficient, value in zip(self.values(), features, strict=True))
        ensure_nonnegative_int64(result)
        return result


class MemoryCalibrationGpu(BaseModel):
    """One ordered FfnAgent GPU recorded by a calibration run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1, description="CUDA device marketing name.")
    compute_capability: tuple[int, int] = Field(description="CUDA major and minor compute capability.")
    total_memory_bytes: int = Field(ge=1, description="Total CUDA device memory in bytes.")
    uuid: str = Field(min_length=1, description="GPU UUID retained as non-matching provenance.")

    @model_validator(mode="after")
    def validate_compute_capability(self) -> MemoryCalibrationGpu:
        """Reject negative compute-capability components."""

        if any(value < 0 for value in self.compute_capability):
            raise ValueError("GPU compute capability must contain non-negative integers")
        return self


class MemoryCalibrationEnvironment(BaseModel):
    """Hardware and software compatibility facts for one calibration Profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    native_abi_version: int = Field(ge=1, description="xpool native ABI used for calibration.")
    ffnagent_gpus: tuple[MemoryCalibrationGpu, ...] = Field(
        min_length=1,
        description="FfnAgent GPUs in configured FfnAgent index order.",
    )
    cuda_driver_version: int = Field(ge=1, description="CUDA driver API integer version.")
    cuda_runtime_version: int = Field(ge=1, description="CUDA runtime API integer version.")
    mps_active_thread_percentage: int = Field(ge=1, le=100, description="CUDA MPS active-thread percentage.")
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
        le=SIGNED_INT64_MAX,
        description="Minimum nonnegative H0 prediction headroom retained for diagnostics.",
    )
    coefficients: FfnMemoryCalibrationCoefficients = Field(description="Qualified FFN opaque-overhead envelope.")


class XpoolMemoryCalibrationProfile(BaseModel):
    """Complete environment-qualified xpool device-memory calibration input."""

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

        ensure_nonnegative_int64(self.retained_bytes, self.peak_bytes)
        if self.retained_bytes > self.peak_bytes:
            raise ValueError("retained device memory cannot exceed startup peak memory")


def cuda_versions() -> tuple[int, int]:
    """Read CUDA driver and runtime versions without creating a CUDA context."""

    runtime = importlib.import_module("cuda.bindings.runtime")
    driver_error, driver_version = runtime.cudaDriverGetVersion()
    runtime_error, runtime_version = runtime.cudaRuntimeGetVersion()
    if driver_error != runtime.cudaError_t.cudaSuccess or runtime_error != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"failed to query CUDA versions: driver={driver_error.name}, runtime={runtime_error.name}")
    return int(driver_version), int(runtime_version)


def validate_memory_calibration_profile(profile: XpoolMemoryCalibrationProfile) -> None:
    """Require one Profile to match the installed Host and configured geometry.

    Args:
        profile: Strict Profile parsed from the configured path.

    Raises:
        RuntimeError: If the Profile is incompatible or Host evidence cannot be
            observed.

    Side Effects:
        Runs the serialized CUDA MPS controller probe.
    """

    config = get_global_config()
    mps = probe_mps_controller()
    if not mps.online or mps.active_thread_percentage is None:
        raise RuntimeError(f"memory calibration requires a reachable MPS controller: {mps.diagnostic}")
    driver_version, runtime_version = cuda_versions()
    environment = profile.environment
    expected = {
        "native ABI": ABI_VERSION,
        "FfnAgent count": len(config.devices.ffn_cuda_devices),
        "AtnAgent count": len(config.devices.atn_cuda_devices),
        "Executor Lane count": config.scheduler.ffn_concurrency,
        "CUDA driver": driver_version,
        "CUDA runtime": runtime_version,
        "MPS active-thread percentage": mps.active_thread_percentage,
        "Torch": importlib.metadata.version("torch"),
        "Triton": importlib.metadata.version("triton"),
        "SGLang": importlib.metadata.version("sglang"),
        "sglang-kernel": importlib.metadata.version("sglang-kernel"),
        "NVSHMEM": importlib.metadata.version("nvidia-nvshmem-cu13"),
    }
    actual = {
        "native ABI": environment.native_abi_version,
        "FfnAgent count": len(environment.ffnagent_gpus),
        "AtnAgent count": profile.ffn.atnagent_count,
        "Executor Lane count": profile.ffn.executor_lane_count,
        "CUDA driver": environment.cuda_driver_version,
        "CUDA runtime": environment.cuda_runtime_version,
        "MPS active-thread percentage": environment.mps_active_thread_percentage,
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

    path = get_global_config().memory.calibration_path
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

    path = get_global_config().memory.calibration_path
    if path is None:
        raise RuntimeError("memory.calibration_path is required to write a calibration Profile")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(profile.model_dump_json(indent=2) + "\n")
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as error:
        raise RuntimeError(f"failed to write memory calibration Profile {path}: {error}") from error
    finally:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)
