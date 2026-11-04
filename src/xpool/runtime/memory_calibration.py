"""Participant-local environment observations for memory calibration reuse."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import pwd
from pathlib import Path

import torch
import torch.cuda.nccl

import xpool.native
from xpool.memory import MemoryCalibrationEnvironment
from xpool.utils.cuda import cuda_versions


def observe_environment(device: int) -> MemoryCalibrationEnvironment:
    """Read Profile-matching facts before the free-memory baseline.

    Device/software facts and configuration contents identify measured inputs.
    NCCL topology selection follows its environment/config-file precedence.
    Inspection failures raise before facts are published. Unprofiled startup
    skips this compatibility-only observation.
    """

    excluded = {
        "CUBLAS_LOGINFO_DBG",
        "CUBLAS_LOGDEST_DBG",
        "NVSHMEM_INFO",
        "NVSHMEM_VERSION",
        "NVSHMEM_DEBUG",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_DEBUG_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "NCCL_TOPO_FILE",
        "NVSHMEM_CONF_FILE",
        "NCCL_CONF_FILE",
    }
    prefixes = ("NVSHMEM_", "NCCL_", "CUBLAS_", "CUBLASLT_", "TORCH_BLAS_", "TORCH_CUBLAS", "CUDA_MODULE_")
    inputs: dict[str, str | None] = {
        f"environment.{key}": value
        for key, value in os.environ.items()
        if key.startswith(prefixes) and key not in excluded
    }
    failures = []
    inputs.update(
        {
            "torch.allocator_backend": torch.cuda.memory.get_allocator_backend(),
            **{
                f"environment.{name}": os.environ.get(name)
                for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING")
            },
            "torch.preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
            "torch.cublas_workspace_size": str(torch.backends.cuda.cublas_workspace_size()),
            "torch.cublaslt_workspace_size": str(torch.backends.cuda.cublaslt_workspace_size()),
        }
    )
    paths = {"file.nvshmem.system": Path("/etc/nvshmem.conf")}
    if home_directory := os.environ.get("HOME"):
        paths["file.nvshmem.user"] = Path(home_directory) / ".nvshmem.conf"
    if configured := os.environ.get("NVSHMEM_CONF_FILE"):
        paths["file.nvshmem.selected"] = Path(configured)
    if configured := os.environ.get("NCCL_CONF_FILE"):
        paths["file.nccl.selected"] = Path(os.fsdecode(os.fsencode(configured)[:1023]))
    else:
        try:
            nccl_home_directory = pwd.getpwuid(os.getuid()).pw_dir
        except (KeyError, OSError) as error:
            failures.append(f"file.nccl.user (passwd lookup): {error}")
        else:
            paths["file.nccl.user"] = Path(os.fsdecode(os.fsencode(f"{nccl_home_directory}/.nccl.conf")[:1023]))
    paths["file.nccl.system"] = Path("/etc/nccl.conf")

    def read_input(name: str, path: Path) -> bytes | None:
        try:
            contents = path.read_bytes()
        except OSError as error:
            inputs[name] = None
            if not isinstance(error, FileNotFoundError):
                failures.append(f"{name} ({path}): {error}")
            return None
        inputs[name] = hashlib.sha256(contents).hexdigest()
        return contents

    topology_file = os.environ.get("NCCL_TOPO_FILE")
    for name, path in paths.items():
        contents = read_input(name, path)
        if contents is None or not name.startswith("file.nccl.") or topology_file is not None:
            continue
        # NCCL keeps the first assignment, preserves whitespace and limits keys/values to 1023 bytes.
        for line in contents.split(b"\n"):
            if line.startswith(b"#"):
                continue
            key, separator, value = line.partition(b"\0")[0].partition(b"=")
            if separator and key[:1023] == b"NCCL_TOPO_FILE":
                topology_file = os.fsdecode(value[:1023])
                break
    read_input(
        "file.nccl.topology",
        Path("/var/run/nvidia-topologyd/virtualTopology.xml" if topology_file is None else topology_file),
    )
    if failures:
        raise RuntimeError(f"cannot observe memory calibration inputs on device {device}: " + "; ".join(failures))
    driver_version, runtime_version = cuda_versions()
    properties = torch.cuda.get_device_properties(device)
    return MemoryCalibrationEnvironment(
        name=properties.name,
        compute_capability=(properties.major, properties.minor),
        total_memory_bytes=properties.total_memory,
        native_abi_version=xpool.native.ABI_VERSION,
        cuda_driver_version=driver_version,
        cuda_runtime_version=runtime_version,
        torch_version=importlib.metadata.version("torch"),
        triton_version=importlib.metadata.version("triton"),
        sglang_version=importlib.metadata.version("sglang"),
        sglang_kernel_version=importlib.metadata.version("sglang-kernel"),
        nvshmem_version=importlib.metadata.version("nvidia-nvshmem-cu13"),
        nccl_version=torch.cuda.nccl.version(),
        configuration_inputs=tuple(sorted(inputs.items())),
    )
