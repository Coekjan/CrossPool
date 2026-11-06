"""Inherited environment policy for qualification process trees."""

from __future__ import annotations

import os


def qualification_environment() -> dict[str, str]:
    """Copy allowed host settings for qualification subprocesses."""

    inherited_names = (
        "PATH",
        "HOME",
        "LOGNAME",
        "USER",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_LOG_DIRECTORY",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
    )
    compute_prefixes = ("NVSHMEM_", "NCCL_", "CUBLAS_", "CUBLASLT_", "TORCH_BLAS_", "TORCH_CUBLAS", "CUDA_MODULE_")
    diagnostics = {
        "CUBLAS_LOGINFO_DBG",
        "CUBLAS_LOGDEST_DBG",
        "NVSHMEM_INFO",
        "NVSHMEM_VERSION",
        "NVSHMEM_DEBUG",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_DEBUG_FILE",
        "NCCL_TOPO_DUMP_FILE",
    }
    return {
        name: value
        for name, value in os.environ.items()
        if name in inherited_names or (name.startswith(compute_prefixes) and name not in diagnostics)
    }
