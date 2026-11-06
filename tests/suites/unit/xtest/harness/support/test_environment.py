"""Host-setting inheritance shared by qualification launchers."""

from __future__ import annotations

import pytest

from xtest.harness.support.environment import qualification_environment


@pytest.mark.parametrize("configured", (False, True))
def test_qualification_preserves_effective_compute_environment(
    monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    policy = {
        "PATH": "/test/bin",
        "LD_LIBRARY_PATH": "/test/lib",
        "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
        "PYTORCH_ALLOC_CONF": "max_split_size_mb:128",
        "PYTORCH_CUDA_ALLOC_CONF": "",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUBLASLT_WORKSPACE_SIZE": "2048",
        "TORCH_BLAS_PREFER_CUBLASLT": "1",
        "TORCH_CUBLASLT_UNIFIED_WORKSPACE": "0",
        "NVSHMEM_SYMMETRIC_SIZE": "2G",
        "NVSHMEM_CONF_FILE": "/configs/nvshmem.conf",
        "NCCL_ALGO": "Tree",
        "NCCL_CONF_FILE": "/configs/nccl.conf",
        "NCCL_TOPO_FILE": "/configs/topology.xml",
        "CUDA_MODULE_LOADING": "EAGER",
    }
    for name, value in policy.items():
        if configured:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)

    excluded = {
        "CUBLAS_LOGINFO_DBG",
        "CUBLAS_LOGDEST_DBG",
        "NCCL_DEBUG",
        "NCCL_DEBUG_FILE",
        "NVSHMEM_INFO",
        "PYTHONPATH",
        "LD_PRELOAD",
        "XPOOL_UNDECLARED_POLICY",
    }
    for name in excluded:
        monkeypatch.setenv(name, "excluded")
    environment = qualification_environment()
    assert excluded.isdisjoint(environment)

    assert {name: environment[name] for name in policy if name in environment} == (policy if configured else {})
