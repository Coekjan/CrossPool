from __future__ import annotations

import subprocess

import pytest

import tests.harness.runner.gpu
from tests.harness.runner.gpu import GpuPool, query_visible_gpu_total_memory_bytes


def test_pool_normalizes_ordinals_and_uuids_in_user_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,GPU-a,1")

    pool = GpuPool.from_environment()
    try:
        assert pool.uuids == ("GPU-c", "GPU-a", "GPU-b")
        assert pool.physical_index_by_uuid == {"GPU-a": 0, "GPU-b": 1, "GPU-c": 2}
        assert pool.available_count == 3
    finally:
        pool.close()


def test_visible_gpu_memory_normalizes_ordinals_and_uuids_in_user_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,GPU-a,1")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="0, GPU-a, 40960\n1, GPU-b, 81920\n2, GPU-c, 122880\n",
            stderr="",
        ),
    )

    assert query_visible_gpu_total_memory_bytes() == (
        122880 * 1024 * 1024,
        40960 * 1024 * 1024,
        81920 * 1024 * 1024,
    )


@pytest.mark.parametrize(
    ("visibility", "message"),
    [
        ("", "nonempty"),
        ("0,0", "duplicate"),
        ("9", "unknown"),
        ("MIG-abcd", "MIG"),
        ("0,,1", "empty GPU entry"),
    ],
)
def test_pool_rejects_invalid_visibility(
    monkeypatch: pytest.MonkeyPatch,
    visibility: str,
    message: str,
) -> None:
    configure_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)

    with pytest.raises(RuntimeError, match=message):
        GpuPool.from_environment()


def test_pool_leases_and_restores_user_order(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0,1")
    pool = GpuPool.from_environment()
    try:
        first = pool.try_lease(2)
        assert first is not None
        assert first.uuids == ("GPU-c", "GPU-a")
        assert pool.try_lease(2) is None
        second = pool.try_lease(1)
        assert second is not None
        assert second.uuids == ("GPU-b",)

        pool.release(first)
        assert pool.available_count == 2
        pool.release(second)
        assert pool.available_count == 3
        restored = pool.try_lease(3)
        assert restored is not None
        assert restored.uuids == ("GPU-c", "GPU-a", "GPU-b")
        pool.release(restored)
    finally:
        pool.close()


def test_pool_refuses_close_with_active_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    pool = GpuPool.from_environment()
    lease = pool.try_lease(1)
    assert lease is not None

    with pytest.raises(RuntimeError, match="leases remain active"):
        pool.close()

    pool.release(lease)
    pool.close()


def test_physical_gpu_links_preserve_directed_raw_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tests.harness.runner.gpu,
        "query_physical_gpus",
        lambda: {"0": "GPU-a", "1": "GPU-b"},
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="\x1b[4mGPU0 GPU1 CPU Affinity\x1b[0m\nGPU0 X NV12 0-3\nGPU1 PIX X 4-7\n",
            stderr="",
        ),
    )

    assert tests.harness.runner.gpu.query_physical_gpu_links() == {
        ("GPU-a", "GPU-b"): "NV12",
        ("GPU-b", "GPU-a"): "PIX",
    }


def configure_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tests.harness.runner.gpu,
        "query_physical_gpus",
        lambda: {"0": "GPU-a", "1": "GPU-b", "2": "GPU-c"},
    )
