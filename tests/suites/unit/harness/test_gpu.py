from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

import tests.harness.gpu
from tests.harness.gpu import GpuPool


def test_pool_normalizes_ordinals_and_uuids_in_user_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_inventory(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,GPU-a,1")

    pool = GpuPool.from_environment()
    try:
        assert pool.uuids == ("GPU-c", "GPU-a", "GPU-b")
        assert pool.available_count == 3
    finally:
        pool.close()


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visibility: str,
    message: str,
) -> None:
    configure_inventory(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)

    with pytest.raises(RuntimeError, match=message):
        GpuPool.from_environment()


def test_pool_leases_and_restores_user_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure_inventory(tmp_path, monkeypatch)
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


def test_pool_refuses_close_with_active_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure_inventory(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    pool = GpuPool.from_environment()
    lease = pool.try_lease(1)
    assert lease is not None

    with pytest.raises(RuntimeError, match="leases remain active"):
        pool.close()

    pool.release(lease)
    pool.close()


def test_pool_reports_every_conflicting_whole_run_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_inventory(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    first = (tmp_path / "GPU-a.lock").open("a+", encoding="utf-8")
    second = (tmp_path / "GPU-b.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        with pytest.raises(RuntimeError, match=r"GPU-a.*GPU-b"):
            GpuPool.from_environment()
    finally:
        first.close()
        second.close()


def configure_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tests.harness.gpu, "GPU_LOCK_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        tests.harness.gpu,
        "query_physical_gpus",
        lambda: {"0": "GPU-a", "1": "GPU-b", "2": "GPU-c"},
    )
