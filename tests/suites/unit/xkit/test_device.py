from __future__ import annotations

import subprocess

import pytest

import xkit.device
from xkit.device import DevicePool, query_visible_device_total_memory_bytes


def test_pool_keeps_selected_order_separate_from_inventory_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,GPU-00000000-0000-0000-0000-000000000001,1")

    pool = DevicePool.from_environment()
    try:
        assert pool.uuids == (
            "GPU-00000000-0000-0000-0000-000000000003",
            "GPU-00000000-0000-0000-0000-000000000001",
            "GPU-00000000-0000-0000-0000-000000000002",
        )
        assert pool.physical_index_by_uuid == {
            "GPU-00000000-0000-0000-0000-000000000001": 0,
            "GPU-00000000-0000-0000-0000-000000000002": 1,
            "GPU-00000000-0000-0000-0000-000000000003": 2,
        }
        assert pool.available_count == 3
    finally:
        pool.close()


def test_visible_device_memory_follows_selected_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,GPU-00000000-0000-0000-0000-000000000001,1")
    monkeypatch.setattr(
        xkit.device,
        "query_physical_device_details",
        lambda: tuple(
            xkit.device.PhysicalDevice(
                str(index - 1),
                f"GPU-00000000-0000-0000-0000-{index:012x}",
                None,
                None,
                index * 40960 * 1024 * 1024,
                None,
            )
            for index in (1, 2, 3)
        ),
    )

    assert query_visible_device_total_memory_bytes() == (
        122880 * 1024 * 1024,
        40960 * 1024 * 1024,
        81920 * 1024 * 1024,
    )


@pytest.mark.parametrize(
    "visibility",
    [None, ""],
)
def test_pool_requires_explicit_visibility(
    monkeypatch: pytest.MonkeyPatch,
    visibility: str | None,
) -> None:
    configure_inventory(monkeypatch)
    if visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)

    with pytest.raises(RuntimeError, match="explicit nonempty"):
        DevicePool.from_environment()


def test_pool_leases_and_restores_user_order(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0,1")
    pool = DevicePool.from_environment()
    try:
        first = pool.try_lease(2)
        assert first is not None
        assert first.uuids == ("GPU-00000000-0000-0000-0000-000000000003", "GPU-00000000-0000-0000-0000-000000000001")
        assert pool.try_lease(2) is None
        second = pool.try_lease(1)
        assert second is not None
        assert second.uuids == ("GPU-00000000-0000-0000-0000-000000000002",)

        pool.release(first)
        assert pool.available_count == 2
        pool.release(second)
        assert pool.available_count == 3
        restored = pool.try_lease(3)
        assert restored is not None
        assert restored.uuids == (
            "GPU-00000000-0000-0000-0000-000000000003",
            "GPU-00000000-0000-0000-0000-000000000001",
            "GPU-00000000-0000-0000-0000-000000000002",
        )
        pool.release(restored)
    finally:
        pool.close()


def test_pool_refuses_close_with_active_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_inventory(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    pool = DevicePool.from_environment()
    lease = pool.try_lease(1)
    assert lease is not None

    with pytest.raises(RuntimeError, match="leases remain active"):
        pool.close()

    pool.release(lease)
    pool.close()


def test_pool_returns_only_the_active_allocation() -> None:
    uuids = ("GPU-00000000-0000-0000-0000-000000000001",)
    pool = DevicePool(uuids, {uuids[0]: 0})
    first = pool.try_lease(1)
    assert first is not None
    pool.release(first)
    current = pool.try_lease(1)
    assert current is not None

    with pytest.raises(RuntimeError, match="not active"):
        pool.release(first)
    assert pool.available_count == 0

    pool.release(current)
    pool.close()


def test_physical_device_topology_preserves_directed_raw_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert xkit.device.query_physical_device_topology(
        {"0": "GPU-00000000-0000-0000-0000-000000000001", "1": "GPU-00000000-0000-0000-0000-000000000002"}
    ).links == {
        ("GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002"): "NV12",
        ("GPU-00000000-0000-0000-0000-000000000002", "GPU-00000000-0000-0000-0000-000000000001"): "PIX",
    }


def configure_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        xkit.device,
        "visible_uuids",
        lambda: (
            "GPU-00000000-0000-0000-0000-000000000003",
            "GPU-00000000-0000-0000-0000-000000000001",
            "GPU-00000000-0000-0000-0000-000000000002",
        ),
    )
    monkeypatch.setattr(
        xkit.device,
        "query_uuids_mapping",
        lambda: {
            0: "GPU-00000000-0000-0000-0000-000000000001",
            1: "GPU-00000000-0000-0000-0000-000000000002",
            2: "GPU-00000000-0000-0000-0000-000000000003",
        },
    )
