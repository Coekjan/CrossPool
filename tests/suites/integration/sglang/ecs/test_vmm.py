from __future__ import annotations

import pytest
import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.mem_cache.memory_pool import KvBufferDesc

import xpool.integrations.sglang.ecs.vmm
import xtest
from xpool import ecs
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking, KvVmmBundle, KvVmmLayout
from xpool.utils.cuda import CudaDriverError


@pytest.fixture
def layout() -> KvVmmLayout:
    return KvVmmLayout(torch.device("cuda:0"), torch.uint8, (10, 4), 1, 1, 1, 1, 8)


@pytest.fixture
def backing(vmm_driver: None, layout: KvVmmLayout) -> KvVmmBacking:
    return KvVmmBacking(layout, ecs.MemoryQuota(layout.profile.bytes_for(layout.profile.bundle_capacity)))


def test_sparse_mapping_keeps_views_and_refunds_exact_ids(backing: KvVmmBacking) -> None:
    pointers = tuple(tensor.data_ptr() for tensor in backing.tensors)
    assert backing.try_map((3,))
    assert backing.try_map((1, 3))
    assert set(backing.bundles) == {0, 1, 3}
    assert backing.backed_bytes == backing.quota.charged_bytes == 24

    assert backing.unmap((3,)) == 8
    assert backing.unmap((3,)) == 0
    assert set(backing.bundles) == {0, 1}
    assert backing.try_map((4,))
    assert tuple(tensor.data_ptr() for tensor in backing.tensors) == pointers
    with pytest.raises(ValueError, match="unpinned"):
        backing.unmap((0, 4))
    assert set(backing.bundles) == {0, 1, 4}

    backing.close()
    assert not backing.bundles
    assert backing.backed_bytes == backing.quota.charged_bytes == 0
    assert backing.base == 0


def test_two_backings_share_one_quota(backing: KvVmmBacking) -> None:
    quota = backing.quota
    quota.set_limit(24)
    peer = KvVmmBacking(backing.layout, quota)

    assert backing.try_map((2,))
    assert not peer.try_map((2,))
    assert set(peer.bundles) == {0}
    assert quota.charged_bytes == 24

    assert backing.unmap((2,)) == 8
    assert peer.try_map((2,))
    backing.close()
    peer.close()
    assert quota.charged_bytes == 0


@pytest.mark.parametrize("operation", ("cuMemCreate", "cuMemMap", "cuMemSetAccess"))
def test_allocation_exhaustion_rolls_back_only_the_new_attempt(
    backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls = 0

    def exhaust(*args: object) -> tuple[driver.CUresult, ...]:
        nonlocal calls
        calls += 1
        status = driver.CUresult.CUDA_SUCCESS if calls == 1 else driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY
        return (status, calls + 10) if operation == "cuMemCreate" else (status,)

    monkeypatch.setattr(xpool.integrations.sglang.ecs.vmm.driver, operation, exhaust)
    assert not backing.try_map((2, 4))
    assert set(backing.bundles) == {0}
    assert backing.backed_bytes == backing.quota.charged_bytes == 8


def test_checked_mapping_failure_is_not_a_capacity_miss(backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemSetAccess",
        lambda *args: (driver.CUresult.CUDA_ERROR_INVALID_VALUE,),
    )
    with pytest.raises(CudaDriverError, match="cuMemSetAccess"):
        backing.try_map((2, 4))
    assert set(backing.bundles) == {0}
    assert backing.backed_bytes == backing.quota.charged_bytes == 8


def test_terminal_result_retains_created_storage_and_skips_device_cleanup(
    backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemSetAccess",
        lambda *args: (driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS,),
    )
    for operation in ("cuMemUnmap", "cuMemRelease"):
        monkeypatch.setattr(
            xpool.integrations.sglang.ecs.vmm.driver,
            operation,
            lambda *args: pytest.fail("device cleanup after terminal result"),
        )
    with pytest.raises(CudaDriverError, match="cuMemSetAccess"):
        backing.try_map((2, 4))
    assert set(backing.bundles) == {0, 2}
    assert backing.backed_bytes == backing.quota.charged_bytes == 16


def test_unconfirmed_rollback_retains_charge_and_original_failure(
    backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemSetAccess",
        lambda *args: (driver.CUresult.CUDA_ERROR_INVALID_VALUE,),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemUnmap",
        lambda *args: (driver.CUresult.CUDA_ERROR_UNKNOWN,),
    )
    with pytest.raises(CudaDriverError, match="cuMemUnmap") as caught:
        backing.try_map((2, 4))
    assert isinstance(caught.value.__context__, CudaDriverError)
    assert caught.value.__context__.operation == "cuMemSetAccess(xpool elastic context state bundle)"
    assert set(backing.bundles) == {0, 2}
    assert backing.backed_bytes == backing.quota.charged_bytes == 16


def test_handle_release_failure_does_not_refund_physical_ownership(
    backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert backing.try_map((3,))
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemRelease",
        lambda *args: (driver.CUresult.CUDA_ERROR_UNKNOWN,),
    )
    with pytest.raises(CudaDriverError, match="cuMemRelease"):
        backing.unmap((3,))
    assert backing.backed_bytes == backing.quota.charged_bytes == 16
    assert not backing.bundles[3].mapped

    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemRelease",
        lambda *args: (driver.CUresult.CUDA_SUCCESS,),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemUnmap",
        lambda *args: pytest.fail("unmapping an already removed mapping"),
    )
    assert backing.unmap((3,)) == 8
    assert backing.backed_bytes == backing.quota.charged_bytes == 8


@pytest.mark.parametrize("tokens_per_row, page_size, row_bytes", ((1, 1, 4), (1, 4, 10), (4, 4, 10), (3, 2, 10)))
def test_page_incidence_counts_crossing_rows_and_dummy_storage(
    tokens_per_row: int, page_size: int, row_bytes: int
) -> None:
    layout = KvVmmLayout(torch.device("cuda:0"), torch.uint8, (17, row_bytes), tokens_per_row, page_size, 2, 2, 16)
    backing = object.__new__(KvVmmBacking)
    backing.layout = layout
    backing.bundles = {bundle_id: KvVmmBundle(1, mapped=True) for bundle_id in range(layout.profile.bundle_capacity)}
    page_count = layout.token_capacity // page_size
    pages = torch.arange(1, page_count + 1, dtype=torch.int64)
    starts, ends = backing.bundle_spans(pages)
    incidence = {
        page: {
            byte_offset // 16
            for token in range(page * page_size, (page + 1) * page_size)
            for byte_offset in range(token // tokens_per_row * row_bytes, (token // tokens_per_row + 1) * row_bytes)
        }
        for page in range(1, page_count + 1)
    }
    assert [set(range(start, end)) for start, end in zip(starts.tolist(), ends.tolist(), strict=True)] == list(
        incidence.values()
    )
    free_pages = {page for page in incidence if page % 3 != 0}
    expected = {
        bundle_id
        for bundle_id in backing.bundles
        if bundle_id >= layout.profile.floor_bundles
        and all(page in free_pages for page, bundles in incidence.items() if bundle_id in bundles)
    }
    assert set(backing.empty_bundles(torch.tensor(sorted(free_pages), dtype=torch.int64)).tolist()) == expected


@xtest.requirements(device_count=1)
def test_sparse_device_mapping_preserves_other_bundles_and_tensor_addresses() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    row_bytes = 128
    probe = KvVmmBacking.describe(
        device=device,
        storage_dtype=torch.float16,
        token_page_size=1,
        descriptors=(KvBufferDesc("k0", (2, 64), row_bytes=row_bytes, tokens_per_row=1),) * 4,
        buffers_per_layer=2,
    )
    rows_per_bundle = probe.mapping_granularity_bytes // row_bytes
    layout = KvVmmLayout(device, torch.float16, (5 * rows_per_bundle, 64), 1, 1, 2, 2, probe.mapping_granularity_bytes)
    quota = ecs.MemoryQuota(layout.profile.bytes_for(layout.profile.bundle_capacity))
    backing = KvVmmBacking(layout, quota)
    try:
        pointers = tuple(tensor.data_ptr() for tensor in backing.tensors)
        assert backing.try_map((2, 4))
        locations = torch.tensor([0, 2 * rows_per_bundle, 4 * rows_per_bundle], device=device)
        values = torch.arange(3 * 64, dtype=torch.float16, device=device).view(3, 64)
        for index, tensor in enumerate(backing.tensors):
            tensor[locations] = values + index
        torch.cuda.synchronize()

        assert backing.unmap((4,)) == layout.profile.bundle_bytes
        assert backing.try_map((3,))
        for index, tensor in enumerate(backing.tensors):
            torch.testing.assert_close(tensor[locations[:2]], values[:2] + index, rtol=0, atol=0)
            tensor[3 * rows_per_bundle] = index
        torch.cuda.synchronize()
        assert tuple(tensor.data_ptr() for tensor in backing.tensors) == pointers
        assert set(backing.bundles) == {0, 2, 3}
        assert backing.backed_bytes == quota.charged_bytes == 3 * layout.profile.bundle_bytes
        free_pages = torch.arange(1, layout.token_capacity + 1, dtype=torch.int64, device=device)
        free_pages = free_pages[free_pages != 3 * rows_per_bundle]
        assert backing.empty_bundles(free_pages).tolist() == [2]
    finally:
        backing.close()
    assert quota.charged_bytes == 0
