import itertools
from unittest.mock import patch

import pytest
import torch
from tests.suites.integration.sglang.ecs.support import install_vmm_driver

import xpool.integrations.sglang.ecs.state_vmm
import xtest
from xpool import ecs
from xpool.integrations.sglang.ecs.state_vmm import StateBufferLayout, StateVmmBacking
from xpool.utils.cuda import CudaDriverError


@pytest.mark.parametrize("shape,axis", (((3, 7, 5), 1), ((7, 2, 3), 0), ((2, 3, 7), 2)))
def test_unit_coverage_matches_independent_element_addresses(shape: tuple[int, ...], axis: int) -> None:
    tensor = torch.empty(shape, dtype=torch.float32)
    layout = StateBufferLayout(tensor.dtype, shape, tensor.stride(), axis, 16)
    units = (0, 2, 6)
    expected = set()
    for indices in itertools.product(*(range(size) for size in shape)):
        if indices[axis] in units:
            offset = sum(index * stride for index, stride in zip(indices, tensor.stride(), strict=True)) * 4
            expected.update(range(offset // 16, (offset + 3) // 16 + 1))
    assert layout.coverage(units) == tuple(sorted(expected))


def test_mixed_buffers_share_quota_and_pin_exact_nonprefix_granules(monkeypatch: pytest.MonkeyPatch) -> None:
    install_vmm_driver(monkeypatch)
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.state_vmm,
        "tensor_from_pointer",
        lambda pointer, byte_count, **kwargs: torch.empty(byte_count, dtype=torch.uint8),
    )
    tensors = (torch.empty((3, 9, 5), dtype=torch.bfloat16), torch.empty((3, 9, 2, 4), dtype=torch.float32))
    layouts = tuple(StateBufferLayout(t.dtype, tuple(t.shape), t.stride(), 1, 16) for t in tensors)
    quota = ecs.MemoryQuota(1 << 20)
    backing = StateVmmBacking(layouts, quota, device=torch.device("cuda:0"))
    pointers = tuple(t.data_ptr() for t in backing.tensors)
    floor = quota.charged_bytes
    assert floor == sum(profile.floor_bundles * profile.bundle_bytes for profile in backing.profiles)
    assert any(granule > 1 for buffer, granule in backing.pinned)
    acquired = backing.coverage((2, 6))
    quota.set_limit(floor)
    assert not backing.try_map(acquired)
    assert quota.charged_bytes == floor
    quota.set_limit(1 << 20)
    assert backing.try_map(acquired)
    charged = quota.charged_bytes
    assert backing.try_map(acquired)
    assert quota.charged_bytes == charged
    empty = backing.empty_granules((6,))
    assert not set(empty).intersection(backing.coverage((6,)))
    assert backing.unmap(empty) == charged - quota.charged_bytes
    assert tuple(t.data_ptr() for t in backing.tensors) == pointers
    with pytest.raises(ValueError, match="unpinned"):
        backing.unmap(tuple(backing.pinned))
    backing.close()
    backing.close()
    assert quota.charged_bytes == 0


def test_failed_mapping_rolls_back_new_owners_without_refunding_existing_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_vmm_driver(monkeypatch)
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.state_vmm,
        "tensor_from_pointer",
        lambda pointer, byte_count, **kwargs: torch.empty(byte_count, dtype=torch.uint8),
    )
    layout = StateBufferLayout(torch.float32, (8, 4), (4, 1), 0, 16)
    quota = ecs.MemoryQuota(128)
    backing = StateVmmBacking((layout,), quota, device=torch.device("cuda:0"))
    retained = dict(backing.mappings)
    with patch.object(
        xpool.integrations.sglang.ecs.state_vmm.driver, "cuMemSetAccess", side_effect=RuntimeError("map")
    ):
        with pytest.raises(RuntimeError, match="map"):
            backing.try_map(backing.coverage((1, 2)))
    assert backing.mappings == retained
    assert quota.charged_bytes == 16
    backing.close()


@pytest.mark.parametrize("terminal", (False, True))
def test_uncertain_state_mapping_and_release_keep_physical_owners_charged(
    monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    install_vmm_driver(monkeypatch)
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.state_vmm,
        "tensor_from_pointer",
        lambda pointer, byte_count, **kwargs: torch.empty(byte_count, dtype=torch.uint8),
    )
    layout = StateBufferLayout(torch.float32, (8, 4), (4, 1), 0, 16)
    quota = ecs.MemoryQuota(128)
    backing = StateVmmBacking((layout,), quota, device=torch.device("cuda:0"))
    driver = xpool.integrations.sglang.ecs.state_vmm.driver
    if terminal:
        with patch.object(driver, "cuMemSetAccess", return_value=(driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS,)):
            with pytest.raises(CudaDriverError, match="ILLEGAL_ADDRESS"):
                backing.try_map(backing.coverage((1, 2)))
        assert set(backing.mappings) == {(0, 0), (0, 1)}
        assert quota.charged_bytes == 32
    else:
        assert backing.try_map(backing.coverage((1, 2)))
        retained = dict(backing.mappings)
        with patch.object(driver, "cuMemUnmap", side_effect=RuntimeError("unconfirmed release")):
            with pytest.raises(RuntimeError, match="unconfirmed release"):
                backing.unmap(backing.coverage((1, 2)))
        assert backing.mappings == retained and quota.charged_bytes == 48
    # This fake driver restores a healthy domain after the ownership assertion;
    # production terminal errors retire the process instead of retrying here.
    backing.close()
    assert quota.charged_bytes == 0


@xtest.requirements(device_count=1)
def test_sparse_device_storage_preserves_views_and_unrelated_units() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    granularity = StateVmmBacking.granularity(device)
    width = granularity // 4
    layouts = (
        StateBufferLayout(torch.bfloat16, (2, 9, width), (9 * width, width, 1), 1, granularity),
        StateBufferLayout(torch.float32, (2, 9, width), (9 * width, width, 1), 1, granularity),
    )
    quota = ecs.MemoryQuota(sum(layout.reserved_bytes for layout in layouts))
    backing = StateVmmBacking(layouts, quota, device=device)
    try:
        pointers = tuple(t.data_ptr() for t in backing.tensors)
        assert backing.try_map(backing.coverage((2, 6)))
        for tensor in backing.tensors:
            tensor[:, 2].fill_(7)
            tensor[:, 6].fill_(13)
        torch.cuda.synchronize(device)
        backing.unmap(backing.empty_granules((6,)))
        assert backing.try_map(backing.coverage((2,)))
        for tensor in backing.tensors:
            tensor[:, 2].fill_(19)
            assert torch.all(tensor[:, 6] == 13)
        assert pointers == tuple(t.data_ptr() for t in backing.tensors)
    finally:
        torch.cuda.synchronize(device)
        backing.close()
    assert quota.charged_bytes == 0
