from __future__ import annotations

import pytest
import torch

from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)


def test_sparse_growth_publishes_only_backed_pages_before_original_allocation(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
) -> None:
    count = allocator.available_size() // allocator.page_size + 3
    plan = allocator.plan_pages(count)
    assert plan is not None
    assert plan.missing_bundle_ids
    before = allocator.available_size()
    with pytest.raises(RuntimeError, match="incompletely backed"):
        allocator.publish_pages(plan.page_ids)
    assert allocator.available_size() == before

    assert allocator.backing.try_map(plan.missing_bundle_ids)
    allocator.publish_pages(plan.page_ids)
    locations = allocator.alloc(count * allocator.page_size)
    assert locations is not None
    page_ids = torch.unique(locations // allocator.page_size)
    assert len(page_ids) == count
    assert torch.count_nonzero(allocator.backing.missing_bundle_counts(page_ids)) == 0

    allocator.free(locations)
    assert allocator.available_size() == count * allocator.page_size
    logical_free = torch.cat((allocator.get_all_free_pages(), allocator.unavailable_pages))
    assert len(logical_free) == 32 // allocator.page_size
    assert len(logical_free.unique()) == len(logical_free)


def test_deferred_free_keeps_ownership_until_the_upstream_group_closes(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
) -> None:
    plan = allocator.plan_pages(4)
    assert plan is not None
    assert allocator.backing.try_map(plan.missing_bundle_ids)
    allocator.publish_pages(plan.page_ids)
    locations = allocator.alloc(4 * allocator.page_size)
    assert locations is not None
    allocator.free_group_begin()
    allocator.free(locations)
    assert allocator.available_size() == 0
    assert allocator.plan_pages(32 // allocator.page_size) is None

    allocator.free_group_end()
    assert allocator.available_size() == 4 * allocator.page_size
    assert allocator.plan_pages(32 // allocator.page_size) is not None


def test_clear_restores_the_backed_free_partition_after_deferred_release(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
) -> None:
    plan = allocator.plan_pages(4)
    assert plan is not None
    assert allocator.backing.try_map(plan.missing_bundle_ids)
    allocator.publish_pages(plan.page_ids)
    locations = allocator.alloc(allocator.available_size())
    assert locations is not None
    allocator.free(locations)

    allocator.clear()

    pages = torch.arange(1, allocator.size // allocator.page_size + 1)
    resident = allocator.backing.missing_bundle_counts(pages) == 0
    free = allocator.get_all_free_pages()
    torch.testing.assert_close(torch.sort(free).values, pages[resident])
    torch.testing.assert_close(torch.sort(allocator.unavailable_pages).values, pages[~resident])
    assert allocator.available_size() == len(free) * allocator.page_size


def test_retirement_withdraws_staged_pages_before_sort_and_physical_unmap(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
) -> None:
    backing = allocator.backing
    assert backing.try_map((3,))
    available = allocator.unavailable_pages[backing.missing_bundle_counts(allocator.unavailable_pages) == 0]
    allocator.publish_pages(available)
    locations = allocator.alloc(allocator.available_size())
    assert locations is not None
    allocator.free(locations)
    exposed = allocator.get_all_free_pages()
    selected = allocator.intersecting_pages(exposed, (3,))
    expected_excluded = exposed[selected]
    assert expected_excluded.numel()

    allocator.begin_retirement((3,))
    with pytest.raises(RuntimeError, match="during retirement"):
        allocator.clear()
    excluded = allocator.retiring_pages
    assert set(expected_excluded.tolist()).issubset(excluded.tolist())
    allocator.merge_and_sort_free()
    assert not allocator.intersecting_pages(allocator.get_all_free_pages(), (3,)).any()
    assert allocator.plan_pages(32 // allocator.page_size) is None

    assert backing.unmap((3,)) == backing.layout.profile.bundle_bytes
    allocator.finish_retirement()
    assert allocator.plan_pages(32 // allocator.page_size) is not None
    assert torch.count_nonzero(backing.missing_bundle_counts(excluded)) == len(excluded)
