from __future__ import annotations

from typing import cast

import torch
from sglang.srt.mem_cache.memory_pool import KVCache

from xpool.integrations.sglang.kv.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)


def test_token_allocator_withholds_and_readmits_only_free_suffix_ids() -> None:
    allocator = ElasticTokenToKVPoolAllocator(8, torch.float16, "cpu", cast(KVCache, object()), False)
    allocated = allocator.alloc(6)
    assert allocated is not None

    allocator.set_token_capacity(4)
    assert allocator.available_size() == 0
    assert not allocator.suffix_is_free(4)

    allocator.free(allocated[4:])
    assert allocator.suffix_is_free(4)
    assert allocator.withheld_size() == 4

    allocator.set_token_capacity(6)
    assert allocator.available_size() == 2
    assert allocator.withheld_size() == 2
    allocator.clear()
    assert allocator.available_size() == 6
    assert allocator.withheld_size() == 2


def test_paged_allocator_routes_freed_suffix_pages_through_page_ids() -> None:
    allocator = ElasticPagedTokenToKVPoolAllocator(16, 4, torch.float16, "cpu", cast(KVCache, object()), False)
    allocated = allocator.alloc(12)
    assert allocated is not None

    allocator.set_token_capacity(8)
    assert allocator.available_size() == 0
    assert not allocator.suffix_is_free(8)

    allocator.free_segment(allocated[8:], start_pos=8)
    assert allocator.suffix_is_free(8)
    assert allocator.withheld_size() == 8

    allocator.set_token_capacity(12)
    assert allocator.available_size() == 4
    assert allocator.withheld_size() == 4
    allocator.clear()
    assert allocator.available_size() == 12
    assert allocator.withheld_size() == 4


def test_capacity_growth_preserves_upstream_sorting_containers() -> None:
    token_allocator = ElasticTokenToKVPoolAllocator(8, torch.float16, "cpu", cast(KVCache, object()), True)
    token_allocator.set_token_capacity(4)
    token_allocator.set_token_capacity(6)
    assert token_allocator.release_pages is not None
    assert token_allocator.release_pages.tolist() == [5, 6]
    assert token_allocator.alloc(6).tolist() == [1, 2, 3, 4, 5, 6]

    paged_allocator = ElasticPagedTokenToKVPoolAllocator(16, 4, torch.float16, "cpu", cast(KVCache, object()), True)
    paged_allocator.set_token_capacity(8)
    paged_allocator.set_token_capacity(12)
    assert paged_allocator.release_pages is not None
    assert paged_allocator.release_pages.tolist() == [3]
    assert paged_allocator.alloc(12).tolist() == list(range(4, 16))
