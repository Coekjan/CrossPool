from __future__ import annotations

from array import array

import pytest
import sglang.srt.mem_cache.allocator.swa
import torch
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

import xtest
from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.ecs.radix import SglangCacheEviction
from xpool.integrations.sglang.ecs.state import SglangElasticContextState, SglangMemoryRequirement
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


def test_preparation_publishes_all_fully_backed_free_pages_without_exposing_owned_pages(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
    attention_cache: UnifiedRadixCache,
) -> None:
    state = SglangElasticContextState.create(
        allocator.get_kvcache(),
        allocator,
        attention_cache.req_to_token_pool,
        configurator=object.__new__(KVCacheConfigurator),
    )
    state.quota.set_limit(state.profile.reservation_bytes)
    assert state.prepare_allocation((1,))
    held = allocator.alloc(allocator.page_size)
    assert held is not None
    assert state.prepare_allocation((4,))
    free = allocator.get_all_free_pages()
    assert len(free) >= 4
    assert not torch.isin(held // allocator.page_size, free).any()
    assert torch.all(allocator.backing.missing_bundle_counts(allocator.unavailable_pages) > 0)
    allocator.free(held)


def test_joint_forecast_uses_only_unlocked_same_pool_cache(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator, attention_cache: UnifiedRadixCache
) -> None:
    # No sizing calculation runs here; retain the actual upstream owner type.
    state = SglangElasticContextState.create(
        allocator.get_kvcache(),
        allocator,
        attention_cache.req_to_token_pool,
        configurator=object.__new__(KVCacheConfigurator),
    )
    state.bind_cache(attention_cache)
    assert state.prepare_allocation((2,))
    locations = allocator.alloc(allocator.page_size)
    assert locations is not None
    inserted = attention_cache.insert(
        InsertParams(key=RadixKey(array("q", range(allocator.page_size))), value=locations)
    )
    requirement = SglangMemoryRequirement((32 // allocator.page_size,), (1,))
    assert state.required_bytes(requirement) == state.profile.reservation_bytes
    receipt = attention_cache.inc_lock_ref(inserted.last_device_node).to_dec_params()
    assert state.required_bytes(requirement) is None
    attention_cache.dec_lock_ref(inserted.last_device_node, receipt)
    assert state.required_bytes(requirement) == state.profile.reservation_bytes

    logical = state.logical_free_pages()
    allocator.free_group_begin()
    current = allocator.alloc(allocator.page_size)
    assert current is not None
    allocator.free(current)
    assert state.logical_free_pages() == (logical[0] - 1,)
    allocator.free_group_end()
    assert state.logical_free_pages() == logical


@xtest.requirements(device_count=1)
def test_composite_prepares_joint_cost_before_original_full_swa_allocation(
    full_swa_pool: SWAKVPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sglang.srt.mem_cache.allocator.swa, "PagedTokenToKVPoolAllocator", ElasticPagedTokenToKVPoolAllocator
    )
    allocator = SWATokenToKVPoolAllocator(8192, 4096, 4, torch.float16, full_swa_pool.device, full_swa_pool, True)
    configurator = object.__new__(KVCacheConfigurator)
    configurator.forward_stream = torch.cuda.current_stream()
    state = SglangElasticContextState.create(
        full_swa_pool,
        allocator,
        ReqToTokenPool(2, 64, full_swa_pool.device, False),
        configurator=configurator,
    )
    charged = state.quota.charged_bytes
    resident = (allocator.full_available_size(), allocator.swa_available_size())
    growth = sum(profile.bundle_bytes for profile in state.profile.pools)
    state.quota.set_limit(charged + max(profile.bundle_bytes for profile in state.profile.pools))
    assert not state.prepare_allocation((600, 600))
    assert state.quota.charged_bytes == charged
    assert (allocator.full_available_size(), allocator.swa_available_size()) == resident

    state.quota.set_limit(charged + growth)
    assert state.prepare_allocation((600, 600))
    assert state.quota.charged_bytes == charged + growth
    device = full_swa_pool.device
    allocated = allocator.alloc_extend(
        torch.tensor([0], device=device),
        torch.tensor([0]),
        torch.tensor([2400], device=device),
        torch.tensor([2400]),
        torch.tensor([-1], device=device),
        2400,
    )
    assert allocated is not None
    assert torch.all(allocator.full_to_swa_index_mapping[allocated] > 0)
    assert state.logical_free_pages() == (8192 // 4 - 600, 4096 // 4 - 600)
    allocator.free(allocated)
    torch.cuda.synchronize()
    assert state.logical_free_pages() == (8192 // 4, 4096 // 4)
    assert state.quota.charged_bytes == charged + growth
    plan = state.prepare_reclaim(growth)
    assert plan is not None
    state.apply_reclaim(plan)
    assert state.quota.charged_bytes == charged + growth
    assert state.prepare_allocation((0, 0))
    for binding in state.components:
        child = binding.allocator
        assert not torch.isin(child.retiring_pages, child.get_all_free_pages()).any()
    assert sum(state.logical_free_pages()) < 8192 // 4 + 4096 // 4
    torch.cuda.synchronize()
    assert state.finish_retirement(plan) == growth
    assert plan.completed
    assert state.finish_retirement(plan) == 0
    assert state.quota.charged_bytes == charged
    assert state.logical_free_pages() == (8192 // 4, 4096 // 4)


@xtest.requirements(device_count=1)
def test_swa_only_cache_reclaim_preserves_locked_full_prefix_and_actual_paired_mapping(
    full_swa_pool: SWAKVPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sglang.srt.mem_cache.allocator.swa, "PagedTokenToKVPoolAllocator", ElasticPagedTokenToKVPoolAllocator
    )
    device = full_swa_pool.device
    allocator = SWATokenToKVPoolAllocator(8192, 4096, 4, torch.float16, device, full_swa_pool, True)
    request_pool = ReqToTokenPool(2, 64, device, False)
    state = SglangElasticContextState.create(
        full_swa_pool,
        allocator,
        request_pool,
        configurator=object.__new__(KVCacheConfigurator),
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=request_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            sliding_window_size=8,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    state.bind_cache(cache)
    locations = allocator.alloc_extend(
        torch.tensor([0], device=device),
        torch.tensor([0]),
        torch.tensor([16], device=device),
        torch.tensor([16]),
        torch.tensor([-1], device=device),
        16,
    )
    assert locations is not None
    key = RadixKey(array("q", range(16)))
    inserted = cache.insert(InsertParams(key=key, value=locations))
    receipt = cache.inc_lock_ref(inserted.last_device_node).to_dec_params()
    free = state.resident_free_pages()
    assert free == (allocator.full_available_size() // 4, allocator.swa_available_size() // 4)
    assert state.eviction_page_targets(EvictParams(num_tokens=1, swa_num_tokens=5)) == (free[0] + 1, free[1] + 2)
    assert state.eviction_page_targets(EvictParams()) == (0, 0)
    assert state.select_cache_eviction((False, False), release_backing=False) is None
    candidate = state.select_cache_eviction((False, True), release_backing=False)
    assert candidate is not None and candidate.component == ComponentType.SWA
    assert list(candidate.key) == list(key[:8])
    release_candidate = state.select_cache_eviction((False, False), release_backing=True)
    assert release_candidate is not None and release_candidate.component == candidate.component
    assert list(release_candidate.key) == list(candidate.key)
    available = (allocator.full_available_size(), allocator.swa_available_size())
    charged = state.quota.charged_bytes
    plan = state.prepare_reclaim(0, cache_eviction=candidate)
    assert plan is not None
    state.apply_reclaim(plan)
    assert plan.completed and plan.event is None
    assert (allocator.full_available_size(), allocator.swa_available_size()) == (available[0], available[1] + 8)
    assert torch.all(allocator.full_to_swa_index_mapping[locations[:8]] == 0)
    assert torch.all(allocator.full_to_swa_index_mapping[locations[8:]] > 0)
    assert state.quota.charged_bytes == charged
    cache.dec_lock_ref(inserted.last_device_node, receipt)
    # A leaf selected under SWA pressure has a whole-leaf effect, not SWA-only.
    assert SglangCacheEviction(key, ComponentType.SWA).resolve(cache) is None
    leaf = state.select_cache_eviction((False, True), release_backing=False)
    assert leaf is not None and leaf.component == ComponentType.FULL
    assert list(leaf.key) == list(key)
    torch.cuda.synchronize()
