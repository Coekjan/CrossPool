from __future__ import annotations

from array import array

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, IncLockRefResult, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams

from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.ecs.radix import (
    SglangCacheEviction,
    around_acquire_swa_component_lock,
    free_swa_out_of_window_slots,
)
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


@pytest.fixture
def swa_cache(monkeypatch: pytest.MonkeyPatch) -> UnifiedRadixCache:
    pool = SWAKVPool(
        size=64,
        size_swa=64,
        page_size=4,
        dtype=torch.float16,
        head_num=1,
        head_dim=1,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
        enable_alt_stream=False,
    )
    allocator = SWATokenToKVPoolAllocator(64, 64, 4, torch.float16, "cpu", pool, True)
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=ReqToTokenPool(2, 64, "cpu", False),
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            sliding_window_size=5,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    original = SWAComponent.acquire_component_lock

    def acquire(
        component: SWAComponent, node: UnifiedTreeNode, result: IncLockRefResult, lock_host: bool = False
    ) -> IncLockRefResult:
        return around_acquire_swa_component_lock(original, component, node, result, lock_host)

    monkeypatch.setattr(SWAComponent, "acquire_component_lock", acquire)
    monkeypatch.setattr(SWAComponent, "_free_out_of_window_slots", free_swa_out_of_window_slots)
    return cache


@pytest.mark.parametrize("retain_floor", (None, 4))
def test_swa_stash_preserves_a_matchable_window_and_live_locations_after_eviction(
    swa_cache: UnifiedRadixCache, monkeypatch: pytest.MonkeyPatch, retain_floor: int | None
) -> None:
    allocator = swa_cache.token_to_kv_pool_allocator
    assert isinstance(allocator, SWATokenToKVPoolAllocator)
    full = allocator.full_attn_allocator.alloc(20)
    swa = allocator.swa_attn_allocator.alloc(20)
    assert full is not None and swa is not None
    allocator.set_full_to_swa_mapping(full, swa)
    req = Req("window", "", array("q", range(18)), SamplingParams(max_new_tokens=1))
    req.init_next_round_input(swa_cache)
    req.lock_receipt = swa_cache.inc_lock_ref(req.last_node).to_dec_params()
    req.set_extend_range(0, 18)
    assert swa_cache.req_to_token_pool.alloc([req]) is not None
    swa_cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 18)), full[:18])
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 18
    monkeypatch.setattr(swa_cache, "swa_retain_floor", lambda request: retain_floor)
    monkeypatch.setenv("SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS", "true")

    swa_cache.cache_unfinished_req(req, chunked=True)
    assert req.kv.cache_protected_len == 16
    assert req.kv.swa_evicted_seqlen <= (8 if retain_floor is None else retain_floor)
    needed = full[13:18]
    before = allocator.translate_loc_from_full_to_swa(needed).clone()
    swa_cache.evict_for_alloc(EvictParams(num_tokens=64, swa_num_tokens=64))
    assert torch.all(before > 0)
    assert torch.equal(allocator.translate_loc_from_full_to_swa(needed), before)

    req.to_finish = FINISH_ABORT("completed ownership check")
    req.update_finish_state(0)
    release_kv_cache(req, swa_cache)
    swa_cache.evict_for_alloc(EvictParams(num_tokens=64, swa_num_tokens=64))
    assert allocator.full_available_size() == allocator.swa_available_size() == 64
    assert swa_cache.req_to_token_pool.available_size() == 2
    swa_cache.sanity_check()


@pytest.mark.parametrize("release_first", (0, 1))
def test_swa_window_split_preserves_existing_receipts_and_full_prefix_protection(
    swa_cache: UnifiedRadixCache, release_first: int
) -> None:
    allocator = swa_cache.token_to_kv_pool_allocator
    assert isinstance(allocator, SWATokenToKVPoolAllocator)
    full = allocator.full_attn_allocator.alloc(32)
    swa = allocator.swa_attn_allocator.alloc(32)
    assert full is not None and swa is not None
    allocator.set_full_to_swa_mapping(full, swa)
    key = RadixKey(array("q", range(32)))
    swa_cache.insert(InsertParams(key=key, value=full))
    first = swa_cache.match_prefix(MatchPrefixParams(key=key[:16]))
    first_lock = swa_cache.inc_lock_ref(first.last_device_node).to_dec_params()
    second = swa_cache.match_prefix(MatchPrefixParams(key=key[:20]))
    second_lock = swa_cache.inc_lock_ref(second.last_device_node).to_dec_params()
    assert swa_cache.swa_protected_size() == 12
    assert swa_cache.full_protected_size() == 20
    holders = ((first.last_device_node, first_lock), (second.last_device_node, second_lock))
    swa_cache.dec_lock_ref(*holders[release_first])
    assert swa_cache.swa_protected_size() == 8
    swa_cache.dec_lock_ref(*holders[1 - release_first])
    assert swa_cache.swa_protected_size() == swa_cache.full_protected_size() == 0
    swa_cache.evict_for_alloc(EvictParams(num_tokens=64, swa_num_tokens=64))
    assert allocator.full_available_size() == allocator.swa_available_size() == 64
    swa_cache.sanity_check()


def test_exact_logical_leaf_eviction_keeps_shared_prefix_and_peer_branch(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
    attention_cache: UnifiedRadixCache,
) -> None:
    page_size = allocator.page_size
    plan = allocator.plan_pages(8)
    assert plan is not None and allocator.backing.try_map(plan.missing_bundle_ids)
    allocator.publish_pages(plan.page_ids)
    key = RadixKey(array("q", range(4 * page_size)), extra_key="adapter", cache_salt="tenant")
    peer_key = RadixKey(
        array("q", [*range(2 * page_size), *range(100, 100 + 2 * page_size)]),
        extra_key="adapter",
        cache_salt="tenant",
    )
    original = allocator.alloc(4 * page_size)
    peer = allocator.alloc(4 * page_size)
    assert original is not None and peer is not None
    attention_cache.insert(InsertParams(key=key, value=original))
    attention_cache.insert(InsertParams(key=peer_key, value=peer))
    matched = attention_cache.match_prefix(MatchPrefixParams(key=key))
    node = attention_cache.tree_core.node_by_id(matched.last_device_node)
    candidate = SglangCacheEviction.from_node(node, ComponentType.FULL)
    assert list(candidate.key) == list(key)
    assert candidate.key.extra_key == key.extra_key and candidate.key.cache_salt == key.cache_salt
    assert candidate.resolve(attention_cache) is node
    # A partial-node match must not silently split the tree during preparation.
    assert SglangCacheEviction(key[:page_size], ComponentType.FULL).resolve(attention_cache) is None
    receipt = attention_cache.inc_lock_ref(node.id).to_dec_params()
    assert candidate.resolve(attention_cache) is None
    attention_cache.dec_lock_ref(node.id, receipt)

    available = allocator.available_size()
    candidate.apply(attention_cache, node)
    assert allocator.available_size() == available + 2 * page_size
    assert len(attention_cache.match_prefix(MatchPrefixParams(key=key)).device_indices) == 2 * page_size
    assert len(attention_cache.match_prefix(MatchPrefixParams(key=peer_key)).device_indices) == 4 * page_size
    assert candidate.resolve(attention_cache) is None
    assert SglangCacheEviction.select(attention_cache, ComponentType.FULL) is not None
    assert torch.unique(allocator.get_all_free_pages()).numel() == len(allocator.get_all_free_pages())
