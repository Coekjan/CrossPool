from __future__ import annotations

from array import array
from typing import cast

import pytest
import sglang.srt.mem_cache.allocator.swa
import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation import alloc_for_decode
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams

import xtest
from xpool.config import LatencySloConfig
from xpool.integrations.sglang.ecs.allocation import SglangAllocation, SglangCapacityMiss
from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.ecs.capacity import CapacityReconciler, capacity_reconciler_scope
from xpool.integrations.sglang.ecs.state import SglangElasticContextState
from xpool.integrations.sglang.hooks.ecs import complete_allocation
from xpool.native.ecs import InstanceControlChannel
from xpool.runtime.instance import InstanceRankRuntime
from xpool.utils.cuda import CudaDriverError
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


def test_clean_rollback_preserves_continuation_and_actual_sorted_acquisitions(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator, attention_cache: UnifiedRadixCache
) -> None:
    page_size = allocator.page_size
    plan = allocator.plan_pages(4)
    assert plan is not None
    assert allocator.backing.try_map(plan.missing_bundle_ids)
    allocator.publish_pages(plan.page_ids)
    allocated = allocator.alloc(3 * page_size)
    assert allocated is not None
    allocator.free(allocated[: 2 * page_size])
    held = allocated[2 * page_size :]
    free_before = set(allocator.get_all_free_pages().tolist())
    request = Req("chunk", "", array("q", [1] * (3 * page_size)), SamplingParams(max_new_tokens=2))
    request.prefix_indices = held
    request.kv.kv_allocated_len = page_size
    request.kv.kv_committed_len = page_size
    request_pool = attention_cache.req_to_token_pool
    rows = request_pool.alloc([request])
    assert rows is not None
    request_pool.req_to_token[rows[0], :page_size] = held.to(torch.int32)
    mapping = torch.zeros(32 + page_size, dtype=torch.int64)
    mapping[held] = 7
    charged_before = allocator.backing.quota.charged_bytes

    with SglangAllocation(request_pool, attention_cache, allocator, mapping) as attempt:
        attempt.watch([request])
        attempt.batch = ScheduleBatch(reqs=[request])
        locations = allocator.alloc(2 * page_size)
        assert locations is not None
        mapping[locations] = 99
        request_pool.req_to_token[rows[0], page_size : 3 * page_size] = locations.to(torch.int32)
        request.kv.kv_allocated_len = 3 * page_size
        request.kv.kv_committed_len = 3 * page_size
        request.kv.swa_evicted_seqlen = page_size
        attempt.rollback()
        attempt.rollback()

    assert SglangAllocation.current() is None
    assert set(allocator.get_all_free_pages().tolist()) == free_before
    assert len(allocator.get_all_free_pages()) == len(free_before)
    assert request.kv.kv_allocated_len == request.kv.kv_committed_len == page_size
    assert request.kv.swa_evicted_seqlen == page_size
    assert request.kv.req_pool_idx == rows[0]
    assert torch.equal(request_pool.req_to_token[rows[0], :page_size], held.to(torch.int32))
    assert torch.count_nonzero(request_pool.req_to_token[rows[0], page_size : 3 * page_size]) == 0
    assert torch.count_nonzero(mapping[locations]) == 0
    assert torch.all(mapping[held] == 7)
    assert allocator.backing.quota.charged_bytes == charged_before


def test_rollback_returns_new_rows_locks_and_selected_waiting_positions(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator, attention_cache: UnifiedRadixCache
) -> None:
    plan = allocator.plan_pages(2)
    assert plan is not None
    assert allocator.backing.try_map(plan.missing_bundle_ids)
    allocator.publish_pages(plan.page_ids)
    cached = allocator.alloc(allocator.page_size)
    assert cached is not None
    key = RadixKey(array("q", range(allocator.page_size)))
    inserted = attention_cache.insert(InsertParams(key=key, value=cached))
    request = Req("first", "", array("q", [1]), SamplingParams(max_new_tokens=2))
    peer = Req("peer", "", array("q", [2]), SamplingParams(max_new_tokens=2))
    scheduler = object.__new__(Scheduler)
    scheduler.waiting_queue = [request, peer]
    scheduler.chunked_req = None
    request_pool = attention_cache.req_to_token_pool
    rows_before = request_pool.available_size()

    with SglangAllocation(request_pool, attention_cache, allocator, None, scheduler) as attempt:
        attempt.watch([request])
        rows = request_pool.alloc([request])
        assert rows is not None
        attempt.record_rows(rows)
        request_pool.req_to_token[rows[0], :2] = 7
        receipt = attention_cache.inc_lock_ref(inserted.last_device_node).to_dec_params()
        attempt.record_lock(receipt)
        request.lock_receipt = receipt
        scheduler.waiting_queue = [peer]
        scheduler.chunked_req = request
        request.inflight_middle_chunks += 1
        assert attention_cache.evictable_size() == 0
        attempt.rollback()

    assert request_pool.available_size() == rows_before
    assert request.kv.req_pool_idx is None
    assert request.inflight_middle_chunks == 0
    assert scheduler.waiting_queue == [request, peer]
    assert scheduler.chunked_req is None
    assert attention_cache.evictable_size() == allocator.page_size
    assert torch.count_nonzero(request_pool.req_to_token[rows[0]]) == 0


def test_terminal_failure_retains_journal_ownership_without_device_rollback(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator, attention_cache: UnifiedRadixCache
) -> None:
    plan = allocator.plan_pages(1)
    assert plan is not None
    assert allocator.backing.try_map(plan.missing_bundle_ids)
    allocator.publish_pages(plan.page_ids)
    available = allocator.available_size()
    error = CudaDriverError(driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS, "test forward preparation")
    attempt = SglangAllocation(attention_cache.req_to_token_pool, attention_cache, allocator, None)

    with pytest.raises(CudaDriverError) as caught:
        with attempt:
            attempt.batch = ScheduleBatch(reqs=[])
            assert allocator.alloc(allocator.page_size) is not None
            raise error
    assert caught.value is error
    assert SglangAllocation.current() is None
    assert allocator.available_size() == available - allocator.page_size
    assert not attempt.committed and not attempt.rolled_back
    assert sum(len(pages) for owner, pages in attempt.page_allocations) == 1


def test_successful_commit_keeps_the_new_logical_page_owner(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator, attention_cache: UnifiedRadixCache
) -> None:
    plan = allocator.plan_pages(1)
    assert plan is not None
    assert allocator.backing.try_map(plan.missing_bundle_ids)
    allocator.publish_pages(plan.page_ids)
    available = allocator.available_size()
    with SglangAllocation(attention_cache.req_to_token_pool, attention_cache, allocator, None) as attempt:
        attempt.batch = ScheduleBatch(reqs=[])
        assert allocator.alloc(allocator.page_size) is not None
        attempt.commit()
    assert attempt.committed
    assert allocator.available_size() == available - allocator.page_size
    assert SglangAllocation.current() is None


def test_decode_miss_clears_every_actual_row_tail_and_keeps_committed_prefixes(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
    attention_cache: UnifiedRadixCache,
) -> None:
    state = SglangElasticContextState.create(
        allocator.get_kvcache(),
        allocator,
        attention_cache.req_to_token_pool,
        configurator=object.__new__(KVCacheConfigurator),
    )
    state.bind_cache(attention_cache)
    assert state.prepare_allocation((2,))
    prefix = allocator.alloc(2 * allocator.page_size)
    assert prefix is not None
    base = Req("leader", "", array("q", [1] * allocator.page_size), SamplingParams(max_new_tokens=2))
    member = Req("member", "", array("q", [2] * allocator.page_size), SamplingParams(max_new_tokens=2))
    base.prefix_indices = prefix[: allocator.page_size]
    base.kv.kv_allocated_len = base.kv.kv_committed_len = allocator.page_size
    rows = state.request_pool.alloc([base, member])
    assert rows is not None
    state.request_pool.req_to_token[rows, : allocator.page_size] = prefix.reshape(2, allocator.page_size).to(
        torch.int32
    )
    before = state.request_pool.req_to_token[rows, : allocator.page_size].clone()
    batch = ScheduleBatch(
        reqs=[base],
        req_pool_indices_cpu=torch.tensor(rows),
        seq_lens_cpu=torch.tensor([allocator.page_size, allocator.page_size]),
    )
    reconciler = CapacityReconciler(
        cast(InstanceControlChannel, object()),
        0,
        state,
        cast(InstanceRankRuntime, object()),
        LatencySloConfig(ttft_ms=1000, tbt_ms=50),
    )

    def write_then_reject() -> torch.Tensor:
        state.request_pool.req_to_token[rows, allocator.page_size] = 99
        base.kv.kv_allocated_len += 1
        base.kv.kv_committed_len += 1
        raise SglangCapacityMiss("reject after every Decode row was written")

    with capacity_reconciler_scope(reconciler), state.allocation() as attempt:
        with pytest.raises(SglangCapacityMiss):
            complete_allocation(write_then_reject, batch, token_per_req=1)
        assert attempt.rolled_back
    assert torch.equal(state.request_pool.req_to_token[rows, : allocator.page_size], before)
    assert torch.count_nonzero(state.request_pool.req_to_token[rows, allocator.page_size]) == 0
    assert base.kv.kv_allocated_len == base.kv.kv_committed_len == allocator.page_size


@xtest.requirements(device_count=1)
def test_upstream_paired_decode_rollback_restores_partial_page_mapping(
    full_swa_pool: SWAKVPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = full_swa_pool.device
    monkeypatch.setattr(
        sglang.srt.mem_cache.allocator.swa, "PagedTokenToKVPoolAllocator", ElasticPagedTokenToKVPoolAllocator
    )
    allocator = SWATokenToKVPoolAllocator(8192, 4096, 4, torch.float16, device, full_swa_pool, True)
    full = cast(ElasticPagedTokenToKVPoolAllocator, allocator.full_attn_allocator)
    request_pool = ReqToTokenPool(2, 64, device, False)
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=request_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            tree_components=(ComponentType.FULL,),
        )
    )
    initial = allocator.alloc_extend(
        torch.tensor([0], device=device),
        torch.tensor([0]),
        torch.tensor([1], device=device),
        torch.tensor([1]),
        torch.tensor([-1], device=device),
        1,
    )
    assert initial is not None
    request = Req("partial", "", array("q", [1]), SamplingParams(max_new_tokens=2))
    request.prefix_indices = initial
    request.kv.kv_allocated_len = request.kv.kv_committed_len = 1
    rows = request_pool.alloc([request])
    assert rows is not None
    request_pool.req_to_token[rows[0], 0] = initial[0].to(torch.int32)
    model_config = object.__new__(ModelConfig)
    model_config.is_encoder_decoder = False
    batch = ScheduleBatch(
        reqs=[request],
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        model_config=model_config,
        device=device,
        req_pool_indices=torch.tensor(rows, device=device),
        req_pool_indices_cpu=torch.tensor(rows),
        seq_lens=torch.tensor([1], device=device),
        seq_lens_cpu=torch.tensor([1]),
    )
    mapping = allocator.full_to_swa_index_mapping
    before = mapping.clone()
    available = (allocator.full_available_size(), allocator.swa_available_size())

    with SglangAllocation(request_pool, cache, full, mapping) as attempt:
        attempt.watch(batch.reqs)
        attempt.batch = batch
        attempt.token_per_req = 1
        locations = alloc_for_decode(batch, token_per_req=1)
        assert torch.all(mapping[locations] != before[locations])
        assert not attempt.page_allocations
        attempt.rollback()
    torch.cuda.synchronize()

    assert torch.equal(mapping, before)
    assert (allocator.full_available_size(), allocator.swa_available_size()) == available
    assert request.kv.kv_allocated_len == request.kv.kv_committed_len == 1
    assert request_pool.req_to_token[rows[0], 0] == initial[0]
    assert request_pool.req_to_token[rows[0], 1] == 0
