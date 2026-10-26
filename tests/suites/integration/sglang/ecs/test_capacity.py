from __future__ import annotations

import time
from array import array
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import sglang.srt.managers.schedule_batch
import sglang.srt.managers.scheduler
import sglang.srt.mem_cache.allocation
import sglang.srt.mem_cache.allocator.swa
import sglang.srt.mem_cache.common
import torch
import torch.distributed
import torch.multiprocessing
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, _PrefillAdmission
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from tests.suites.integration.sglang.ecs.support import full_swa_pool_scope, install_vmm_driver

import xtest
from xpool import ecs
from xpool.config import LatencySloConfig
from xpool.integrations.sglang.ecs.admission import ElasticPrefillAdder
from xpool.integrations.sglang.ecs.allocation import SglangCapacityMiss
from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.ecs.capacity import CapacityReconciler, capacity_reconciler_scope
from xpool.integrations.sglang.ecs.pool import ElasticMHATokenToKVPool
from xpool.integrations.sglang.ecs.state import (
    SglangElasticContextState,
    SglangMemoryRequirement,
    SglangPrefillProposal,
)
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking, KvVmmLayout
from xpool.integrations.sglang.hooks.ecs import (
    around_alloc_for_extend,
    around_alloc_req_slots,
    around_evict_from_tree_cache,
    complete_allocation,
)
from xpool.native.ecs import (
    InstanceControlChannel,
    QuotaCommand,
    QuotaCompletion,
    QuotaDemand,
)
from xpool.runtime.instance import InstanceRankRuntime
from xtest.harness.support.sglang.fakes import server_args
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


@dataclass(slots=True)
class RecordingChannel:
    """Only record publications; native IPC coherence is qualified by the channel suite."""

    completion: QuotaCompletion | None = None
    demand: QuotaDemand | None = None

    def publish_completion(self, value: QuotaCompletion) -> None:
        self.completion = value

    def publish_demand(self, value: QuotaDemand) -> None:
        self.demand = value


@pytest.mark.parametrize("page_size", (1, 4), ids=("token", "paged"))
@xtest.requirements(device_count=1)
def test_scheduler_delivers_formed_batch_before_reclaiming_for_later_candidate(
    page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    layout = KvVmmBacking.describe(
        device=device,
        storage_dtype=torch.uint8,
        token_page_size=page_size,
        descriptors=(KvBufferDesc("kv", (64 + page_size, 1024 * 1024), row_bytes=1024 * 1024, tokens_per_row=1),),
        buffers_per_layer=1,
    )
    quota = ecs.MemoryQuota(layout.profile.bytes_for(layout.profile.floor_bundles))
    backing = KvVmmBacking(layout, quota)
    try:
        pool = object.__new__(ElasticMHATokenToKVPool)
        pool.backing = backing
        allocator = (
            ElasticTokenToKVPoolAllocator(64, torch.float16, str(device), pool, True)
            if page_size == 1
            else ElasticPagedTokenToKVPoolAllocator(64, page_size, torch.float16, str(device), pool, True)
        )
        request_pool = ReqToTokenPool(4, 64, str(device), False)
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=request_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                tree_components=(ComponentType.FULL,),
            )
        )
        configurator = object.__new__(KVCacheConfigurator)
        configurator.forward_stream = torch.cuda.current_stream()
        state = SglangElasticContextState.create(pool, allocator, request_pool, configurator=configurator)
        state.bind_cache(cache)
        capacity = state.allocation_bytes((16 // page_size,))
        assert capacity is not None
        quota.set_limit(capacity)
        requests = [
            Req(name, "", array("q", [index + 1] * length), SamplingParams(max_new_tokens=1))
            for index, (name, length) in enumerate((("small", 8), ("large", 32)))
        ]
        for request in requests:
            request.time_stats.scheduler_recv_time = time.monotonic()
        model_config = object.__new__(ModelConfig)
        model_config.vocab_size, model_config.is_encoder_decoder = 64, False
        scheduler = object.__new__(Scheduler)
        scheduler.grammar_manager = Mock(has_waiting_grammars=Mock(return_value=False))
        scheduler.enable_priority_preemption = scheduler.is_hybrid_swa = False
        scheduler.waiting_queue, scheduler.chunked_req = list(requests), None
        scheduler.min_free_slots_delayer = None
        scheduler.policy, scheduler.processed_tokens_counter = Mock(), None
        scheduler.chunked_prefill_size, scheduler.dynamic_chunk_sizer = 16, None
        scheduler.tp_worker = Mock(model_runner=Mock(attn_backend=object(), prefill_aware_swa=False))
        scheduler.page_size = page_size
        scheduler.tree_cache, scheduler.token_to_kv_pool_allocator = cache, allocator
        scheduler.new_token_ratio_tracker = Mock(current=1.0)
        scheduler.max_prefill_tokens, scheduler.max_prefill_bs, scheduler.max_running_requests = 64, 4, 4
        scheduler.is_mixed_chunk = scheduler.enable_priority_scheduling = False
        scheduler.priority_scheduling_preemption_threshold, scheduler.dllm_config = 0, None
        scheduler.model_config, scheduler.spec_algorithm = model_config, SpeculativeAlgorithm.NONE
        scheduler.enable_overlap = scheduler.enable_lora = False
        scheduler.req_to_token_pool = request_pool
        monkeypatch.setattr(scheduler, "get_num_allocatable_reqs", Mock(return_value=4))
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hicache_storage = False
        scheduler.enable_hierarchical_cache = scheduler.enable_unified_cache_external_linker = False
        scheduler.truncation_align_size = None
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.load_inquirer = Mock()
        channel = RecordingChannel()
        reconciler = CapacityReconciler(
            cast(InstanceControlChannel, channel),
            0,
            state,
            cast(InstanceRankRuntime, object()),
            LatencySloConfig(ttft_ms=1000, tbt_ms=50),
            active_quota_bytes=(quota.limit_bytes,),
            completed_sequence=1,
            service_ceiling_bytes=(state.profile.reservation_bytes,),
            scheduler=scheduler,
        )
        original_allocate = sglang.srt.mem_cache.allocation.alloc_for_extend
        original_rows = sglang.srt.mem_cache.allocation.alloc_req_slots
        monkeypatch.setattr(sglang.srt.managers.scheduler, "PrefillAdder", ElasticPrefillAdder)
        monkeypatch.setattr(
            sglang.srt.managers.schedule_batch,
            "alloc_for_extend",
            lambda batch: around_alloc_for_extend(original_allocate, batch),
        )
        monkeypatch.setattr(
            sglang.srt.mem_cache.allocation,
            "alloc_req_slots",
            lambda owner, reqs, tree: around_alloc_req_slots(original_rows, owner, reqs, tree),
        )
        monkeypatch.setattr(
            sglang.srt.mem_cache.common,
            "evict_from_tree_cache",
            lambda tree, count: around_evict_from_tree_cache(lambda *args: None, tree, count),
        )
        # Allocation imports this helper by value, as the production hook scanner handles.
        monkeypatch.setattr(
            sglang.srt.mem_cache.allocation,
            "evict_from_tree_cache",
            sglang.srt.mem_cache.common.evict_from_tree_cache,
        )
        with get_parallel().override(attn_tp_size=1, attn_tp_rank=0), capacity_reconciler_scope(reconciler):
            reconciler.begin_scheduling(scheduler)
            with state.allocation(scheduler) as attempt:
                batch, _ = Scheduler._get_new_batch_prefill_raw(scheduler, None, scheduler.running_batch)
                assert batch is not None and batch.reqs == [requests[0]]
                assert attempt.committed and batch.out_cache_loc.numel() == 8
                row = requests[0].kv.req_pool_idx
                assert row is not None
                assert torch.equal(request_pool.req_to_token[row, :8].to(torch.int64), batch.out_cache_loc)
            reconciler.finish_scheduling(scheduler)
            assert scheduler.waiting_queue == [requests[1]]
            assert channel.demand is not None and channel.demand.required_bytes is None
            assert reconciler.pending_retirement is None
            reconciler.accept_command(QuotaCommand(2, (state.profile.reservation_bytes,)))
            reconciler.begin_scheduling(scheduler)
            with state.allocation(scheduler) as attempt:
                later, _ = Scheduler._get_new_batch_prefill_raw(scheduler, None, scheduler.running_batch)
                assert later is not None and later.reqs == [requests[1]]
                assert attempt.committed and later.out_cache_loc.numel() == 16
            assert not scheduler.waiting_queue
    finally:
        torch.cuda.synchronize()
        backing.close()


def test_prefill_waits_for_whole_input_budget_but_prepares_no_forecast_output(
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
    state.quota.set_limit(4 * allocator.backing.layout.profile.bundle_bytes)
    channel = RecordingChannel()
    reconciler = CapacityReconciler(
        cast(InstanceControlChannel, channel),
        0,
        state,
        cast(InstanceRankRuntime, object()),
        LatencySloConfig(ttft_ms=1000, tbt_ms=50),
        admission_free_pages=state.logical_free_pages(),
        service_ceiling_bytes=(state.profile.reservation_bytes,),
    )
    request = Req("prefill", "", array("q", range(20)), SamplingParams(max_new_tokens=4))
    request.init_next_round_input(attention_cache)
    request.time_stats.scheduler_recv_time = time.monotonic()
    adder = ElasticPrefillAdder(
        allocator.page_size,
        attention_cache,
        allocator,
        ScheduleBatch(reqs=[]),
        1.0,
        32,
        16,
        max_running_requests=4,
    )
    available = allocator.available_size()
    with get_parallel().override(attn_tp_size=1, attn_tp_rank=0), capacity_reconciler_scope(reconciler):
        with state.allocation() as attempt:
            assert adder.add_one_req(request, False, None) is AddReqResult.NO_TOKEN
            assert not adder.can_run_list and request.extend_range is None
            assert allocator.available_size() == available
            attempt.rollback()
        state.quota.set_limit(state.profile.reservation_bytes)
        with state.allocation() as attempt:
            proposal = adder.proposal_for_admission(request, _PrefillAdmission(0, 16, 0, True))
            adder.add_one_req(request, False, None)
            assert adder.can_run_list == [request]
            assert request.extend_range is not None and request.extend_range.length == 16
            assert proposal.memory.forecast_pages[0] > proposal.preparation_pages[0]
            forecast = state.required_bytes(proposal.memory)
            assert forecast is not None and state.quota.charged_bytes < forecast <= state.quota.limit_bytes
            assert allocator.available_size() >= 20
            backed = allocator.available_size()
            waiting = Req("waiting", "", array("q", range(4)), SamplingParams(max_new_tokens=1))
            waiting.init_next_round_input(attention_cache)
            assert adder.add_one_req(waiting, True, None) is AddReqResult.OTHER
            assert adder.can_run_list == [request] and waiting.extend_range is None
            attempt.rollback()
    assert request.extend_range is None
    assert request.kv.req_pool_idx is None
    assert allocator.available_size() == backed


def test_chunk_owner_keeps_residency_and_smaller_quota_pending_through_final_allocation(
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
    state.quota.set_limit(state.profile.reservation_bytes)
    assert state.prepare_allocation((8 // allocator.page_size,))
    locations = allocator.alloc(allocator.page_size)
    assert locations is not None
    attention_cache.insert(InsertParams(key=RadixKey(array("q", range(allocator.page_size))), value=locations))
    owner = Req("owner", "", array("q", range(20)), SamplingParams(max_new_tokens=4))
    scheduler = object.__new__(Scheduler)
    scheduler.chunked_req = owner
    scheduler.waiting_queue = []
    scheduler.running_batch = ScheduleBatch(reqs=[])
    channel = RecordingChannel()
    backed = state.quota.charged_bytes
    reconciler = CapacityReconciler(
        cast(InstanceControlChannel, channel),
        0,
        state,
        cast(InstanceRankRuntime, object()),
        LatencySloConfig(ttft_ms=1000, tbt_ms=50),
        active_quota_bytes=(state.quota.limit_bytes,),
        completed_sequence=1,
        scheduler=scheduler,
    )
    with get_parallel().override(attn_tp_size=1, attn_tp_rank=0), capacity_reconciler_scope(reconciler):
        reconciler.accept_command(QuotaCommand(2, (backed,)))
        reconciler.begin_scheduling(scheduler)
        assert channel.completion is None and reconciler.completed_sequence == 1
        assert reconciler.reclaim(16, (allocator.available_size() // allocator.page_size + 1,))
        assert attention_cache.full_evictable_size() == 0
        assert state.quota.charged_bytes == backed and reconciler.pending_retirement is None
        with state.allocation(scheduler) as attempt:
            scheduler.chunked_req = None
            assert reconciler.has_chunk_owner
            reconciler.begin_scheduling(scheduler)
            assert channel.completion is None
            waiting = Req("waiting", "", array("q", range(4)), SamplingParams(max_new_tokens=1))
            waiting.init_next_round_input(attention_cache)
            adder = ElasticPrefillAdder(
                allocator.page_size, attention_cache, allocator, scheduler.running_batch, 1.0, 32, 16
            )
            assert adder.add_one_req(waiting, False, None) is AddReqResult.OTHER
            assert not adder.can_run_list
            attempt.commit()
            assert not reconciler.has_chunk_owner
        reconciler.begin_scheduling(scheduler)
    assert channel.completion is not None and channel.completion.sequence == 2
    assert state.quota.limit_bytes == backed


@xtest.parameterize("needs_external_quota", (False, True))
def test_prefill_local_reclamation_keeps_admission_open_until_capacity_can_progress(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
    attention_cache: UnifiedRadixCache,
    monkeypatch: pytest.MonkeyPatch,
    needs_external_quota: bool,
) -> None:
    configurator = object.__new__(KVCacheConfigurator)
    configurator.forward_stream = Mock(spec=torch.cuda.Stream)
    event = Mock(spec=torch.cuda.Event)
    event.query.return_value = True
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    state = SglangElasticContextState.create(
        allocator.get_kvcache(), allocator, attention_cache.req_to_token_pool, configurator=configurator
    )
    state.bind_cache(attention_cache)
    bundle_bytes = allocator.backing.layout.profile.bundle_bytes
    ceiling_bundles = (7 if allocator.page_size == 1 else 8) if needs_external_quota else 8
    quota_bundles = ceiling_bundles - int(needs_external_quota)
    state.quota.set_limit(quota_bundles * bundle_bytes)
    prepared_tokens = quota_bundles * 4 - allocator.page_size if needs_external_quota else 16
    assert state.prepare_allocation((prepared_tokens // allocator.page_size,))
    locations = allocator.alloc(8 if needs_external_quota else 16)
    assert locations is not None
    if needs_external_quota:
        attention_cache.insert(InsertParams(key=RadixKey(array("q", [1] * 8)), value=locations))
    else:
        attention_cache.insert(InsertParams(key=RadixKey(array("q", [2] * 8)), value=locations[:8]))
        attention_cache.insert(InsertParams(key=RadixKey(array("q", [3] * 8)), value=locations[8:]))
    channel = RecordingChannel()
    reconciler = CapacityReconciler(
        cast(InstanceControlChannel, channel),
        0,
        state,
        cast(InstanceRankRuntime, object()),
        LatencySloConfig(ttft_ms=1000, tbt_ms=50),
        service_ceiling_bytes=(ceiling_bundles * bundle_bytes,),
        active_quota_bytes=(state.quota.limit_bytes,),
        completed_sequence=1,
    )
    input_tokens = (24 if allocator.page_size == 1 else 20) if needs_external_quota else 16
    request = Req("pressure", "", array("q", [1] * input_tokens), SamplingParams(max_new_tokens=1))
    request.init_next_round_input(attention_cache)
    request.time_stats.scheduler_recv_time = time.monotonic()
    scheduler = object.__new__(Scheduler)
    scheduler.running_batch = ScheduleBatch(reqs=[])
    scheduler.waiting_queue, scheduler.chunked_req = [request], None
    admitted = False
    with get_parallel().override(attn_tp_size=1, attn_tp_rank=0), capacity_reconciler_scope(reconciler):
        for _ in range(allocator.backing.layout.profile.bundle_capacity + 2):
            reconciler.begin_scheduling(scheduler)
            reconciler.sync_admission_budget()
            adder = ElasticPrefillAdder(
                allocator.page_size, attention_cache, allocator, ScheduleBatch(reqs=[]), 1.0, 32, 4
            )
            with state.allocation() as attempt:
                result = adder.add_one_req(request, False, None)
                admitted = adder.can_run_list == [request]
                attempt.rollback()
            if admitted:
                break
            reconciler.finish_scheduling(scheduler)
            if result is AddReqResult.NO_TOKEN:
                assert needs_external_quota and reconciler.pending_retirement is None
                demand = channel.demand
                assert demand is not None and demand.required_bytes == (ceiling_bundles * bundle_bytes,)
                reconciler.accept_command(QuotaCommand(2, (ceiling_bundles * bundle_bytes,)))
            else:
                # Only a stable external shortage may latch the engine gate.
                assert result is AddReqResult.OTHER
                assert channel.demand is not None and channel.demand.required_bytes is None
    assert admitted
    assert state.quota.limit_bytes == ceiling_bundles * bundle_bytes


def run_tp_capacity_rank(rank: int, rendezvous: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    group = object.__new__(GroupCoordinator)
    group.cpu_group = torch.distributed.group.WORLD
    group.rank = group.rank_in_group = rank
    group.ranks, group.world_size, group.mq_broadcaster = [0, 1], 2, None
    args = server_args()
    channel = RecordingChannel()
    with (
        pytest.MonkeyPatch.context() as patch,
        get_context().override_server_args(cuda_graph_config=args.cuda_graph_config),
        get_parallel().override(attn_tp_size=2, attn_tp_rank=rank, tp_group=group),
    ):
        install_vmm_driver(patch)
        layout = KvVmmLayout(torch.device("cuda:0"), torch.uint8, (33, 4), 1, 1, 1, 1, 16)
        quota = ecs.MemoryQuota(layout.profile.bytes_for(layout.profile.floor_bundles))
        backing = KvVmmBacking(layout, quota)
        pool = object.__new__(ElasticMHATokenToKVPool)
        pool.backing = backing
        allocator = ElasticTokenToKVPoolAllocator(32, torch.float16, "cpu", pool, True)
        request_pool = ReqToTokenPool(4, 64, "cpu", False)
        configurator = object.__new__(KVCacheConfigurator)
        configurator.forward_stream = Mock(spec=torch.cuda.Stream)
        state = SglangElasticContextState.create(pool, allocator, request_pool, configurator=configurator)
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=request_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tree_components=(ComponentType.FULL,),
            )
        )
        state.bind_cache(cache)
        reconciler = CapacityReconciler(
            cast(InstanceControlChannel, channel),
            0,
            state,
            cast(InstanceRankRuntime, object()),
            LatencySloConfig(ttft_ms=1000, tbt_ms=50),
            service_ceiling_bytes=(144, 144),
        )
        try:
            reconciler.accept_command(QuotaCommand(1, (144, 48)))
            reconciler.begin_scheduling(None)
            assert channel.completion is not None and channel.completion.sequence == 1
            assert state.prepare_allocation((5 + rank,))
            held = allocator.alloc(2) if rank == 1 else None
            local_free = allocator.available_size()
            reconciler.sync_admission_budget()
            assert reconciler.admission_free_pages == (30,)
            assert allocator.available_size() == local_free
            if held is not None:
                allocator.free(held)

            def propose(limit: int | None) -> SglangPrefillProposal:
                assert limit is not None
                return SglangPrefillProposal(
                    _PrefillAdmission(0, limit, 0, True), SglangMemoryRequirement((limit + 2,), (limit,)), (limit,)
                )

            adder = ElasticPrefillAdder(1, cache, allocator, ScheduleBatch(reqs=[]), 1.0, 32, 16)
            with capacity_reconciler_scope(reconciler):
                result = adder.coordinate_prefill(propose, chunk_tokens_limit=16)
                assert isinstance(result, SglangPrefillProposal) and result.admission.extend_len == 9
                assert (
                    adder.coordinate_prefill(
                        lambda limit: AddReqResult.NO_TOKEN if rank == 1 else propose(limit), chunk_tokens_limit=16
                    )
                    is AddReqResult.NO_TOKEN
                )
                assert (
                    adder.coordinate_prefill(
                        lambda limit: AddReqResult.OTHER if rank == 0 else propose(limit), chunk_tokens_limit=16
                    )
                    is AddReqResult.OTHER
                )
                selected = Req("selected", "", array("q", [1]), SamplingParams(max_new_tokens=1))
                adder.can_run_list.append(selected)
                assert (
                    adder.coordinate_prefill(
                        lambda limit: AddReqResult.NO_TOKEN if rank == 1 else propose(limit), chunk_tokens_limit=16
                    )
                    is AddReqResult.OTHER
                )
                available = allocator.available_size()
                with pytest.MonkeyPatch.context() as preparation:
                    if rank == 1:
                        preparation.setattr(SglangElasticContextState, "prepare_allocation", lambda owner, pages: False)
                    assert (
                        adder.coordinate_prefill(
                            lambda limit: SglangPrefillProposal(
                                _PrefillAdmission(0, 1, 0, False), SglangMemoryRequirement((2,), (1,)), (1,)
                            ),
                            chunk_tokens_limit=1,
                        )
                        is AddReqResult.OTHER
                    )
                assert allocator.available_size() == available and reconciler.pending_retirement is None
                adder.can_run_list.clear()

            locations = allocator.alloc(2)
            assert locations is not None
            cache.insert(InsertParams(key=RadixKey(array("q", [7, 8])), value=locations))
            scheduler = object.__new__(Scheduler)
            scheduler.running_batch = ScheduleBatch(reqs=[])
            scheduler.waiting_queue, scheduler.chunked_req = [], None
            if rank == 1:
                reconciler.defer_eviction(EvictParams(num_tokens=2))
            reconciler.begin_scheduling(scheduler)
            assert cache.full_evictable_size() == 0
            assert state.logical_free_pages() == (32,)
            assert reconciler.admission_free_pages == (32,)

            request = Req("blocked", "", array("q", [1]), SamplingParams(max_new_tokens=1))
            request.time_stats.scheduler_recv_time = time.monotonic()
            scheduler.waiting_queue = [request]
            reconciler.record_prefill_requirement(request, SglangMemoryRequirement((14,), (1,)))
            reconciler.finish_scheduling(scheduler)
            if rank == 0:
                demand = channel.demand
                assert demand is not None and demand.evaluated_sequence == 1
                assert demand.required_bytes == (144, 64)
                assert demand.deadline_monotonic_ns is not None

            event = Mock(spec=torch.cuda.Event)
            event.query.return_value = True
            patch.setattr(torch.cuda, "Event", lambda: event)
            with capacity_reconciler_scope(reconciler):
                assert (
                    adder.coordinate_prefill(
                        lambda limit: SglangPrefillProposal(
                            _PrefillAdmission(0, 1, 0, True), SglangMemoryRequirement((14,), (1,)), (1,)
                        ),
                        chunk_tokens_limit=1,
                    )
                    is AddReqResult.OTHER
                )
                assert (reconciler.pending_retirement is not None) == (rank == 1)
                assert (
                    adder.coordinate_prefill(lambda limit: AddReqResult.NO_TOKEN, chunk_tokens_limit=1)
                    is AddReqResult.OTHER
                )
            reconciler.finish_retirement()

            batch = ScheduleBatch(reqs=[])
            available = allocator.available_size()

            def clean_allocation() -> torch.Tensor:
                if rank == 1:
                    raise SglangCapacityMiss("local clean shortage")
                locations = allocator.alloc(1)
                assert locations is not None
                return locations

            with capacity_reconciler_scope(reconciler), state.allocation() as attempt:
                with pytest.raises(SglangCapacityMiss):
                    complete_allocation(clean_allocation, batch, token_per_req=None)
                assert attempt.rolled_back and not attempt.committed
            assert allocator.available_size() == available

            original_error = RuntimeError("local allocation failure")

            def failed_allocation() -> torch.Tensor:
                if rank == 1:
                    raise original_error
                locations = allocator.alloc(1)
                assert locations is not None
                return locations

            with capacity_reconciler_scope(reconciler), pytest.raises(RuntimeError) as caught:
                with state.allocation() as attempt:
                    complete_allocation(failed_allocation, batch, token_per_req=None)
            if rank == 1:
                assert caught.value is original_error
            assert not attempt.rolled_back and not attempt.committed
            assert len(attempt.page_allocations) == (1 if rank == 0 else 0)
            assert allocator.available_size() == available - (1 if rank == 0 else 0)
        finally:
            backing.close()
            torch.distributed.destroy_process_group()


def test_tp_capacity_matches_rejections_chunk_bounds_and_cache_effects_without_replacing_local_free_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Spawn imports this collected module outside the pytest parent's sys.path.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5]))
    torch.multiprocessing.spawn(run_tp_capacity_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)


def run_tp_composite_rank(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    group = object.__new__(GroupCoordinator)
    group.cpu_group = torch.distributed.group.WORLD
    group.rank = group.rank_in_group = rank
    group.ranks, group.world_size, group.mq_broadcaster = [0, 1], 2, None
    args = server_args()
    try:
        with (
            get_context().override_server_args(cuda_graph_config=args.cuda_graph_config, page_size=4),
            get_parallel().override(attn_tp_size=2, attn_tp_rank=rank, tp_group=group),
            pytest.MonkeyPatch.context() as patch,
            full_swa_pool_scope(device) as pool,
        ):
            patch.setattr(
                sglang.srt.mem_cache.allocator.swa, "PagedTokenToKVPoolAllocator", ElasticPagedTokenToKVPoolAllocator
            )
            allocator = SWATokenToKVPoolAllocator(8192, 4096, 4, torch.float16, str(device), pool, True)
            configurator = object.__new__(KVCacheConfigurator)
            configurator.forward_stream = torch.cuda.current_stream()
            state = SglangElasticContextState.create(
                pool, allocator, ReqToTokenPool(2, 64, str(device), False), configurator=configurator
            )
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=state.request_pool,
                    token_to_kv_pool_allocator=allocator,
                    page_size=4,
                    sliding_window_size=8,
                    tree_components=(ComponentType.FULL, ComponentType.SWA),
                )
            )
            state.bind_cache(cache)
            channel = RecordingChannel()
            reconciler = CapacityReconciler(
                cast(InstanceControlChannel, channel),
                0,
                state,
                cast(InstanceRankRuntime, object()),
                LatencySloConfig(ttft_ms=1000, tbt_ms=50),
            )
            floor = state.profile.floor_bytes
            growth = sum(profile.bundle_bytes for profile in state.profile.pools)
            reconciler.accept_command(QuotaCommand(1, (floor + 2 * growth, floor + growth)))
            reconciler.begin_scheduling(None)
            batch = ScheduleBatch(reqs=[])
            free_before = state.logical_free_pages()
            mapping_before = allocator.full_to_swa_index_mapping.clone()

            def allocate() -> torch.Tensor:
                if not state.prepare_allocation((600, 600)):
                    raise SglangCapacityMiss("joint backing could not be prepared")
                locations = allocator.alloc_extend(
                    torch.tensor([0], device=device),
                    torch.tensor([0]),
                    torch.tensor([2400], device=device),
                    torch.tensor([2400]),
                    torch.tensor([-1], device=device),
                    2400,
                )
                assert locations is not None
                return locations

            with (
                capacity_reconciler_scope(reconciler),
                pytest.MonkeyPatch.context() as injection,
                state.allocation() as attempt,
            ):
                if rank == 1:

                    def reject(*args: object, **kwargs: object) -> torch.Tensor:
                        raise SglangCapacityMiss("SWA child clean shortage after Full acquisition")

                    injection.setattr(allocator.swa_attn_allocator, "alloc_extend", reject)
                with pytest.raises(SglangCapacityMiss):
                    complete_allocation(allocate, batch, token_per_req=None)
                assert attempt.rolled_back and not attempt.committed
            assert state.logical_free_pages() == free_before
            assert torch.equal(allocator.full_to_swa_index_mapping, mapping_before)
            assert state.quota.charged_bytes == floor + growth

            with capacity_reconciler_scope(reconciler), state.allocation() as attempt:
                locations = complete_allocation(allocate, batch, token_per_req=None)
                assert attempt.committed
            assert torch.all(allocator.full_to_swa_index_mapping[locations] > 0)
            cache.insert(InsertParams(key=RadixKey(array("q", range(2400))), value=locations))
            if rank == 0:
                # Only this rank has an empty physical bundle before logical eviction.
                assert state.components[0].allocator.backing.try_map((2,))
            scheduler = object.__new__(Scheduler)
            scheduler.running_batch = ScheduleBatch(reqs=[])
            scheduler.waiting_queue, scheduler.chunked_req = [], None
            reconciler.accept_command(QuotaCommand(2, (floor, floor)))
            while reconciler.completed_sequence < 2:
                retirement = reconciler.pending_retirement
                if retirement is not None:
                    torch.cuda.synchronize()
                    assert retirement.event is not None and retirement.event.query()
                reconciler.begin_scheduling(scheduler)
                assert state.quota.charged_bytes == sum(
                    binding.allocator.backing.backed_bytes for binding in state.components
                )
                if reconciler.pending_retirement is not None:
                    assert state.quota.charged_bytes > floor
                    assert channel.completion is not None and channel.completion.sequence == 1
            assert reconciler.pending_retirement is None
            assert state.quota.charged_bytes == state.quota.limit_bytes == floor
            assert channel.completion is not None and channel.completion.sequence == 2
            assert channel.completion.backed_bytes == floor
            assert state.logical_free_pages() == free_before
    finally:
        torch.distributed.destroy_process_group()


@xtest.requirements(device_count=2)
def test_tp_composite_rolls_back_both_children_and_retires_backing_before_smaller_quota_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5]))
    torch.multiprocessing.spawn(run_tp_composite_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
