from __future__ import annotations

from array import array
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.distributed
import torch.multiprocessing
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, retract_all
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation import alloc_token_slots
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams

import xpool.integrations.sglang.kv.capacity
import xpool.native
from xpool.config import LatencySloConfig
from xpool.integrations.sglang.hooks.kv import (
    ElasticPrefillAdder,
    around_prefill_add_one_req,
    capacity_reconciler_scope,
)
from xpool.integrations.sglang.kv.allocator import ElasticPagedTokenToKVPoolAllocator, ElasticTokenToKVPoolAllocator
from xpool.integrations.sglang.kv.capacity import CapacityReconciler
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.runtime.instance import InstanceRankRuntime
from xtest.harness.support.sglang.fakes import ServerArgs
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)
TEST_SLO = LatencySloConfig(ttft_ms=1000, tbt_ms=50)


class FakeBacking:
    """Minimal compound backing used by reconciliation tests."""

    capacity_profile = SimpleNamespace(floor_bundles=1)

    def __init__(self, backed_bundles: int = 2) -> None:
        self.backed_bundles = backed_bundles
        self.resize_calls: list[int] = []

    def usable_tokens(self, bundle_count: int) -> int:
        return bundle_count * 4

    def required_bundles(self, token_count: int) -> int:
        return (token_count + 3) // 4

    def resize(self, bundle_count: int) -> None:
        self.resize_calls.append(bundle_count)
        self.backed_bundles = bundle_count


class FakeControlChannel:
    """Single-partition in-memory Control Channel."""

    def __init__(self, command: xpool.native.kv.KvCapacityCommand | None = None, ceiling: int = 4) -> None:
        self.commands = [command]
        self.ceiling = ceiling
        self.initial_backing: list[int] = []
        self.capture_complete = False
        self.completions: list[xpool.native.kv.KvCapacityCompletion] = []
        self.demands: list[xpool.native.kv.KvCapacityDemand] = []

    def publish_initial_backing(self, bundles: int) -> None:
        self.initial_backing.append(bundles)

    def publish_capture_complete(self) -> None:
        self.capture_complete = True

    def service_ceiling(self) -> int:
        return self.ceiling

    def read_commands(self) -> list[xpool.native.kv.KvCapacityCommand | None]:
        return self.commands

    def publish_completion(self, completion: xpool.native.kv.KvCapacityCompletion) -> None:
        self.completions.append(completion)

    def publish_demand(self, demand: xpool.native.kv.KvCapacityDemand) -> None:
        self.demands.append(demand)


def make_reconciler(
    channel: FakeControlChannel,
    backing: FakeBacking,
    allocator: object,
    request_pool: object,
    *,
    command: xpool.native.kv.KvCapacityCommand | None = None,
    active_bundles: int = 0,
    applied_sequence: int = 0,
    completed_sequence: int = 0,
) -> CapacityReconciler:
    return CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceControlChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, backing),
        allocator=cast(ElasticTokenToKVPoolAllocator, allocator),
        request_pool=cast(ReqToTokenPool, request_pool),
        instance_rank=cast(InstanceRankRuntime, object()),
        slo=TEST_SLO,
        command=command,
        active_bundles=active_bundles,
        applied_sequence=applied_sequence,
        completed_sequence=completed_sequence,
    )


def create_prefill_cache(
    page_size: int = 1,
) -> tuple[UnifiedRadixCache, ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator, ReqToTokenPool]:
    request_pool = ReqToTokenPool(4, 64, "cpu", False)
    # CPU allocation never dereferences the KV tensor storage.
    storage = cast(KVCache, object())
    allocator = (
        ElasticTokenToKVPoolAllocator(40, torch.float16, "cpu", storage, False)
        if page_size == 1
        else ElasticPagedTokenToKVPoolAllocator(40, page_size, torch.float16, "cpu", storage, False)
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=request_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            tree_components=(ComponentType.FULL,),
        )
    )
    return cache, allocator, request_pool


@pytest.mark.parametrize(("page_size", "prefix_length", "occupied"), [(1, 0, 32), (4, 0, 32), (4, 2, 40)])
def test_chunk_continuation_bounds_negative_forecast_by_allocatable_slots(
    page_size: int, prefix_length: int, occupied: int
) -> None:
    cache, allocator, request_pool = create_prefill_cache(page_size)
    slots = allocator.alloc(occupied)
    assert slots is not None
    request = Req("chunk", "", array("q", range(30)), SamplingParams(max_new_tokens=1))
    request.init_next_round_input(cache)
    request.prefix_indices = slots[:prefix_length]
    request.time_stats.scheduler_recv_time = 10.0
    reconciler = make_reconciler(
        FakeControlChannel(), FakeBacking(10), allocator, request_pool, active_bundles=10, applied_sequence=1
    )
    adder = ElasticPrefillAdder(page_size, cache, allocator, ScheduleBatch(reqs=[]), 1.0, 16, 16)
    adder.rem_total_token_offset = 100
    assert adder.rem_total_tokens < 0
    free_before = allocator.available_size()

    with get_parallel().override(attn_tp_rank=0), capacity_reconciler_scope(reconciler):
        retained = adder.add_chunked_req(request)

    assert retained is request
    assert adder.can_run_list == [request]
    assert request.extend_range is not None
    assert request.extend_range.length == (8 if prefix_length == 0 else 2)
    new_pages = (request.extend_range.end + page_size - 1) // page_size - (prefix_length + page_size - 1) // page_size
    assert new_pages * page_size <= free_before
    if page_size == 1:
        allocated = alloc_token_slots(cache, request.extend_range.length)
        assert len(allocated) == request.extend_range.length
        assert allocator.available_size() == 0


def test_legally_admitted_chunk_parks_then_retraction_allows_progress_and_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, allocator, request_pool = create_prefill_cache()
    channel = FakeControlChannel(ceiling=20)
    backing = FakeBacking(10)
    reconciler = make_reconciler(
        channel, backing, allocator, request_pool, active_bundles=10, applied_sequence=1, completed_sequence=1
    )
    chunk = Req("chunk", "", array("q", range(37)), SamplingParams(max_new_tokens=1))
    peer = Req("peer", "", array("q", range(100, 107)), SamplingParams(max_new_tokens=20))
    for request in (chunk, peer):
        request.init_next_round_input(cache)
        request.time_stats.scheduler_recv_time = 10.0
    scheduler = SimpleNamespace(
        chunked_req=None,
        waiting_queue=[],
        running_batch=ScheduleBatch(reqs=[]),
        tree_cache=cache,
        enable_overlap=False,
        schedule_stream="execution",
    )

    def prepare_selected(requests: list[Req]) -> None:
        assert request_pool.alloc(requests) is not None
        for request in requests:
            extent = request.extend_range
            assert extent is not None
            slots = alloc_token_slots(cache, extent.length)
            request_pool.write((request.kv.req_pool_idx, slice(extent.start, extent.end)), slots)
            request.kv.kv_allocated_len = request.kv.kv_committed_len = extent.end
            cache.cache_unfinished_req(request, chunked=extent.end < len(request.full_untruncated_fill_ids))

    class Event:
        ready = False

        def record(self, stream: object) -> None:
            assert stream == "execution"

        def query(self) -> bool:
            return self.ready

    event = Event()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1), capacity_reconciler_scope(reconciler):
        adder = ElasticPrefillAdder(1, cache, allocator, scheduler.running_batch, 1.0, 15, 15)
        # Alignment leaves room for a complete peer Prefill beside the first chunk.
        assert adder.add_one_req(chunk, False, 8) is AddReqResult.CONTINUE
        adder.add_one_req(peer, True, 8)
        assert adder.can_run_list == [chunk, peer]
        assert adder.new_chunked_req is chunk
        prepare_selected(adder.can_run_list)
        peer.output_ids.append(1000)
        scheduler.running_batch = ScheduleBatch(reqs=[peer])
        scheduler.chunked_req = chunk
        reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=4))

        while allocator.available_size() > 0:
            reconciler.begin_scheduling(cast(Scheduler, scheduler))
            assert allocator.token_capacity == 40
            chunk.init_next_round_input()
            adder = ElasticPrefillAdder(1, cache, allocator, scheduler.running_batch, 1.0, 15, 15)
            scheduler.chunked_req = adder.add_chunked_req(chunk)
            assert scheduler.chunked_req is chunk
            assert chunk.extend_range is not None
            assert 0 < chunk.extend_range.length <= allocator.available_size()
            prepare_selected(adder.can_run_list)

        chunk.init_next_round_input()
        original_extent = chunk.extend_range
        adder = ElasticPrefillAdder(1, cache, allocator, scheduler.running_batch, 1.0, 15, 15)
        assert adder.add_chunked_req(chunk) is chunk
        assert adder.can_run_list == []
        assert chunk.extend_range == original_extent
        assert adder.add_one_req(peer, True, 8) is AddReqResult.OTHER
        reconciler.finish_scheduling(cast(Scheduler, scheduler))
        demand = channel.demands[-1]
        assert (demand.evaluated_sequence, demand.requested_bundles, demand.deadline_monotonic_ns) == (
            1,
            17,
            11_000_000_000,
        )
        assert reconciler.applied_sequence == reconciler.completed_sequence == 1
        assert backing.backed_bundles == 10

        retract_all(
            reqs=[peer],
            req_to_token_pool=request_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        assert not peer.kv.holds_kv
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.waiting_queue = [peer]
        chunk.init_next_round_input()
        adder = ElasticPrefillAdder(1, cache, allocator, scheduler.running_batch, 1.0, 15, 15)
        scheduler.chunked_req = adder.add_chunked_req(chunk)
        assert scheduler.chunked_req is None
        prepare_selected(adder.can_run_list)
        assert len(chunk.prefix_indices) == len(chunk.origin_input_ids)
        reconciler.finish_scheduling(cast(Scheduler, scheduler))
        assert channel.demands[-1].requested_bundles is None

        reconciler.begin_scheduling(cast(Scheduler, scheduler))
        assert reconciler.active_bundles == 10
        assert allocator.token_capacity == 40
        release_kv_cache(chunk, cache, is_insert=False)
        reconciler.begin_scheduling(cast(Scheduler, scheduler))
        assert allocator.token_capacity == 16
        assert allocator.suffix_is_free(16)
        assert not reconciler.draining
        assert reconciler.applied_sequence == 2
        assert reconciler.completed_sequence == 1
        assert backing.backed_bundles == 10
        assert channel.completions == []
        event.ready = True
        reconciler.begin_scheduling(cast(Scheduler, scheduler))

    assert backing.backed_bundles == 4
    assert reconciler.completed_sequence == 2
    assert [(item.sequence, item.backed_bundles) for item in channel.completions] == [(2, 4)]


def run_tp2_readiness_vote(rank: int, rendezvous_uri: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous_uri,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        get_context().set_server_args(ServerArgs(model_path="dummy", tp_size=2, dp_size=1, enable_dp_attention=False))
        reconciler = make_reconciler(FakeControlChannel(), FakeBacking(), object(), object())
        with get_parallel().override(
            attn_tp_size=2,
            tp_group=SimpleNamespace(cpu_group=torch.distributed.group.WORLD),
        ):
            assert not reconciler.vote_ready(rank == 0)
            assert reconciler.vote_ready(True)
    finally:
        torch.distributed.destroy_process_group()


def test_tp2_readiness_vote_requires_every_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5]))
    torch.multiprocessing.spawn(
        run_tp2_readiness_vote,
        args=((tmp_path / "readiness-vote").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_post_capture_finalization_publishes_floor_once_then_activates_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=3)
    channel = FakeControlChannel(command)
    backing = FakeBacking()
    capacities: list[int] = []
    allocator = SimpleNamespace(set_token_capacity=capacities.append)
    resets: list[None] = []
    request_pool = SimpleNamespace(reset_aux_cache_allocator=lambda: resets.append(None))
    model_runner = SimpleNamespace(device="cuda:0", memory_pool_config=None)
    reconciler = make_reconciler(channel, backing, allocator, request_pool)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        result = reconciler.finalize_after_capture(cast(ModelRunner, model_runner))

    assert backing.resize_calls == [1, 3]
    assert channel.initial_backing == [1]
    assert channel.capture_complete
    assert [(item.sequence, item.backed_bundles) for item in channel.completions] == [(1, 3)]
    assert capacities == [4, 12]
    assert resets == [None, None]
    assert reconciler.applied_sequence == reconciler.completed_sequence == 1
    assert result.max_total_num_tokens == 16


def test_reclaim_drains_until_the_exact_suffix_becomes_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1)
    channel = FakeControlChannel(command)
    backing = FakeBacking(8)
    reconciler = make_reconciler(
        channel,
        backing,
        SimpleNamespace(),
        SimpleNamespace(),
        command=command,
        active_bundles=8,
        applied_sequence=1,
        completed_sequence=1,
    )
    probes: list[int] = []

    def select_nodes(tree_cache: object, allocator: object, token_capacity: int) -> tuple[int, ...] | None:
        probes.append(token_capacity)
        return None if len(probes) == 1 else (3, 2)

    monkeypatch.setattr(xpool.integrations.sglang.kv.capacity, "select_suffix_reclaim_nodes", select_nodes)
    scheduler = cast(Scheduler, SimpleNamespace(chunked_req=None, tree_cache=object()))

    monkeypatch.setattr(CapacityReconciler, "vote_ready", lambda self, ready: False)
    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.begin_scheduling(scheduler)
        assert reconciler.draining
        reconciler.begin_scheduling(scheduler)

    assert probes == [4, 4]
    assert channel.completions == []
    assert backing.resize_calls == []
    assert reconciler.active_bundles == 8
    assert reconciler.applied_sequence == 1
    assert reconciler.completed_sequence == 1


def test_growth_maps_before_exposing_capacity_and_completes_at_the_boundary() -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=3)
    channel = FakeControlChannel(command)
    backing = FakeBacking(2)
    capacities: list[int] = []

    def expose_capacity(value: int) -> None:
        assert backing.resize_calls == [3]
        capacities.append(value)

    running_batch = SimpleNamespace(batch_is_full=True)
    reconciler = make_reconciler(
        channel,
        backing,
        SimpleNamespace(set_token_capacity=expose_capacity),
        SimpleNamespace(reset_aux_cache_allocator=lambda: None),
        command=command,
        active_bundles=2,
        applied_sequence=1,
        completed_sequence=1,
    )
    scheduler = cast(Scheduler, SimpleNamespace(chunked_req=object(), running_batch=running_batch))

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.begin_scheduling(scheduler)

    assert backing.resize_calls == [3]
    assert capacities == [12]
    assert not running_batch.batch_is_full
    assert [(item.sequence, item.backed_bundles) for item in channel.completions] == [(2, 3)]
    assert reconciler.active_bundles == 3
    assert reconciler.applied_sequence == reconciler.completed_sequence == 2


def test_common_reclaim_admits_on_both_ranks_while_local_events_retire_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1)

    class Event:
        ready = False
        stream: object = None

        def record(self, stream: object) -> None:
            self.stream = stream

        def query(self) -> bool:
            return self.ready

    events = [Event(), Event()]
    event_factory = iter(events)
    monkeypatch.setattr(torch.cuda, "Event", lambda: next(event_factory))
    ranks: list[tuple[CapacityReconciler, Scheduler, FakeControlChannel, FakeBacking]] = []
    for rank in range(2):
        channel = FakeControlChannel(command)
        backing = FakeBacking(8)
        cache, allocator, request_pool = create_prefill_cache()
        allocator.set_token_capacity(32)
        reconciler = make_reconciler(
            channel,
            backing,
            allocator,
            request_pool,
            command=command,
            active_bundles=8,
            applied_sequence=1,
            completed_sequence=1,
        )
        scheduler = cast(
            Scheduler,
            SimpleNamespace(
                chunked_req=None,
                tree_cache=cache,
                running_batch=ScheduleBatch(reqs=[], batch_is_full=True),
                enable_overlap=rank == 1,
                schedule_stream="schedule",
                forward_stream="forward",
            ),
        )
        # The real TP vote has separate coverage; start at its common all-ready commit.
        reconciler.apply_command(scheduler, ())
        ranks.append((reconciler, scheduler, channel, backing))
        assert allocator.token_capacity == 4
        assert not reconciler.draining
        assert not scheduler.running_batch.batch_is_full
        assert reconciler.applied_sequence == 2
        assert reconciler.completed_sequence == 1
        assert backing.backed_bundles == 8
        assert channel.completions == []

    assert [event.stream for event in events] == ["schedule", "forward"]
    events[0].ready = True
    for rank, (reconciler, scheduler, channel, backing) in enumerate(ranks):
        reconciler.begin_scheduling(scheduler)
        assert backing.backed_bundles == (1 if rank == 0 else 8)
        assert reconciler.completed_sequence == (2 if rank == 0 else 1)
        request = Req("new-prefill", "", array("q", [1]), SamplingParams(max_new_tokens=1))
        request.init_next_round_input(scheduler.tree_cache)
        request.time_stats.scheduler_recv_time = 10.0
        adder = ElasticPrefillAdder(1, scheduler.tree_cache, reconciler.allocator, scheduler.running_batch, 1.0, 15, 15)
        with get_parallel().override(attn_tp_rank=rank), capacity_reconciler_scope(reconciler):
            result = around_prefill_add_one_req(PrefillAdder.add_one_req, adder, request, False, None)
        assert result is AddReqResult.CONTINUE
        assert adder.can_run_list == [request]
        assert request.extend_range is not None
        slots = alloc_token_slots(scheduler.tree_cache, request.extend_range.length)
        assert len(slots) == 1
        assert all(1 <= slot <= 4 for slot in slots.tolist())
        assert reconciler.allocator.available_size() == 3
        if rank == 1:
            assert channel.completions == []

    events[1].ready = True
    reconciler, scheduler, channel, backing = ranks[1]
    reconciler.begin_scheduling(scheduler)
    assert backing.backed_bundles == 1
    assert [(item.sequence, item.backed_bundles) for item in channel.completions] == [(2, 1)]


class TimedRequest(SimpleNamespace):
    """Hashable request stub carrying SGLang scheduler timestamps."""

    __hash__ = object.__hash__


def test_prefill_demand_uses_scheduler_receipt_deadline_until_the_request_leaves() -> None:
    channel = FakeControlChannel()
    backing = FakeBacking()
    request = cast(
        Req,
        TimedRequest(
            time_stats=SimpleNamespace(
                scheduler_recv_time=10.0,
                last_decode_finish_time=0.0,
                last_prefill_finished_time=0.0,
            )
        ),
    )
    reconciler = make_reconciler(
        channel,
        backing,
        object(),
        object(),
        active_bundles=2,
        applied_sequence=1,
        completed_sequence=1,
    )

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.record_prefill_requirement(request, 13)
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[request], chunked_req=None)))
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[], chunked_req=request)))
        reconciler.applied_sequence = 2
        reconciler.record_prefill_requirement(request, 9)
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[], chunked_req=request)))
        reconciler.record_prefill_requirement(request, None)
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[], chunked_req=request)))

    assert [
        (demand.evaluated_sequence, demand.requested_bundles, demand.deadline_monotonic_ns)
        for demand in channel.demands
    ] == [
        (1, 4, 11_000_000_000),
        (2, 3, 11_000_000_000),
        (2, None, None),
    ]


def test_decode_batch_keeps_its_target_with_earliest_request_deadline() -> None:
    channel = FakeControlChannel()
    requests = (
        cast(
            Req,
            TimedRequest(
                time_stats=SimpleNamespace(
                    scheduler_recv_time=1.0,
                    last_decode_finish_time=20.0,
                    last_prefill_finished_time=19.0,
                )
            ),
        ),
        cast(
            Req,
            TimedRequest(
                time_stats=SimpleNamespace(
                    scheduler_recv_time=2.0,
                    last_decode_finish_time=0.0,
                    last_prefill_finished_time=18.0,
                )
            ),
        ),
    )
    reconciler = make_reconciler(
        channel,
        FakeBacking(),
        object(),
        object(),
        active_bundles=2,
        applied_sequence=1,
        completed_sequence=1,
    )

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.record_decode_requirement(requests, 17)
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=list(requests), chunked_req=None)))

    demand = channel.demands[-1]
    assert (demand.evaluated_sequence, demand.requested_bundles, demand.deadline_monotonic_ns) == (
        1,
        5,
        18_050_000_000,
    )


def test_new_command_cannot_supersede_an_unfinished_operation() -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1)
    reconciler = make_reconciler(
        FakeControlChannel(command),
        FakeBacking(),
        object(),
        object(),
        command=command,
        applied_sequence=1,
        completed_sequence=1,
    )

    with pytest.raises(RuntimeError, match="superseded"):
        reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=3, target_bundles=4))
    with pytest.raises(RuntimeError, match="target changed"):
        reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=2))
