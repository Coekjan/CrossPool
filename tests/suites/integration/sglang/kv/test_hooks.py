from __future__ import annotations

import subprocess
import sys
from array import array
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
import sglang.srt.mem_cache.allocation
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import SchedulerInvariantChecker
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats, SchedulerPoolStatsObserver
from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_parallel

import xpool.integrations.sglang.hooks.kv
import xpool.native
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.sglang.runtime import published_sglang_config
from xpool.config import LatencySloConfig, XpoolConfig
from xpool.integrations.sglang.hooks.kv import (
    ElasticPrefillAdder,
    after_check_decode_mem,
    after_pool_stats,
    around_check_full_pool,
    around_prefill_add_one_req,
    around_request_broadcast,
    around_retract_decode,
    capacity_reconciler_scope,
    compute_kv_reservation_budget,
    validate_kv_seams,
)
from xpool.integrations.sglang.kv.allocator import ElasticTokenToKVPoolAllocator
from xpool.integrations.sglang.kv.capacity import CapacityReconciler
from xpool.integrations.sglang.kv.radix import evict_suffix_reclaim_nodes, select_suffix_reclaim_nodes
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.runtime.instance import InstanceRankRuntime

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, published_sglang_config.__name__)


def test_reservation_budget_uses_static_device_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(
        config=XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0], "device_memory_utilization": 0.75},
                "ffn": {"devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )
    )
    get_context().override("test", mem_fraction_static=0.6)
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.kv.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=1_000),
    )

    configurator = cast(KVCacheConfigurator, SimpleNamespace(device="cuda:0"))
    assert compute_kv_reservation_budget(configurator, 200) == 600


def test_elastic_pool_construction_hooks_reach_configurator_aliases() -> None:
    script = """
import sglang.srt.mem_cache.kv_cache_configurator as configurator
import sglang.srt.model_executor.model_runner_components.kv_pool_runtime as kv_pool_runtime
from sglang.srt.plugins.hook_registry import HookRegistry
from xpool.integrations.sglang.kv.allocator import ElasticPagedTokenToKVPoolAllocator, ElasticTokenToKVPoolAllocator
from xpool.integrations.sglang.hooks.kv import KvHookSet
from xpool.integrations.sglang.kv.pool import ElasticMHATokenToKVPool, ElasticMLATokenToKVPool

original_resize_id = id(kv_pool_runtime.compute_post_capture_kv_resize)
for hook in KvHookSet().hooks():
    HookRegistry.register(hook.target, hook.handler, hook.kind)
HookRegistry.apply_hooks()

assert configurator.MHATokenToKVPool is ElasticMHATokenToKVPool
assert configurator.MLATokenToKVPool is ElasticMLATokenToKVPool
assert configurator.TokenToKVPoolAllocator is ElasticTokenToKVPoolAllocator
assert configurator.PagedTokenToKVPoolAllocator is ElasticPagedTokenToKVPoolAllocator
assert id(kv_pool_runtime.compute_post_capture_kv_resize) != original_resize_id
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_pinned_sglang_kv_seams_match() -> None:
    validate_kv_seams()


def test_elastic_prefill_adder_uses_active_prefix_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    adder = ElasticPrefillAdder.__new__(ElasticPrefillAdder)
    adder.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 7, token_capacity=12)
    adder.tree_cache = object()
    adder.rem_total_token_offset = 3
    adder.cur_rem_token_offset = 5
    monkeypatch.setattr(xpool.integrations.sglang.hooks.kv, "admission_evictable_size", lambda cache, size: 4)

    assert adder.rem_total_tokens == 8
    assert adder.cur_rem_tokens == 6


def test_allocation_after_suffix_reclaim_evicts_active_cache() -> None:
    allocator = ElasticTokenToKVPoolAllocator(12, torch.float16, "cpu", cast(KVCache, object()), False)
    tree_cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=cast(ReqToTokenPool, object()),
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    for first_token, length in ((11, 4), (21, 8)):
        slots = allocator.alloc(length)
        assert slots is not None
        tree_cache.insert(InsertParams(key=RadixKey(array("i", range(first_token, first_token + length))), value=slots))

    selected = select_suffix_reclaim_nodes(tree_cache, allocator, token_capacity=8)
    assert selected is not None
    allocator.set_token_capacity(8)
    evict_suffix_reclaim_nodes(tree_cache, selected)
    assert allocator.suffix_is_free(8)
    assert allocator.available_size() == 4

    slots = sglang.srt.mem_cache.allocation.alloc_token_slots(tree_cache, 5)

    assert len(slots) == 5
    assert all(1 <= slot <= 8 for slot in slots.tolist())
    assert allocator.withheld_size() == 4


def test_request_broadcast_carries_and_consumes_capacity_commands() -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=4)
    channel = SimpleNamespace(read_commands=lambda: [command])
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceControlChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, object()),
        allocator=cast(ElasticTokenToKVPoolAllocator, object()),
        request_pool=cast(ReqToTokenPool, object()),
        instance_rank=cast(InstanceRankRuntime, object()),
        slo=LatencySloConfig(ttft_ms=1000, tbt_ms=50),
    )

    def broadcast(receiver: SchedulerRequestReceiver, values: list[object] | None) -> list[object]:
        assert values is not None
        return values

    with capacity_reconciler_scope(reconciler):
        requests = around_request_broadcast(broadcast, cast(SchedulerRequestReceiver, object()), ["request"])

    assert requests == ["request"]
    assert reconciler.command is command


@pytest.mark.parametrize("event_loop", [Scheduler.event_loop_normal, Scheduler.event_loop_overlap])
def test_paused_scheduler_does_not_enter_capacity_planning(event_loop: Callable[[Scheduler], None]) -> None:
    class PausedScheduler:
        gracefully_exit = False
        _engine_paused = True
        request_receiver = SimpleNamespace(recv_requests=lambda: [])

        def process_input_requests(self, requests: list[object]) -> None:
            self.gracefully_exit = True

        def get_next_batch_to_run(self, **kwargs: object) -> None:
            pytest.fail("paused iteration entered capacity planning")

    event_loop(cast(Scheduler, PausedScheduler()))


def test_pool_accounting_treats_withheld_suffix_as_free() -> None:
    allocator = ElasticTokenToKVPoolAllocator(10, torch.float16, "cpu", cast(KVCache, object()), False)
    allocator.set_token_capacity(6)
    stats = PoolStats(
        full_num_used=5,
        full_token_usage=0.5,
        full_available_size=3,
        full_evictable_size=2,
    )
    observer = cast(
        SchedulerPoolStatsObserver,
        SimpleNamespace(token_to_kv_pool_allocator=allocator, max_total_num_tokens=6),
    )

    adjusted = after_pool_stats(stats, observer)
    seen: list[PoolStats] = []

    def check(checker: SchedulerInvariantChecker, value: PoolStats, uncached: int) -> tuple[bool, str]:
        seen.append(value)
        return False, ""

    around_check_full_pool(
        check,
        cast(
            SchedulerInvariantChecker,
            SimpleNamespace(token_to_kv_pool_allocator=allocator, max_total_num_tokens=6),
        ),
        stats,
    )

    assert adjusted.full_num_used == 1
    assert adjusted.full_token_usage == pytest.approx(1 / 6)
    assert seen[0].full_available_size == 3


def test_authoritative_admission_failures_publish_quantified_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[xpool.native.kv.KvCapacityDemand] = []
    channel = SimpleNamespace(publish_demand=published.append)

    class Request(SimpleNamespace):
        __hash__ = object.__hash__

    request = cast(
        Req,
        Request(
            sampling_params=SimpleNamespace(max_new_tokens=4),
            output_ids=[],
            full_untruncated_fill_ids=[1, 2, 3, 4],
            prefix_indices=[],
            kv=SimpleNamespace(holds_mamba=False),
            time_stats=SimpleNamespace(
                scheduler_recv_time=10.0,
                last_decode_finish_time=10.5,
                last_prefill_finished_time=10.25,
            ),
        ),
    )
    allocator = ElasticTokenToKVPoolAllocator(8, torch.float16, "cpu", cast(KVCache, object()), False)
    allocated = allocator.alloc(6)
    assert allocated is not None
    adder = ElasticPrefillAdder.__new__(ElasticPrefillAdder)
    adder.token_to_kv_pool_allocator = allocator
    adder.tree_cache = object()
    adder.rem_total_token_offset = 0
    adder.page_size = 1
    adder.can_run_list = []
    adder._mamba_slot_cost = 0
    monkeypatch.setattr(xpool.integrations.sglang.hooks.kv, "admission_evictable_size", lambda cache, size: 0)
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceControlChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, SimpleNamespace(required_bundles=lambda tokens: (tokens + 3) // 4)),
        allocator=allocator,
        request_pool=cast(ReqToTokenPool, object()),
        instance_rank=cast(InstanceRankRuntime, object()),
        slo=LatencySloConfig(ttft_ms=1000, tbt_ms=50),
        active_bundles=2,
        applied_sequence=1,
        completed_sequence=1,
    )

    with get_parallel().override(attn_tp_rank=0), capacity_reconciler_scope(reconciler):
        result = around_prefill_add_one_req(
            lambda current_adder, current_req, has_chunked, alignment: (
                current_adder.rem_total_tokens,
                AddReqResult.NO_TOKEN,
            )[1],
            adder,
            request,
            False,
            None,
        )
        batch = cast(
            ScheduleBatch,
            SimpleNamespace(
                reqs=[request],
                token_to_kv_pool_allocator=allocator,
                new_tokens_required_next_decode=lambda: 3,
            ),
        )
        after_check_decode_mem(False, batch, selected_indices=[0])
        after_check_decode_mem(False, batch)
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[request])))

    assert result is AddReqResult.NO_TOKEN
    assert [
        (demand.evaluated_sequence, demand.requested_bundles, demand.deadline_monotonic_ns) for demand in published
    ] == [(1, 3, 10_550_000_000)]


def test_drain_stops_new_prefill_admission() -> None:
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceControlChannel, object()),
        command_index=0,
        backing=cast(KvVmmBacking, object()),
        allocator=cast(ElasticTokenToKVPoolAllocator, object()),
        request_pool=cast(ReqToTokenPool, object()),
        instance_rank=cast(InstanceRankRuntime, object()),
        slo=LatencySloConfig(ttft_ms=1000, tbt_ms=50),
        command=xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1),
        active_bundles=2,
        completed_sequence=1,
    )

    with capacity_reconciler_scope(reconciler):
        result = around_prefill_add_one_req(
            lambda *_: pytest.fail("draining must stop new prefill admission"),
            ElasticPrefillAdder.__new__(ElasticPrefillAdder),
            cast(Req, object()),
            False,
            None,
        )

    assert result is AddReqResult.NO_TOKEN


def test_decode_retraction_preserves_all_ordinary_requests_below_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = ElasticTokenToKVPoolAllocator(2, torch.float16, "cpu", cast(KVCache, object()), False)
    allocator.set_token_capacity(1)
    requests = [SimpleNamespace(beam_group=None), SimpleNamespace(beam_group=None)]
    retracted: list[object] = []

    def retract(**kwargs: object) -> None:
        retracted.extend(cast(list[object], kwargs["reqs"]))

    monkeypatch.setattr(xpool.integrations.sglang.hooks.kv, "retract_all", retract)
    monkeypatch.setattr(ScheduleBatch, "_get_decode_retraction_order", staticmethod(lambda reqs: [0, 1]))
    batch = SimpleNamespace(
        reqs=requests,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=object(),
        tree_cache=object(),
        hisparse_coordinator=None,
        check_decode_mem=lambda selected_indices=None: False,
    )
    batch.filter_batch = lambda keep_indices: setattr(batch, "reqs", [])

    result = around_retract_decode(
        lambda value: pytest.fail("temporary elastic shortage delegated to upstream abort path"),
        cast(ScheduleBatch, batch),
    )

    assert result == (requests, 0.0, [])
    assert retracted == requests
