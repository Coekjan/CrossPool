from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.invariant_checker import SchedulerInvariantChecker
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats, SchedulerPoolStatsObserver
from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.runtime_context import get_context, get_parallel

import xpool.integrations.sglang.hooks.kv
import xpool.native
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.sglang.runtime import published_sglang_config
from xpool.config import XpoolConfig
from xpool.integrations.sglang.hooks.kv import (
    ElasticPrefillAdder,
    after_check_decode_mem,
    after_pool_stats,
    after_prefill_add_one_req,
    around_check_full_pool,
    around_request_broadcast,
    around_retract_decode,
    capacity_reconciler_scope,
    compute_kv_reservation_budget,
    validate_kv_seams,
)
from xpool.integrations.sglang.kv.allocator import ElasticTokenToKVPoolAllocator
from xpool.integrations.sglang.kv.capacity import CapacityReconciler
from xpool.integrations.sglang.kv.vmm import KvVmmBacking

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, published_sglang_config.__name__)


def test_reservation_budget_uses_static_device_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(
        config=XpoolConfig.from_mapping(
            {
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


def test_request_broadcast_carries_and_consumes_capacity_commands() -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=4, active_bundles=3)
    channel = SimpleNamespace(read_commands=lambda: [command])
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceCapacityChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, object()),
        allocator=cast(ElasticTokenToKVPoolAllocator, object()),
        request_pool=cast(ReqToTokenPool, object()),
    )

    def broadcast(receiver: SchedulerRequestReceiver, values: list[object] | None) -> list[object]:
        assert values is not None
        return values

    with capacity_reconciler_scope(reconciler):
        requests = around_request_broadcast(broadcast, cast(SchedulerRequestReceiver, object()), ["request"])

    assert requests == ["request"]
    assert reconciler.command is command


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
        SimpleNamespace(token_to_kv_pool_allocator=allocator, max_total_num_tokens=10),
    )

    adjusted = after_pool_stats(stats, observer)
    seen: list[PoolStats] = []

    def check(checker: SchedulerInvariantChecker, value: PoolStats, uncached: int) -> tuple[bool, str]:
        seen.append(value)
        return False, ""

    around_check_full_pool(
        check,
        cast(SchedulerInvariantChecker, SimpleNamespace(token_to_kv_pool_allocator=allocator)),
        stats,
    )

    assert adjusted.full_num_used == 1
    assert adjusted.full_token_usage == 0.1
    assert seen[0].full_available_size == 7


def test_authoritative_admission_failures_publish_one_pressure_edge() -> None:
    published: list[int | None] = []
    channel = SimpleNamespace(publish_pressure=lambda value: published.append(value))
    request = cast(Req, object())
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceCapacityChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, object()),
        allocator=cast(ElasticTokenToKVPoolAllocator, object()),
        request_pool=cast(ReqToTokenPool, object()),
        command=xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=2, active_bundles=2),
    )

    with get_parallel().override(attn_tp_rank=0), capacity_reconciler_scope(reconciler):
        after_prefill_add_one_req(
            AddReqResult.NO_TOKEN,
            cast(PrefillAdder, SimpleNamespace(can_run_list=[])),
            request,
            False,
            None,
        )
        after_check_decode_mem(False, cast(ScheduleBatch, SimpleNamespace(reqs=[request])), selected_indices=[0])
        after_check_decode_mem(False, cast(ScheduleBatch, SimpleNamespace(reqs=[request])))

    assert published == [2]


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
