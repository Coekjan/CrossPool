from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.invariant_checker import SchedulerInvariantChecker
from sglang.srt.managers.scheduler_components.pool_stats_observer import SchedulerPoolStatsObserver
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.scheduler_stage_metrics import SchedulerStageMetricsRecorder
from sglang.srt.runtime_context import get_context

import xpool.integrations.sglang.hooks.ecs
from xpool.config import LatencySloConfig, XpoolConfig
from xpool.integrations.sglang import hooks
from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.ecs.capacity import CapacityReconciler, capacity_reconciler_scope
from xpool.integrations.sglang.ecs.state import SglangElasticContextState
from xpool.integrations.sglang.hooks.ecs import after_pool_stats, around_check_full_pool
from xpool.native.ecs import InstanceControlChannel
from xpool.runtime.instance import InstanceRankRuntime
from xtest.harness.support.config import TEST_MODEL_ID, install_test_config, reset_global_config
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, published_sglang_config.__name__)


def test_virtual_reservation_uses_device_envelope_instead_of_launch_order_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0], "device_memory_utilization": 0.75},
                "ffn": {"devices": [1]},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            }
        )
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.ecs.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=1000),
    )
    configurator = object.__new__(KVCacheConfigurator)
    configurator.device = "cuda:0"
    with get_context().override_server_args(mem_fraction_static=0.6):
        assert hooks.ecs.compute_reservation_budget(configurator, 200) == 600
        assert hooks.ecs.compute_reservation_budget(configurator, 800) == 600


def test_statistics_exclude_free_unavailable_ids_but_conservation_checks_the_full_reservation(
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
    reconciler = CapacityReconciler(
        cast(InstanceControlChannel, object()),
        0,
        state,
        cast(InstanceRankRuntime, object()),
        LatencySloConfig(ttft_ms=1000, tbt_ms=50),
    )
    observer = SchedulerPoolStatsObserver(
        tree_cache=attention_cache,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=state.request_pool,
        session_controller=SimpleNamespace(sessions={}),
        hisparse_coordinator=None,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        enable_hisparse=False,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=16,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
    )
    checker = SchedulerInvariantChecker(
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        disaggregation_mode=DisaggregationMode.NULL,
        page_size=allocator.page_size,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=16,
        tree_cache=attention_cache,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=state.request_pool,
        pool_stats_observer=observer,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
        scheduler_stage_metrics=SchedulerStageMetricsRecorder(enabled=False),
    )
    with capacity_reconciler_scope(reconciler):
        stats = after_pool_stats(observer.get_pool_stats(), observer)
        assert stats.full_num_used == 0 and stats.full_token_usage == 0
        assert stats.full_available_size == allocator.available_size()
        assert not around_check_full_pool(SchedulerInvariantChecker._check_full_pool, checker, stats)[0]
        damaged = replace(stats, full_available_size=stats.full_available_size - allocator.page_size)
        assert around_check_full_pool(SchedulerInvariantChecker._check_full_pool, checker, damaged)[0]
    assert checker.max_total_num_tokens == observer.max_total_num_tokens == 16
