"""Elastic Context State adaptations of the pinned SGLang allocation boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import signature
from typing import cast

import torch
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import SchedulerInvariantChecker
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats, SchedulerPoolStatsObserver
from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver
from sglang.srt.mem_cache import allocation, common
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams, EvictResult
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator, _InitializedPools, _PoolSizes
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.kv_pool_runtime import PostCaptureKVResize
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_schedule

from xpool.config import get_global_config
from xpool.integrations.sglang.adapter import SglangInstanceRankRuntime
from xpool.integrations.sglang.ecs.admission import (
    ElasticPrefillAdder,
    check_decode_mem,
    prepare_for_decode,
    retract_decode,
)
from xpool.integrations.sglang.ecs.allocation import (
    SglangAllocation,
    SglangAllocationStatus,
    SglangCapacityMiss,
)
from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.ecs.capacity import (
    CapacityReconciler,
    capacity_reconciler_scope,
    current_capacity_reconciler,
)
from xpool.integrations.sglang.ecs.construction import current_state_adapter_type
from xpool.integrations.sglang.ecs.implementations.kv_radix import (
    around_acquire_swa_component_lock,
    free_swa_out_of_window_slots,
)
from xpool.integrations.sglang.ecs.pool import (
    ElasticMHATokenToKVPool,
    ElasticMLATokenToKVPool,
)
from xpool.integrations.sglang.hooks.registry import SglangHook, SglangHookSet
from xpool.native.ecs import QuotaCommand


@dataclass(frozen=True, slots=True)
class ElasticContextStateQuotaCommandBatch:
    """Quota commands carried through SGLang's existing TP request broadcast."""

    commands: tuple[QuotaCommand | None, ...]


@dataclass(slots=True)
class CapacityRequestReceiver:
    """Lend the runner's capacity owner to the original receive iteration."""

    receiver: SchedulerRequestReceiver
    reconciler: CapacityReconciler

    def recv_requests(self, local_reqs: list[AbortReq] | None = None) -> list[object]:
        with capacity_reconciler_scope(self.reconciler):
            return self.receiver.recv_requests(local_reqs=local_reqs)


def compute_reservation_budget(configurator: KVCacheConfigurator, pre_model_load_memory: int) -> int:
    """Reserve virtual capacity independently of other models' launch order."""

    utilization = min(get_global_config().atn.device_memory_utilization, get_schedule().mem_fraction_static)
    budget = int(torch.cuda.get_device_properties(configurator.device).total_memory * utilization)
    return current_state_adapter_type().reservation_budget(configurator, budget)


def compute_post_capture_resize(
    model_runner: ModelRunner, *, draft_runners: tuple[ModelRunner, ...] = ()
) -> PostCaptureKVResize:
    """Adopt the startup byte grant and derive stable engine limits once."""

    runtime = SglangInstanceRankRuntime.require(model_runner)
    if runtime.capacity_reconciler is None:
        raise RuntimeError("xpool post-capture attention finalization requires its capacity owner")
    return runtime.capacity_reconciler.finalize_after_capture(model_runner)


def after_scheduler_init_request_receiver(result: None, scheduler: Scheduler) -> None:
    """Bind the real cache and Scheduler before the first request iteration."""

    runtime = SglangInstanceRankRuntime.require(scheduler.tp_worker.model_runner)
    reconciler = runtime.capacity_reconciler
    if reconciler is None:
        raise RuntimeError("xpool scheduler requires an attached elastic context state capacity owner")
    reconciler.state.bind_cache(scheduler.tree_cache)
    reconciler.scheduler = scheduler
    scheduler.request_receiver = CapacityRequestReceiver(scheduler.request_receiver, reconciler)


def around_request_broadcast(
    original_fn: Callable[[SchedulerRequestReceiver, list[object] | None, list[object] | None], list[object]],
    receiver: SchedulerRequestReceiver,
    recv_reqs: list[object] | None,
    local_reqs: list[object] | None = None,
) -> list[object]:
    """Deliver exactly one common command snapshot through the existing broadcast."""

    reconciler = current_capacity_reconciler()
    if recv_reqs is not None:
        recv_reqs.append(ElasticContextStateQuotaCommandBatch(tuple(reconciler.channel.read_commands())))
    requests = original_fn(receiver, recv_reqs, local_reqs)
    markers = [
        (index, value)
        for index, value in enumerate(requests)
        if isinstance(value, ElasticContextStateQuotaCommandBatch)
    ]
    if len(markers) != 1:
        raise RuntimeError("xpool elastic context state broadcast must deliver exactly one command snapshot")
    index, marker = markers[0]
    requests.pop(index)
    reconciler.accept_command(marker.commands[reconciler.command_index])
    return requests


def around_scheduler_get_next_batch_to_run[R](
    original_fn: Callable[[Scheduler, ScheduleBatch, ScheduleBatch | None], R],
    scheduler: Scheduler,
    running_batch: ScheduleBatch,
    last_batch: ScheduleBatch | None,
) -> R:
    """Progress physical retirement and establish common inputs before planning."""

    runtime = SglangInstanceRankRuntime.require(scheduler.tp_worker.model_runner)
    reconciler = runtime.capacity_reconciler
    if reconciler is None:
        raise RuntimeError("xpool scheduler planning requires its elastic context state capacity owner")
    with capacity_reconciler_scope(reconciler):
        reconciler.begin_scheduling(scheduler)
        result = original_fn(scheduler, running_batch, last_batch)
        reconciler.finish_scheduling(scheduler)
    return result


def around_get_new_batch_prefill_raw(
    original_fn: Callable[
        [Scheduler, PrefillDelayerSinglePassExecutor | None, ScheduleBatch], tuple[ScheduleBatch | None, ScheduleBatch]
    ],
    scheduler: Scheduler,
    prefill_delayer_single_pass: PrefillDelayerSinglePassExecutor | None,
    running_batch: ScheduleBatch,
) -> tuple[ScheduleBatch | None, ScheduleBatch]:
    """Journal tentative selection until the complete Extend allocation commits."""

    with current_capacity_reconciler().state.allocation(scheduler) as attempt:
        try:
            result = original_fn(scheduler, prefill_delayer_single_pass, running_batch)
        except SglangCapacityMiss:
            return None, running_batch
        if not attempt.committed:
            attempt.commit()
        return result


def complete_allocation[R](operation: Callable[[], R], batch: ScheduleBatch, *, token_per_req: int | None) -> R:
    """Own the single matched verdict for all child acquisitions and table writes."""

    attempt = SglangAllocation.current()
    if attempt is None:
        raise RuntimeError("xpool elastic context state allocation requires its preparation journal")
    attempt.watch(batch.reqs)
    attempt.batch, attempt.token_per_req = batch, token_per_req
    if token_per_req is not None:
        attempt.record_decode(batch.req_pool_indices_cpu, batch.seq_lens_cpu, token_per_req)
    error: BaseException | None = None
    result: R | None = None
    try:
        current_capacity_reconciler().state.before_allocation(batch, token_per_req=token_per_req)
        result = operation()
        status = SglangAllocationStatus.READY
    except SglangCapacityMiss as failure:
        error, status = failure, SglangAllocationStatus.CAPACITY_MISS
    except BaseException as failure:
        error, status = failure, SglangAllocationStatus.FAILED
    finally:
        attempt.batch = attempt.token_per_req = None
    verdict = current_capacity_reconciler().allocation_verdict(status)
    if verdict is SglangAllocationStatus.FAILED:
        if status is SglangAllocationStatus.FAILED and error is not None:
            raise error
        raise RuntimeError("xpool elastic context state allocation failed on a TP peer") from error
    if verdict is SglangAllocationStatus.CAPACITY_MISS:
        attempt.rollback()
        raise SglangCapacityMiss("xpool elastic context state allocation could not commit on every TP rank") from error
    attempt.commit()
    # READY establishes the original operation's return type, including a
    # legitimately nullable R; no replacement result is constructed here.
    return cast(R, result)


def around_alloc_for_extend(
    original_fn: Callable[[ScheduleBatch], tuple[torch.Tensor, torch.Tensor, torch.Tensor]], batch: ScheduleBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return complete_allocation(lambda: original_fn(batch), batch, token_per_req=None)


def around_alloc_for_decode(
    original_fn: Callable[[ScheduleBatch, int], torch.Tensor], batch: ScheduleBatch, token_per_req: int
) -> torch.Tensor:
    return complete_allocation(lambda: original_fn(batch, token_per_req), batch, token_per_req=token_per_req)


def around_alloc_req_slots(
    original_fn: Callable[[ReqToTokenPool, list[Req], BasePrefixCache | None], list[int]],
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Retain only new row receipts; continuations keep their existing rows."""

    new_rows = tuple(req.kv.req_pool_idx is None for req in reqs)
    rows = original_fn(req_to_token_pool, reqs, tree_cache)
    attempt = SglangAllocation.current()
    if attempt is not None and attempt.batch is not None:
        attempt.record_rows([row for row, acquired in zip(rows, new_rows, strict=True) if acquired])
    return rows


def around_evict_from_tree_cache(
    original_fn: Callable[[BasePrefixCache | None, int], None], tree_cache: BasePrefixCache | None, num_tokens: int
) -> None:
    """Prepare the actual batch requirement rather than the helper's conservative estimate."""

    attempt = SglangAllocation.current()
    if attempt is None or attempt.batch is None:
        return original_fn(tree_cache, num_tokens)
    state = current_capacity_reconciler().state
    pages = state.allocation_counts(attempt.batch, token_per_req=attempt.token_per_req)
    if not state.prepare_allocation(pages):
        raise SglangCapacityMiss("xpool elastic context state cannot realize the selected batch under its byte grant")


def around_cache_evict(
    original_fn: Callable[[UnifiedRadixCache, EvictParams], EvictResult],
    cache: UnifiedRadixCache,
    params: EvictParams,
) -> EvictResult:
    """Defer unmatched local cache requests to the next common logical boundary."""

    current_capacity_reconciler().defer_eviction(params)
    return EvictResult()


def after_pool_stats(result: PoolStats, observer: SchedulerPoolStatsObserver) -> PoolStats:
    return current_capacity_reconciler().state.pool_stats(result, observer)


def around_check_full_pool(
    original_fn: Callable[[SchedulerInvariantChecker, PoolStats, int], tuple[bool, str]],
    checker: SchedulerInvariantChecker,
    ps: PoolStats,
    uncached: int = 0,
) -> tuple[bool, str]:
    return current_capacity_reconciler().state.check_pool(original_fn, checker, ps, uncached, window=False)


def around_check_swa_pool(
    original_fn: Callable[[SchedulerInvariantChecker, PoolStats, int], tuple[bool, str]],
    checker: SchedulerInvariantChecker,
    ps: PoolStats,
    uncached: int = 0,
) -> tuple[bool, str]:
    return current_capacity_reconciler().state.check_pool(original_fn, checker, ps, uncached, window=True)


def around_init_pools(
    original_fn: Callable[..., _InitializedPools],
    configurator: KVCacheConfigurator,
    *,
    sizes: _PoolSizes,
    req_to_token_pool: ReqToTokenPool | None,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None,
) -> _InitializedPools:
    """Prepare every floor under one ledger before the original engine factory.

    A construction failure releases only safely quiesced resources. Terminal
    device errors or failed cleanup retain the owners in the exception chain.
    """

    return current_state_adapter_type().initialize_pools(
        original_fn,
        configurator,
        sizes=sizes,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
    )


def validate_seams() -> None:
    """Check pinned call shapes where the adaptation retains upstream mutation order."""

    expected_parameters = (
        (KVCacheConfigurator._init_pools, ("self", "sizes", "req_to_token_pool", "token_to_kv_pool_allocator")),
        (
            PrefillAdder._select_prefill_admission,
            ("self", "req", "total_tokens", "host_hit_length", "swa_host_hit_length", "truncation_align_size"),
        ),
        (PrefillAdder._commit_prefill_admission, ("self", "req", "admission", "mamba_gap_reserve")),
        (Scheduler._get_new_batch_prefill_raw, ("self", "prefill_delayer_single_pass", "running_batch")),
        (allocation.alloc_req_slots, ("req_to_token_pool", "reqs", "tree_cache")),
        (allocation.alloc_for_extend, ("batch",)),
        (allocation.alloc_for_decode, ("batch", "token_per_req")),
        (common.evict_from_tree_cache, ("tree_cache", "num_tokens")),
        (ScheduleBatch.prepare_for_decode, ("self",)),
        (ScheduleBatch.retract_decode, ("self",)),
        (ScheduleBatch.check_decode_mem, ("self", "selected_indices")),
    )
    for target, expected in expected_parameters:
        if tuple(signature(target).parameters) != expected:
            raise RuntimeError(f"xpool sglang attention seam changed: {target.__qualname__}{signature(target)}")


class ElasticContextStateHookSet(SglangHookSet):
    """Pool construction, matched scheduling, allocation and reservation accounting."""

    def hooks(self) -> tuple[SglangHook, ...]:
        validate_seams()
        return (
            SglangHook("sglang.srt.mem_cache.memory_pool.MHATokenToKVPool", ElasticMHATokenToKVPool, HookType.REPLACE),
            SglangHook("sglang.srt.mem_cache.memory_pool.MLATokenToKVPool", ElasticMLATokenToKVPool, HookType.REPLACE),
            SglangHook(
                "sglang.srt.mem_cache.allocator.token.TokenToKVPoolAllocator",
                ElasticTokenToKVPoolAllocator,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator",
                ElasticPagedTokenToKVPoolAllocator,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._profile_available_bytes",
                compute_reservation_budget,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.model_executor.model_runner_components.kv_pool_runtime.compute_post_capture_kv_resize",
                compute_post_capture_resize,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._init_pools",
                around_init_pools,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.scheduler.Scheduler.init_request_receiver",
                after_scheduler_init_request_receiver,
                HookType.AFTER,
            ),
            SglangHook(
                "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver._broadcast_reqs_across_ranks",
                around_request_broadcast,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
                around_scheduler_get_next_batch_to_run,
                HookType.AROUND,
            ),
            SglangHook("sglang.srt.managers.schedule_policy.PrefillAdder", ElasticPrefillAdder, HookType.REPLACE),
            SglangHook(
                "sglang.srt.managers.scheduler.Scheduler._get_new_batch_prefill_raw",
                around_get_new_batch_prefill_raw,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.schedule_batch.ScheduleBatch.check_decode_mem", check_decode_mem, HookType.REPLACE
            ),
            SglangHook(
                "sglang.srt.managers.schedule_batch.ScheduleBatch.retract_decode", retract_decode, HookType.REPLACE
            ),
            SglangHook(
                "sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_decode",
                prepare_for_decode,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.mem_cache.common.evict_from_tree_cache", around_evict_from_tree_cache, HookType.AROUND
            ),
            SglangHook("sglang.srt.mem_cache.allocation.alloc_req_slots", around_alloc_req_slots, HookType.AROUND),
            SglangHook("sglang.srt.mem_cache.allocation.alloc_for_extend", around_alloc_for_extend, HookType.AROUND),
            SglangHook("sglang.srt.mem_cache.allocation.alloc_for_decode", around_alloc_for_decode, HookType.AROUND),
            SglangHook(
                "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.evict", around_cache_evict, HookType.AROUND
            ),
            SglangHook(
                "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.evict_for_alloc",
                around_cache_evict,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.mem_cache.unified_cache.components.swa.SWAComponent.acquire_component_lock",
                around_acquire_swa_component_lock,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.mem_cache.unified_cache.components.swa.SWAComponent._free_out_of_window_slots",
                free_swa_out_of_window_slots,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.managers.scheduler_components.pool_stats_observer.SchedulerPoolStatsObserver.get_pool_stats",
                after_pool_stats,
                HookType.AFTER,
            ),
            SglangHook(
                "sglang.srt.managers.scheduler_components.invariant_checker.SchedulerInvariantChecker._check_full_pool",
                around_check_full_pool,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.scheduler_components.invariant_checker.SchedulerInvariantChecker._check_swa_pool",
                around_check_swa_pool,
                HookType.AROUND,
            ),
        )
