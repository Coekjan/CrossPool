"""SGLang hook handlers for elastic KV capacity."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from inspect import signature

import torch
from sglang.srt.beam_search.batch_tail import beam_retraction_order
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, retract_all
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS, AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import SchedulerInvariantChecker
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import NewTokenRatioTracker
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats, SchedulerPoolStatsObserver
from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.kv_pool_runtime import (
    PostCaptureKVResize,
)
from sglang.srt.model_executor.model_runner_components.kv_pool_runtime import (
    compute_post_capture_kv_resize as sglang_compute_post_capture_kv_resize,
)
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_schedule

import xpool.native
from xpool.config import get_global_config
from xpool.integrations.sglang.adapter import SglangInstanceRankRuntime
from xpool.integrations.sglang.hooks.registry import SglangHook, SglangHookSet
from xpool.integrations.sglang.kv.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.kv.capacity import CapacityReconciler
from xpool.integrations.sglang.kv.pool import ElasticMHATokenToKVPool, ElasticMLATokenToKVPool
from xpool.integrations.sglang.kv.radix import admission_evictable_size

capacity_reconciler_context = ContextVar[CapacityReconciler | None](
    "xpool_capacity_reconciler",
    default=None,
)


@contextmanager
def capacity_reconciler_scope(reconciler: CapacityReconciler) -> Iterator[None]:
    """Expose one reconciler to nested SGLang scheduling hooks."""

    token = capacity_reconciler_context.set(reconciler)
    try:
        yield
    finally:
        capacity_reconciler_context.reset(token)


def current_capacity_reconciler() -> CapacityReconciler:
    """Return the reconciler in the current scheduling scope."""

    reconciler = capacity_reconciler_context.get()
    if reconciler is None:
        raise RuntimeError("xpool kv scheduling hook ran outside a capacity reconciler scope")
    return reconciler


@dataclass(frozen=True, slots=True)
class KvCapacityCommandBatch:
    """Capacity commands carried through SGLang's existing TP broadcast."""

    commands: tuple[xpool.native.kv.KvCapacityCommand | None, ...]


@dataclass(slots=True)
class CapacityRequestReceiver:
    """Enter capacity scope around the original SGLang receiver."""

    receiver: SchedulerRequestReceiver
    reconciler: CapacityReconciler

    def recv_requests(self) -> list[object]:
        """Delegate one receive iteration inside the reconciler scope."""

        with capacity_reconciler_scope(self.reconciler):
            return self.receiver.recv_requests()


class ElasticPrefillAdder(PrefillAdder):
    """Use the elastic allocator's active prefix for admission budgets."""

    tree_cache: UnifiedRadixCache
    token_to_kv_pool_allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator
    track_remaining_tokens = False
    minimum_remaining_tokens: float | None = None

    @property
    def rem_total_tokens(self) -> float:
        allocator = self.token_to_kv_pool_allocator
        remaining = (
            allocator.available_size()
            + admission_evictable_size(self.tree_cache, allocator.token_capacity)
            - self.rem_total_token_offset
        )
        if self.track_remaining_tokens:
            self.minimum_remaining_tokens = (
                remaining if self.minimum_remaining_tokens is None else min(self.minimum_remaining_tokens, remaining)
            )
        return remaining

    @property
    def cur_rem_tokens(self) -> int:
        allocator = self.token_to_kv_pool_allocator
        return (
            allocator.available_size()
            + admission_evictable_size(self.tree_cache, allocator.token_capacity)
            - self.cur_rem_token_offset
        )


def compute_kv_reservation_budget(
    configurator: KVCacheConfigurator,
    pre_model_load_memory: int,
) -> int:
    """Return the launch-order-independent virtual KV reservation budget."""

    utilization = min(
        get_global_config().atn.device_memory_utilization,
        get_schedule().mem_fraction_static,
    )
    return int(torch.cuda.get_device_properties(configurator.device).total_memory * utilization)


def compute_post_capture_kv_resize(model_runner: ModelRunner) -> PostCaptureKVResize:
    """Finalize elastic backing through the runner-owned capacity reconciler."""

    runtime = SglangInstanceRankRuntime.require(model_runner)
    if runtime.kv_capacity is None:
        raise RuntimeError("xpool post-capture kv finalization requires an attached capacity reconciler")
    return runtime.kv_capacity.finalize_after_capture(model_runner)


def after_scheduler_init_request_receiver(result: None, scheduler: Scheduler) -> None:
    """Wrap the assigned receiver with the runner-owned reconciler scope."""

    if not isinstance(scheduler.tree_cache, UnifiedRadixCache) or not isinstance(
        scheduler.tree_cache.tree_core, UnifiedTreeCore
    ):
        raise RuntimeError("xpool elastic kv requires SGLang UnifiedRadixCache with the Python UnifiedTreeCore")
    runtime = SglangInstanceRankRuntime.require(scheduler.tp_worker.model_runner)
    if runtime.kv_capacity is None:
        raise RuntimeError("xpool scheduler receiver requires an attached capacity reconciler")
    scheduler.request_receiver = CapacityRequestReceiver(scheduler.request_receiver, runtime.kv_capacity)


def around_request_broadcast(
    original_fn: Callable[[SchedulerRequestReceiver, list[object] | None], list[object]],
    receiver: SchedulerRequestReceiver,
    recv_reqs: list[object] | None,
) -> list[object]:
    """Carry coherent capacity commands through SGLang's request broadcast."""

    reconciler = current_capacity_reconciler()
    if recv_reqs is not None:
        recv_reqs.append(KvCapacityCommandBatch(tuple(reconciler.channel.read_commands())))
    requests = original_fn(receiver, recv_reqs)
    markers = [
        (index, request) for index, request in enumerate(requests) if isinstance(request, KvCapacityCommandBatch)
    ]
    if len(markers) != 1:
        raise RuntimeError("xpool capacity broadcast must deliver exactly one command batch")
    marker_index, marker = markers[0]
    requests.pop(marker_index)
    reconciler.accept_command(marker.commands[reconciler.command_index])
    return requests


def around_scheduler_get_next_batch_to_run[R](
    original_fn: Callable[[Scheduler, ScheduleBatch, ScheduleBatch | None], R],
    scheduler: Scheduler,
    running_batch: ScheduleBatch,
    last_batch: ScheduleBatch | None,
) -> R:
    """Reconcile capacity around one ordinary SGLang planning iteration."""

    runtime = SglangInstanceRankRuntime.require(scheduler.tp_worker.model_runner)
    if runtime.kv_capacity is None:
        raise RuntimeError("xpool scheduler planning requires an attached capacity reconciler")
    reconciler = runtime.kv_capacity
    reconciler.begin_scheduling(scheduler)
    with capacity_reconciler_scope(reconciler):
        result = original_fn(scheduler, running_batch, last_batch)
    reconciler.finish_scheduling(scheduler)
    return result


def around_prefill_add_one_req(
    original_fn: Callable[[PrefillAdder, Req, bool, int | None], AddReqResult],
    adder: PrefillAdder,
    req: Req,
    has_chunked_req: bool,
    truncation_align_size: int | None,
) -> AddReqResult:
    """Record authoritative prefill capacity rejection."""

    if not isinstance(adder, ElasticPrefillAdder):
        return original_fn(adder, req, has_chunked_req, truncation_align_size)
    reconciler = current_capacity_reconciler()
    if reconciler.draining:
        return AddReqResult.NO_TOKEN

    adder.minimum_remaining_tokens = None
    adder.track_remaining_tokens = True
    try:
        result = original_fn(adder, req, has_chunked_req, truncation_align_size)
    finally:
        adder.track_remaining_tokens = False
    remaining = adder.minimum_remaining_tokens
    if result is AddReqResult.NO_TOKEN and req not in adder.can_run_list and remaining is not None:
        max_new_tokens = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        required_tokens = (
            len(req.full_untruncated_fill_ids)
            - len(req.prefix_indices)
            + max_new_tokens
            + adder.page_size
            + adder._mamba_gap_budget_for_req(req)
        )
        shortfall = math.floor(required_tokens - remaining) + 1
        if shortfall > 0:
            reconciler.record_prefill_requirement(
                req,
                adder.token_to_kv_pool_allocator.token_capacity + shortfall,
            )
    return result


def after_check_decode_mem(
    result: bool,
    batch: ScheduleBatch,
    selected_indices: list[int] | None = None,
) -> bool:
    """Record authoritative full-batch Decode capacity rejection."""

    if not result and selected_indices is None:
        required_tokens = batch.new_tokens_required_next_decode()
        allocator = batch.token_to_kv_pool_allocator
        if isinstance(allocator, ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator):
            current_capacity_reconciler().record_decode_requirement(
                batch.reqs,
                allocator.token_capacity + required_tokens - allocator.available_size(),
            )
    return result


def around_retract_decode(
    original_fn: Callable[[ScheduleBatch], tuple[list[Req], float, list[Req]]],
    batch: ScheduleBatch,
) -> tuple[list[Req], float, list[Req]]:
    """Preserve ordinary requests blocked only by temporary elastic capacity."""

    allocator = batch.token_to_kv_pool_allocator
    if (
        not isinstance(allocator, ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator)
        or allocator.token_capacity >= allocator.reserved_token_capacity
        or any(req.beam_group is not None for req in batch.reqs)
    ):
        return original_fn(batch)

    order = beam_retraction_order(ScheduleBatch._get_decode_retraction_order(batch.reqs), batch.reqs)
    if not order or batch.check_decode_mem(selected_indices=[order[0]]):
        return original_fn(batch)

    retracted = list(batch.reqs)
    retract_all(
        reqs=retracted,
        req_to_token_pool=batch.req_to_token_pool,
        token_to_kv_pool_allocator=batch.token_to_kv_pool_allocator,
        tree_cache=batch.tree_cache,
        hisparse_coordinator=batch.hisparse_coordinator,
    )
    batch.filter_batch(keep_indices=[])
    return retracted, NewTokenRatioTracker.estimate_new_token_ratio_after_retract(batch.reqs), []


def after_pool_stats(result: PoolStats, observer: SchedulerPoolStatsObserver) -> PoolStats:
    """Exclude withheld free suffix slots from published pool use."""

    allocator = observer.token_to_kv_pool_allocator
    if not isinstance(allocator, ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator):
        return result
    used = (
        observer.max_total_num_tokens
        - result.full_available_size
        - result.full_evictable_size
        - (observer.max_total_num_tokens - allocator.token_capacity)
    )
    return replace(
        result,
        full_num_used=used,
        full_token_usage=used / observer.max_total_num_tokens,
    )


def around_check_full_pool(
    original_fn: Callable[[SchedulerInvariantChecker, PoolStats, int], tuple[bool, str]],
    checker: SchedulerInvariantChecker,
    pool_stats: PoolStats,
    uncached: int = 0,
) -> tuple[bool, str]:
    """Count withheld free slots when checking fixed-pool conservation."""

    allocator = checker.token_to_kv_pool_allocator
    if isinstance(allocator, ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator):
        pool_stats = replace(
            pool_stats,
            full_available_size=(
                pool_stats.full_available_size + checker.max_total_num_tokens - allocator.token_capacity
            ),
        )
    return original_fn(checker, pool_stats, uncached)


def validate_kv_seams() -> None:
    """Reject a pinned SGLang release whose hooked call shapes changed."""

    expected_parameters = (
        (KVCacheConfigurator._profile_available_bytes, ("self", "pre_model_load_memory")),
        (sglang_compute_post_capture_kv_resize, ("model_runner",)),
        (Scheduler.init_request_receiver, ("self",)),
        (PagedTokenToKVPoolAllocator._release_page_ids, ("self", "page_ids")),
        (SchedulerRequestReceiver._broadcast_reqs_across_ranks, ("self", "recv_reqs")),
        (Scheduler.get_next_batch_to_run, ("self", "running_batch", "last_batch")),
        (PrefillAdder.add_one_req, ("self", "req", "has_chunked_req", "truncation_align_size")),
        (ScheduleBatch.check_decode_mem, ("self", "selected_indices")),
        (ScheduleBatch.retract_decode, ("self",)),
        (ScheduleBatch._get_decode_retraction_order, ("reqs",)),
        (ScheduleBatch.filter_batch, ("self", "chunked_req_to_exclude", "keep_indices")),
        (
            retract_all,
            (
                "reqs",
                "req_to_token_pool",
                "token_to_kv_pool_allocator",
                "tree_cache",
                "hisparse_coordinator",
                "offload_kv",
            ),
        ),
        (SchedulerPoolStatsObserver._get_token_info, ("self",)),
        (SchedulerInvariantChecker._check_full_pool, ("self", "ps", "uncached")),
    )
    for target, expected in expected_parameters:
        actual = tuple(signature(target).parameters)
        if actual != expected:
            raise RuntimeError(f"xpool sglang kv seam changed: {target.__qualname__}{signature(target)}")


class KvHookSet(SglangHookSet):
    """Pinned construction, scheduling, and accounting hooks for elastic KV."""

    def hooks(self) -> tuple[SglangHook, ...]:
        validate_kv_seams()
        return (
            SglangHook(
                "sglang.srt.managers.schedule_policy.PrefillAdder",
                ElasticPrefillAdder,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool",
                ElasticMHATokenToKVPool,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.mem_cache.memory_pool.MLATokenToKVPool",
                ElasticMLATokenToKVPool,
                HookType.REPLACE,
            ),
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
                compute_kv_reservation_budget,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.model_executor.model_runner_components.kv_pool_runtime.compute_post_capture_kv_resize",
                compute_post_capture_kv_resize,
                HookType.REPLACE,
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
            SglangHook(
                "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req",
                around_prefill_add_one_req,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.schedule_batch.ScheduleBatch.check_decode_mem",
                after_check_decode_mem,
                HookType.AFTER,
            ),
            SglangHook(
                "sglang.srt.managers.schedule_batch.ScheduleBatch.retract_decode",
                around_retract_decode,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.scheduler_components.pool_stats_observer.SchedulerPoolStatsObserver._get_token_info",
                after_pool_stats,
                HookType.AFTER,
            ),
            SglangHook(
                "sglang.srt.managers.scheduler_components.invariant_checker.SchedulerInvariantChecker._check_full_pool",
                around_check_full_pool,
                HookType.AROUND,
            ),
        )
