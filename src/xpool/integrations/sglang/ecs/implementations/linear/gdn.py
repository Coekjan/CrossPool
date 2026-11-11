"""Gated DeltaNet state option for the pinned SGLang hybrid pool lifecycle."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Literal, cast

import torch
from sglang.srt.configs.mamba_utils import BaseLinearStateParams, Mamba2CacheParams
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch, set_mamba_track_indices_from_reqs
from sglang.srt.managers.schedule_policy import PrefillAdder, _PrefillAdmission
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import SchedulerInvariantChecker
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats, SchedulerPoolStatsObserver
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator, _InitializedPools, _PoolSizes
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    KVCache,
    MambaPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType, EvictLayer
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore, UnifiedTreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_exec, get_memory, get_schedule, mamba_track_grid
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

from xpool import ecs
from xpool.integrations.sglang.ecs.allocation import SglangAllocation, SglangCapacityMiss
from xpool.integrations.sglang.ecs.implementations.kv import KvStateAdapter, SglangReclaimPlan
from xpool.integrations.sglang.ecs.implementations.kv_radix import SglangCacheEviction
from xpool.integrations.sglang.ecs.interfaces import (
    SglangMemoryRequirement,
    SglangPrefillProposal,
    StateCacheProposal,
    StateReclaimPlan,
)
from xpool.integrations.sglang.ecs.pool import prepared_backing_scope
from xpool.integrations.sglang.ecs.state_vmm import StateBufferLayout, StateVmmBacking
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.utils.cuda import is_terminal_device_error
from xpool.utils.integer import ensure_uint64

prepared_state_backing = ContextVar[StateVmmBacking | None]("xpool_gdn_prepared_backing", default=None)
checkpoint_request = ContextVar[Req | None]("xpool_gdn_checkpoint_request", default=None)


def validate_settings() -> None:
    config = get_exec().mamba
    if not config.enable_mamba_extra_buffer or config.enable_mamba_extra_buffer_lazy:
        raise ValueError("xpool linear state requires the non-lazy extra_buffer radix strategy")
    if any(
        (
            config.enable_int8_mamba_checkpoint,
            config.enable_linear_replayssm,
            config.enable_linear_replayssm_spec,
            config.enable_mamba_cache_stochastic_rounding,
            get_memory().enable_page_major_kv_layout,
            get_memory().enable_unified_memory,
        )
    ):
        raise ValueError("xpool linear state option does not support the selected state encoding or layout")


def state_layouts(
    params: Mamba2CacheParams, slots: int, layers: int, granularity: int
) -> tuple[StateBufferLayout, ...]:
    descriptions: list[tuple[torch.dtype, tuple[int, ...]]] = [
        (params.dtype.conv, (layers, slots + 1, *shape)) for shape in params.shape.conv
    ]
    descriptions.append((params.dtype.temporal, (layers, slots + 1, *params.shape.temporal)))
    result = []
    for dtype, shape in descriptions:
        stride = 1
        strides = []
        for size in reversed(shape):
            strides.append(stride)
            stride *= size
        result.append(StateBufferLayout(dtype, shape, tuple(reversed(strides)), 1, granularity))
    return tuple(result)


class ElasticMambaPool(MambaPool):
    """Use prepared stable views without eagerly touching the virtual reservation."""

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: list[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: int | None = None,
        speculative_eagle_topk: int | None = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
    ) -> None:
        backing = prepared_state_backing.get()
        if backing is None:
            raise RuntimeError("xpool linear state requires its prepared backing")
        # The pinned factory passes request capacity and top-k metadata even
        # when speculation is disabled. Only an actual draft allocation would
        # introduce another state geometry.
        ensure_uint64(size, spec_state_size)
        if (
            speculative_num_draft_tokens is not None
            or enable_linear_replayssm
            or enable_linear_replayssm_spec
            or envelope_layout
        ):
            raise ValueError("xpool linear state option requires ordinary non-replay state")
        if not isinstance(cache_params, Mamba2CacheParams):
            raise TypeError("xpool linear state requires the supported cache parameter shape")
        expected = state_layouts(cache_params, size, len(mamba_layer_ids), backing.layouts[0].mapping_granularity_bytes)
        if expected != backing.layouts:
            raise ValueError("xpool linear state factory differs from its prepared geometry")
        self.backing = backing
        self.size = size
        self.device = device
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.num_mamba_layers = len(mamba_layer_ids)
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.enable_linear_replayssm = False
        self.enable_linear_replayssm_spec = False
        self.linear_replayssm_cache_len = linear_replayssm_cache_len
        self.replayssm_spec_fold = False
        self.replayssm_is_kda = False
        self.replayssm_write_pos = None
        self.replayssm_spec_write_pos = None
        self.replayssm_cache_base = None
        self.replayssm_is_flush = None
        self.debug_memory_pool = envs.SGLANG_DEBUG_MEMORY_POOL.get()
        self.conv_shard_groups = cache_params.shape.conv_shard_groups
        self.conv_slice_axis = 0
        self.mamba_cache = self.State(conv=list(backing.tensors[:-1]), temporal=backing.tensors[-1])
        self.clear_slots(torch.tensor([0], dtype=torch.int64, device=device))

    @property
    def mem_usage(self) -> float:
        return self.backing.backed_bytes / (1 << 30)

    def close(self) -> None:
        self.mamba_cache = self.State(conv=[], temporal=torch.empty(0))
        self.backing.close()


class ElasticMambaSlotAllocator(MambaSlotAllocator):
    """Deliver only fully resident slots, retaining allocation epochs for rollback."""

    def __init__(self, size: int, device: str, backing: StateVmmBacking) -> None:
        self.backing = backing
        self.owned: dict[int, int] = {}
        self.epoch = 0
        self.retiring: set[int] = set()
        self.unit_granules = tuple(frozenset(backing.coverage((slot,))) for slot in range(size + 1))
        super().__init__(size, device)

    def clear(self) -> None:
        if self.retiring:
            raise RuntimeError("xpool cannot reset linear state during physical retirement")
        super().clear()
        self.owned.clear()
        self.publish_resident()

    def publish_resident(self) -> None:
        free = tuple(slot for slot in range(1, self.size + 1) if slot not in self.owned and slot not in self.retiring)
        resident = [slot for slot in free if self.unit_granules[slot].issubset(self.backing.mappings)]
        self.free_slots = torch.tensor(resident, dtype=torch.int64, device=self.device)

    def plan_slots(self, count: int) -> tuple[int, ...] | None:
        ensure_uint64(count)
        candidates = [slot for slot in range(1, self.size + 1) if slot not in self.owned and slot not in self.retiring]
        if count > len(candidates):
            return None
        covered = set(self.backing.mappings)
        selected = []
        for _ in range(count):
            slot = min(
                candidates,
                key=lambda unit: (
                    sum(
                        self.backing.layouts[buffer].mapping_granularity_bytes
                        for buffer, granule in self.unit_granules[unit].difference(covered)
                    ),
                    unit,
                ),
            )
            selected.append(slot)
            covered.update(self.unit_granules[slot])
            candidates.remove(slot)
        return tuple(selected)

    def _do_alloc(self, need_size: int) -> torch.Tensor | None:
        ensure_uint64(need_size)
        result = super()._do_alloc(need_size)
        if result is not None:
            for slot in result.tolist():
                self.epoch += 1
                self.owned[slot] = self.epoch
        return result

    def free(self, free_index: torch.Tensor) -> None:
        slots = free_index.tolist()
        if len(set(slots)) != len(slots) or any(slot not in self.owned for slot in slots):
            raise RuntimeError("xpool linear state slot was released without ownership")
        for slot in slots:
            del self.owned[slot]
        super().free(free_index)

    def alloc_group_begin(self, num_reqs: int) -> None:
        # Opportunistic queue-wide preallocation would hide the resident slots
        # from joint quota planning. Individual acquisitions retain exact epochs.
        self._alloc_iter = None

    def alloc_group_end(self) -> None:
        self._alloc_iter = None


class ElasticHybridReqToTokenPool(HybridReqToTokenPool):
    """Select elastic state only inside the model-owned construction scope."""

    mamba_pool_cls: type[MambaPool] = MambaPool
    state_adapter: GdnStateAdapter | None = None

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: list[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int | None = None,
        speculative_eagle_topk: int | None = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        short_conv_layer_ids: list[int] | None = None,
        short_conv_state_shape: tuple[int, int] | None = None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
    ) -> None:
        backing = prepared_state_backing.get()
        if backing is not None:
            if short_conv_layer_ids or short_conv_state_shape is not None or ngram_context_len:
                raise ValueError("xpool linear state option does not own auxiliary slot-state buffers")
            self.mamba_pool_cls = ElasticMambaPool
        super()._init_mamba_pool(
            mamba_size,
            mamba_spec_state_size,
            cache_params,
            mamba_layer_ids,
            device,
            enable_mamba_extra_buffer,
            # Upstream annotates this nullable default as int.
            cast(int, speculative_num_draft_tokens),
            speculative_eagle_topk,
            enable_linear_replayssm,
            linear_replayssm_cache_len,
            mamba_envelope_layout,
            enable_linear_replayssm_spec,
            short_conv_layer_ids,
            short_conv_state_shape,
            ngram_context_len,
            ngram_eos_token_id,
        )
        if backing is not None:
            self.mamba_allocator = ElasticMambaSlotAllocator(mamba_size, device, backing)

    def free_mamba_cache(self, req: Req, mamba_ping_pong_track_buffer_to_keep: int | None = None) -> None:
        if self.state_adapter is not None:
            lease = self.state_adapter.leases.pop(req, None)
            if lease is not None:
                self.mamba_allocator.free(lease)
            self.state_adapter.sources.pop(req, None)
        super().free_mamba_cache(req, mamba_ping_pong_track_buffer_to_keep)


@dataclass(slots=True)
class GdnStateAdapter:
    """Compose ordinary KV ownership with the concrete linear-state option."""

    kv: KvStateAdapter
    request_pool: ElasticHybridReqToTokenPool
    root_pool: HybridLinearKVPool
    slots: ElasticMambaSlotAllocator
    sources: dict[Req, torch.Tensor] = field(default_factory=dict)
    leases: dict[Req, torch.Tensor] = field(default_factory=dict)

    @property
    def backing(self) -> StateVmmBacking:
        return self.slots.backing

    @property
    def quota(self) -> ecs.MemoryQuota:
        return self.kv.quota

    @property
    def pool_count(self) -> int:
        return self.kv.pool_count + 1

    @property
    def profile(self) -> ecs.ElasticContextStateProfile:
        return ecs.ElasticContextStateProfile(self.kv.profile.pools + self.backing.profiles)

    @classmethod
    def reservation_budget(cls, configurator: KVCacheConfigurator, budget_bytes: int) -> int:
        return int(configurator._handle_max_mamba_cache(budget_bytes / (1 << 30)) * (1 << 30))

    @staticmethod
    def describe_layouts(configurator: KVCacheConfigurator, sizes: _PoolSizes):
        # The generic KV helper accepts the actual Full-layer count; recurrent
        # geometry never enters its configuration or storage implementation.
        if configurator.mambaish_config is None:
            raise ValueError("xpool linear state requires hybrid layer geometry")
        return KvStateAdapter.describe_layouts(
            configurator, sizes, layer_counts=(len(configurator.mambaish_config.full_attention_layer_ids),)
        )

    @classmethod
    def initialize_pools(
        cls,
        original_fn: Callable[..., _InitializedPools],
        configurator: KVCacheConfigurator,
        *,
        sizes: _PoolSizes,
        req_to_token_pool: ReqToTokenPool | None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None,
    ) -> _InitializedPools:
        validate_settings()
        kv_layouts = cls.describe_layouts(configurator, sizes)
        config = configurator.mambaish_config
        if config is None or get_schedule().max_mamba_cache_size is None:
            raise RuntimeError("xpool linear state sizing did not resolve its slot limit")
        precision = config.mamba2_cache_params.dtype
        if precision.conv is not torch.bfloat16 or precision.temporal is not torch.float32:
            raise ValueError("xpool linear state requires BF16 convolution and FP32 matrix state")
        device = torch.device(configurator.device)
        if device.index is None:
            device = torch.device(device.type, configurator.gpu_id)
        layouts = state_layouts(
            config.mamba2_cache_params,
            get_schedule().max_mamba_cache_size,
            len(config.mamba2_cache_params.layers),
            StateVmmBacking.granularity(device),
        )
        pinned_cost = sum(len(layout.coverage((0,))) * layout.mapping_granularity_bytes for layout in layouts)
        quota = ecs.MemoryQuota(
            sum(layout.profile.bytes_for(layout.profile.floor_bundles) for layout in kv_layouts) + pinned_cost
        )
        kv_backings: list[KvVmmBacking] = []
        state: StateVmmBacking | None = None
        try:
            for layout in kv_layouts:
                kv_backings.append(KvVmmBacking(layout, quota))
            state = StateVmmBacking(layouts, quota, device=device)
            token = prepared_state_backing.set(state)
            try:
                with prepared_backing_scope(tuple(kv_backings)):
                    return original_fn(
                        configurator,
                        sizes=sizes,
                        req_to_token_pool=req_to_token_pool,
                        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                    )
            finally:
                prepared_state_backing.reset(token)
        except BaseException as error:
            if not is_terminal_device_error(error):
                torch.cuda.synchronize(device)
                if state is not None:
                    state.close()
                for backing in reversed(kv_backings):
                    backing.close()
            raise

    @classmethod
    def create(
        cls,
        pool: KVCache,
        allocator: BaseTokenToKVPoolAllocator | None,
        request_pool: ReqToTokenPool | None,
        *,
        configurator: KVCacheConfigurator,
    ) -> GdnStateAdapter:
        if not isinstance(pool, HybridLinearKVPool) or not isinstance(request_pool, ElasticHybridReqToTokenPool):
            raise TypeError("xpool linear state requires the real hybrid pool composition")
        slots = request_pool.mamba_allocator
        if not isinstance(slots, ElasticMambaSlotAllocator):
            raise TypeError("xpool linear state requires its elastic slot allocator")
        kv = KvStateAdapter.create(pool.full_kv_pool, allocator, request_pool, configurator=configurator)
        if kv.quota is not slots.backing.quota:
            raise ValueError("xpool hybrid state pools must share one quota")
        result = cls(kv, request_pool, pool, slots)
        request_pool.state_adapter = result
        return result

    def bind_cache(self, cache: BasePrefixCache) -> None:
        self.kv.bind_cache(cache)

    def logical_free_units(self) -> tuple[int, ...]:
        return (*self.kv.logical_free_units(), self.slots.size - len(self.slots.owned) - len(self.slots.retiring))

    def resident_free_units(self) -> tuple[int, ...]:
        return (*self.kv.resident_free_units(), self.slots.available_size())

    def eviction_targets(self, params: EvictParams) -> tuple[int, ...]:
        return (
            *self.kv.eviction_targets(params),
            self.slots.available_size() + params.mamba_num if params.mamba_num else 0,
        )

    def allocation_bytes(self, counts: tuple[int, ...]) -> int | None:
        kv_bytes = self.kv.allocation_bytes(counts[:-1])
        slots = self.slots.plan_slots(counts[-1])
        if kv_bytes is None or slots is None:
            return None
        return kv_bytes + self.backing.missing_bytes(slots)

    def required_bytes(self, requirement: SglangMemoryRequirement) -> int | None:
        counts = requirement.forecast_units
        cache = self.kv.cache
        available_cache = 0 if cache is None else cache.mamba_evictable_size()
        slots = self.slots.plan_slots(max(0, counts[-1] - available_cache))
        kv_requirement = SglangMemoryRequirement(counts[:-1], requirement.allocation_units[:-1])
        kv_bytes = self.kv.required_bytes(kv_requirement)
        return None if kv_bytes is None or slots is None else kv_bytes + self.backing.missing_bytes(slots)

    def prepare_allocation(self, counts: tuple[int, ...]) -> bool:
        slots = self.slots.plan_slots(counts[-1])
        required = self.allocation_bytes(counts)
        if slots is None or required is None or required > self.quota.limit_bytes:
            return False
        if not self.kv.prepare_allocation(counts[:-1]) or not self.backing.try_map(self.backing.coverage(slots)):
            return False
        self.slots.publish_resident()
        return self.slots.available_size() >= counts[-1]

    def close(self) -> None:
        self.kv.close()
        cast(ElasticMambaPool, self.request_pool.mamba_pool).close()

    def slots_needed(self, requests: Iterable[Req]) -> int:
        return sum(
            int(not req.kv.holds_mamba)
            + (
                self.request_pool.mamba_ping_pong_track_buffer_size
                if req.kv.mamba_ping_pong_track_buffer is None
                else 0
            )
            + int(req not in self.leases)
            for req in requests
        )

    def prefill_budget(
        self, adder: PrefillAdder, kind: Literal["total", "current", "window"], free: tuple[int, ...]
    ) -> float:
        return self.kv.prefill_budget(adder, kind, free[:-1])

    def prefill_proposal(
        self, adder: PrefillAdder, request: Req, admission: _PrefillAdmission, *, continuation: bool
    ) -> SglangPrefillProposal:
        base = self.kv.prefill_proposal(adder, request, admission, continuation=continuation)
        required = self.slots_needed((*adder.can_run_list, request))
        return SglangPrefillProposal(
            admission,
            SglangMemoryRequirement((*base.memory.forecast_units, required), (*base.memory.allocation_units, required)),
            (*base.preparation_units, required),
        )

    def commit_prefill(self, request: Req) -> None:
        if request not in self.leases:
            lease = self.slots.alloc(1)
            if lease is None:
                raise RuntimeError("xpool prepared checkpoint replacement was unavailable")
            self.leases[request] = lease
        source = self.sources.pop(request, None)
        if source is not None:
            if not request.kv.holds_mamba:
                slot = self.slots.alloc(1)
                if slot is None:
                    raise RuntimeError("xpool prepared private state was unavailable")
                request.kv.mamba_pool_idx = slot[0]
            request.kv.mamba_cow_src_index = source
            request.kv.mamba_needs_clear = False

    def before_allocation(self, batch: ScheduleBatch, *, token_per_req: int | None) -> None:
        if not self.prepare_allocation(self.allocation_counts(batch, token_per_req=token_per_req)):
            raise SglangCapacityMiss("xpool hybrid state cannot back the selected batch")
        for request in batch.reqs:
            self.commit_prefill(request)

    def allocation_counts(self, batch: ScheduleBatch, *, token_per_req: int | None) -> tuple[int, ...]:
        return (*self.kv.allocation_counts(batch, token_per_req=token_per_req), self.slots_needed(batch.reqs))

    def decode_requirement(
        self, batch: ScheduleBatch, selected_indices: list[int] | None = None
    ) -> SglangMemoryRequirement:
        base = self.kv.decode_requirement(batch, selected_indices)
        requests = batch.reqs if selected_indices is None else [batch.reqs[index] for index in selected_indices]
        slots = self.slots_needed(requests)
        return SglangMemoryRequirement((*base.forecast_units, slots), (*base.allocation_units, slots))

    def stage_decode(self, batch: ScheduleBatch) -> ScheduleBatch:
        return replace(
            self.kv.stage_decode(batch),
            mamba_cow_src_indices=None,
            mamba_cow_dst_indices=None,
            mamba_clear_indices=None,
        )

    def finish_decode(self, batch: ScheduleBatch) -> None:
        # Preserve the pinned engine's tracking metadata after the common
        # allocation verdict and sequence-counter updates.
        interval = mamba_track_grid(batch.tree_cache.page_size)
        if batch.reqs:
            set_mamba_track_indices_from_reqs(batch)
        else:
            batch.mamba_track_indices = torch.empty(0, dtype=torch.int64, device=batch.device)
            batch.mamba_track_buffer_indices = []
        remainders = batch.seq_lens_cpu % interval
        mask = remainders == 0
        batch.mamba_track_mask_cpu = mask.tolist()
        batch.mamba_track_mask_next_cpu = (remainders == interval - 1).tolist() if batch.enable_overlap else None
        batch.mamba_decode_batch_idx_cpu = [req.decode_batch_idx for req in batch.reqs]
        batch.mamba_track_mask = mask.pin_memory().to(device=batch.device, non_blocking=True)

    def allocation(self, scheduler: Scheduler | None = None) -> SglangAllocation:
        result = self.kv.allocation(scheduler)
        result.receipts.append(GdnAllocationReceipt(self, self.slots.epoch, dict(self.leases)))
        return result

    def reference_config(self, ceiling_bytes: int) -> MemoryPoolConfig:
        # Price a legal request, including its insertion obligation. Virtual
        # capacity is an address limit; charging its entire reservation here
        # would impose a fixed physical split and defeat joint admission.
        request_slots = 2 + self.request_pool.mamba_ping_pong_track_buffer_size
        if self.slots.size < request_slots:
            raise MemoryError("xpool linear state capacity cannot hold a request and its checkpoint obligation")
        units = tuple(range(request_slots + 1))
        state_bytes = sum(
            len(layout.coverage(units)) * layout.mapping_granularity_bytes for layout in self.backing.layouts
        )
        if ceiling_bytes < state_bytes + self.kv.profile.floor_bytes:
            raise MemoryError("xpool hybrid service ceiling cannot hold a legal request")
        return self.kv.reference_config(ceiling_bytes - state_bytes)

    def pool_stats(self, result: PoolStats, observer: SchedulerPoolStatsObserver) -> PoolStats:
        result = self.kv.pool_stats(result, observer)
        used = len(self.slots.owned) - (result.mamba_evictable_size or 0)
        return replace(result, mamba_num_used=used, mamba_usage=used / self.slots.size)

    def check_pool(
        self,
        original_fn: Callable[[SchedulerInvariantChecker, PoolStats, int], tuple[bool, str]],
        checker: SchedulerInvariantChecker,
        ps: PoolStats,
        uncached: int,
        *,
        window: bool,
    ) -> tuple[bool, str]:
        return self.kv.check_pool(original_fn, checker, ps, uncached, window=window)

    def select_cache_eviction(self, shortages: tuple[bool, ...], *, release_backing: bool) -> StateCacheProposal | None:
        if self.kv.cache is None:
            raise RuntimeError("xpool linear cache is not bound")
        if shortages[-1] or release_backing:
            selected = GdnCacheEviction.select_state(self.kv.cache)
            if selected is not None:
                return selected.proposal(self.kv.cache, self.cache_pools)
        selected = self.kv.select_cache_operation(shortages[:-1], release_backing=release_backing)
        return None if selected is None else selected.proposal(self.kv.cache, self.cache_pools)

    @property
    def cache_pools(self) -> tuple[tuple[ComponentType, int], ...]:
        return (*self.kv.cache_pools, (ComponentType.MAMBA, 1))

    def prepare_reclaim(
        self, release_bytes: int, *, cache_eviction: StateCacheProposal | None = None
    ) -> GdnReclaimPlan | None:
        if cache_eviction is None:
            kv_plan = self.kv.prepare_reclaim(release_bytes)
        else:
            cache = self.kv.cache
            if cache is None or cache_eviction.pool_index >= len(self.cache_pools):
                return None
            ct = self.cache_pools[cache_eviction.pool_index][0]
            cls = GdnCacheEviction if ct == ComponentType.MAMBA else SglangCacheEviction
            operation = cls(cache_eviction.key.radix_key(), ct)
            node = operation.resolve(cache)
            if node is None or operation.proposal(cache, self.cache_pools) != cache_eviction:
                return None
            kv_plan = SglangReclaimPlan(release_bytes, operation, node, ((),) * self.kv.pool_count)
        granules = self.backing.empty_granules(tuple(self.slots.owned))
        if kv_plan is None and (release_bytes == 0 or not granules):
            return None
        return GdnReclaimPlan(kv_plan, release_bytes)

    def apply_reclaim(self, plan: StateReclaimPlan) -> None:
        concrete = cast(GdnReclaimPlan, plan)
        if concrete.applied:
            raise RuntimeError("xpool linear state reclaim operation was already applied")
        concrete.applied = True
        if concrete.kv is not None:
            self.kv.apply_reclaim(concrete.kv)
        if concrete.release_bytes:
            empty = self.backing.empty_granules(tuple(self.slots.owned))
            cost = 0
            selected = []
            for granule in empty:
                selected.append(granule)
                cost += self.backing.layouts[granule[0]].mapping_granularity_bytes
                if cost >= concrete.release_bytes:
                    break
            concrete.granules = tuple(selected)
            selected_set = set(selected)
            self.slots.retiring = {
                slot
                for slot in range(1, self.slots.size + 1)
                if selected_set.intersection(self.backing.coverage((slot,)))
            }
            self.slots.publish_resident()
            if selected:
                concrete.event = torch.cuda.Event()
                stream = self.kv.configurator.forward_stream
                # Recurrent execution joins its attention-side auxiliary
                # streams before returning. Join scheduler-side state copies
                # as well when that caller uses another current stream.
                stream.wait_stream(torch.cuda.current_stream(self.backing.device))
                concrete.event.record(stream)
        concrete.completed = concrete.event is None and (concrete.kv is None or concrete.kv.completed)

    def finish_retirement(self, plan: StateReclaimPlan) -> int:
        concrete = cast(GdnReclaimPlan, plan)
        if not concrete.applied:
            raise RuntimeError("xpool linear state cannot retire an unapplied reclaim operation")
        if concrete.completed:
            return 0
        released = self.kv.finish_retirement(concrete.kv) if concrete.kv is not None else 0
        if concrete.event is not None and concrete.event.query():
            released += self.backing.unmap(concrete.granules)
            concrete.event = None
            self.slots.retiring.clear()
            self.slots.publish_resident()
        concrete.completed = concrete.event is None and (concrete.kv is None or concrete.kv.completed)
        return released


@dataclass(slots=True)
class GdnReclaimPlan:
    kv: SglangReclaimPlan | None
    release_bytes: int
    granules: tuple[tuple[int, int], ...] = ()
    event: torch.cuda.Event | None = None
    applied: bool = False
    completed: bool = False


@dataclass(slots=True)
class GdnAllocationReceipt:
    adapter: GdnStateAdapter
    epoch: int
    leases: dict[Req, torch.Tensor]
    requests: dict[Req, ReqKvInfo] = field(default_factory=dict)

    def watch(self, requests: Iterable[Req]) -> None:
        for req in requests:
            if req not in self.requests:
                state = replace(req.kv)
                if state.mamba_ping_pong_track_buffer is not None:
                    state.mamba_ping_pong_track_buffer = state.mamba_ping_pong_track_buffer.clone()
                self.requests[req] = state

    def commit(self) -> None:
        self.adapter.sources.clear()
        self.requests.clear()

    def rollback(self) -> None:
        acquired = tuple(slot for slot, epoch in self.adapter.slots.owned.items() if epoch > self.epoch)
        if acquired:
            self.adapter.slots.free(torch.tensor(acquired, dtype=torch.int64, device=self.adapter.slots.device))
        for req, state in self.requests.items():
            row = req.kv.req_pool_idx
            if row is not None:
                self.adapter.request_pool.req_index_to_mamba_index_mapping[row] = (
                    0 if state.mamba_pool_idx is None else state.mamba_pool_idx
                )
                mapping = self.adapter.request_pool.req_index_to_mamba_ping_pong_track_buffer_mapping
                mapping[row] = 0 if state.mamba_ping_pong_track_buffer is None else state.mamba_ping_pong_track_buffer
            req.kv.mamba_pool_idx = state.mamba_pool_idx
            req.kv.mamba_cow_src_index = state.mamba_cow_src_index
            req.kv.mamba_needs_clear = state.mamba_needs_clear
            req.kv.mamba_ping_pong_track_buffer = state.mamba_ping_pong_track_buffer
            req.kv.mamba_next_track_idx = state.mamba_next_track_idx
            req.kv.mamba_last_track_idx = state.mamba_last_track_idx
            req.kv.mamba_last_track_seqlen = state.mamba_last_track_seqlen
        self.adapter.leases = dict(self.leases)
        self.adapter.sources.clear()
        self.requests.clear()


@dataclass(frozen=True, slots=True)
class GdnCacheEviction(SglangCacheEviction):
    @classmethod
    def select_state(cls, cache: UnifiedRadixCache) -> SglangCacheEviction | None:
        owner = cache.components.get(ComponentType.MAMBA)
        if not isinstance(owner, MambaComponent):
            return None
        owner.evict_device_start(1)
        try:
            node = owner._evict_device_cursor
        finally:
            owner.evict_device_end()
        if node is None:
            return None
        tree = cast(UnifiedTreeCore, cache.tree_core)
        if node in tree.evictable_device_leaves:
            return SglangCacheEviction.from_node(node, ComponentType.FULL)
        return cls.from_node(node, ComponentType.MAMBA)

    def resolve(self, cache: UnifiedRadixCache) -> UnifiedTreeNode | None:
        tree = cast(UnifiedTreeCore, cache.tree_core)
        matched, node_id, pinned = tree.match_full_device_prefix(self.key)
        if matched != pinned or matched != len(self.key):
            return None
        node = tree.node_by_id(node_id)
        data = node.component_data[ComponentType.MAMBA]
        owner = tree.components_by_type[ComponentType.MAMBA]
        if data.value is None or data.lock_ref:
            return None
        if tree._should_cascade_evict_component(
            node,
            owner,
            tree.components_by_type[ComponentType.FULL],
            EvictLayer.DEVICE,
            tree._is_cascade_evict_leaf(node, EvictLayer.DEVICE),
        ):
            return None
        return node

    def apply(self, cache: UnifiedRadixCache, node: UnifiedTreeNode) -> None:
        tree = cast(UnifiedTreeCore, cache.tree_core)
        tracker: dict[ComponentType, int] = defaultdict(int)
        device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        trigger = tree.components_by_type[ComponentType.MAMBA]
        tree._evict_component_and_detach_lru(
            node,
            trigger,
            target=EvictLayer.DEVICE,
            tracker=tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        tree._cascade_evict(node, trigger, tracker, device_frees=device_frees, host_frees=host_frees)
        cache._drain_device_frees(device_frees)
        cache._drain_host_frees(host_frees)


def adapter_of(cache: BasePrefixCache) -> GdnStateAdapter | None:
    pool = cache.req_to_token_pool
    return pool.state_adapter if isinstance(pool, ElasticHybridReqToTokenPool) else None


def around_build_allocator(
    original_fn: Callable[..., BaseTokenToKVPoolAllocator],
    configurator: KVCacheConfigurator,
    *,
    sizes: _PoolSizes,
    token_to_kv_pool: KVCache,
    is_dsv4_model: bool,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None,
) -> BaseTokenToKVPoolAllocator:
    elastic = prepared_state_backing.get() is not None and isinstance(token_to_kv_pool, HybridLinearKVPool)
    result = original_fn(
        configurator,
        sizes=sizes,
        token_to_kv_pool=token_to_kv_pool.full_kv_pool if elastic else token_to_kv_pool,
        is_dsv4_model=is_dsv4_model,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
    )
    if elastic:
        # The allocator derives backing from the Full child, but callers still
        # borrow the actual hybrid cache for recurrent execution.
        result._kvcache = token_to_kv_pool
    return result


def around_finalize_match(
    original_fn: Callable[[MambaComponent, MatchPrefixParams, MatchResult], MatchResult],
    component: MambaComponent,
    params: MatchPrefixParams,
    result: MatchResult,
) -> MatchResult:
    adapter = adapter_of(component.cache)
    if adapter is None or not params.cow_mamba or params.req is None:
        return original_fn(component, params, result)
    # This is the engine's normal committed match, not a pure preflight lookup.
    # Defer only private slot acquisition until joint admission has prepared it.
    result = original_fn(component, replace(params, cow_mamba=False), result)
    request = params.req
    adapter.sources.pop(request, None)
    source = cast(UnifiedTreeCore, component.tree_core).get_component_device_value(
        result.best_match_node, ComponentType.MAMBA
    )
    if source is not None:
        adapter.sources[request] = source
    return result


def around_prepare_checkpoint(
    original_fn: Callable[[MambaComponent, Req, InsertParams, int, bool], int | None],
    component: MambaComponent,
    req: Req,
    insert_params: InsertParams,
    token_ids_len: int,
    is_finished: bool,
) -> int | None:
    if adapter_of(component.cache) is None:
        return original_fn(component, req, insert_params, token_ids_len, is_finished)
    token = checkpoint_request.set(req)
    try:
        return original_fn(component, req, insert_params, token_ids_len, is_finished)
    finally:
        checkpoint_request.reset(token)


def around_checkpoint_slot(
    original_fn: Callable[[MambaComponent], torch.Tensor],
    component: MambaComponent,
) -> torch.Tensor:
    adapter = adapter_of(component.cache)
    if adapter is None:
        return original_fn(component)
    request = checkpoint_request.get()
    lease = None if request is None else adapter.leases.pop(request, None)
    if lease is None:
        raise RuntimeError("xpool checkpoint insertion has no prepared replacement slot")
    return lease


def around_check_state_pool(
    original_fn: Callable[[SchedulerInvariantChecker, PoolStats], tuple[bool, str]],
    checker: SchedulerInvariantChecker,
    stats: PoolStats,
) -> tuple[bool, str]:
    adapter = adapter_of(checker.tree_cache)
    if adapter is None:
        return original_fn(checker, stats)
    unavailable = adapter.slots.size - len(adapter.slots.owned) - adapter.slots.available_size()
    available = cast(int, stats.mamba_available_size) + unavailable + len(adapter.leases)
    return original_fn(checker, replace(stats, mamba_available_size=available))


def state_hooks() -> tuple[SglangHook, ...]:
    return (
        SglangHook(
            "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool_allocator",
            around_build_allocator,
            HookType.AROUND,
        ),
        SglangHook(
            "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool", ElasticHybridReqToTokenPool, HookType.REPLACE
        ),
        SglangHook(
            "sglang.srt.mem_cache.unified_cache.components.mamba.MambaComponent.finalize_match_result_in_cache",
            around_finalize_match,
            HookType.AROUND,
        ),
        SglangHook(
            "sglang.srt.mem_cache.unified_cache.components.mamba.MambaComponent.prepare_for_caching_req",
            around_prepare_checkpoint,
            HookType.AROUND,
        ),
        SglangHook(
            "sglang.srt.mem_cache.unified_cache.components.mamba.MambaComponent._alloc_mamba_slot",
            around_checkpoint_slot,
            HookType.AROUND,
        ),
        SglangHook(
            "sglang.srt.managers.scheduler_components.invariant_checker.SchedulerInvariantChecker._check_mamba_pool",
            around_check_state_pool,
            HookType.AROUND,
        ),
    )
