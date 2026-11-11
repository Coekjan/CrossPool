"""Implementation-neutral resource demands and serving-state contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder, _PrefillAdmission
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import SchedulerInvariantChecker
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats, SchedulerPoolStatsObserver
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator, _InitializedPools, _PoolSizes
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

from xpool import ecs
from xpool.integrations.sglang.ecs.allocation import SglangAllocation
from xpool.integrations.sglang.ecs.radix import StateCacheProposal
from xpool.utils.integer import ensure_uint64

type StateResourceCounts = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SglangMemoryRequirement:
    """Forecast and immediate acquisitions, in each pool's native allocation unit."""

    forecast_units: tuple[int, ...]
    allocation_units: tuple[int, ...]

    def __post_init__(self) -> None:
        ensure_uint64(*self.forecast_units, *self.allocation_units)
        if (
            not self.forecast_units
            or len(self.forecast_units) != len(self.allocation_units)
            or any(
                forecast < allocation
                for forecast, allocation in zip(self.forecast_units, self.allocation_units, strict=True)
            )
        ):
            raise ValueError("xpool elastic context state forecast must cover each immediate allocation")


@dataclass(frozen=True, slots=True)
class SglangPrefillProposal:
    """Pure selection with forecast, pre-lock residency and actual acquisitions."""

    admission: _PrefillAdmission
    memory: SglangMemoryRequirement
    preparation_units: tuple[int, ...]

    def __post_init__(self) -> None:
        ensure_uint64(*self.preparation_units)
        if len(self.preparation_units) != len(self.memory.forecast_units) or any(
            not allocation <= preparation <= forecast
            for forecast, preparation, allocation in zip(
                self.memory.forecast_units, self.preparation_units, self.memory.allocation_units, strict=True
            )
        ):
            raise ValueError("xpool Prefill preparation must cover acquisition within its forecast")


class StateReclaimPlan(Protocol):
    """Implementation-owned physical retirement and terminal observation."""

    completed: bool


class SglangStateAdapter(Protocol):
    """Rank-local state semantics behind shared byte-quota coordination.

    All mutations run on the scheduler thread. Logical IDs, cache policies and
    physical inventories remain in the selected implementation. Resource
    vectors retain one immutable order throughout the rank's lifetime.
    """

    @classmethod
    def create(
        cls,
        pool: KVCache,
        allocator: BaseTokenToKVPoolAllocator | None,
        request_pool: ReqToTokenPool | None,
        *,
        configurator: KVCacheConfigurator,
    ) -> SglangStateAdapter: ...
    @classmethod
    def initialize_pools(
        cls,
        original_fn: Callable[..., _InitializedPools],
        configurator: KVCacheConfigurator,
        *,
        sizes: _PoolSizes,
        req_to_token_pool: ReqToTokenPool | None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None,
    ) -> _InitializedPools: ...
    @classmethod
    def reservation_budget(cls, configurator: KVCacheConfigurator, budget_bytes: int) -> int: ...
    @property
    def profile(self) -> ecs.ElasticContextStateProfile: ...
    @property
    def quota(self) -> ecs.MemoryQuota: ...
    @property
    def pool_count(self) -> int: ...
    def bind_cache(self, cache: BasePrefixCache) -> None: ...
    def logical_free_units(self) -> StateResourceCounts: ...
    def resident_free_units(self) -> StateResourceCounts: ...
    def eviction_targets(self, params: EvictParams) -> StateResourceCounts: ...
    def select_cache_eviction(
        self, shortages: tuple[bool, ...], *, release_backing: bool
    ) -> StateCacheProposal | None: ...
    def required_bytes(self, requirement: SglangMemoryRequirement) -> int | None: ...
    def allocation_counts(self, batch: ScheduleBatch, *, token_per_req: int | None) -> StateResourceCounts: ...
    def allocation_bytes(self, counts: StateResourceCounts) -> int | None: ...
    def prepare_allocation(self, counts: StateResourceCounts) -> bool:
        """Map and publish delivery capacity, with no logical acquisition or eviction.

        Return False for a clean shortage. Device/cleanup uncertainty raises
        with physical owners and their quota charge retained. No TP collective
        occurs inside this rank-local operation.
        """
        ...

    def allocation(self, scheduler: Scheduler | None = None) -> SglangAllocation: ...
    def reference_config(self, ceiling_bytes: int) -> MemoryPoolConfig:
        """Price stable engine limits under a byte ceiling without resizing pools."""
        ...

    def prepare_reclaim(
        self, release_bytes: int, *, cache_eviction: StateCacheProposal | None = None
    ) -> StateReclaimPlan | None: ...
    def apply_reclaim(self, plan: StateReclaimPlan) -> None:
        """Apply the resolved effect after all-ready, withdraw IDs and fence users."""
        ...

    def finish_retirement(self, plan: StateReclaimPlan) -> int:
        """Poll locally and refund confirmed physical release in bytes.

        Pending fences and completed receipts return zero. Failed releases
        retain unconfirmed owners and charge, and fail the generation.
        """
        ...

    def prefill_budget(
        self, adder: PrefillAdder, kind: Literal["total", "current", "window"], free: StateResourceCounts
    ) -> float: ...
    def prefill_proposal(
        self, adder: PrefillAdder, request: Req, admission: _PrefillAdmission, *, continuation: bool
    ) -> SglangPrefillProposal: ...
    def decode_requirement(
        self, batch: ScheduleBatch, selected_indices: list[int] | None = None
    ) -> SglangMemoryRequirement: ...
    def commit_prefill(self, request: Req) -> None:
        """Acquire prepared request transitions into the current watched journal."""
        ...

    def before_allocation(self, batch: ScheduleBatch, *, token_per_req: int | None) -> None: ...
    def stage_decode(self, batch: ScheduleBatch) -> ScheduleBatch: ...
    def finish_decode(self, batch: ScheduleBatch) -> None: ...
    def pool_stats(self, result: PoolStats, observer: SchedulerPoolStatsObserver) -> PoolStats: ...
    def check_pool(
        self,
        original_fn: Callable[[SchedulerInvariantChecker, PoolStats, int], tuple[bool, str]],
        checker: SchedulerInvariantChecker,
        ps: PoolStats,
        uncached: int,
        *,
        window: bool,
    ) -> tuple[bool, str]: ...
    def close(self) -> None: ...
