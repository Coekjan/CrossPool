"""SGLang's concrete Elastic Context State owners and joint physical-capacity calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.distributed
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import _PrefillAdmission
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator, _PoolSizes
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, KVCache, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore, UnifiedTreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig, create_memory_pool_configurator
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils.common import get_num_new_pages

from xpool import ecs
from xpool.integrations.sglang.ecs.allocation import SglangAllocation
from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
    SglangPagePlan,
)
from xpool.integrations.sglang.ecs.pool import ElasticMHATokenToKVPool, ElasticMLATokenToKVPool, backing_of
from xpool.integrations.sglang.ecs.radix import SglangCacheEviction
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking, KvVmmLayout
from xpool.utils import align_up
from xpool.utils.integer import ensure_uint64


@dataclass(frozen=True, slots=True)
class SglangMemoryRequirement:
    """Forecast and immediate acquisitions, in each real component's page unit."""

    forecast_pages: tuple[int, ...]
    allocation_pages: tuple[int, ...]

    def __post_init__(self) -> None:
        ensure_uint64(*self.forecast_pages, *self.allocation_pages)
        if len(self.forecast_pages) != len(self.allocation_pages) or any(
            forecast < allocation
            for forecast, allocation in zip(self.forecast_pages, self.allocation_pages, strict=True)
        ):
            raise ValueError("xpool elastic context state forecast must cover each immediate allocation")


@dataclass(frozen=True, slots=True)
class SglangPrefillProposal:
    """Pure selection with forecast, pre-lock residency and actual page acquisition."""

    admission: _PrefillAdmission
    memory: SglangMemoryRequirement
    preparation_pages: tuple[int, ...]

    def __post_init__(self) -> None:
        ensure_uint64(*self.preparation_pages)
        if len(self.preparation_pages) != len(self.memory.forecast_pages) or any(
            not allocation <= preparation <= forecast
            for forecast, preparation, allocation in zip(
                self.memory.forecast_pages, self.preparation_pages, self.memory.allocation_pages, strict=True
            )
        ):
            raise ValueError("xpool Prefill preparation must cover acquisition within its forecast")


@dataclass(slots=True)
class SglangReclaimPlan:
    """Prepared logical effect and actual local physical retirement ownership."""

    release_bytes: int
    cache_eviction: SglangCacheEviction | None
    cache_node: UnifiedTreeNode | None
    bundle_ids: tuple[tuple[int, ...], ...]
    event: torch.cuda.Event | None = None
    applied: bool = False
    completed: bool = False


@dataclass(frozen=True, slots=True)
class SglangKvComponent:
    """Associate a real KV allocator with its Radix cache component."""

    component: ComponentType
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator


@dataclass(slots=True)
class SglangElasticContextState:
    """Assemble actual engine owners without replacing its composite pool or cache.

    The runner creates this state after pool construction; the Scheduler binds
    its cache later. Every backing shares one physical permission ledger.
    """

    pool: KVCache
    allocator: BaseTokenToKVPoolAllocator
    request_pool: ReqToTokenPool
    configurator: KVCacheConfigurator
    components: tuple[SglangKvComponent, ...]
    cache: UnifiedRadixCache | None = field(init=False, default=None)

    @classmethod
    def create(
        cls,
        pool: KVCache,
        allocator: BaseTokenToKVPoolAllocator | None,
        request_pool: ReqToTokenPool | None,
        *,
        configurator: KVCacheConfigurator,
    ) -> SglangElasticContextState:
        """Check the factory's real component wiring before serving registration."""

        if request_pool is None:
            raise ValueError("xpool elastic context state request pool is unavailable after pool construction")
        if isinstance(pool, SWAKVPool) and isinstance(allocator, SWATokenToKVPoolAllocator):
            pools = (pool.full_kv_pool, pool.swa_kv_pool)
            children = (allocator.full_attn_allocator, allocator.swa_attn_allocator)
            identities = (ComponentType.FULL, ComponentType.SWA)
        elif isinstance(allocator, ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator):
            pools = (pool,)
            children = (allocator,)
            identities = (ComponentType.FULL,)
        else:
            raise TypeError("xpool elastic context state requires the supported elastic pool/allocator composition")
        components: list[SglangKvComponent] = []
        backings = tuple(backing_of(child) for child in pools)
        for identity, child, backing in zip(identities, children, backings, strict=True):
            if not isinstance(child, ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator):
                raise TypeError("xpool elastic context state composite requires elastic child allocators")
            if child.backing is not backing or child.page_size != backing.layout.token_page_size:
                raise ValueError("xpool elastic context state allocator differs from its component backing")
            if backing.quota is not backings[0].quota or backing.layout.device != backings[0].layout.device:
                raise ValueError("xpool elastic context state components must share one device and byte quota")
            components.append(SglangKvComponent(identity, child))
        return cls(pool, allocator, request_pool, configurator, tuple(components))

    @staticmethod
    def describe_layouts(configurator: KVCacheConfigurator, sizes: _PoolSizes) -> tuple[KvVmmLayout, ...]:
        """Describe the pinned factory's resolved MHA/MLA components before any mapping.

        Child constructors compare these descriptions with their actual buffer
        descriptors. Configuration resolution and original pool construction
        still run once, in the upstream owner.
        """

        model = configurator.model_config
        page_size = configurator.pool_page_size
        dtype = configurator.kv_cache_dtype
        storage_dtype = (
            torch.uint8 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz) else dtype
        )
        if configurator.is_hybrid_swa:
            if sizes.full_max_total_num_tokens is None or sizes.swa_max_total_num_tokens is None:
                raise ValueError("xpool elastic context state composite requires both resolved pool sizes")
            token_counts = (sizes.full_max_total_num_tokens, sizes.swa_max_total_num_tokens)
            layer_counts = (len(model.full_attention_layer_ids), len(model.swa_attention_layer_ids))
        else:
            token_counts = (sizes.max_total_num_tokens,)
            layer_counts = (configurator.layer_info.num_effective_layers,)
        layouts: list[KvVmmLayout] = []
        for index, (token_count, layer_count) in enumerate(zip(token_counts, layer_counts, strict=True)):
            if configurator.use_mla_backend:
                latent_dim = (
                    model.swa_kv_lora_rank + model.swa_qk_rope_head_dim
                    if index == 1
                    else model.kv_lora_rank + model.qk_rope_head_dim
                )
                shapes = ((token_count + page_size, 1, latent_dim),)
            else:
                head_count = model.get_num_kv_heads(get_parallel().attn_tp_size, get_parallel().attn_dcp_size)
                head_dim, value_dim = model.head_dim, model.v_head_dim
                if configurator.is_hybrid_swa:
                    # The ordinary hybrid factory defaults V to K unless it
                    # explicitly supplies compressed Full/SWA geometry.
                    value_dim = head_dim
                    if configurator.is_hybrid_swa_compress:
                        value_dim = model.v_head_dim
                        if index == 1:
                            head_count = max(
                                1, model.hf_text_config.swa_num_key_value_heads // get_parallel().attn_tp_size
                            )
                            head_dim, value_dim = model.swa_head_dim, model.swa_v_head_dim
                if head_dim is None or value_dim is None:
                    raise ValueError("xpool elastic context state requires resolved K/V head dimensions")
                shapes = (
                    (token_count + page_size, head_count, head_dim),
                    (token_count + page_size, head_count, value_dim),
                )
            descriptors = tuple(
                KvBufferDesc(
                    f"buffer{buffer_index}-{layer}",
                    shape,
                    row_bytes=shape[1] * shape[2] * storage_dtype.itemsize,
                    tokens_per_row=1,
                )
                for buffer_index, shape in enumerate(shapes)
                for layer in range(layer_count)
            )
            layouts.append(
                KvVmmBacking.describe(
                    device=torch.device(configurator.device),
                    storage_dtype=storage_dtype,
                    token_page_size=page_size,
                    descriptors=descriptors,
                    buffers_per_layer=len(shapes),
                )
            )
        return tuple(layouts)

    @property
    def profile(self) -> ecs.ElasticContextStateProfile:
        """Derive immutable physical bounds from the actual component descriptions."""

        return ecs.ElasticContextStateProfile(
            tuple(binding.allocator.backing.layout.profile for binding in self.components)
        )

    @property
    def quota(self) -> ecs.MemoryQuota:
        """Return the single ledger shared by the assembled component backings."""

        return self.components[0].allocator.backing.quota

    def bind_cache(self, cache: BasePrefixCache) -> None:
        """Bind the Scheduler's real Python TreeCore once, without mirroring its state."""

        if not isinstance(cache, UnifiedRadixCache) or not isinstance(cache.tree_core, UnifiedTreeCore):
            raise TypeError("xpool elastic context state requires UnifiedRadixCache with the Python UnifiedTreeCore")
        if cache.token_to_kv_pool_allocator is not self.allocator or cache.req_to_token_pool is not self.request_pool:
            raise ValueError("xpool elastic context state cache differs from its registered pool owners")
        if self.cache is not None and self.cache is not cache:
            raise RuntimeError("xpool elastic context state state already has a different cache")
        self.cache = cache

    def logical_free_pages(self) -> tuple[int, ...]:
        """Count unowned quanta, excluding deferred group frees and current retirement."""

        return tuple(
            len(binding.allocator.get_all_free_pages())
            + len(binding.allocator.unavailable_pages)
            - len(binding.allocator.retiring_pages)
            for binding in self.components
        )

    def resident_free_pages(self) -> tuple[int, ...]:
        """Count immediately deliverable free pages in component-profile order."""

        return tuple(binding.allocator.available_size() // binding.allocator.page_size for binding in self.components)

    def eviction_page_targets(self, params: EvictParams) -> tuple[int, ...]:
        """Convert engine eviction requests to desired resident-free page counts."""

        requested = {ComponentType.FULL: params.num_tokens, ComponentType.SWA: params.swa_num_tokens}
        return tuple(
            available + (requested[binding.component] + binding.allocator.page_size - 1) // binding.allocator.page_size
            if requested[binding.component] > 0
            else 0
            for binding, available in zip(self.components, self.resident_free_pages(), strict=True)
        )

    def select_cache_eviction(
        self, shortages: tuple[bool, ...], *, release_backing: bool
    ) -> SglangCacheEviction | None:
        """Select the first eligible component effect for the leader's common needs."""

        if self.cache is None:
            raise RuntimeError("xpool elastic context state cache reclamation requires a bound cache")
        for binding, shortage in zip(self.components, shortages, strict=True):
            if shortage or release_backing:
                candidate = SglangCacheEviction.select(self.cache, binding.component)
                if candidate is not None:
                    return candidate
        return None

    def required_bytes(self, requirement: SglangMemoryRequirement) -> int | None:
        """Price one joint forecast using resident holes and eligible same-pool cache."""

        if len(requirement.forecast_pages) != len(self.components):
            raise ValueError("xpool elastic context state requirement differs from its component count")
        required = self.quota.charged_bytes
        for binding, page_count in zip(self.components, requirement.forecast_pages, strict=True):
            allocator = binding.allocator
            if self.cache is not None:
                cached = self.cache.tree_core.component_evictable_size(binding.component) // allocator.page_size
                page_count = max(0, page_count - cached)
            plan = allocator.plan_pages(page_count)
            if plan is None:
                return None
            required += len(plan.missing_bundle_ids) * allocator.backing.layout.profile.bundle_bytes
        ensure_uint64(required)
        return required

    def allocation_pages(self, batch: ScheduleBatch, *, token_per_req: int | None) -> tuple[int, ...]:
        """Calculate actual paired acquisitions after committed window releases."""

        page_size = self.components[0].allocator.page_size
        if token_per_req is None:
            if batch.extend_num_tokens is None:
                raise ValueError("xpool elastic context state requires a prepared Extend batch")
            pages = (
                batch.extend_num_tokens
                if page_size == 1
                else cast(
                    int,
                    get_num_new_pages(
                        seq_lens=batch.seq_lens_cpu,
                        page_size=page_size,
                        prefix_lens=torch.tensor(batch.prefix_lens, dtype=torch.int64),
                    ),
                )
            )
        else:
            if token_per_req != 1:
                raise ValueError("xpool elastic context state Decode allocation requires one token per request")
            pages = (
                batch.seq_lens_cpu.numel()
                if page_size == 1
                else int(torch.count_nonzero(batch.seq_lens_cpu % page_size == 0).item())
            )
        return (pages,) * len(self.components)

    def plan_allocation(self, page_counts: tuple[int, ...]) -> tuple[SglangPagePlan, ...] | None:
        """Describe one exact joint realization without mapping or acquiring ownership."""

        ensure_uint64(*page_counts)
        if len(page_counts) != len(self.components):
            raise ValueError("xpool elastic context state allocation differs from its component count")
        plans = tuple(
            binding.allocator.plan_pages(count) for binding, count in zip(self.components, page_counts, strict=True)
        )
        if any(plan is None for plan in plans):
            return None
        return tuple(plan for plan in plans if plan is not None)

    def allocation_bytes(self, page_counts: tuple[int, ...]) -> int | None:
        """Price immediate acquisitions without treating cached owners as already free."""

        plans = self.plan_allocation(page_counts)
        if plans is None:
            return None
        return self.quota.charged_bytes + sum(
            len(plan.missing_bundle_ids) * binding.allocator.backing.layout.profile.bundle_bytes
            for binding, plan in zip(self.components, plans, strict=True)
        )

    def prepare_allocation(self, page_counts: tuple[int, ...]) -> bool:
        """Jointly back enough deliverable pages before any composite acquisition.

        A later clean miss retains successful mappings as charged reusable
        storage. Checked device errors propagate without publishing free IDs.
        """

        complete = self.plan_allocation(page_counts)
        if complete is None:
            return False
        cost = sum(
            len(plan.missing_bundle_ids) * binding.allocator.backing.layout.profile.bundle_bytes
            for binding, plan in zip(self.components, complete, strict=True)
        )
        if cost > self.quota.limit_bytes - self.quota.charged_bytes:
            return False
        for binding, plan in zip(self.components, complete, strict=True):
            if not binding.allocator.backing.try_map(plan.missing_bundle_ids):
                return False
        for binding in self.components:
            allocator = binding.allocator
            candidates = allocator.unavailable_pages[~torch.isin(allocator.unavailable_pages, allocator.retiring_pages)]
            allocator.publish_pages(candidates[allocator.backing.missing_bundle_counts(candidates) == 0])
        return all(available >= count for available, count in zip(self.resident_free_pages(), page_counts, strict=True))

    def allocation(self, scheduler: Scheduler | None = None) -> SglangAllocation:
        """Create one journal over the real Full owner and optional paired mapping."""

        if self.cache is None:
            raise RuntimeError("xpool elastic context state allocation requires the Scheduler's cache")
        mapping = (
            self.allocator.full_to_swa_index_mapping if isinstance(self.allocator, SWATokenToKVPoolAllocator) else None
        )
        return SglangAllocation(self.request_pool, self.cache, self.components[0].allocator, mapping, scheduler)

    def reference_bytes(self, config: MemoryPoolConfig) -> int | None:
        """Price a stable engine reference with exact dummy and VMM rounding costs."""

        tokens = (
            {ComponentType.FULL: config.full_max_total_num_tokens, ComponentType.SWA: config.swa_max_total_num_tokens}
            if isinstance(self.pool, SWAKVPool)
            else {ComponentType.FULL: config.max_total_num_tokens}
        )
        total = 0
        for binding in self.components:
            count = tokens[binding.component]
            layout = binding.allocator.backing.layout
            if count is None or not 0 < count <= layout.token_capacity:
                return None
            rows = (count + layout.token_page_size + layout.tokens_per_row - 1) // layout.tokens_per_row
            bundles = (
                align_up(rows * layout.row_bytes, layout.mapping_granularity_bytes) // layout.mapping_granularity_bytes
            )
            total += layout.profile.bytes_for(bundles)
        return total

    def reference_config(self, ceiling_bytes: int) -> MemoryPoolConfig:
        """Derive stable limits once from the immutable ceiling, never from live quota.

        The byte-budget entry may synchronize upstream constraints; run it once.
        The correction search uses only the configurator's local sizing policy,
        followed by one matched TP minimum of corresponding logical capacities.
        Request storage and capture concurrency remain the original owners'.
        """

        ensure_uint64(ceiling_bytes)
        if ceiling_bytes < self.profile.floor_bytes:
            raise MemoryError("xpool elastic context state service ceiling cannot hold its bootstrap floors")
        configurator = self.configurator
        reference = configurator.config_from_budget(
            ceiling_bytes, cap_tokens=self.components[0].allocator.backing.layout.token_capacity
        )
        policy = create_memory_pool_configurator(configurator)
        page_size = self.components[0].allocator.page_size
        minimum_pages = 1
        if isinstance(self.pool, SWAKVPool):
            ratio = get_schedule().swa_full_tokens_ratio
            window = configurator.sliding_window_size
            minimum_swa_pages = 1 if window is None else window // page_size + 2
            minimum_pages = max(1, math.ceil(minimum_swa_pages / ratio))
            # Match the upstream float multiplication and page-floor exactly.
            while int(minimum_pages * page_size * ratio) // page_size < minimum_swa_pages:
                minimum_pages += 1
        lower, upper = minimum_pages, reference.max_total_num_tokens // page_size
        if lower > upper:
            raise MemoryError("xpool elastic context state service ceiling cannot hold a legal engine reference")
        best: MemoryPoolConfig | None = None
        while lower <= upper:
            middle = (lower + upper) // 2
            candidate = policy.calculate_pool_sizes_from_max_tokens(middle * page_size, page_size)
            cost = self.reference_bytes(candidate)
            if cost is not None and cost <= ceiling_bytes:
                best = candidate
                lower = middle + 1
            else:
                upper = middle - 1
        if best is None:
            raise MemoryError("xpool elastic context state service ceiling cannot hold a legal engine reference")
        capacities = torch.tensor(
            [best.max_total_num_tokens, best.full_max_total_num_tokens or 0, best.swa_max_total_num_tokens or 0],
            dtype=torch.int64,
        )
        parallel = get_parallel()
        if parallel.attn_tp_size > 1:
            group = parallel.attn_tp_group.cpu_group if parallel.enable_dp_attention else parallel.tp_group.cpu_group
            torch.distributed.all_reduce(capacities, op=torch.distributed.ReduceOp.MIN, group=group)
        total_tokens, full_tokens, swa_tokens = capacities.tolist()
        return MemoryPoolConfig(
            max_total_num_tokens=total_tokens,
            max_running_requests=self.request_pool.size,
            full_max_total_num_tokens=full_tokens if isinstance(self.pool, SWAKVPool) else None,
            swa_max_total_num_tokens=swa_tokens if isinstance(self.pool, SWAKVPool) else None,
        )

    def empty_bundle_ids(self, release_bytes: int) -> tuple[tuple[int, ...], ...]:
        """Select stable unowned physical bundles, never crediting deferred frees."""

        if release_bytes == 0:
            return ((),) * len(self.components)
        remaining = release_bytes
        selected: list[tuple[int, ...]] = []
        for binding in self.components:
            allocator = binding.allocator
            backing = allocator.backing
            free = torch.cat((allocator.get_all_free_pages(), allocator.unavailable_pages))
            free = free[~torch.isin(free, allocator.retiring_pages)]
            empty = torch.sort(backing.empty_bundles(free)).values
            count = (remaining + backing.layout.profile.bundle_bytes - 1) // backing.layout.profile.bundle_bytes
            bundles = tuple(empty[:count].tolist())
            remaining = max(0, remaining - len(bundles) * backing.layout.profile.bundle_bytes)
            selected.append(bundles)
        return tuple(selected)

    def prepare_reclaim(
        self, release_bytes: int, *, cache_eviction: SglangCacheEviction | None = None
    ) -> SglangReclaimPlan | None:
        """Prepare the exact common cache effect, or only currently unowned storage."""

        ensure_uint64(release_bytes)
        node = None
        if cache_eviction is not None:
            if self.cache is None:
                raise RuntimeError("xpool elastic context state cache reclamation requires a bound cache")
            node = cache_eviction.resolve(self.cache)
            if node is None:
                return None
        bundles = self.empty_bundle_ids(release_bytes)
        if cache_eviction is None and not any(bundles):
            return None
        return SglangReclaimPlan(release_bytes, cache_eviction, node, bundles)

    def apply_reclaim(self, plan: SglangReclaimPlan) -> None:
        """Commit agreed cache frees, withdraw actual empty IDs, then record their users."""

        if plan.applied:
            raise RuntimeError("xpool elastic context state reclaim operation was already applied")
        if plan.cache_eviction is not None:
            if self.cache is None or plan.cache_node is None:
                raise RuntimeError("xpool elastic context state reclaim lost its prepared cache owner")
            plan.cache_eviction.apply(self.cache, plan.cache_node)
            # Only actual component frees establish post-eviction emptiness.
            plan.bundle_ids = self.empty_bundle_ids(plan.release_bytes)
        for binding, bundles in zip(self.components, plan.bundle_ids, strict=True):
            if bundles:
                binding.allocator.begin_retirement(bundles)
        plan.applied = True
        if any(plan.bundle_ids):
            plan.event = torch.cuda.Event()
            plan.event.record(self.configurator.forward_stream)
        else:
            plan.completed = True

    def finish_retirement(self, plan: SglangReclaimPlan) -> int:
        """Release exact mappings only after their execution dependency has completed."""

        if not plan.applied:
            raise RuntimeError("xpool elastic context state cannot retire an unapplied reclaim operation")
        if plan.completed or plan.event is None or not plan.event.query():
            return 0
        released = 0
        for binding, bundles in zip(self.components, plan.bundle_ids, strict=True):
            if bundles:
                released += binding.allocator.backing.unmap(bundles)
                binding.allocator.finish_retirement()
        plan.completed = True
        return released

    def close(self) -> None:
        """Drop engine views and release every backing after the lifecycle stopped users."""

        match self.pool:
            case SWAKVPool():
                # Upstream annotates children as KVCache; create() validated
                # their concrete elastic owners before assembling this state.
                for child in (self.pool.full_kv_pool, self.pool.swa_kv_pool):
                    cast(ElasticMHATokenToKVPool | ElasticMLATokenToKVPool, child).close()
            case ElasticMHATokenToKVPool() | ElasticMLATokenToKVPool():
                self.pool.close()
