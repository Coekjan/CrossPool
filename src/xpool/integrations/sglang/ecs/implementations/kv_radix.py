"""Exact logical eviction effects on the pinned Python Unified Radix owners."""

from __future__ import annotations

import time
from array import array
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache import common
from sglang.srt.mem_cache.base_prefix_cache import IncLockRefResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType, EvictLayer
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore, UnifiedTreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from xpool.integrations.sglang.ecs.radix import StateCacheEffect, StateCacheKey, StateCacheProposal


def free_swa_out_of_window_slots(component: SWAComponent, req: Req, pre_len: int) -> None:
    """Retain a matchable window before the actual page-aligned insert endpoint."""

    page_size = component.cache.page_size
    insert_end = (pre_len + 1) // page_size * page_size
    retain_floor = max(0, insert_end - component.full_window_pages * page_size)
    existing_floor = component.cache.swa_retain_floor(req)
    if existing_floor is not None:
        retain_floor = min(retain_floor, existing_floor)
    common.free_swa_out_of_window_slots(
        req,
        pre_len,
        sliding_window_size=component.sliding_window_size,
        page_size=page_size,
        req_to_token_pool=component.cache.req_to_token_pool,
        token_to_kv_pool_allocator=component.cache.token_to_kv_pool_allocator,
        retain_floor=retain_floor,
    )


def around_acquire_swa_component_lock(
    original_fn: Callable[[SWAComponent, UnifiedTreeNode, IncLockRefResult, bool], IncLockRefResult],
    component: SWAComponent,
    node: UnifiedTreeNode,
    result: IncLockRefResult,
    lock_host: bool = False,
) -> IncLockRefResult:
    """Split the live window boundary while preserving upstream receipts and counters."""

    if not lock_host:
        tree = cast(UnifiedTreeCore, component.tree_core)
        current = node
        remaining = component.sliding_window_size
        while current.parent is not None and remaining > 0:
            data = current.component_data[ComponentType.SWA]
            key = cast(RadixKey, current.key)
            length = len(key)
            tail = (remaining + tree.page_size - 1) // tree.page_size * tree.page_size
            if data.value is not None and length > tail:
                tree._split_node(key, current, length - tail)
                length = tail
            remaining -= length
            current = current.parent
    return original_fn(component, node, result, lock_host)


@dataclass(frozen=True, slots=True)
class SglangCacheEviction:
    """Owned root-to-node identity and one effect agreed by the TP participants.

    FULL removes a device leaf with its normal component cascades. SWA removes
    only that component; it must not cascade to FULL on any participant.
    """

    key: RadixKey
    component: ComponentType

    def proposal(
        self, cache: UnifiedRadixCache, pools: tuple[tuple[ComponentType, int], ...]
    ) -> StateCacheProposal | None:
        """Snapshot the exact logical cascade before the all-rank readiness vote.

        Each pool supplies its upstream component and native allocation-unit
        width. Host-tier operations are outside this implementation's contract.
        """

        node = self.resolve(cache)
        if node is None:
            return None
        tree = cast(UnifiedTreeCore, cache.tree_core)
        effects = []

        def collect(current: UnifiedTreeNode, components: tuple[ComponentType, ...]) -> None:
            key = StateCacheKey.capture(self.from_node(current, self.component).key)
            for component in components:
                index, width = next((i, width) for i, (ct, width) in enumerate(pools) if ct == component)
                value = current.component_data[component].value
                if value is not None and len(value):
                    effects.append(StateCacheEffect(key, index, (len(value) + width - 1) // width))

        if self.component == ComponentType.FULL:
            collect(node, tuple(ct for ct, _ in pools))
            # Deleting a Full leaf also removes childless tombstone ancestors.
            # Predict only the exact chain that the pinned owner will visit.
            parent = node.parent
            while parent is not None and parent is not tree.root_node:
                if (
                    len(parent.children) != 1
                    or any(cd.lock_ref or cd.host_lock_ref for cd in parent.component_data)
                    or parent.component_data[ComponentType.FULL].value is not None
                ):
                    break
                collect(parent, tuple(ct for ct, _ in pools))
                if parent.component_data[ComponentType.FULL].host_value is not None:
                    break
                parent = parent.parent
        else:
            trigger = tree.components_by_type[self.component]
            cascades = tuple(
                comp.component_type
                for comp in tree.components
                if tree._should_cascade_evict_component(
                    node, trigger, comp, EvictLayer.DEVICE, tree._is_cascade_evict_leaf(node, EvictLayer.DEVICE)
                )
            )
            collect(node, (self.component, *cascades))
        index = next(i for i, (ct, _) in enumerate(pools) if ct == self.component)
        return StateCacheProposal(
            StateCacheKey.capture(self.key),
            index,
            "endpoint" if self.component == ComponentType.FULL else "resource",
            tuple(effects),
        )

    @classmethod
    def from_node(cls, node: UnifiedTreeNode, component: ComponentType) -> SglangCacheEviction:
        """Copy effective edge tokens and namespace, including bigram boundaries."""

        edges: list[RadixKey] = []
        while node.parent is not None:
            if node.key is None:
                raise ValueError("xpool elastic context state cache node has no edge key")
            edges.append(node.key)
            node = node.parent
        if not edges:
            raise ValueError("xpool elastic context state cannot evict the cache root")
        first = edges[-1]
        tokens = array("q")
        for edge in reversed(edges):
            raw = edge.raw_token_ids()
            tokens.extend(raw[1:] if first.is_bigram and tokens else raw)
        return cls(RadixKey(tokens, first.extra_key, is_bigram=first.is_bigram, cache_salt=first.cache_salt), component)

    @classmethod
    def select(cls, cache: UnifiedRadixCache, component: ComponentType) -> SglangCacheEviction | None:
        """Borrow the original selector's first candidate without evicting its data.

        The pinned selector exposes a heap for Full and an LRU cursor for SWA.
        Start/end own only that short-lived traversal, not a logical mutation.
        """

        owner = cache.components.get(component)
        if not isinstance(owner, FullComponent | SWAComponent):
            return None
        owner.evict_device_start(1)
        try:
            if isinstance(owner, FullComponent):
                node = owner._evict_device_heap[0][1] if owner._evict_device_heap else None
            else:
                node = owner._evict_device_cursor
        finally:
            owner.evict_device_end()
        if node is None:
            return None
        tree = cast(UnifiedTreeCore, cache.tree_core)
        if (
            component == ComponentType.SWA
            and node in tree.evictable_device_leaves
            and (not tree.enable_session_radix_cache or owner._can_evict_leaf_atomically(node))
        ):
            component = ComponentType.FULL
        return cls.from_node(node, component)

    def resolve(self, cache: UnifiedRadixCache) -> UnifiedTreeNode | None:
        """Check the exact existing endpoint and specified cascade without splitting."""

        if not len(self.key) or self.component not in (ComponentType.FULL, ComponentType.SWA):
            return None
        tree = cast(UnifiedTreeCore, cache.tree_core)
        matched, node_id, pinned = tree.match_full_device_prefix(self.key)
        if matched != pinned or matched != len(self.key):
            return None
        node = tree.node_by_id(node_id)
        if self.component == ComponentType.FULL:
            return node if node in tree.evictable_device_leaves else None
        trigger = tree.components_by_type.get(ComponentType.SWA)
        data = node.component_data[ComponentType.SWA]
        if trigger is None or data.value is None or data.lock_ref != 0:
            return None
        if tree._should_cascade_evict_component(
            node,
            trigger,
            tree.components_by_type[ComponentType.FULL],
            EvictLayer.DEVICE,
            tree._is_cascade_evict_leaf(node, EvictLayer.DEVICE),
        ):
            return None
        return node

    def apply(self, cache: UnifiedRadixCache, node: UnifiedTreeNode) -> None:
        """Consume the prepared exact operation and every actual component free.

        The coordinator has obtained a common readiness vote and permitted no
        intervening cache mutation. No public LRU walk chooses another victim.
        """

        tree = cast(UnifiedTreeCore, cache.tree_core)
        started_at = time.perf_counter()
        if self.component == ComponentType.FULL:
            result = tree.evict_device_leaf(node.id, cache.is_write_back)
            if result.backup_kv is not None:
                raise RuntimeError("xpool elastic context state eviction does not support write-back KV")
            tracker, device_frees, host_frees = result.tracker, result.device_frees, result.host_frees
        else:
            tracker: dict[ComponentType, int] = defaultdict(int)
            device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
            host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
            trigger = tree.components_by_type[ComponentType.SWA]
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
        cache.update_eviction_metrics(tracker.get(ComponentType.FULL, 0), started_at)
