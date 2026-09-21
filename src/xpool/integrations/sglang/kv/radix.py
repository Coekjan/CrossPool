"""Active-prefix queries for SGLang's Python Unified TreeCore."""

from __future__ import annotations

import time
from typing import cast

import torch
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentDeviceSlot
from sglang.srt.mem_cache.unified_cache.components import BASE_COMPONENT_TYPE
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore, UnifiedTreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from xpool.integrations.sglang.kv.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)


def admission_evictable_size(tree_cache: UnifiedRadixCache, token_capacity: int) -> int:
    """Count unlocked cached IDs inside the allocator's active prefix."""

    if token_capacity == 0:
        return 0

    values: list[torch.Tensor] = []
    tree_core = cast(UnifiedTreeCore, tree_cache.tree_core)
    pending = list(tree_core.root_node.children.values())
    while pending:
        node = pending.pop()
        component = node.component_data[BASE_COMPONENT_TYPE]
        if component.lock_ref == 0 and component.value is not None:
            values.append(component.value)
        pending.extend(node.children.values())

    if not values:
        return 0
    ids = torch.cat(values)
    lower = tree_cache.page_size
    return int(torch.count_nonzero((ids >= lower) & (ids < lower + token_capacity)).item())


def select_suffix_reclaim_nodes(
    tree_cache: UnifiedRadixCache,
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
    token_capacity: int,
) -> tuple[int, ...] | None:
    """Select an unlocked child-first node set that can release the complete suffix."""

    page_size = allocator.page_size
    target_page = token_capacity // page_size
    selected: list[UnifiedTreeNode] = []
    selected_pages: list[torch.Tensor] = []
    blocked = False

    def visit(node: UnifiedTreeNode, inherited: bool) -> None:
        nonlocal blocked
        component = node.component_data[BASE_COMPONENT_TYPE]
        value = component.value
        owns_suffix = value is not None and bool(torch.any(value // page_size > target_page).item())
        include = inherited or owns_suffix
        for child in node.children.values():
            visit(child, include)
        if not include:
            return
        if component.lock_ref != 0:
            blocked = True
            return
        selected.append(node)
        if value is not None:
            selected_pages.append(value // page_size)

    tree_core = cast(UnifiedTreeCore, tree_cache.tree_core)
    for child in tree_core.root_node.children.values():
        visit(child, False)
    if blocked:
        return None

    free_pages = torch.cat((allocator.get_all_free_pages(), allocator.withheld_pages))
    covered = [free_pages, *selected_pages]
    suffix_pages = torch.unique(torch.cat(covered))
    covered_count = int(torch.count_nonzero(suffix_pages > target_page).item())
    reserved_pages = allocator.reserved_token_capacity // page_size
    if covered_count != reserved_pages - target_page:
        return None
    return tuple(node.id for node in selected)


def evict_suffix_reclaim_nodes(tree_cache: UnifiedRadixCache, node_ids: tuple[int, ...]) -> None:
    """Evict one prepared child-first node set and consume every returned cache action."""

    started_at = time.perf_counter()
    evicted_tokens = 0
    for node_id in node_ids:
        result = tree_cache.tree_core.evict_device_leaf(node_id, tree_cache.is_write_back)
        if result.backup_kv is not None:
            raise RuntimeError("xpool elastic kv reclaim does not support write-back cache eviction")
        for component_type in list(result.device_frees):
            tree_cache.components[component_type].apply_component_action(
                FreeComponentDeviceSlot(result.device_frees.pop(component_type), component_type=component_type)
            )
        for component_type in list(result.host_frees):
            tree_cache.components[component_type].free_host_values(result.host_frees.pop(component_type))
        evicted_tokens += result.tracker[BASE_COMPONENT_TYPE]
    tree_cache.update_eviction_metrics(evicted_tokens, started_at)
