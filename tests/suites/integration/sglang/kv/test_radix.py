from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import torch
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.unified_cache.components import BASE_COMPONENT_TYPE
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from xpool.integrations.sglang.kv.allocator import ElasticTokenToKVPoolAllocator
from xpool.integrations.sglang.kv.radix import admission_evictable_size, select_suffix_reclaim_nodes


def make_node(*, values: list[int], locked: bool = False) -> UnifiedTreeNode:
    node = UnifiedTreeNode((BASE_COMPONENT_TYPE,))
    component = node.component_data[BASE_COMPONENT_TYPE]
    component.value = torch.tensor(values)
    component.lock_ref = int(locked)
    return node


def test_admission_evictable_size_counts_only_unlocked_active_ids() -> None:
    root = make_node(values=[])
    unlocked = make_node(values=[3, 4, 7, 11, 12])
    locked = make_node(values=[5, 6], locked=True)
    child = make_node(values=[8, 9])
    root.children = {0: unlocked, 1: locked}
    unlocked.children = {2: child}
    tree_cache = SimpleNamespace(
        page_size=4,
        tree_core=SimpleNamespace(root_node=root),
    )

    cache = cast(UnifiedRadixCache, tree_cache)
    assert admission_evictable_size(cache, token_capacity=8) == 5
    assert admission_evictable_size(cache, token_capacity=0) == 0


def test_suffix_reclaim_selection_is_logically_consistent_across_ranks() -> None:
    def rank_state(
        *, reverse_branches: bool
    ) -> tuple[UnifiedRadixCache, ElasticTokenToKVPoolAllocator, dict[int, str]]:
        allocator = ElasticTokenToKVPoolAllocator(12, torch.float16, "cpu", cast(KVCache, object()), False)
        slots = allocator.alloc(10)
        assert slots is not None
        root = make_node(values=[])
        shared = make_node(values=slots[:4].tolist())
        branch_a = make_node(values=slots[4:8].tolist())
        leaf_a = make_node(values=slots[8:9].tolist())
        branch_b = make_node(values=slots[9:10].tolist())
        branch_a.children = {"leaf-a": leaf_a}
        shared.children = (
            {"branch-b": branch_b, "branch-a": branch_a}
            if reverse_branches
            else {"branch-a": branch_a, "branch-b": branch_b}
        )
        root.children = {"shared": shared}
        cache = cast(UnifiedRadixCache, SimpleNamespace(page_size=1, tree_core=SimpleNamespace(root_node=root)))
        labels = {
            shared.id: "shared",
            branch_a.id: "shared/branch-a",
            leaf_a.id: "shared/branch-a/leaf-a",
            branch_b.id: "shared/branch-b",
        }
        return cache, allocator, labels

    selections: list[set[str]] = []
    for reverse_branches in (False, True):
        cache, allocator, labels = rank_state(reverse_branches=reverse_branches)
        selected = select_suffix_reclaim_nodes(cache, allocator, token_capacity=6)
        assert selected is not None
        selections.append({labels[node_id] for node_id in selected})

    assert selections == [
        {"shared/branch-a", "shared/branch-a/leaf-a", "shared/branch-b"},
        {"shared/branch-a", "shared/branch-a/leaf-a", "shared/branch-b"},
    ]
