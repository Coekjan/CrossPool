from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import torch
from sglang.srt.mem_cache.unified_cache.components import BASE_COMPONENT_TYPE
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from xpool.integrations.sglang.kv.radix import admission_evictable_size


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
