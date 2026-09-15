"""Active-prefix queries for SGLang's Python Unified TreeCore."""

from __future__ import annotations

from typing import cast

import torch
from sglang.srt.mem_cache.unified_cache.components import BASE_COMPONENT_TYPE
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


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
