from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from tests.suites.integration.sglang.ecs.support import full_swa_pool_scope, install_vmm_driver

from xpool import ecs
from xpool.integrations.sglang.ecs.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.ecs.pool import ElasticMHATokenToKVPool
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking, KvVmmLayout


@pytest.fixture
def vmm_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    install_vmm_driver(monkeypatch)


@pytest.fixture(params=(1, 4), ids=("token", "paged"))
def allocator(
    request: pytest.FixtureRequest, vmm_driver: None
) -> ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator:
    page_size = request.param
    layout = KvVmmLayout(torch.device("cuda:0"), torch.uint8, (32 + page_size, 4), 1, page_size, 1, 1, 16)
    quota = ecs.MemoryQuota(layout.profile.bytes_for(layout.profile.bundle_capacity))
    pool = object.__new__(ElasticMHATokenToKVPool)
    pool.backing = KvVmmBacking(layout, quota)
    if page_size == 1:
        return ElasticTokenToKVPoolAllocator(32, torch.float16, "cpu", pool, True)
    return ElasticPagedTokenToKVPoolAllocator(32, page_size, torch.float16, "cpu", pool, True)


@pytest.fixture
def attention_cache(
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator, published_sglang_config: None
) -> UnifiedRadixCache:
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=ReqToTokenPool(4, 64, "cpu", False),
            token_to_kv_pool_allocator=allocator,
            page_size=allocator.page_size,
            tree_components=(ComponentType.FULL,),
        )
    )


@pytest.fixture
def full_swa_pool(published_sglang_config: None) -> Iterator[SWAKVPool]:
    device = torch.device("cuda", torch.cuda.current_device())
    with full_swa_pool_scope(device) as pool:
        yield pool
