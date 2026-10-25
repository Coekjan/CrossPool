from __future__ import annotations

from typing import cast

import pytest
import torch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

import xtest
from xpool.integrations.sglang.ecs.pool import ElasticMHATokenToKVPool
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


@xtest.requirements(device_count=1)
def test_upstream_full_swa_factory_consumes_prepared_shared_quota_storage(full_swa_pool: SWAKVPool) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    full = cast(ElasticMHATokenToKVPool, full_swa_pool.full_kv_pool)
    swa = cast(ElasticMHATokenToKVPool, full_swa_pool.swa_kv_pool)
    quota = full.backing.quota
    assert full_swa_pool.post_capture_backed_bytes == quota.charged_bytes == quota.limit_bytes
    quota.set_limit(quota.limit_bytes + full.backing.layout.profile.bundle_bytes)
    assert full.backing.try_map((2,))
    assert not swa.backing.try_map((1,))

    full_location = 2 * full.backing.layout.mapping_granularity_bytes // full.backing.layout.row_bytes
    locations = torch.tensor([full_location], device=device)
    keys = torch.randn(1, 8, 64, dtype=torch.float16, device=device)
    values = torch.randn_like(keys)
    full.set_kv_buffer(
        RadixAttention(num_heads=8, head_dim=64, scaling=64**-0.5, num_kv_heads=8, layer_id=0),
        locations,
        keys,
        values,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(full.k_buffer[0][locations], keys, rtol=0, atol=0)
    torch.testing.assert_close(full.v_buffer[0][locations], values, rtol=0, atol=0)
    assert full_swa_pool.post_capture_backed_bytes == quota.charged_bytes
