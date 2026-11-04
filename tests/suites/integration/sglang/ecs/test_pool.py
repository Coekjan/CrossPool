from __future__ import annotations

from typing import cast

import pytest
import torch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
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
    full_locations = full_location + torch.tensor([1, 7, 31], device=device)
    swa_locations = torch.tensor([19, 3, 47], device=device)
    mapping = torch.zeros(full_swa_pool.size + full_swa_pool.page_size, dtype=torch.int64, device=device)
    mapping[full_locations] = swa_locations
    full_swa_pool.register_mapping(mapping)
    translated = full_swa_pool.translate_loc_from_full_to_swa(full_locations)
    torch.testing.assert_close(translated, swa_locations, rtol=0, atol=0)
    assert not torch.equal(full_locations, translated)
    locations = KVWriteLoc(full_locations, swa_loc=translated)

    expected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_id in (0, 1, 2):
        keys = torch.arange(3 * 8 * 64, dtype=torch.float16, device=device).view(3, 8, 64) + 2 * layer_id + 1
        values = -keys
        expected[layer_id] = keys, values
        full_swa_pool.set_kv_buffer(
            RadixAttention(num_heads=8, head_dim=64, scaling=64**-0.5, num_kv_heads=8, layer_id=layer_id),
            locations,
            keys,
            values,
        )
    torch.cuda.synchronize()
    for layer_id, expected_key_buffer, expected_value_buffer, physical_locations in (
        (0, full.k_buffer[0], full.v_buffer[0], full_locations),
        (1, full.k_buffer[1], full.v_buffer[1], full_locations),
        (2, swa.k_buffer[0], swa.v_buffer[0], swa_locations),
    ):
        key_buffer, value_buffer = full_swa_pool.get_kv_buffer(layer_id)
        assert key_buffer.data_ptr() == expected_key_buffer.data_ptr()
        assert value_buffer.data_ptr() == expected_value_buffer.data_ptr()
        assert full_swa_pool.get_key_buffer(layer_id).data_ptr() == expected_key_buffer.data_ptr()
        assert full_swa_pool.get_value_buffer(layer_id).data_ptr() == expected_value_buffer.data_ptr()
        assert not key_buffer.is_contiguous()
        assert not value_buffer.is_contiguous()
        keys, values = expected[layer_id]
        torch.testing.assert_close(key_buffer[physical_locations], keys, rtol=0, atol=0)
        torch.testing.assert_close(value_buffer[physical_locations], values, rtol=0, atol=0)
    assert full_swa_pool.post_capture_backed_bytes == quota.charged_bytes
