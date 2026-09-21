from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from sglang.srt.layers.radix_attention import RadixAttention

import xpool.integrations.sglang.kv.vmm
from tests.harness.support.sglang.runtime import published_sglang_config
from xpool.integrations.sglang.kv.pool import ElasticMHATokenToKVPool, ElasticMLATokenToKVPool
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.service.wire import KvCapacityPartitionProfile


def test_partial_bundle_mapping_retains_actual_progress_for_close(monkeypatch: pytest.MonkeyPatch) -> None:
    backing = object.__new__(KvVmmBacking)
    backing.capacity_profile = KvCapacityPartitionProfile(
        bundle_bytes=2048,
        bundle_capacity=2,
        floor_bundles=1,
        token_capacity=1,
        mapping_granularity_bytes=2048,
        row_bytes=2048,
        tokens_per_row=1,
        token_page_size=1,
    )
    backing.base = 4096
    backing.reserved_bytes = 4096
    backing.allocation_properties = object()
    backing.access_descriptors = [object()]
    backing.backed_bundles = 0
    backing.raw_storage = None
    backing.tensors = []

    failure = object()
    unmapped: list[int] = []
    released: list[int] = []
    freed: list[tuple[int, int]] = []
    next_handle = 0

    def create(byte_count: int, properties: object, flags: int) -> int:
        nonlocal next_handle
        assert byte_count == backing.capacity_profile.bundle_bytes
        assert properties is backing.allocation_properties
        assert flags == 0
        handle = next_handle
        next_handle += 1
        return handle

    def map_bundle(address: int, byte_count: int, offset: int, handle: int, flags: int) -> None:
        assert byte_count == backing.capacity_profile.bundle_bytes
        assert offset == flags == 0
        assert handle in (0, 1)

    def set_access(address: int, byte_count: int, descriptors: list[object], count: int) -> object | None:
        assert byte_count == backing.capacity_profile.bundle_bytes
        assert descriptors is backing.access_descriptors
        assert count == 1
        return failure if address == backing.base + backing.capacity_profile.bundle_bytes else None

    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemCreate", create)
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemMap", map_bundle)
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemSetAccess", set_access)
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver,
        "cuMemUnmap",
        lambda address, byte_count: unmapped.append(address),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver,
        "cuMemRelease",
        lambda handle: released.append(handle),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver,
        "cuMemAddressFree",
        lambda address, byte_count: freed.append((address, byte_count)),
    )

    def check(result: object, message: str) -> object:
        if result is failure:
            raise RuntimeError(message)
        return result

    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm, "check_drv", check)

    with pytest.raises(RuntimeError, match="cuMemSetAccess"):
        backing.resize(2)

    assert backing.backed_bundles == 1
    assert unmapped == [backing.base + backing.capacity_profile.bundle_bytes]
    assert released == [0, 1]

    backing.close()

    assert backing.backed_bundles == 0
    assert backing.base == 0
    assert unmapped == [4096 + 2048, 4096]
    assert freed == [(4096, 4096)]


@pytest.mark.requires_cuda()
def test_mha_pool_preserves_views_across_reversible_bundle_mapping() -> None:
    pool = ElasticMHATokenToKVPool(2048, 1, torch.float16, 8, 64, 2, "cuda", False)
    try:
        keys_by_layer = pool.k_buffer
        values_by_layer = pool.v_buffer
        pointers = tuple(tensor.data_ptr() for tensor in (*keys_by_layer, *values_by_layer))
        pool.backing.resize(pool.backing.capacity_profile.bundle_capacity)

        locations = torch.tensor([1, 2048], device="cuda")
        keys = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
        values = torch.randn_like(keys)
        pool.set_kv_buffer(cast(RadixAttention, SimpleNamespace(layer_id=0)), locations, keys, values)
        torch.cuda.synchronize()
        assert torch.equal(keys_by_layer[0][locations], keys)
        assert torch.equal(values_by_layer[0][locations], values)

        graph_input = keys_by_layer[0][1]
        graph_output = torch.empty_like(graph_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.add(graph_input, 1, out=graph_output)
        pool.backing.resize(pool.backing.capacity_profile.floor_bundles)
        pool.backing.resize(pool.backing.capacity_profile.bundle_capacity)
        graph.replay()
        torch.cuda.synchronize()

        assert tuple(tensor.data_ptr() for tensor in (*keys_by_layer, *values_by_layer)) == pointers
        assert torch.equal(graph_output, graph_input + 1)
    finally:
        pool.close()


@pytest.mark.requires_cuda()
@pytest.mark.usefixtures(published_sglang_config.__name__)
def test_mla_pool_writes_compound_strided_views() -> None:
    pool = ElasticMLATokenToKVPool(2048, 1, torch.float16, 512, 64, 2, "cuda", False)
    try:
        locations = torch.tensor([1, 2], device="cuda")
        hidden_states = torch.randn(2, 1, 576, device="cuda", dtype=torch.float16)
        pool.set_kv_buffer(
            cast(RadixAttention, SimpleNamespace(layer_id=0)),
            locations,
            hidden_states,
            torch.empty(0, device="cuda"),
        )
        torch.cuda.synchronize()
        assert torch.equal(pool.kv_buffer[0][locations], hidden_states)
    finally:
        pool.close()
