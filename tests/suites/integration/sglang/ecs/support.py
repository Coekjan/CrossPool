from __future__ import annotations

import itertools
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.mem_cache.memory_pool import KvBufferDesc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

import xpool.integrations.sglang.ecs.vmm
from xpool import ecs
from xpool.integrations.sglang.ecs.pool import ElasticMHATokenToKVPool, prepared_backing_scope
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking


def install_vmm_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real backing/allocator ownership with CPU storage instead of device mappings."""

    handles = itertools.count(1)
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemAddressReserve",
        lambda *args: (driver.CUresult.CUDA_SUCCESS, 4096),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm,
        "tensor_from_pointer",
        lambda pointer, byte_count, **kwargs: torch.empty(byte_count, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.vmm.driver,
        "cuMemCreate",
        lambda *args: (driver.CUresult.CUDA_SUCCESS, next(handles)),
    )
    for operation in ("cuMemMap", "cuMemSetAccess", "cuMemUnmap", "cuMemRelease", "cuMemAddressFree"):
        monkeypatch.setattr(
            xpool.integrations.sglang.ecs.vmm.driver,
            operation,
            lambda *args: (driver.CUresult.CUDA_SUCCESS,),
        )


@contextmanager
def full_swa_pool_scope(device: torch.device) -> Iterator[SWAKVPool]:
    """Own the small real composite used by component and multi-rank tests."""

    layouts = tuple(
        KvVmmBacking.describe(
            device=device,
            storage_dtype=torch.float16,
            token_page_size=4,
            descriptors=tuple(
                KvBufferDesc(f"buffer{index}", (size + 4, 8, 64), row_bytes=1024, tokens_per_row=1)
                for index in range(2 * layer_count)
            ),
            buffers_per_layer=2,
        )
        for size, layer_count in ((8192, 2), (4096, 1))
    )
    quota = ecs.MemoryQuota(sum(layout.profile.bytes_for(layout.profile.floor_bundles) for layout in layouts))
    backings: list[KvVmmBacking] = []
    try:
        for layout in layouts:
            backings.append(KvVmmBacking(layout, quota))
        with prepared_backing_scope(tuple(backings)):
            pool = SWAKVPool(
                size=8192,
                size_swa=4096,
                page_size=4,
                dtype=torch.float16,
                head_num=8,
                head_dim=64,
                swa_attention_layer_ids=[2],
                full_attention_layer_ids=[0, 1],
                device=str(device),
                full_kv_pool_class=ElasticMHATokenToKVPool,
                swa_kv_pool_class=ElasticMHATokenToKVPool,
            )
        yield pool
    finally:
        for backing in backings:
            backing.close()
