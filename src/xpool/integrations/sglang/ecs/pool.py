"""Elastic children consumed by SGLang's ordinary pool factories."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, KVCache, MHATokenToKVPool, MLATokenToKVPool

from xpool.integrations.sglang.ecs.vmm import KvVmmBacking, KvVmmLayout

prepared_backings = ContextVar[Iterator[KvVmmBacking] | None]("xpool_prepared_attention_backings", default=None)


@contextmanager
def prepared_backing_scope(backings: tuple[KvVmmBacking, ...]) -> Iterator[None]:
    """Lend precharged backings to one original pool-factory call, in component order.

    The startup owner retains cleanup responsibility. Children validate their
    actual geometry when claiming a backing; this scope owns only the binding.
    """

    pending = iter(backings)
    token = prepared_backings.set(pending)
    try:
        yield
        if next(pending, None) is not None:
            raise RuntimeError("xpool elastic context state factory did not consume every prepared backing")
    finally:
        prepared_backings.reset(token)


def take_prepared_backing(layout: KvVmmLayout) -> KvVmmBacking:
    """Claim the next prepared component after checking the real factory geometry."""

    pending = prepared_backings.get()
    backing = None if pending is None else next(pending, None)
    if backing is None:
        raise RuntimeError("xpool elastic context state pool requires a prepared backing")
    if backing.layout != layout:
        raise ValueError("xpool elastic context state pool geometry differs from its prepared backing")
    return backing


class ElasticMHATokenToKVPool(MHATokenToKVPool):
    """MHA/GQA child with stable compound views from the shared-quota startup owner."""

    backing: KvVmmBacking
    k_buffer: list[torch.Tensor]
    v_buffer: list[torch.Tensor]

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: int | None = None,
        swa_head_num: int | None = None,
        swa_head_dim: int | None = None,
        swa_v_head_dim: int | None = None,
        start_layer: int | None = None,
        end_layer: int | None = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        kv_cache_layout: str | None = None,
        quant_method: object | None = None,
        post_capture_active: bool = False,
        allocation_label: str | None = None,
    ) -> None:
        """Preserve the pinned MHA constructor while selecting prepared elastic storage."""

        super().__init__(
            size,
            page_size,
            dtype,
            head_num,
            head_dim,
            layer_num,
            device,
            enable_memory_saver,
            v_head_dim,
            swa_head_num,
            swa_head_dim,
            swa_v_head_dim,
            start_layer,
            end_layer,
            enable_alt_stream,
            enable_kv_cache_copy,
            kv_cache_layout,
            quant_method,
            True,
            allocation_label,
        )

    def _alloc_post_capture_buffers(self) -> None:
        """Install the prepared K/V views in upstream descriptor order."""

        self.backing = take_prepared_backing(
            KvVmmBacking.describe(
                device=torch.device(self.device),
                storage_dtype=self.store_dtype,
                token_page_size=self.page_size,
                descriptors=self._build_kv_buffer_descs(),
                buffers_per_layer=2,
            )
        )
        self._assign_post_capture_tensors(self.backing.tensors)

    def _store_kv_layer(self, layer_idx: int, loc: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor) -> None:
        """Write through indexing that preserves the compound views' row strides."""

        self.k_buffer[layer_idx][loc] = cache_k
        self.v_buffer[layer_idx][loc] = cache_v

    @property
    def post_capture_backed_bytes(self) -> int:
        """Report the child's actual physical storage through the upstream pool interface."""

        return self.backing.backed_bytes

    def close(self) -> None:
        """Drop borrowed engine views before releasing this child's physical backing."""

        self.k_buffer = []
        self.v_buffer = []
        self.backing.close()


class ElasticMLATokenToKVPool(MLATokenToKVPool):
    """MLA child with stable combined-KV views from the shared-quota startup owner."""

    backing: KvVmmBacking
    kv_buffer: list[torch.Tensor]
    kv_cache_dim: int

    def _create_buffers(self) -> None:
        """Install the prepared combined-KV views before upstream publishes pointers."""

        self.post_capture_active = True
        shape = (self.size + self.page_size, 1, self.kv_cache_dim)
        descriptors = tuple(
            KvBufferDesc(f"kv{layer}", shape, row_bytes=self.kv_cache_dim * self.store_dtype.itemsize, tokens_per_row=1)
            for layer in range(self.layer_num)
        )
        self.backing = take_prepared_backing(
            KvVmmBacking.describe(
                device=torch.device(self.device),
                storage_dtype=self.store_dtype,
                token_page_size=self.page_size,
                descriptors=descriptors,
                buffers_per_layer=1,
            )
        )
        self.kv_buffer = self.backing.tensors

    @property
    def post_capture_backed_bytes(self) -> int:
        """Report the child's actual physical storage through the upstream pool interface."""

        return self.backing.backed_bytes

    def close(self) -> None:
        """Drop borrowed engine views before releasing this child's physical backing."""

        self.kv_buffer = []
        self.backing.close()


def backing_of(pool: KVCache) -> KvVmmBacking:
    """Resolve a supported elastic child; composites retain their own pool ownership."""

    match pool:
        case ElasticMHATokenToKVPool() | ElasticMLATokenToKVPool():
            return pool.backing
        case _:
            raise TypeError("xpool elastic context state allocator requires an elastic token-backed child pool")
