"""Elastic compound-layout SGLang KV pool implementations."""

from __future__ import annotations

import torch
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, MHATokenToKVPool, MLATokenToKVPool

from xpool.integrations.sglang.kv.vmm import KvVmmBacking


class ElasticMHATokenToKVPool(MHATokenToKVPool):
    """MHA/GQA pool backed by one reversible compound VMM reservation."""

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
        """Construct the pinned upstream pool while selecting elastic post-capture storage."""

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
        """Install stable strided K/V views in upstream descriptor order."""

        self.backing = KvVmmBacking(
            device=torch.device(self.device),
            storage_dtype=self.store_dtype,
            token_page_size=self.page_size,
            descriptors=self._build_kv_buffer_descs(),
            buffers_per_layer=2,
        )
        self._assign_post_capture_tensors(self.backing.tensors)

    def _store_kv_layer(
        self,
        layer_idx: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Write one layer through indexing that honors compound-view strides."""

        self.k_buffer[layer_idx][loc] = cache_k
        self.v_buffer[layer_idx][loc] = cache_v

    def close(self) -> None:
        """Drop engine views before releasing their sole VMM backing owner."""

        self.k_buffer = []
        self.v_buffer = []
        self.backing.close()


class ElasticMLATokenToKVPool(MLATokenToKVPool):
    """MLA pool backed by one reversible compound VMM reservation."""

    backing: KvVmmBacking
    kv_buffer: list[torch.Tensor]
    kv_cache_dim: int

    def _create_buffers(self) -> None:
        """Install stable strided combined-KV views before pointer publication."""

        self.post_capture_active = True
        shape = (self.size + self.page_size, 1, self.kv_cache_dim)
        row_bytes = self.kv_cache_dim * self.store_dtype.itemsize
        descriptors = [
            KvBufferDesc(f"kv{layer}", shape, row_bytes=row_bytes, tokens_per_row=1) for layer in range(self.layer_num)
        ]
        self.backing = KvVmmBacking(
            device=torch.device(self.device),
            storage_dtype=self.store_dtype,
            token_page_size=self.page_size,
            descriptors=descriptors,
            buffers_per_layer=1,
        )
        self.kv_buffer = self.backing.tensors

    def close(self) -> None:
        """Drop engine views before releasing their sole VMM backing owner."""

        self.kv_buffer = []
        self.backing.close()
