"""Concrete KV acquisitions and restoration behind the allocation journal interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils.common import Range


@dataclass(frozen=True, slots=True)
class SglangRequestCheckpoint:
    """Fields changed before allocation publication, excluding committed window frees."""

    request: Req
    req_pool_idx: int | None
    extend_range: Range | None
    lock_receipt: DecLockRefParams
    kv_allocated_len: int
    kv_committed_len: int
    inflight_middle_chunks: int

    @classmethod
    def capture(cls, request: Req) -> SglangRequestCheckpoint:
        """Retain a request's first pre-attempt state after committed prefix matching."""

        return cls(
            request,
            request.kv.req_pool_idx,
            request.extend_range,
            request.lock_receipt,
            request.kv.kv_allocated_len,
            request.kv.kv_committed_len,
            request.inflight_middle_chunks,
        )

    def restore(self) -> None:
        """Restore tentative metadata while preserving actual cache/window releases."""

        self.request.kv.req_pool_idx = self.req_pool_idx
        self.request.extend_range = self.extend_range
        self.request.lock_receipt = self.lock_receipt
        self.request.kv.kv_allocated_len = self.kv_allocated_len
        self.request.kv.kv_committed_len = self.kv_committed_len
        self.request.inflight_middle_chunks = self.inflight_middle_chunks


@dataclass(slots=True)
class KvAllocationReceipt:
    """Own real KV acquisitions, request rows and engine lock receipts."""

    request_pool: ReqToTokenPool
    cache: UnifiedRadixCache
    full_allocator: TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator
    full_to_swa_mapping: torch.Tensor | None
    checkpoints: dict[Req, SglangRequestCheckpoint] = field(default_factory=dict)
    page_allocations: list[tuple[TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator, torch.Tensor]] = field(
        default_factory=list
    )
    mapping_snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(default_factory=list)
    request_rows: list[int] = field(default_factory=list)
    lock_receipts: list[tuple[int, DecLockRefParams]] = field(default_factory=list)
    decode_rows: tuple[torch.Tensor, torch.Tensor, int] | None = None

    def watch(self, requests: Iterable[Req]) -> None:
        for request in requests:
            if request not in self.checkpoints:
                self.checkpoints[request] = SglangRequestCheckpoint.capture(request)

    def record_decode(self, rows: torch.Tensor, lengths: torch.Tensor, step: int) -> None:
        self.decode_rows = rows, lengths, step

    def record_allocation(
        self,
        allocator: BaseTokenToKVPoolAllocator,
        unit_ids: torch.Tensor,
        locations: torch.Tensor,
    ) -> None:
        """Retain actual page ownership before gathering old paired-mapping values.

        The Full child snapshots every returned location, even if it acquired
        zero pages. Advanced indexing owns the old values independently of the
        composite's subsequent setter or Decode's direct mapping writes.
        """

        if unit_ids.numel():
            self.page_allocations.append(
                (cast(TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator, allocator), unit_ids)
            )
        if allocator is self.full_allocator and self.full_to_swa_mapping is not None:
            self.mapping_snapshots.append((self.full_to_swa_mapping, locations, self.full_to_swa_mapping[locations]))

    def record_rows(self, rows: list[int]) -> None:
        """Retain newly allocated request rows, excluding reused continuation rows."""

        self.request_rows.extend(rows)

    def record_lock(self, receipt: DecLockRefParams) -> None:
        """Retain an actual newly acquired cache lock, using its original release receipt."""

        if receipt.node_id is not None:
            self.lock_receipts.append((receipt.node_id, receipt))

    def commit(self) -> None:
        self.checkpoints.clear()
        self.page_allocations.clear()
        self.mapping_snapshots.clear()
        self.request_rows.clear()
        self.lock_receipts.clear()
        self.decode_rows = None

    def rollback(self) -> None:
        while self.mapping_snapshots:
            mapping, locations, values = self.mapping_snapshots[-1]
            mapping[locations] = values
            self.mapping_snapshots.pop()
        if self.decode_rows is not None:
            rows, lengths, step = self.decode_rows
            positions = lengths[:, None] + torch.arange(step)
            self.request_pool.req_to_token[rows[:, None], positions] = 0
            self.decode_rows = None
        else:
            for checkpoint in self.checkpoints.values():
                request = checkpoint.request
                if checkpoint.req_pool_idx is not None:
                    start = max(checkpoint.kv_allocated_len, len(request.prefix_indices))
                    self.request_pool.req_to_token[checkpoint.req_pool_idx, start : request.kv.kv_allocated_len] = 0
        while self.page_allocations:
            allocator, pages = self.page_allocations[-1]
            allocator.free_page_ids(pages)
            self.page_allocations.pop()
        if self.request_rows:
            self.request_pool.req_to_token[self.request_rows] = 0
            self.request_pool.free_rows(self.request_rows)
            self.request_rows.clear()
        while self.lock_receipts:
            node, receipt = self.lock_receipts[-1]
            self.cache.dec_lock_ref(node, receipt)
            self.lock_receipts.pop()
        for checkpoint in self.checkpoints.values():
            checkpoint.restore()
        self.checkpoints.clear()
