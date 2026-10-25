"""Unpublished SGLang allocation ownership through one common TP verdict."""

from __future__ import annotations

from collections.abc import Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import IntEnum
from types import TracebackType

import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils.common import Range

allocation_context = ContextVar["SglangAllocation | None"]("xpool_attention_allocation", default=None)


class SglangCapacityMiss(RuntimeError):
    """Unwind a known clean shortage to the owning matched allocation decision."""


class SglangAllocationStatus(IntEnum):
    """CPU-group MAX verdict for the complete component/rank allocation window."""

    READY = 0
    CAPACITY_MISS = 1
    FAILED = 2


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
class SglangAllocation:
    """Journal actual acquisitions; commit or explicitly roll back after a common verdict.

    Entry precedes tentative selection/locks. The complete allocation wrapper
    binds ``batch`` and ``token_per_req`` only for real child acquisition.
    Exceptions never trigger implicit device cleanup: the failure owner retains
    terminal or uncertain resources for Generation shutdown.
    """

    request_pool: ReqToTokenPool
    cache: UnifiedRadixCache
    full_allocator: TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator
    full_to_swa_mapping: torch.Tensor | None
    scheduler: Scheduler | None = None
    batch: ScheduleBatch | None = field(init=False, default=None)
    token_per_req: int | None = field(init=False, default=None)
    checkpoints: dict[Req, SglangRequestCheckpoint] = field(init=False, default_factory=dict)
    page_allocations: list[tuple[TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator, torch.Tensor]] = field(
        init=False, default_factory=list
    )
    mapping_snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(init=False, default_factory=list)
    request_rows: list[int] = field(init=False, default_factory=list)
    lock_receipts: list[tuple[int, DecLockRefParams]] = field(init=False, default_factory=list)
    waiting_positions: dict[Req, int] = field(init=False, default_factory=dict)
    prior_chunk_owner: Req | None = field(init=False, default=None)
    decode_rows: tuple[torch.Tensor, torch.Tensor, int] | None = field(init=False, default=None)
    committed: bool = field(init=False, default=False)
    rolled_back: bool = field(init=False, default=False)
    context_token: Token[SglangAllocation | None] | None = field(init=False, default=None)

    @classmethod
    def current(cls) -> SglangAllocation | None:
        """Return the scheduler thread's current tentative allocation, if any."""

        return allocation_context.get()

    def __enter__(self) -> SglangAllocation:
        """Bind one attempt; nested or reused attempts are invalid lifecycle transitions."""

        if self.current() is not None or self.context_token is not None:
            raise RuntimeError("xpool elastic context state allocation is already active or resolved")
        if self.scheduler is not None:
            self.prior_chunk_owner = self.scheduler.chunked_req
        self.context_token = allocation_context.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore scope without converting an uncertain failure into a clean rollback."""

        token = self.context_token
        if token is None or self.current() is not self:
            raise RuntimeError("xpool elastic context state allocation was not entered")
        allocation_context.reset(token)
        if exc_value is None and not (self.committed or self.rolled_back):
            raise RuntimeError("xpool elastic context state allocation exited without a common verdict")

    def watch(self, requests: Iterable[Req]) -> None:
        """Checkpoint each actual request once, before this attempt changes it."""

        for request in requests:
            if request in self.checkpoints:
                continue
            self.checkpoints[request] = SglangRequestCheckpoint.capture(request)
            if self.scheduler is not None:
                position = next(
                    (index for index, queued in enumerate(self.scheduler.waiting_queue) if queued is request), None
                )
                if position is not None:
                    self.waiting_positions[request] = position

    def record_allocation(
        self,
        allocator: TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator,
        page_ids: torch.Tensor,
        locations: torch.Tensor,
    ) -> None:
        """Retain actual page ownership before gathering old paired-mapping values.

        The Full child snapshots every returned location, even if it acquired
        zero pages. Advanced indexing owns the old values independently of the
        composite's subsequent setter or Decode's direct mapping writes.
        """

        if page_ids.numel():
            self.page_allocations.append((allocator, page_ids))
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
        """Publish the attempt only after every component and TP rank reported READY."""

        if self.current() is not self or self.committed or self.rolled_back:
            raise RuntimeError("xpool elastic context state allocation cannot commit outside its unresolved scope")
        self.committed = True
        self.checkpoints.clear()
        self.page_allocations.clear()
        self.mapping_snapshots.clear()
        self.request_rows.clear()
        self.lock_receipts.clear()
        self.waiting_positions.clear()
        self.decode_rows = None

    def rollback(self) -> None:
        """Restore unpublished ownership after a common clean miss; retain physical backing.

        Successful steps consume their receipts. A cleanup error propagates
        with the remaining owners retained; it is not another capacity miss.
        Committed cache eviction, window progression and retraction stay intact.
        """

        if self.current() is not self or self.committed:
            raise RuntimeError("xpool elastic context state allocation cannot roll back outside its uncommitted scope")
        if self.rolled_back:
            return
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
        if self.scheduler is not None:
            for request, position in sorted(self.waiting_positions.items(), key=lambda entry: entry[1]):
                if request not in self.scheduler.waiting_queue:
                    self.scheduler.waiting_queue.insert(position, request)
            self.scheduler.chunked_req = self.prior_chunk_owner
        self.checkpoints.clear()
        self.waiting_positions.clear()
        self.rolled_back = True
