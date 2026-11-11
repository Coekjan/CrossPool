"""Allocation transaction ordering and common TP publication ownership."""

from __future__ import annotations

from collections.abc import Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import IntEnum
from types import TracebackType

import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams

from xpool.integrations.sglang.ecs.receipts import StateAllocationJournal, StateAllocationReceipt

allocation_context = ContextVar["SglangAllocation | None"]("xpool_attention_allocation", default=None)


class SglangCapacityMiss(RuntimeError):
    """Unwind a known clean shortage to the owning matched allocation decision."""


class SglangAllocationStatus(IntEnum):
    """CPU-group MAX verdict for the complete allocation window."""

    READY = 0
    CAPACITY_MISS = 1
    FAILED = 2


@dataclass(slots=True)
class SglangAllocation:
    """Retain implementation receipts until one matched TP verdict.

    Exceptions retain unconfirmed owners. Cleanup follows an explicit clean
    miss, after all peers have agreed to roll back unpublished ownership.
    """

    journal: StateAllocationJournal
    scheduler: Scheduler | None = None
    batch: ScheduleBatch | None = field(init=False, default=None)
    token_per_req: int | None = field(init=False, default=None)
    waiting_positions: dict[Req, int] = field(init=False, default_factory=dict)
    watched: set[Req] = field(init=False, default_factory=set)
    prior_chunk_owner: Req | None = field(init=False, default=None)
    receipts: list[StateAllocationReceipt] = field(init=False, default_factory=list)
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
        requests = tuple(requests)
        self.journal.watch(requests)
        for receipt in self.receipts:
            receipt.watch(requests)
        for request in requests:
            if request in self.watched:
                continue
            self.watched.add(request)
            if self.scheduler is not None:
                position = next(
                    (index for index, queued in enumerate(self.scheduler.waiting_queue) if queued is request), None
                )
                if position is not None:
                    self.waiting_positions[request] = position

    def record_allocation(
        self, allocator: BaseTokenToKVPoolAllocator, unit_ids: torch.Tensor, locations: torch.Tensor
    ) -> None:
        self.journal.record_allocation(allocator, unit_ids, locations)

    def record_rows(self, rows: list[int]) -> None:
        self.journal.record_rows(rows)

    def record_lock(self, receipt: DecLockRefParams) -> None:
        self.journal.record_lock(receipt)

    def record_decode(self, rows: torch.Tensor, lengths: torch.Tensor, step: int) -> None:
        self.journal.record_decode(rows, lengths, step)

    def commit(self) -> None:
        if self.current() is not self or self.committed or self.rolled_back:
            raise RuntimeError("xpool allocation cannot commit outside its unresolved scope")
        for receipt in self.receipts:
            receipt.commit()
        self.journal.commit()
        self.receipts.clear()
        self.waiting_positions.clear()
        self.watched.clear()
        self.committed = True

    def rollback(self) -> None:
        if self.current() is not self or self.committed:
            raise RuntimeError("xpool allocation cannot roll back outside its uncommitted scope")
        if self.rolled_back:
            return
        while self.receipts:
            self.receipts[-1].rollback()
            self.receipts.pop()
        self.journal.rollback()
        if self.scheduler is not None:
            for request, position in sorted(self.waiting_positions.items(), key=lambda entry: entry[1]):
                if request not in self.scheduler.waiting_queue:
                    self.scheduler.waiting_queue.insert(position, request)
            self.scheduler.chunked_req = self.prior_chunk_owner
        self.waiting_positions.clear()
        self.watched.clear()
        self.rolled_back = True
