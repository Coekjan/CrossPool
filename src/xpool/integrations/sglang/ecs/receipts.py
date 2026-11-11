"""Ownership receipts used by common allocation transactions."""

from collections.abc import Iterable
from typing import Protocol

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams


class StateAllocationReceipt(Protocol):
    """Retain unpublished implementation-owned state until the TP verdict.

    Register before any acquisition. Rollback consumes acquired ownership once
    and preserves unconsumed owners when cleanup fails. Commit transfers live
    obligations to the real request/cache owner.
    """

    def watch(self, requests: Iterable[Req]) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


class StateAllocationJournal(StateAllocationReceipt, Protocol):
    """Record actual engine allocation, row and lock receipts before publication."""

    def record_allocation(
        self, allocator: BaseTokenToKVPoolAllocator, unit_ids: torch.Tensor, locations: torch.Tensor
    ) -> None: ...
    def record_rows(self, rows: list[int]) -> None: ...
    def record_lock(self, receipt: DecLockRefParams) -> None: ...
    def record_decode(self, rows: torch.Tensor, lengths: torch.Tensor, step: int) -> None: ...
