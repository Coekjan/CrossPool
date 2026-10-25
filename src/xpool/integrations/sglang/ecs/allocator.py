"""Resident free IDs and sparse backing plans for the pinned SGLang allocators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.utils.common import get_num_new_pages

from xpool.integrations.sglang.ecs.allocation import SglangAllocation
from xpool.integrations.sglang.ecs.pool import backing_of
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking


@dataclass(frozen=True, slots=True)
class SglangPagePlan:
    """Short-lived free-page realization and its exact missing physical bundle IDs."""

    page_ids: torch.Tensor
    missing_bundle_ids: tuple[int, ...]


class ElasticPageAllocator(BaseTokenToKVPoolAllocator):
    """Shared sparse partition operations for the token and paged child allocators.

    Upstream owns allocation, sorting and deferred frees. Only mapping/reclaim
    boundaries move unowned IDs between resident and unavailable partitions.
    ``retiring_pages`` retains withdrawn IDs until confirmed physical release.
    """

    backing: KvVmmBacking
    free_pages: torch.Tensor
    unavailable_pages: torch.Tensor
    retiring_pages: torch.Tensor

    def clear(self) -> None:
        """Rebuild the sparse free partition after the engine has released logical owners."""

        if self.retiring_pages.numel():
            raise RuntimeError("xpool elastic context state cannot clear an allocator during retirement")
        super().clear()
        resident = self.backing.missing_bundle_counts(self.free_pages) == 0
        self.unavailable_pages = self.free_pages[~resident]
        self.free_pages = self.free_pages[resident]

    def record_acquisition(self, before: torch.Tensor, locations: torch.Tensor | None) -> None:
        """Journal actual consumed IDs after upstream sorting, only in a real allocation phase."""

        attempt = SglangAllocation.current()
        if locations is not None and attempt is not None and attempt.batch is not None:
            acquired = before[: len(before) - len(self.free_pages)]
            attempt.record_allocation(
                cast(TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator, self), acquired, locations
            )

    def plan_pages(self, page_count: int) -> SglangPagePlan | None:
        """Prefer resident holes, then partial backing and stable virtual-page order.

        This computes physical cost without acquiring a page or changing its
        owner. Deferred group frees and the pending retirement are ineligible.
        """

        resident = self.get_all_free_pages()
        if page_count <= len(resident):
            return SglangPagePlan(resident[:page_count], ())
        candidates = self.unavailable_pages[~torch.isin(self.unavailable_pages, self.retiring_pages)]
        if len(resident) + len(candidates) < page_count:
            return None
        candidates = torch.sort(candidates).values
        starts, ends = self.backing.bundle_spans(candidates)
        missing_counts = self.backing.missing_bundle_counts(candidates)
        partial = missing_counts < ends - starts
        candidates = candidates[torch.argsort(partial.to(torch.int8), descending=True, stable=True)]
        selected = candidates[: page_count - len(resident)]
        starts, ends = self.backing.bundle_spans(selected)
        profile = self.backing.layout.profile
        differences = torch.zeros(profile.bundle_capacity + 1, dtype=torch.int64, device=selected.device)
        differences.scatter_add_(0, starts, torch.ones_like(starts))
        differences.scatter_add_(0, ends, -torch.ones_like(ends))
        touched = differences.cumsum(0)[:-1] > 0
        mapped = torch.tensor(tuple(self.backing.bundles), dtype=torch.int64, device=selected.device)
        touched[mapped] = False
        missing_ids = tuple(touched.nonzero().flatten().tolist())
        return SglangPagePlan(torch.cat((resident, selected)), missing_ids)

    def publish_pages(self, page_ids: torch.Tensor) -> None:
        """Expose selected unavailable IDs only after their complete backing is ready."""

        selected = torch.isin(self.unavailable_pages, page_ids)
        pages = self.unavailable_pages[selected]
        if not pages.numel():
            return
        if bool(torch.isin(pages, self.retiring_pages).any().item()):
            raise RuntimeError("xpool elastic context state cannot publish retiring pages")
        if bool((self.backing.missing_bundle_counts(pages) != 0).any().item()):
            raise RuntimeError("xpool elastic context state cannot publish incompletely backed pages")
        self.unavailable_pages = self.unavailable_pages[~selected]
        # Both concrete children provide the upstream page-unit release API.
        cast(TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator, self).free_page_ids(pages)

    def intersecting_pages(self, page_ids: torch.Tensor, bundle_ids: tuple[int, ...]) -> torch.Tensor:
        """Select free pages touching a proposed physical retirement, on device."""

        selected = torch.zeros(self.backing.layout.profile.bundle_capacity, dtype=torch.int64, device=page_ids.device)
        selected[torch.tensor(bundle_ids, dtype=torch.int64, device=page_ids.device)] = 1
        prefix = torch.cat((selected.new_zeros(1), selected.cumsum(0)))
        starts, ends = self.backing.bundle_spans(page_ids)
        return prefix[ends] != prefix[starts]

    def begin_retirement(self, bundle_ids: tuple[int, ...]) -> None:
        """Withdraw and retain intersecting free IDs until physical unmap.

        The caller has proved the bundles completely unowned. Subclasses keep
        each upstream mergeable free container out of the retirement set.
        """

        if self.retiring_pages.numel():
            raise RuntimeError("xpool elastic context state allocator already has a pending retirement")
        exposed = self.get_all_free_pages()
        pages = torch.cat((exposed, self.unavailable_pages))
        self.retiring_pages = pages[self.intersecting_pages(pages, bundle_ids)]
        new_unavailable = exposed[self.intersecting_pages(exposed, bundle_ids)]
        self.unavailable_pages = torch.cat((self.unavailable_pages, new_unavailable))

    def finish_retirement(self) -> None:
        """End exclusion after confirmed physical unmap; IDs remain unavailable."""

        self.retiring_pages = self.retiring_pages[:0]


class ElasticTokenToKVPoolAllocator(ElasticPageAllocator, TokenToKVPoolAllocator):
    """Ordinary token allocation over fully backed, one-based free IDs."""

    release_pages: torch.Tensor

    def __init__(self, size: int, dtype: torch.dtype, device: str, kvcache: KVCache, need_sort: bool) -> None:
        self.backing = backing_of(kvcache)
        self.retiring_pages = torch.empty(0, dtype=torch.int64, device=device)
        super().__init__(size, dtype, device, kvcache, need_sort)

    def alloc(self, need_size: int) -> torch.Tensor | None:
        """Keep original token selection and record only its actual new ownership."""

        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()
        before = self.free_pages
        locations = super().alloc(need_size)
        self.record_acquisition(before, locations)
        return locations

    def begin_retirement(self, bundle_ids: tuple[int, ...]) -> None:
        """Exclude IDs from both immediate and sort-deferred token containers."""

        super().begin_retirement(bundle_ids)
        self.free_pages = self.free_pages[~self.intersecting_pages(self.free_pages, bundle_ids)]
        self.release_pages = self.release_pages[~self.intersecting_pages(self.release_pages, bundle_ids)]


class ElasticPagedTokenToKVPoolAllocator(ElasticPageAllocator, PagedTokenToKVPoolAllocator):
    """Paged allocation whose immediate and staged free IDs always have full backing."""

    def __init__(
        self, size: int, page_size: int, dtype: torch.dtype, device: str, kvcache: KVCache, need_sort: bool
    ) -> None:
        self.backing = backing_of(kvcache)
        self.retiring_pages = torch.empty(0, dtype=torch.int64, device=device)
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

    def alloc(self, need_size: int) -> torch.Tensor | None:
        """Keep original paged selection and record ownership in page units."""

        if need_size // self.page_size > len(self.free_pages):
            self.merge_and_sort_free()
        before = self.free_pages
        locations = super().alloc(need_size)
        self.record_acquisition(before, locations)
        return locations

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int | None = None,
    ) -> torch.Tensor | None:
        """Retain the upstream merge and kernel, then receipt the actual consumed prefix."""

        if extend_num_tokens // self.page_size + len(prefix_lens) + 1 > len(self.free_pages):
            self.merge_and_sort_free()
        before = self.free_pages
        if num_new_pages is None:
            # The pinned helper returns a CPU .item() integer despite its Tensor annotation.
            num_new_pages = cast(
                int, get_num_new_pages(seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu)
            )
        locations = super().alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        self.record_acquisition(before, locations)
        return locations

    def alloc_decode(
        self, seq_lens: torch.Tensor, seq_lens_cpu: torch.Tensor, last_loc: torch.Tensor
    ) -> torch.Tensor | None:
        """Record returned mapping locations even when Decode reuses a partial page."""

        if len(seq_lens) > len(self.free_pages):
            self.merge_and_sort_free()
        before = self.free_pages
        locations = super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        self.record_acquisition(before, locations)
        return locations

    def begin_retirement(self, bundle_ids: tuple[int, ...]) -> None:
        """Exclude IDs from immediate and every sort-deferred page container."""

        super().begin_retirement(bundle_ids)
        self.free_pages = self.free_pages[~self.intersecting_pages(self.free_pages, bundle_ids)]
        self.staged_pages = [staged[~self.intersecting_pages(staged, bundle_ids)] for staged in self.staged_pages]
        self.num_staged_pages = sum(len(staged) for staged in self.staged_pages)
