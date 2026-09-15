"""SGLang allocators with an adjustable admitted logical prefix."""

from __future__ import annotations

import torch
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache


def free_suffix_is_complete(
    containers: tuple[torch.Tensor, ...],
    cutoff: int,
    reserved_capacity: int,
) -> bool:
    """Return whether unique free IDs cover the complete suffix above ``cutoff``."""

    free_ids = torch.cat(containers)
    return int(torch.count_nonzero(free_ids > cutoff).item()) == reserved_capacity - cutoff


class ElasticTokenToKVPoolAllocator(TokenToKVPoolAllocator):
    """Token allocator that retains but withholds free suffix slot IDs."""

    free_pages: torch.Tensor
    release_pages: torch.Tensor

    def __init__(self, size: int, dtype: torch.dtype, device: str, kvcache: KVCache, need_sort: bool) -> None:
        self.reserved_token_capacity = size
        self.token_capacity = size
        self.withheld_pages = torch.empty(0, dtype=torch.int64, device=device)
        super().__init__(size, dtype, device, kvcache, need_sort)

    def set_token_capacity(self, token_capacity: int) -> None:
        """Expose exactly the free one-based IDs within ``token_capacity``."""

        if not 0 <= token_capacity <= self.reserved_token_capacity:
            raise ValueError("xpool kv token capacity is outside its reservation")
        if token_capacity > self.token_capacity:
            admitted = self.withheld_pages[self.withheld_pages <= token_capacity]
            self.withheld_pages = self.withheld_pages[self.withheld_pages > token_capacity]
            super().free(admitted)
        elif token_capacity < self.token_capacity:
            free_pages = self.free_pages
            release_pages = self.release_pages
            self.withheld_pages = torch.cat(
                (
                    self.withheld_pages,
                    free_pages[free_pages > token_capacity],
                    release_pages[release_pages > token_capacity],
                )
            )
            self.free_pages = free_pages[free_pages <= token_capacity]
            self.release_pages = release_pages[release_pages <= token_capacity]
        self.token_capacity = token_capacity

    def free(self, free_index: torch.Tensor) -> None:
        """Return active IDs to SGLang and retain suffix IDs outside admission."""

        if free_index.numel() == 0:
            return
        if self.free_group is not None:
            self.free_group.append(self._copy_for_free_group(free_index))
            return
        active = free_index[free_index <= self.token_capacity]
        withheld = free_index[free_index > self.token_capacity]
        super().free(active)
        if withheld.numel():
            self.withheld_pages = torch.cat((self.withheld_pages, withheld))

    def suffix_is_free(self, token_capacity: int) -> bool:
        """Return whether every reserved ID above ``token_capacity`` is currently free."""

        if not 0 <= token_capacity <= self.reserved_token_capacity:
            raise ValueError("xpool kv token capacity is outside its reservation")
        return free_suffix_is_complete(
            (self.free_pages, self.release_pages, self.withheld_pages),
            token_capacity,
            self.reserved_token_capacity,
        )

    def withheld_size(self) -> int:
        """Return the number of free token slots withheld from allocation."""

        return len(self.withheld_pages)

    def clear(self) -> None:
        """Rebuild free and withheld sets without changing the applied cutoff."""

        super().clear()
        self.free_pages = self.free_pages[self.free_pages <= self.token_capacity]
        self.withheld_pages = torch.arange(
            self.token_capacity + 1,
            self.reserved_token_capacity + 1,
            dtype=torch.int64,
            device=self.device,
        )


class ElasticPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    """Paged allocator that retains but withholds free suffix page IDs."""

    free_pages: torch.Tensor
    release_pages: torch.Tensor

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ) -> None:
        self.reserved_token_capacity = size
        self.token_capacity = size
        self.withheld_pages = torch.empty(0, dtype=torch.int64, device=device)
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

    def set_token_capacity(self, token_capacity: int) -> None:
        """Expose exactly the free one-based pages within ``token_capacity``."""

        if token_capacity % self.page_size != 0:
            raise ValueError("xpool paged kv token capacity must be page aligned")
        if not 0 <= token_capacity <= self.reserved_token_capacity:
            raise ValueError("xpool kv token capacity is outside its reservation")
        old_pages = self.token_capacity // self.page_size
        new_pages = token_capacity // self.page_size
        if new_pages > old_pages:
            admitted = self.withheld_pages[self.withheld_pages <= new_pages]
            self.withheld_pages = self.withheld_pages[self.withheld_pages > new_pages]
            super()._release_page_ids(admitted)
        elif new_pages < old_pages:
            free_pages = self.free_pages
            release_pages = self.release_pages
            self.withheld_pages = torch.cat(
                (
                    self.withheld_pages,
                    free_pages[free_pages > new_pages],
                    release_pages[release_pages > new_pages],
                )
            )
            self.free_pages = free_pages[free_pages <= new_pages]
            self.release_pages = release_pages[release_pages <= new_pages]
        self.token_capacity = token_capacity

    def _release_page_ids(self, *page_ids: torch.Tensor) -> None:
        """Return active pages to SGLang and retain pages outside admission."""

        pages = torch.cat(page_ids)
        active_pages = self.token_capacity // self.page_size
        super()._release_page_ids(pages[pages <= active_pages])
        withheld = pages[pages > active_pages]
        if withheld.numel():
            self.withheld_pages = torch.cat((self.withheld_pages, withheld))

    def suffix_is_free(self, token_capacity: int) -> bool:
        """Return whether every reserved page above ``token_capacity`` is currently free."""

        if token_capacity % self.page_size != 0:
            raise ValueError("xpool paged kv token capacity must be page aligned")
        if not 0 <= token_capacity <= self.reserved_token_capacity:
            raise ValueError("xpool kv token capacity is outside its reservation")
        target_pages = token_capacity // self.page_size
        return free_suffix_is_complete(
            (self.free_pages, self.release_pages, self.withheld_pages),
            target_pages,
            self.reserved_token_capacity // self.page_size,
        )

    def withheld_size(self) -> int:
        """Return the number of token slots withheld from allocation."""

        return len(self.withheld_pages) * self.page_size

    def clear(self) -> None:
        """Rebuild free and withheld sets without changing the applied cutoff."""

        super().clear()
        active_pages = self.token_capacity // self.page_size
        self.free_pages = self.free_pages[self.free_pages <= active_pages]
        self.withheld_pages = torch.arange(
            active_pages + 1,
            self.reserved_token_capacity // self.page_size + 1,
            dtype=torch.int64,
            device=self.device,
        )
