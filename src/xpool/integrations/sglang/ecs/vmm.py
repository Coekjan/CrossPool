"""Sparse compound VMM storage under one rank-local Elastic Context State byte quota."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.mem_cache.memory_pool import KvBufferDesc
from sglang.srt.utils.cuda_vmm_utils import (
    make_device_allocation_prop,
    make_rw_access_desc,
    tensor_from_pointer,
)

from xpool import ecs
from xpool.utils import align_up
from xpool.utils.cuda import CudaDriverError, check_driver, is_terminal_device_error
from xpool.utils.integer import ensure_uint64


@dataclass(frozen=True, slots=True)
class KvVmmLayout:
    """Uniform component geometry for stable time-major compound storage.

    One physical bundle covers the same row interval in every layer/buffer.
    Logical page zero is dummy storage; its covering bundles stay pinned.
    Describing a layout owns neither a Tensor nor a device allocation.
    """

    device: torch.device
    storage_dtype: torch.dtype
    shape: tuple[int, ...]
    tokens_per_row: int
    token_page_size: int
    layer_count: int
    buffers_per_layer: int
    mapping_granularity_bytes: int
    profile: ecs.PoolProfile = field(init=False)

    def __post_init__(self) -> None:
        ensure_uint64(
            *self.shape,
            self.tokens_per_row,
            self.token_page_size,
            self.layer_count,
            self.buffers_per_layer,
            self.mapping_granularity_bytes,
        )
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("xpool elastic context state backing requires an indexed device")
        if not self.shape or any(dimension == 0 for dimension in self.shape):
            raise ValueError("xpool elastic context state backing requires positive component dimensions")
        if 0 in (
            self.tokens_per_row,
            self.token_page_size,
            self.layer_count,
            self.buffers_per_layer,
            self.mapping_granularity_bytes,
        ):
            raise ValueError("xpool elastic context state backing requires positive storage geometry")
        if self.token_capacity < self.token_page_size:
            raise ValueError("xpool elastic context state reservation requires a usable page after dummy storage")
        granularity = self.mapping_granularity_bytes
        floor_rows = (self.token_page_size + self.tokens_per_row - 1) // self.tokens_per_row
        object.__setattr__(
            self,
            "profile",
            ecs.PoolProfile(
                bundle_bytes=granularity * self.layer_count * self.buffers_per_layer,
                bundle_capacity=align_up(self.shape[0] * self.row_bytes, granularity) // granularity,
                floor_bundles=align_up(floor_rows * self.row_bytes, granularity) // granularity,
            ),
        )

    @property
    def row_bytes(self) -> int:
        """Return one component row's byte stride before layer/buffer interleaving."""

        return math.prod(self.shape[1:]) * self.storage_dtype.itemsize

    @property
    def token_capacity(self) -> int:
        """Return reserved logical tokens excluding the dummy token page."""

        return self.shape[0] * self.tokens_per_row - self.token_page_size


@dataclass(slots=True)
class KvVmmBundle:
    """Retain a physical handle and actual mapping progress through cleanup failures."""

    handle: driver.CUmemGenericAllocationHandle
    mapped: bool = False

    def close(self, address: int, byte_count: int) -> None:
        """Remove the mapping before releasing its physical-allocation handle."""

        if self.mapped:
            (status,) = driver.cuMemUnmap(address, byte_count)
            check_driver(status, "cuMemUnmap(xpool elastic context state bundle)")
            self.mapped = False
        (status,) = driver.cuMemRelease(self.handle)
        check_driver(status, "cuMemRelease(xpool elastic context state bundle)")


class KvVmmBacking:
    """Own stable Tensor views and exact physical bundle IDs under a shared quota.

    The scheduler serializes changes. Callers withdraw affected logical IDs and
    wait for their actual device users before unmapping or closing storage.
    """

    @classmethod
    def describe(
        cls,
        *,
        device: torch.device,
        storage_dtype: torch.dtype,
        token_page_size: int,
        descriptors: Sequence[KvBufferDesc],
        buffers_per_layer: int,
    ) -> KvVmmLayout:
        """Validate uniform descriptors and query granularity without mapping storage.

        The caller has already selected the device. All component descriptors
        must have the same shape, row-byte stride and tokens-per-row geometry.
        """

        if device.type != "cuda":
            raise ValueError("xpool elastic context state backing requires a device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        ensure_uint64(token_page_size, buffers_per_layer)
        if token_page_size == 0 or buffers_per_layer == 0 or not descriptors:
            raise ValueError("xpool elastic context state backing requires positive storage geometry")
        if len(descriptors) % buffers_per_layer:
            raise ValueError("xpool elastic context state descriptors do not form complete layers")
        first = descriptors[0]
        geometry = (first.shape, first.row_bytes, first.tokens_per_row)
        if any(
            (descriptor.shape, descriptor.row_bytes, descriptor.tokens_per_row) != geometry
            for descriptor in descriptors
        ):
            raise ValueError("xpool elastic context state components require identical geometry")
        if math.prod(first.shape[1:]) * storage_dtype.itemsize != first.row_bytes:
            raise ValueError("xpool elastic context state row bytes disagree with component shape and dtype")
        properties = make_device_allocation_prop(device.index, handle_types=None)
        status, granularity = driver.cuMemGetAllocationGranularity(
            properties, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        check_driver(status, "cuMemGetAllocationGranularity(xpool elastic context state)")
        return KvVmmLayout(
            device=device,
            storage_dtype=storage_dtype,
            shape=first.shape,
            tokens_per_row=first.tokens_per_row,
            token_page_size=token_page_size,
            layer_count=len(descriptors) // buffers_per_layer,
            buffers_per_layer=buffers_per_layer,
            mapping_granularity_bytes=int(granularity),
        )

    def __init__(self, layout: KvVmmLayout, quota: ecs.MemoryQuota) -> None:
        """Reserve virtual storage, expose compound views and charge/map the dummy floor."""

        self.layout = layout
        self.quota = quota
        self.base = 0
        self.bundles: dict[int, KvVmmBundle] = {}
        self.raw_storage: torch.Tensor | None = None
        self.tensors: list[torch.Tensor] = []
        self.allocation_properties = make_device_allocation_prop(layout.device.index, handle_types=None)
        self.access_descriptors = [make_rw_access_desc(layout.device.index)]
        reserved_bytes = layout.profile.bytes_for(layout.profile.bundle_capacity)
        status, base = driver.cuMemAddressReserve(reserved_bytes, layout.mapping_granularity_bytes, 0, 0)
        check_driver(status, "cuMemAddressReserve(xpool elastic context state)")
        self.base = int(base)
        try:
            self.raw_storage = tensor_from_pointer(self.base, reserved_bytes, device_id=layout.device.index)
            logical_bytes = layout.shape[0] * layout.row_bytes * layout.layer_count * layout.buffers_per_layer
            logical = (
                self.raw_storage[:logical_bytes]
                .view(layout.storage_dtype)
                .view(layout.shape[0], layout.layer_count, layout.buffers_per_layer, *layout.shape[1:])
            )
            self.tensors = [
                logical[:, layer, buffer, ...]
                for buffer in range(layout.buffers_per_layer)
                for layer in range(layout.layer_count)
            ]
            if not self.try_map(tuple(range(layout.profile.floor_bundles))):
                raise MemoryError("xpool elastic context state bootstrap floor could not be backed")
        except BaseException as error:
            if not self.bundles and not is_terminal_device_error(error):
                self.close()
            raise

    @property
    def backed_bytes(self) -> int:
        """Return actual retained physical bytes, including incomplete mapping ownership."""

        return len(self.bundles) * self.layout.profile.bundle_bytes

    def try_map(self, bundle_ids: tuple[int, ...]) -> bool:
        """Back exact missing IDs, charging before allocation and publishing on success.

        Quota shortage and safely rolled-back allocation exhaustion return
        False. Other errors propagate. Unconfirmed cleanup and terminal device
        failures retain the physical inventory and its quota charge.
        """

        profile = self.layout.profile
        ensure_uint64(*bundle_ids)
        if any(bundle_id >= profile.bundle_capacity for bundle_id in bundle_ids):
            raise ValueError("xpool elastic context state bundle ID exceeds its reservation")
        if self.base == 0:
            raise RuntimeError("xpool elastic context state backing is closed")
        missing = tuple(sorted(set(bundle_ids).difference(self.bundles)))
        byte_count = len(missing) * profile.bundle_bytes
        if not self.quota.try_reserve(byte_count):
            return False

        acquired: list[int] = []
        try:
            for bundle_id in missing:
                status, handle = driver.cuMemCreate(profile.bundle_bytes, self.allocation_properties, 0)
                check_driver(status, "cuMemCreate(xpool elastic context state bundle)")
                bundle = KvVmmBundle(handle)
                self.bundles[bundle_id] = bundle
                acquired.append(bundle_id)
                address = self.base + bundle_id * profile.bundle_bytes
                (status,) = driver.cuMemMap(address, profile.bundle_bytes, 0, handle, 0)
                check_driver(status, "cuMemMap(xpool elastic context state bundle)")
                bundle.mapped = True
                (status,) = driver.cuMemSetAccess(
                    address, profile.bundle_bytes, self.access_descriptors, len(self.access_descriptors)
                )
                check_driver(status, "cuMemSetAccess(xpool elastic context state bundle)")
        except BaseException as error:
            self.quota.release((len(missing) - len(acquired)) * profile.bundle_bytes)
            error.add_note(
                f"requested_bundles={len(missing)} retained_bytes={self.backed_bytes} "
                f"quota_bytes={self.quota.limit_bytes}"
            )
            if is_terminal_device_error(error):
                raise
            for bundle_id in reversed(acquired):
                self.bundles[bundle_id].close(self.base + bundle_id * profile.bundle_bytes, profile.bundle_bytes)
                del self.bundles[bundle_id]
                self.quota.release(profile.bundle_bytes)
            if isinstance(error, CudaDriverError) and error.error_code == driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
                return False
            raise
        return True

    def unmap(self, bundle_ids: tuple[int, ...]) -> int:
        """Release exact unowned, retired IDs and refund only confirmed physical releases.

        The pinned dummy floor cannot be released here. An absent bundle is
        already free. Partial cleanup failure retains all unresolved charges.
        """

        profile = self.layout.profile
        ensure_uint64(*bundle_ids)
        if any(not profile.floor_bundles <= bundle_id < profile.bundle_capacity for bundle_id in bundle_ids):
            raise ValueError("xpool elastic context state unmap requires unpinned bundle IDs within its reservation")
        if self.base == 0:
            raise RuntimeError("xpool elastic context state backing is closed")
        released_bytes = 0
        for bundle_id in sorted(set(bundle_ids).intersection(self.bundles)):
            self.bundles[bundle_id].close(self.base + bundle_id * profile.bundle_bytes, profile.bundle_bytes)
            del self.bundles[bundle_id]
            self.quota.release(profile.bundle_bytes)
            released_bytes += profile.bundle_bytes
        return released_bytes

    def bundle_spans(self, page_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return half-open physical-bundle intervals touched by logical token pages.

        IDs use the child allocator's page units and exclude dummy page zero.
        Conversion stays on the input Tensor's device, including boundary rows
        that contain multiple tokens or straddle physical bundles.
        """

        layout = self.layout
        first_rows = page_ids * layout.token_page_size // layout.tokens_per_row
        past_rows = ((page_ids + 1) * layout.token_page_size + layout.tokens_per_row - 1) // layout.tokens_per_row
        return (
            first_rows * layout.row_bytes // layout.mapping_granularity_bytes,
            (past_rows * layout.row_bytes + layout.mapping_granularity_bytes - 1) // layout.mapping_granularity_bytes,
        )

    def missing_bundle_counts(self, page_ids: torch.Tensor) -> torch.Tensor:
        """Count each page's missing physical bundles on device for cold-path planning."""

        missing = torch.ones(self.layout.profile.bundle_capacity, dtype=torch.int64, device=page_ids.device)
        mapped = torch.tensor(tuple(self.bundles), dtype=torch.int64, device=page_ids.device)
        missing[mapped] = 0
        prefix = torch.cat((missing.new_zeros(1), missing.cumsum(0)))
        starts, ends = self.bundle_spans(page_ids)
        return prefix[ends] - prefix[starts]

    def empty_bundles(self, free_page_ids: torch.Tensor) -> torch.Tensor:
        """Find mapped unpinned bundles whose intersecting pages are all logically free.

        Supply the complete unique free-page snapshot after deferred frees and
        allocation scopes permit it. Interval differences count crossing pages
        on device; only the mapping/reclaim owner transfers selected IDs.
        """

        layout = self.layout
        profile = layout.profile
        bundle_ids = torch.tensor(
            tuple(bundle_id for bundle_id in self.bundles if bundle_id >= profile.floor_bundles),
            dtype=torch.int64,
            device=free_page_ids.device,
        )
        starts, ends = self.bundle_spans(free_page_ids)
        differences = torch.zeros(profile.bundle_capacity + 1, dtype=torch.int64, device=free_page_ids.device)
        differences.scatter_add_(0, starts, torch.ones_like(starts))
        differences.scatter_add_(0, ends, -torch.ones_like(ends))
        free_counts = differences.cumsum(0)[:-1]

        first_rows = bundle_ids * layout.mapping_granularity_bytes // layout.row_bytes
        past_rows = ((bundle_ids + 1) * layout.mapping_granularity_bytes + layout.row_bytes - 1) // layout.row_bytes
        past_rows = past_rows.clamp(max=layout.shape[0])
        past_page = layout.token_capacity // layout.token_page_size + 1
        first_pages = (first_rows * layout.tokens_per_row // layout.token_page_size).clamp(min=1, max=past_page)
        past_pages = ((past_rows * layout.tokens_per_row + layout.token_page_size - 1) // layout.token_page_size).clamp(
            min=1, max=past_page
        )
        return bundle_ids[free_counts[bundle_ids] == past_pages - first_pages]

    def close(self) -> None:
        """Drop owned views and release actual mappings/handles before the virtual range."""

        if self.base == 0:
            return
        self.tensors.clear()
        self.raw_storage = None
        profile = self.layout.profile
        for bundle_id in tuple(self.bundles):
            self.bundles[bundle_id].close(self.base + bundle_id * profile.bundle_bytes, profile.bundle_bytes)
            del self.bundles[bundle_id]
            self.quota.release(profile.bundle_bytes)
        (status,) = driver.cuMemAddressFree(self.base, profile.bytes_for(profile.bundle_capacity))
        check_driver(status, "cuMemAddressFree(xpool elastic context state)")
        self.base = 0
