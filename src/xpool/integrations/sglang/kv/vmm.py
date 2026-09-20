"""Reversible compound CUDA VMM backing for one SGLang KV pool."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cuda.bindings.driver as driver  # ty: ignore[unresolved-import]
import torch
from sglang.srt.mem_cache.memory_pool import KvBufferDesc
from sglang.srt.utils.cuda_vmm_utils import (
    check_drv,
    get_device_granularity,
    make_device_allocation_prop,
    make_rw_access_desc,
    tensor_from_pointer,
)

from xpool.service.wire import KvCapacityPartitionProfile
from xpool.utils import align_up


class KvVmmBacking:
    """Own one stable compound virtual range and its mapped bundle prefix."""

    def __init__(
        self,
        *,
        device: torch.device,
        storage_dtype: torch.dtype,
        token_page_size: int,
        descriptors: Sequence[KvBufferDesc],
        buffers_per_layer: int,
    ) -> None:
        """Reserve the compound range, expose ordered views, and map the bootstrap floor."""

        if device.type != "cuda":
            raise ValueError("xpool elastic kv backing requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if token_page_size <= 0 or buffers_per_layer <= 0 or not descriptors:
            raise ValueError("xpool elastic kv backing geometry must be positive")
        if len(descriptors) % buffers_per_layer != 0:
            raise ValueError("xpool elastic kv descriptors do not form complete layers")

        first = descriptors[0]
        if not first.shape or first.shape[0] <= 0:
            raise ValueError("xpool elastic kv descriptors require a positive row count")
        geometry = (first.shape, first.row_bytes, first.tokens_per_row)
        if any(
            (descriptor.shape, descriptor.row_bytes, descriptor.tokens_per_row) != geometry
            for descriptor in descriptors
        ):
            raise ValueError("xpool elastic kv compound components require identical geometry")

        rows = int(first.shape[0])
        row_shape = tuple(int(dimension) for dimension in first.shape[1:])
        row_bytes = int(first.row_bytes)
        if math.prod(row_shape) * storage_dtype.itemsize != row_bytes:
            raise ValueError("xpool elastic kv descriptor row bytes disagree with its shape and dtype")

        token_page_size = int(token_page_size)
        mapping_granularity_bytes = get_device_granularity(device.index)
        layer_count = len(descriptors) // buffers_per_layer
        tokens_per_row = int(first.tokens_per_row)
        token_capacity = rows * tokens_per_row - token_page_size
        if token_capacity <= 0:
            raise ValueError("xpool elastic kv reservation does not contain a usable token page")

        component_span_bytes = rows * row_bytes
        bundle_capacity = align_up(component_span_bytes, mapping_granularity_bytes) // mapping_granularity_bytes
        bundle_bytes = mapping_granularity_bytes * len(descriptors)
        bootstrap_rows = (token_page_size + tokens_per_row - 1) // tokens_per_row
        floor_bundles = max(
            1,
            (bootstrap_rows * row_bytes + mapping_granularity_bytes - 1) // mapping_granularity_bytes,
        )
        self.capacity_profile = KvCapacityPartitionProfile(
            bundle_bytes=bundle_bytes,
            bundle_capacity=bundle_capacity,
            floor_bundles=floor_bundles,
            token_capacity=token_capacity,
            mapping_granularity_bytes=mapping_granularity_bytes,
            row_bytes=row_bytes,
            tokens_per_row=tokens_per_row,
            token_page_size=token_page_size,
        )
        self.reserved_bytes = bundle_capacity * bundle_bytes
        self.base = 0
        self.backed_bundles = 0
        self.raw_storage: torch.Tensor | None = None
        self.tensors: list[torch.Tensor] = []

        allocation_properties = make_device_allocation_prop(device.index)
        self.access_descriptors = [make_rw_access_desc(device.index)]
        with torch.cuda.device(device):
            self.base = int(
                check_drv(
                    driver.cuMemAddressReserve(
                        self.reserved_bytes,
                        mapping_granularity_bytes,
                        0,
                        0,
                    ),
                    "cuMemAddressReserve(xpool kv)",
                )
            )
            try:
                self.raw_storage = tensor_from_pointer(
                    self.base,
                    self.reserved_bytes,
                    device_id=device.index,
                )
                logical_elements = rows * layer_count * buffers_per_layer * math.prod(row_shape)
                logical = (
                    self.raw_storage[: logical_elements * storage_dtype.itemsize]
                    .view(storage_dtype)
                    .view(
                        rows,
                        layer_count,
                        buffers_per_layer,
                        *row_shape,
                    )
                )
                self.tensors = [
                    logical[:, layer, buffer, ...]
                    for buffer in range(buffers_per_layer)
                    for layer in range(layer_count)
                ]
                self.allocation_properties = allocation_properties
                self.resize(floor_bundles)
            except BaseException:
                self.close()
                raise

    def usable_tokens(self, bundle_count: int) -> int:
        """Return the page-aligned allocator capacity safely covered by a bundle prefix."""

        return self.capacity_profile.usable_tokens(bundle_count)

    def required_bundles(self, token_count: int) -> int:
        """Return the smallest compound-bundle count covering ``token_count`` tokens."""

        if token_count < 0:
            raise ValueError("xpool elastic kv token demand must be nonnegative")
        profile = self.capacity_profile
        required_rows = math.ceil((token_count + profile.token_page_size) / profile.tokens_per_row)
        return math.ceil(required_rows * profile.row_bytes / profile.mapping_granularity_bytes)

    def resize(self, bundle_count: int) -> None:
        """Map or unmap complete tail bundles until the physical prefix reaches ``bundle_count``."""

        profile = self.capacity_profile
        if not profile.floor_bundles <= bundle_count <= profile.bundle_capacity:
            raise ValueError("xpool elastic kv bundle count is outside its service range")
        if self.base == 0:
            raise RuntimeError("xpool elastic kv backing is closed")

        while self.backed_bundles < bundle_count:
            address = self.base + self.backed_bundles * profile.bundle_bytes
            handle = check_drv(
                driver.cuMemCreate(profile.bundle_bytes, self.allocation_properties, 0),
                "cuMemCreate(xpool kv bundle)",
            )
            mapped = False
            try:
                check_drv(
                    driver.cuMemMap(address, profile.bundle_bytes, 0, handle, 0),
                    "cuMemMap(xpool kv bundle)",
                )
                mapped = True
                check_drv(
                    driver.cuMemSetAccess(
                        address,
                        profile.bundle_bytes,
                        self.access_descriptors,
                        len(self.access_descriptors),
                    ),
                    "cuMemSetAccess(xpool kv bundle)",
                )
                check_drv(driver.cuMemRelease(handle), "cuMemRelease(xpool kv bundle)")
                handle = None
            except BaseException:
                if mapped:
                    check_drv(driver.cuMemUnmap(address, profile.bundle_bytes), "cuMemUnmap(xpool kv rollback)")
                if handle is not None:
                    check_drv(driver.cuMemRelease(handle), "cuMemRelease(xpool kv rollback)")
                raise
            self.backed_bundles += 1

        while self.backed_bundles > bundle_count:
            address = self.base + (self.backed_bundles - 1) * profile.bundle_bytes
            check_drv(driver.cuMemUnmap(address, profile.bundle_bytes), "cuMemUnmap(xpool kv bundle)")
            self.backed_bundles -= 1

    def close(self) -> None:
        """Drop every Tensor view, unmap the actual prefix, and release the virtual range."""

        if self.base == 0:
            return
        self.tensors.clear()
        self.raw_storage = None
        bundle_bytes = self.capacity_profile.bundle_bytes
        while self.backed_bundles:
            address = self.base + (self.backed_bundles - 1) * bundle_bytes
            check_drv(driver.cuMemUnmap(address, bundle_bytes), "cuMemUnmap(xpool kv close)")
            self.backed_bundles -= 1
        check_drv(driver.cuMemAddressFree(self.base, self.reserved_bytes), "cuMemAddressFree(xpool kv)")
        self.base = 0
