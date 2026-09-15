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

        self.token_page_size = int(token_page_size)
        self.mapping_granularity_bytes = get_device_granularity(device.index)
        layer_count = len(descriptors) // buffers_per_layer
        self.row_bytes = row_bytes
        self.tokens_per_row = int(first.tokens_per_row)
        self.token_capacity = rows * self.tokens_per_row - self.token_page_size
        if self.token_capacity <= 0:
            raise ValueError("xpool elastic kv reservation does not contain a usable token page")

        component_span_bytes = rows * self.row_bytes
        self.bundle_capacity = (
            align_up(component_span_bytes, self.mapping_granularity_bytes) // self.mapping_granularity_bytes
        )
        self.bundle_bytes = self.mapping_granularity_bytes * len(descriptors)
        bootstrap_rows = (self.token_page_size + self.tokens_per_row - 1) // self.tokens_per_row
        self.floor_bundles = max(
            1,
            (bootstrap_rows * self.row_bytes + self.mapping_granularity_bytes - 1) // self.mapping_granularity_bytes,
        )
        if self.floor_bundles > self.bundle_capacity:
            raise ValueError("xpool elastic kv bootstrap floor exceeds its reservation")

        self.reserved_bytes = self.bundle_capacity * self.bundle_bytes
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
                        self.mapping_granularity_bytes,
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
                self.resize(self.floor_bundles)
            except BaseException:
                self.close()
                raise

    def partition_profile(self) -> KvCapacityPartitionProfile:
        """Project immutable backing geometry into the Instance registration value."""

        return KvCapacityPartitionProfile(
            bundle_bytes=self.bundle_bytes,
            bundle_capacity=self.bundle_capacity,
            floor_bundles=self.floor_bundles,
            token_capacity=self.token_capacity,
            mapping_granularity_bytes=self.mapping_granularity_bytes,
            row_bytes=self.row_bytes,
            tokens_per_row=self.tokens_per_row,
            token_page_size=self.token_page_size,
        )

    def usable_tokens(self, bundle_count: int) -> int:
        """Return the page-aligned allocator capacity safely covered by a bundle prefix."""

        if not 0 <= bundle_count <= self.bundle_capacity:
            raise ValueError("xpool elastic kv bundle count is outside its reservation")
        backed_tokens = (
            bundle_count * self.mapping_granularity_bytes // self.row_bytes * self.tokens_per_row - self.token_page_size
        )
        return min(self.token_capacity, max(0, backed_tokens)) // self.token_page_size * self.token_page_size

    def resize(self, bundle_count: int) -> None:
        """Map or unmap complete tail bundles until the physical prefix reaches ``bundle_count``."""

        if not self.floor_bundles <= bundle_count <= self.bundle_capacity:
            raise ValueError("xpool elastic kv bundle count is outside its service range")
        if self.base == 0:
            raise RuntimeError("xpool elastic kv backing is closed")

        while self.backed_bundles < bundle_count:
            address = self.base + self.backed_bundles * self.bundle_bytes
            handle = check_drv(
                driver.cuMemCreate(self.bundle_bytes, self.allocation_properties, 0),
                "cuMemCreate(xpool kv bundle)",
            )
            mapped = False
            try:
                check_drv(driver.cuMemMap(address, self.bundle_bytes, 0, handle, 0), "cuMemMap(xpool kv bundle)")
                mapped = True
                check_drv(
                    driver.cuMemSetAccess(
                        address,
                        self.bundle_bytes,
                        self.access_descriptors,
                        len(self.access_descriptors),
                    ),
                    "cuMemSetAccess(xpool kv bundle)",
                )
                check_drv(driver.cuMemRelease(handle), "cuMemRelease(xpool kv bundle)")
                handle = None
            except BaseException:
                if mapped:
                    check_drv(driver.cuMemUnmap(address, self.bundle_bytes), "cuMemUnmap(xpool kv rollback)")
                if handle is not None:
                    check_drv(driver.cuMemRelease(handle), "cuMemRelease(xpool kv rollback)")
                raise
            self.backed_bundles += 1

        while self.backed_bundles > bundle_count:
            address = self.base + (self.backed_bundles - 1) * self.bundle_bytes
            check_drv(driver.cuMemUnmap(address, self.bundle_bytes), "cuMemUnmap(xpool kv bundle)")
            self.backed_bundles -= 1

    def close(self) -> None:
        """Drop every Tensor view, unmap the actual prefix, and release the virtual range."""

        if self.base == 0:
            return
        self.tensors.clear()
        self.raw_storage = None
        while self.backed_bundles:
            address = self.base + (self.backed_bundles - 1) * self.bundle_bytes
            check_drv(driver.cuMemUnmap(address, self.bundle_bytes), "cuMemUnmap(xpool kv close)")
            self.backed_bundles -= 1
        check_drv(driver.cuMemAddressFree(self.base, self.reserved_bytes), "cuMemAddressFree(xpool kv)")
        self.base = 0
