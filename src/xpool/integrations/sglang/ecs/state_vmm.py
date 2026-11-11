"""Descriptor-driven stable virtual storage for retained state allocation units."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.utils.cuda_vmm_utils import make_device_allocation_prop, make_rw_access_desc, tensor_from_pointer

from xpool import ecs
from xpool.integrations.sglang.ecs.vmm import KvVmmBundle
from xpool.utils import align_up
from xpool.utils.cuda import CudaDriverError, check_driver, is_terminal_device_error
from xpool.utils.integer import ensure_uint64


@dataclass(frozen=True, slots=True)
class StateBufferLayout:
    """Contiguous tensor geometry with one logical allocation-unit axis.

    Strides count elements. A unit can select disjoint byte intervals when its
    axis is inside the tensor. Descriptions own no physical or virtual storage.
    """

    dtype: torch.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    unit_axis: int
    mapping_granularity_bytes: int

    def __post_init__(self) -> None:
        ensure_uint64(*self.shape, *self.strides, self.unit_axis, self.mapping_granularity_bytes)
        if not self.shape or any(size == 0 for size in self.shape):
            raise ValueError("state storage requires positive tensor dimensions")
        if not 0 <= self.unit_axis < len(self.shape) or self.mapping_granularity_bytes == 0:
            raise ValueError("state storage requires a bounded unit axis and positive mapping granularity")
        stride = 1
        expected: list[int] = []
        for size in reversed(self.shape):
            expected.append(stride)
            stride *= size
        if self.strides != tuple(reversed(expected)):
            raise ValueError("state storage requires contiguous tensor strides")
        ensure_uint64(self.byte_count, self.reserved_bytes)

    @property
    def byte_count(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def reserved_bytes(self) -> int:
        return align_up(self.byte_count, self.mapping_granularity_bytes)

    def coverage(self, unit_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Return exact physical granules intersecting the selected logical units."""

        ensure_uint64(*unit_ids)
        capacity = self.shape[self.unit_axis]
        if any(unit >= capacity for unit in unit_ids):
            raise ValueError("state allocation unit exceeds its reservation")
        width = self.strides[self.unit_axis] * self.dtype.itemsize
        outer_count = math.prod(self.shape[: self.unit_axis])
        outer_stride = capacity * width
        granularity = self.mapping_granularity_bytes
        result: set[int] = set()
        for outer, unit in itertools.product(range(outer_count), unit_ids):
            first = outer * outer_stride + unit * width
            result.update(range(first // granularity, (first + width + granularity - 1) // granularity))
        return tuple(sorted(result))


class StateVmmBacking:
    """Own stable mixed-dtype views and sparse physical mappings under one quota.

    The scheduler owner supplies logical pins and ensures all device users have
    completed before unmapping. Logical ownership and state initialization stay
    with the selected state implementation.
    """

    @staticmethod
    def granularity(device: torch.device, *, recommended: bool = True) -> int:
        if device.type != "cuda" or device.index is None:
            raise ValueError("state backing requires an indexed CUDA device")
        properties = make_device_allocation_prop(device.index, handle_types=None)
        flag = (
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            if recommended
            else driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        status, granularity = driver.cuMemGetAllocationGranularity(properties, flag)
        check_driver(status, "cuMemGetAllocationGranularity(xpool state)")
        return int(granularity)

    def __init__(
        self,
        layouts: tuple[StateBufferLayout, ...],
        quota: ecs.MemoryQuota,
        *,
        device: torch.device,
        pinned_units: tuple[int, ...] = (0,),
    ) -> None:
        if device.type != "cuda" or device.index is None or not layouts:
            raise ValueError("state backing requires buffer layouts and an indexed CUDA device")
        if len({layout.shape[layout.unit_axis] for layout in layouts}) != 1:
            raise ValueError("state buffers must describe the same logical allocation-unit capacity")
        self.layouts = layouts
        self.quota = quota
        self.device = device
        self.properties = make_device_allocation_prop(device.index, handle_types=None)
        self.access = [make_rw_access_desc(device.index)]
        self.bases: list[int] = []
        self.raw_storage: list[torch.Tensor] = []
        self.tensors: list[torch.Tensor] = []
        self.mappings: dict[tuple[int, int], KvVmmBundle] = {}
        self.pinned = frozenset(self.coverage(pinned_units))
        try:
            for layout in layouts:
                status, base = driver.cuMemAddressReserve(layout.reserved_bytes, layout.mapping_granularity_bytes, 0, 0)
                check_driver(status, "cuMemAddressReserve(xpool state)")
                self.bases.append(int(base))
                raw = tensor_from_pointer(int(base), layout.reserved_bytes, device_id=device.index)
                self.raw_storage.append(raw)
                self.tensors.append(raw[: layout.byte_count].view(layout.dtype).view(layout.shape))
            if not self.try_map(tuple(sorted(self.pinned))):
                raise MemoryError("xpool state bootstrap floor could not be backed")
        except BaseException as error:
            if not is_terminal_device_error(error):
                self.close()
            raise

    @property
    def profiles(self) -> tuple[ecs.PoolProfile, ...]:
        return tuple(
            ecs.PoolProfile(
                layout.mapping_granularity_bytes,
                layout.reserved_bytes // layout.mapping_granularity_bytes,
                sum(buffer == index for buffer, granule in self.pinned),
            )
            for index, layout in enumerate(self.layouts)
        )

    @property
    def backed_bytes(self) -> int:
        return sum(self.layouts[buffer].mapping_granularity_bytes for buffer, granule in self.mappings)

    def coverage(self, unit_ids: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        return tuple(
            (index, granule) for index, layout in enumerate(self.layouts) for granule in layout.coverage(unit_ids)
        )

    def missing_bytes(self, unit_ids: tuple[int, ...]) -> int:
        return sum(
            self.layouts[buffer].mapping_granularity_bytes
            for buffer, granule in self.coverage(unit_ids)
            if (buffer, granule) not in self.mappings
        )

    def try_map(self, granules: tuple[tuple[int, int], ...]) -> bool:
        """Charge before mapping; a clean shortage preserves all prior mappings."""

        for buffer, granule in granules:
            ensure_uint64(buffer, granule)
            if buffer >= len(self.layouts) or granule >= (
                self.layouts[buffer].reserved_bytes // self.layouts[buffer].mapping_granularity_bytes
            ):
                raise ValueError("state granule exceeds its reservation")
        if len(self.bases) != len(self.layouts) or any(base == 0 for base in self.bases):
            raise RuntimeError("state backing is closed")
        missing = tuple(sorted(set(granules).difference(self.mappings)))
        cost = sum(self.layouts[buffer].mapping_granularity_bytes for buffer, granule in missing)
        if not self.quota.try_reserve(cost):
            return False
        acquired: list[tuple[int, int]] = []
        try:
            for key in missing:
                buffer, granule = key
                size = self.layouts[buffer].mapping_granularity_bytes
                status, handle = driver.cuMemCreate(size, self.properties, 0)
                check_driver(status, "cuMemCreate(xpool state)")
                mapping = KvVmmBundle(handle)
                self.mappings[key] = mapping
                acquired.append(key)
                address = self.bases[buffer] + granule * size
                (status,) = driver.cuMemMap(address, size, 0, handle, 0)
                check_driver(status, "cuMemMap(xpool state)")
                mapping.mapped = True
                (status,) = driver.cuMemSetAccess(address, size, self.access, len(self.access))
                check_driver(status, "cuMemSetAccess(xpool state)")
        except BaseException as error:
            retained = sum(self.layouts[buffer].mapping_granularity_bytes for buffer, granule in acquired)
            self.quota.release(cost - retained)
            error.add_note(f"retained_state_bytes={self.backed_bytes}")
            if is_terminal_device_error(error):
                raise
            for key in reversed(acquired):
                self.release_mapping(key)
            if isinstance(error, CudaDriverError) and error.error_code == driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
                return False
            raise
        return True

    def release_mapping(self, key: tuple[int, int]) -> int:
        buffer, granule = key
        size = self.layouts[buffer].mapping_granularity_bytes
        self.mappings[key].close(self.bases[buffer] + granule * size, size)
        del self.mappings[key]
        self.quota.release(size)
        return size

    def unmap(self, granules: tuple[tuple[int, int], ...]) -> int:
        """Release fence-confirmed unowned granules; pinned granules stay resident."""

        if set(granules).intersection(self.pinned):
            raise ValueError("state unmap requires unpinned granules")
        return sum(self.release_mapping(key) for key in sorted(set(granules).intersection(self.mappings)))

    def empty_granules(self, owned_units: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        protected = self.pinned.union(self.coverage(owned_units))
        return tuple(sorted(set(self.mappings).difference(protected)))

    def close(self) -> None:
        """Release exact physical owners before freeing virtual ranges after quiescence."""

        self.tensors.clear()
        self.raw_storage.clear()
        for key in tuple(self.mappings):
            self.release_mapping(key)
        for index, base in enumerate(self.bases):
            if base:
                (status,) = driver.cuMemAddressFree(base, self.layouts[index].reserved_bytes)
                check_driver(status, "cuMemAddressFree(xpool state)")
                self.bases[index] = 0
