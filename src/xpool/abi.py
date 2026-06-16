"""Versioned Python view of the xpool native FFN descriptor ABI.

This module mirrors `src/cext/include/abi.hpp`. Keep enum values, field order,
struct formats, alignment assumptions, and byte sizes in lockstep with the C++
header until xpool grows generated ABI bindings.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

ABI_VERSION = 1


class ForwardMode(IntEnum):
    DECODE = 1
    EXTEND = 2


class TensorDType(IntEnum):
    BF16 = 1
    FP16 = 2
    FP32 = 3


class DescriptorStatus(IntEnum):
    EMPTY = 0
    PUBLISHED = 1
    GRANTED = 2
    DONE = 3
    FAILED = 4


@dataclass(frozen=True, slots=True)
class FfnRequestDescriptor:
    """Packed FFN request descriptor shared with native code."""

    STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIQIIIIIIQQQIIII")

    sequence: int
    instance_id: int
    model_id: int
    layer_id: int
    forward_mode: ForwardMode
    dtype: TensorDType
    status: DescriptorStatus
    input_ptr: int
    output_ptr: int
    scratch_ptr: int
    num_tokens: int
    hidden_size: int
    slot_id: int = 0
    reserved: int = 0

    @classmethod
    def byte_size(cls) -> int:
        return cls.STRUCT.size

    def pack(self) -> bytes:
        return self.STRUCT.pack(
            ABI_VERSION,
            self.byte_size(),
            self.sequence,
            self.instance_id,
            self.model_id,
            self.layer_id,
            int(self.forward_mode),
            int(self.dtype),
            int(self.status),
            self.input_ptr,
            self.output_ptr,
            self.scratch_ptr,
            self.num_tokens,
            self.hidden_size,
            self.slot_id,
            self.reserved,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "FfnRequestDescriptor":
        values = cls.STRUCT.unpack(payload)
        abi_version = values[0]
        descriptor_bytes = values[1]
        if abi_version != ABI_VERSION:
            raise ValueError(f"unsupported ABI version: {abi_version}")
        if descriptor_bytes != cls.byte_size():
            raise ValueError(f"unexpected descriptor size: {descriptor_bytes}")
        return cls(
            sequence=values[2],
            instance_id=values[3],
            model_id=values[4],
            layer_id=values[5],
            forward_mode=ForwardMode(values[6]),
            dtype=TensorDType(values[7]),
            status=DescriptorStatus(values[8]),
            input_ptr=values[9],
            output_ptr=values[10],
            scratch_ptr=values[11],
            num_tokens=values[12],
            hidden_size=values[13],
            slot_id=values[14],
            reserved=values[15],
        )


@dataclass(frozen=True, slots=True)
class FfnResultDescriptor:
    """Packed FFN result descriptor shared with native code."""

    STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIQIIIIQ")

    sequence: int
    status: DescriptorStatus
    error_code: int
    slot_id: int
    reserved: int = 0
    output_ptr: int = 0

    @classmethod
    def byte_size(cls) -> int:
        return cls.STRUCT.size

    def pack(self) -> bytes:
        return self.STRUCT.pack(
            ABI_VERSION,
            self.byte_size(),
            self.sequence,
            int(self.status),
            self.error_code,
            self.slot_id,
            self.reserved,
            self.output_ptr,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "FfnResultDescriptor":
        values = cls.STRUCT.unpack(payload)
        abi_version = values[0]
        descriptor_bytes = values[1]
        if abi_version != ABI_VERSION:
            raise ValueError(f"unsupported ABI version: {abi_version}")
        if descriptor_bytes != cls.byte_size():
            raise ValueError(f"unexpected descriptor size: {descriptor_bytes}")
        return cls(
            sequence=values[2],
            status=DescriptorStatus(values[3]),
            error_code=values[4],
            slot_id=values[5],
            reserved=values[6],
            output_ptr=values[7],
        )
