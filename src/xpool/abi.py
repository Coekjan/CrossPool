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

ABI_VERSION = 3


class ForwardMode(IntEnum):
    """Runtime forward modes accepted by the xpool FFN descriptor ABI.

    Attributes:
        EXTEND: Extend/prefill-mode FFN request for a contiguous prompt-token batch.
        DECODE: Decode-mode FFN request for one scheduler decode step.
    """

    EXTEND = 1
    DECODE = 2


class TensorDType(IntEnum):
    """Hidden-state tensor dtypes represented in packed FFN descriptors.

    Attributes:
        BF16: bfloat16 hidden states.
        FP16: IEEE float16 hidden states.
        FP32: IEEE float32 hidden states.
    """

    BF16 = 1
    FP16 = 2
    FP32 = 3


class DescriptorStatus(IntEnum):
    """Lifecycle states shared by FFN request and result descriptors.

    Attributes:
        EMPTY: Descriptor lane is available for a new request.
        PUBLISHED: Attention-side shim has published a request for polling.
        GRANTED: Device agent has granted the communication slot.
        DONE: FFN execution completed and the output descriptor is valid.
        FAILED: FFN execution failed and the result descriptor carries an error code.
    """

    EMPTY = 0
    PUBLISHED = 1
    GRANTED = 2
    DONE = 3
    FAILED = 4


@dataclass(frozen=True, slots=True)
class FfnRequestDescriptor:
    """Packed FFN request descriptor shared with native code.

    Attributes:
        STRUCT: Little-endian struct packer that mirrors ``FfnRequestDescriptor`` in ``abi.hpp``.
        sequence: Monotonic lane sequence used to distinguish graph replays and stale slots.
        instance_id: Integer runtime instance index from xpool config declaration order.
        model_id: Integer model index selecting the FFN weight set on the executor side.
        layer_id: Decoder layer id whose FFN implementation should consume this request.
        forward_mode: Decode or extend mode represented in the native ABI.
        dtype: Hidden-state dtype used by both the request input and output tensor.
        status: Request-lane state machine value.
        input_offset: Byte offset of the input hidden-state tensor inside the shared arena.
        output_offset: Byte offset of the output hidden-state tensor inside the shared arena.
        scratch_offset: Byte offset of per-request scratch space, or zero when no scratch is used.
        num_tokens: Number of token rows in the contiguous ``[num_tokens, hidden_size]`` tensor.
        hidden_size: Hidden dimension columns in the FFN input/output tensor.
        slot_id: Communication-slot id granted to this request and echoed in the result.
        reserved: Future ABI extension field; producers must write zero.
    """

    STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIQIIIIIIQQQIIII")

    sequence: int
    instance_id: int
    model_id: int
    layer_id: int
    forward_mode: ForwardMode
    dtype: TensorDType
    status: DescriptorStatus
    input_offset: int
    output_offset: int
    scratch_offset: int
    num_tokens: int
    hidden_size: int
    slot_id: int = 0
    reserved: int = 0

    @classmethod
    def byte_size(cls) -> int:
        """Return the packed descriptor byte size.

        Returns:
            Size in bytes of the Python packer, which must match
            ``xpool::kFfnRequestDescriptorBytes``.
        """

        return cls.STRUCT.size

    def pack(self) -> bytes:
        """Serialize this descriptor into the native little-endian ABI layout.

        Returns:
            Packed bytes suitable for writing into a shared descriptor lane.

        Side Effects:
            None; the dataclass is immutable and packing allocates only the returned ``bytes`` object.
        """

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
            self.input_offset,
            self.output_offset,
            self.scratch_offset,
            self.num_tokens,
            self.hidden_size,
            self.slot_id,
            self.reserved,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "FfnRequestDescriptor":
        """Decode and validate a native request descriptor payload.

        Args:
            payload: Byte payload read from a request descriptor lane.

        Returns:
            Immutable Python view of the decoded request descriptor.

        Raises:
            ValueError: If the payload length, ABI version, descriptor byte size, enum values,
                or reserved field do not match the active ABI.
        """

        if len(payload) != cls.byte_size():
            raise ValueError(f"unexpected descriptor payload length: {len(payload)}")
        values = cls.STRUCT.unpack(payload)
        abi_version = values[0]
        descriptor_bytes = values[1]
        reserved = values[15]
        if abi_version != ABI_VERSION:
            raise ValueError(f"unsupported ABI version: {abi_version}")
        if descriptor_bytes != cls.byte_size():
            raise ValueError(f"unexpected descriptor size: {descriptor_bytes}")
        if reserved != 0:
            raise ValueError(f"reserved descriptor field must be zero, got {reserved}")
        return cls(
            sequence=values[2],
            instance_id=values[3],
            model_id=values[4],
            layer_id=values[5],
            forward_mode=ForwardMode(values[6]),
            dtype=TensorDType(values[7]),
            status=DescriptorStatus(values[8]),
            input_offset=values[9],
            output_offset=values[10],
            scratch_offset=values[11],
            num_tokens=values[12],
            hidden_size=values[13],
            slot_id=values[14],
            reserved=reserved,
        )


@dataclass(frozen=True, slots=True)
class FfnResultDescriptor:
    """Packed FFN result descriptor shared with native code.

    Attributes:
        STRUCT: Little-endian struct packer that mirrors ``FfnResultDescriptor`` in ``abi.hpp``.
        sequence: Request sequence completed by this result; consumers must match it.
        status: Result-lane state, normally ``DONE`` or ``FAILED``.
        error_code: Native executor error code; zero means success.
        slot_id: Communication-slot id released or failed by this result.
        reserved: Future ABI extension field; producers must write zero.
        output_offset: Byte offset of the completed output tensor inside the shared arena.
    """

    STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIQIIIIQ")

    sequence: int
    status: DescriptorStatus
    error_code: int
    slot_id: int
    reserved: int = 0
    output_offset: int = 0

    @classmethod
    def byte_size(cls) -> int:
        """Return the packed result descriptor byte size.

        Returns:
            Size in bytes of the Python packer, which must match
            ``xpool::kFfnResultDescriptorBytes``.
        """

        return cls.STRUCT.size

    def pack(self) -> bytes:
        """Serialize this result descriptor into the native little-endian ABI layout.

        Returns:
            Packed bytes suitable for writing into a shared result descriptor lane.

        Side Effects:
            None; the dataclass is immutable and packing allocates only the returned ``bytes`` object.
        """

        return self.STRUCT.pack(
            ABI_VERSION,
            self.byte_size(),
            self.sequence,
            int(self.status),
            self.error_code,
            self.slot_id,
            self.reserved,
            self.output_offset,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "FfnResultDescriptor":
        """Decode and validate a native result descriptor payload.

        Args:
            payload: Byte payload read from a result descriptor lane.

        Returns:
            Immutable Python view of the decoded result descriptor.

        Raises:
            ValueError: If the payload length, ABI version, descriptor byte size, enum values,
                or reserved field do not match the active ABI.
        """

        if len(payload) != cls.byte_size():
            raise ValueError(f"unexpected result payload length: {len(payload)}")
        values = cls.STRUCT.unpack(payload)
        abi_version = values[0]
        descriptor_bytes = values[1]
        reserved = values[6]
        if abi_version != ABI_VERSION:
            raise ValueError(f"unsupported ABI version: {abi_version}")
        if descriptor_bytes != cls.byte_size():
            raise ValueError(f"unexpected descriptor size: {descriptor_bytes}")
        if reserved != 0:
            raise ValueError(f"reserved result field must be zero, got {reserved}")
        return cls(
            sequence=values[2],
            status=DescriptorStatus(values[3]),
            error_code=values[4],
            slot_id=values[5],
            reserved=reserved,
            output_offset=values[7],
        )
