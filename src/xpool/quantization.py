"""Explicit MXFP4 checkpoint sources and bounded Host decoding."""

from __future__ import annotations

from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["MXFP4_DECODE_BLOCK_LIMIT", "Mxfp4TensorSource", "decode_mxfp4_blocks"]

MXFP4_DECODE_BLOCK_LIMIT = 4096
type PositiveDimension = Annotated[int, Field(strict=True, gt=0)]


class Mxfp4TensorSource(BaseModel):
    """Canonical packed source with an explicit Expert and logical row view.

    Shapes describe physical storage before selection. Each input group has
    32 E2M1 values in 16 bytes and one E8M0 exponent byte. This declaration
    identifies source values independently of their execution representation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    blocks_key: str = Field(strict=True, min_length=1, description="Exact packed-byte safetensors key.")
    scales_key: str = Field(strict=True, min_length=1, description="Exact exponent-byte safetensors key.")
    decoded_shape: (
        tuple[PositiveDimension, PositiveDimension] | tuple[PositiveDimension, PositiveDimension, PositiveDimension]
    ) = Field(description="Physical decoded [N,K] or [E,N,K] geometry before selection.")
    expert_index: int | None = Field(strict=True, ge=0, description="Selected stacked Expert, absent for a matrix.")
    row_part: Literal["all", "even", "odd"] = Field(description="Logical row view after Expert selection.")

    @model_validator(mode="after")
    def validate_selection(self) -> Mxfp4TensorSource:
        """Require canonical groups and a valid, unambiguous matrix view."""

        if self.blocks_key == self.scales_key:
            raise ValueError("MXFP4 blocks and scales keys must differ")
        if self.decoded_shape[-1] % 32:
            raise ValueError("MXFP4 input width must be divisible by 32")
        if len(self.decoded_shape) == 3:
            if self.expert_index is None or self.expert_index >= self.decoded_shape[0]:
                raise ValueError("stacked MXFP4 source requires an in-range Expert index")
        elif self.expert_index is not None:
            raise ValueError("matrix MXFP4 source cannot select an Expert index")
        if self.row_part != "all" and self.decoded_shape[-2] % 2:
            raise ValueError("split MXFP4 rows require an even physical row count")
        return self

    @property
    def logical_shape(self) -> tuple[int, int]:
        """Return the selected [output,input] matrix shape without storing it."""

        rows, columns = self.decoded_shape[-2:]
        return rows if self.row_part == "all" else rows // 2, columns

    @property
    def blocks_shape(self) -> tuple[int, ...]:
        """Return the exact canonical packed-byte header shape."""

        return (*self.decoded_shape[:-1], self.decoded_shape[-1] // 32, 16)

    @property
    def scales_shape(self) -> tuple[int, ...]:
        """Return the exact canonical exponent-byte header shape."""

        return (*self.decoded_shape[:-1], self.decoded_shape[-1] // 32)


def decode_mxfp4_blocks(blocks: torch.Tensor, scales: torch.Tensor, *, out: torch.Tensor) -> None:
    """Decode one bounded CPU tile into caller-owned contiguous BF16/FP16.

    Args:
        blocks: Contiguous CPU uint8 packed bytes shaped [B,16].
        scales: Contiguous CPU uint8 shared exponents shaped [B].
        out: Contiguous CPU BF16/FP16 destination shaped [B,32].

    Raises:
        ValueError: If tensor boundaries, the block limit, a reserved exponent,
            or a non-finite intermediate or converted value is encountered.

    Side Effects:
        Writes ``out`` and retains no resources. A failed call can partially
        write private output, which the caller must not publish.
    """

    for name, tensor in (("blocks", blocks), ("scales", scales), ("out", out)):
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"MXFP4 {name} must be contiguous CPU storage")
    if blocks.dtype is not torch.uint8 or blocks.ndim != 2 or blocks.shape[1] != 16:
        raise ValueError("MXFP4 blocks must be uint8 [B,16]")
    block_count = blocks.shape[0]
    if not 1 <= block_count <= MXFP4_DECODE_BLOCK_LIMIT:
        raise ValueError(f"MXFP4 tile must contain 1..{MXFP4_DECODE_BLOCK_LIMIT} blocks")
    if scales.dtype is not torch.uint8 or scales.shape != (block_count,):
        raise ValueError("MXFP4 scales must be uint8 [B]")
    if out.dtype not in (torch.bfloat16, torch.float16) or out.shape != (block_count, 32):
        raise ValueError("MXFP4 output must be BF16/FP16 [B,32]")
    if bool((scales == 255).any()):
        raise ValueError("MXFP4 scale byte 255 is reserved")

    values = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
        dtype=torch.float32,
        device="cpu",
    )
    indices = torch.stack((blocks & 15, blocks >> 4), dim=-1).reshape(block_count, 32).long()
    decoded = torch.ldexp(values[indices], scales.to(torch.int32).unsqueeze(1) - 127)
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("MXFP4 decoded FP32 values are non-finite")
    out.copy_(decoded)
    if not bool(torch.isfinite(out).all()):
        raise ValueError(f"MXFP4 values overflow {out.dtype}")
