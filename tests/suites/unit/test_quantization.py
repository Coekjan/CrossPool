from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from xpool.quantization import MXFP4_DECODE_BLOCK_LIMIT, Mxfp4TensorSource, decode_mxfp4_blocks


def scalar_decode(blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    magnitudes = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    rows = []
    for block, scale in zip(blocks.tolist(), scales.tolist(), strict=True):
        row = []
        for byte in block:
            for code in (byte % 16, byte // 16):
                sign = -1.0 if code >= 8 else 1.0
                row.append(math.ldexp(sign * magnitudes[code % 8], scale - 127))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64).to(dtype)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("scale", (0, 100, 103, 104, 126, 127, 128, 139))
def test_decode_matches_scalar_values_order_rounding_and_signed_zero(dtype: torch.dtype, scale: int) -> None:
    blocks = torch.tensor([[code + 16 * (15 - code) for code in range(16)]], dtype=torch.uint8)
    scales = torch.tensor([scale], dtype=torch.uint8)
    out = torch.empty((1, 32), dtype=dtype)

    assert decode_mxfp4_blocks(blocks, scales, out=out) is None

    expected = scalar_decode(blocks, scales, dtype)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(out), torch.signbit(expected))


@pytest.mark.parametrize("scale", (252, 254))
def test_decode_large_finite_bf16_exponents(scale: int) -> None:
    blocks = torch.full((2, 16), 0x22, dtype=torch.uint8)
    scales = torch.full((2,), scale, dtype=torch.uint8)
    out = torch.empty((2, 32), dtype=torch.bfloat16)
    decode_mxfp4_blocks(blocks, scales, out=out)
    torch.testing.assert_close(out, scalar_decode(blocks, scales, out.dtype), rtol=0, atol=0)


def test_decoder_ignores_non_cpu_default_factory_device() -> None:
    blocks = torch.full((1, 16), 0x22, dtype=torch.uint8)
    scales = torch.tensor([127], dtype=torch.uint8)
    out = torch.empty((1, 32), dtype=torch.float16)
    with torch.device("meta"):
        decode_mxfp4_blocks(blocks, scales, out=out)
    assert torch.equal(out, torch.ones_like(out))


@pytest.mark.parametrize(
    ("scale", "dtype", "match"),
    ((255, torch.bfloat16, "reserved"), (254, torch.bfloat16, "FP32"), (141, torch.float16, "overflow")),
)
def test_decode_rejects_reserved_and_nonfinite_values(scale: int, dtype: torch.dtype, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        decode_mxfp4_blocks(
            torch.full((1, 16), 0x77, dtype=torch.uint8),
            torch.tensor([scale], dtype=torch.uint8),
            out=torch.empty((1, 32), dtype=dtype),
        )


@pytest.mark.parametrize(
    "invalid",
    (
        "blocks_dtype",
        "blocks_shape",
        "scales_dtype",
        "scales_shape",
        "out_dtype",
        "out_shape",
        "blocks_stride",
        "scales_stride",
        "out_stride",
        "device",
    ),
)
def test_decoder_validates_tensor_boundary(invalid: str) -> None:
    blocks = torch.zeros((2, 16), dtype=torch.uint8)
    scales = torch.full((2,), 127, dtype=torch.uint8)
    out = torch.empty((2, 32), dtype=torch.float16)
    if invalid == "blocks_dtype":
        blocks = blocks.long()
    elif invalid == "blocks_shape":
        blocks = blocks[:, :15].contiguous()
    elif invalid == "scales_dtype":
        scales = scales.long()
    elif invalid == "scales_shape":
        scales = scales.view(2, 1)
    elif invalid == "out_dtype":
        out = out.float()
    elif invalid == "out_shape":
        out = out[:, :31].contiguous()
    elif invalid == "blocks_stride":
        blocks = torch.zeros((2, 32), dtype=torch.uint8)[:, ::2]
    elif invalid == "scales_stride":
        scales = torch.full((4,), 127, dtype=torch.uint8)[::2]
    elif invalid == "out_stride":
        out = torch.empty((2, 64), dtype=torch.float16)[:, ::2]
    else:
        blocks = torch.empty((2, 16), dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError):
        decode_mxfp4_blocks(blocks, scales, out=out)


@pytest.mark.parametrize("count", (0, MXFP4_DECODE_BLOCK_LIMIT, MXFP4_DECODE_BLOCK_LIMIT + 1))
def test_decoder_block_limit(count: int) -> None:
    blocks = torch.zeros((count, 16), dtype=torch.uint8)
    scales = torch.full((count,), 127, dtype=torch.uint8)
    out = torch.empty((count, 32), dtype=torch.bfloat16)
    if count == MXFP4_DECODE_BLOCK_LIMIT:
        decode_mxfp4_blocks(blocks, scales, out=out)
        assert torch.count_nonzero(out) == 0
    else:
        with pytest.raises(ValueError, match="blocks"):
            decode_mxfp4_blocks(blocks, scales, out=out)


@pytest.mark.parametrize(
    ("shape", "index", "part", "logical"),
    (((6, 64), None, "all", (6, 64)), ((6, 64), None, "odd", (3, 64)), ((2, 6, 64), 1, "even", (3, 64))),
)
def test_source_selection_shapes_and_roundtrip(
    shape: tuple[int, ...],
    index: int | None,
    part: str,
    logical: tuple[int, int],
) -> None:
    source = Mxfp4TensorSource.model_validate(
        {
            "blocks_key": "blocks",
            "scales_key": "scales",
            "decoded_shape": shape,
            "expert_index": index,
            "row_part": part,
        }
    )
    assert source.logical_shape == logical
    assert source.blocks_shape == (*shape[:-1], shape[-1] // 32, 16)
    assert source.scales_shape == (*shape[:-1], shape[-1] // 32)
    assert "logical_shape" not in source.model_dump()
    assert Mxfp4TensorSource.model_validate_json(source.model_dump_json()) == source
    with pytest.raises(ValidationError):
        source.blocks_key = "replacement"


@pytest.mark.parametrize(
    "changes",
    (
        {"blocks_key": ""},
        {"scales_key": "blocks"},
        {"blocks_key": 1},
        {"unknown": 0},
        {"decoded_shape": (True, 32)},
        {"decoded_shape": (2, False, 32)},
        {"decoded_shape": (2, 0)},
        {"decoded_shape": (2, 31)},
        {"decoded_shape": (2, 32, 1, 1)},
        {"decoded_shape": ("2", 32)},
        {"expert_index": True},
        {"expert_index": 0},
        {"row_part": "bad"},
        {"decoded_shape": (3, 32), "row_part": "odd"},
        {"decoded_shape": (2, 4, 32)},
        {"decoded_shape": (2, 4, 32), "expert_index": 2},
        {"decoded_shape": (2, 4, 32), "expert_index": -1},
    ),
)
def test_source_rejects_invalid_or_ambiguous_selection(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Mxfp4TensorSource.model_validate(
            {
                "blocks_key": "blocks",
                "scales_key": "scales",
                "decoded_shape": (4, 32),
                "expert_index": None,
                "row_part": "all",
                **changes,
            }
        )
