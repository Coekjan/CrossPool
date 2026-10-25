from __future__ import annotations

import pytest
import torch
from pydantic import TypeAdapter

from xpool.quantization import (
    Mxfp4,
    QuantizationSpec,
    TensorLayout,
    TensorSlice,
    Unquantized,
    WeightLayout,
    WeightSlices,
)


@pytest.mark.parametrize("shape", [(17, 64), (8, 17, 64)])
def test_weight_layout_preserves_floating_or_encodes_input_blocks(shape: tuple[int, ...]) -> None:
    assert Unquantized().weight_layout(logical_shape=shape, dtype=torch.bfloat16) == WeightLayout(
        values=TensorLayout(shape, torch.bfloat16, 0), scales=None
    )
    assert Mxfp4().weight_layout(logical_shape=shape, dtype=torch.bfloat16) == WeightLayout(
        values=TensorLayout((*shape[:-1], 32), torch.uint8, 0),
        scales=TensorLayout((*shape[:-1], 2), torch.uint8, 127),
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_weight_slices_preserve_logical_ownership_across_encodings(axis: int) -> None:
    shape = (8, 96, 2880)
    selection = TensorSlice(axis, 0 if axis == 0 else 32, 4 if axis == 0 else 64)
    assert Unquantized().weight_slices(logical_shape=shape, selection=selection) == WeightSlices(selection, None)
    packed = Mxfp4().weight_slices(logical_shape=shape, selection=selection)
    if axis == 2:
        assert packed == WeightSlices(TensorSlice(axis, 16, 32), TensorSlice(axis, 1, 2))
    else:
        assert packed == WeightSlices(selection, selection)
    assert Mxfp4().slice_alignment(logical_shape=shape, axis=axis) == (32 if axis == 2 else 1)


def test_empty_tp_source_suffix_has_valid_empty_physical_slices() -> None:
    selection = TensorSlice(2, 2880, 0)
    assert Mxfp4().weight_slices(logical_shape=(8, 64, 2880), selection=selection) == WeightSlices(
        TensorSlice(2, 1440, 0), TensorSlice(2, 90, 0)
    )


def test_strided_output_rows_preserve_count_and_encoding_blocks() -> None:
    selection = TensorSlice(1, 1, 3, step=2)
    source = torch.arange(2 * 6 * 64).view(2, 6, 64)
    selection.validate_bounds(tuple(source.shape))
    assert torch.equal(source[selection.indices()], source[:, 1::2])
    for scheme in (Unquantized(), Mxfp4()):
        projected = scheme.weight_slices(logical_shape=tuple(source.shape), selection=selection)
        assert projected.values == selection
    assert Mxfp4().weight_slices(logical_shape=tuple(source.shape), selection=selection).scales == selection
    with pytest.raises(ValueError, match="exceeds"):
        TensorSlice(1, 1, 4, step=2).validate_bounds(tuple(source.shape))
    with pytest.raises(ValueError, match="contiguous complete"):
        Mxfp4().weight_slices(logical_shape=(2, 128), selection=TensorSlice(1, 0, 32, step=2))


@pytest.mark.parametrize("step", (0, -1, True, 1.5))
def test_slice_steps_require_positive_integers(step: int) -> None:
    with pytest.raises(ValueError, match="step"):
        TensorSlice(0, 0, 1, step=step)


@pytest.mark.parametrize("shape", [(), (64,), (0, 64), (True, 64), (1.5, 64)])
def test_schemes_reject_invalid_logical_geometry(shape: tuple[int, ...]) -> None:
    for scheme in (Unquantized(), Mxfp4()):
        with pytest.raises(ValueError):
            scheme.weight_layout(logical_shape=shape, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "selection",
    [TensorSlice(3, 0, 1), TensorSlice(2, 2880, 32), TensorSlice(2, 1, 32), TensorSlice(2, 0, 31)],
)
def test_packed_slices_reject_out_of_bounds_or_incomplete_blocks(selection: TensorSlice) -> None:
    with pytest.raises(ValueError):
        Mxfp4().weight_slices(logical_shape=(8, 64, 2880), selection=selection)


@pytest.mark.parametrize("shape", [(8, 64, 65), (8, 64, 16)])
def test_packed_layout_rejects_incomplete_scale_blocks(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        Mxfp4().weight_layout(logical_shape=shape, dtype=torch.bfloat16)


def test_encoding_declarations_round_trip_as_immutable_hashable_values() -> None:
    adapter = TypeAdapter(QuantizationSpec)
    for declaration in (Unquantized(), Mxfp4()):
        restored = adapter.validate_json(adapter.dump_json(declaration))
        assert restored == declaration
        assert hash(restored) == hash(declaration)
        with pytest.raises(ValueError):
            declaration.kind = "invalid"  # ty: ignore[invalid-assignment]
