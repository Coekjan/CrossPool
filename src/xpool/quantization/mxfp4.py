"""Packed FP4 values with E8M0 scaling for each 32 input elements."""

from __future__ import annotations

from typing import Literal

import torch

from xpool.quantization.base import QuantizationScheme, TensorLayout, TensorSlice, WeightLayout, WeightSlices

__all__ = ["Mxfp4"]


class Mxfp4(QuantizationScheme):
    """Two FP4 values per byte and one E8M0 scale byte per 32 input values."""

    kind: Literal["mxfp4"] = "mxfp4"

    def validate_shape(self, logical_shape: tuple[int, ...]) -> None:
        """Require complete input scale blocks independently of kernel tiles."""

        super().validate_shape(logical_shape)
        if logical_shape[-1] % 32:
            raise ValueError("mxfp4 input width must contain complete 32-value blocks")

    def weight_layout(self, *, logical_shape: tuple[int, ...], dtype: torch.dtype) -> WeightLayout:
        """Describe packed values and canonical strided E8M0 scales.

        Effective floating precision does not change the encoded byte dtypes.
        Scale byte 127 represents unit scale for neutral zero-valued weights.
        """

        self.validate_shape(logical_shape)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("mxfp4 weights require an effective floating dtype")
        return WeightLayout(
            values=TensorLayout((*logical_shape[:-1], logical_shape[-1] // 2), torch.uint8, 0),
            scales=TensorLayout((*logical_shape[:-1], logical_shape[-1] // 32), torch.uint8, 127),
        )

    def slice_alignment(self, *, logical_shape: tuple[int, ...], axis: int) -> int:
        """Require complete scale blocks only along the input axis."""

        super().slice_alignment(logical_shape=logical_shape, axis=axis)
        return 32 if axis == len(logical_shape) - 1 else 1

    def weight_slices(self, *, logical_shape: tuple[int, ...], selection: TensorSlice) -> WeightSlices:
        """Translate a source interval to packed value and scale intervals."""

        self.validate_shape(logical_shape)
        selection.validate_bounds(logical_shape)
        if selection.axis != len(logical_shape) - 1:
            return WeightSlices(values=selection, scales=selection)
        if selection.step != 1 or selection.start % 32 or selection.length % 32:
            raise ValueError("mxfp4 input slices must preserve contiguous complete 32-value blocks")
        return WeightSlices(
            values=TensorSlice(selection.axis, selection.start // 2, selection.length // 2),
            scales=TensorSlice(selection.axis, selection.start // 32, selection.length // 32),
        )
