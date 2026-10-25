"""Ordinary floating weight storage through the common representation contract."""

from __future__ import annotations

from typing import Literal

import torch

from xpool.quantization.base import QuantizationScheme, TensorLayout, TensorSlice, WeightLayout, WeightSlices

__all__ = ["Unquantized"]


class Unquantized(QuantizationScheme):
    """Retain logical weight shape and effective floating precision unchanged."""

    kind: Literal["unquantized"] = "unquantized"

    def weight_layout(self, *, logical_shape: tuple[int, ...], dtype: torch.dtype) -> WeightLayout:
        """Describe one floating allocation, with no scale storage."""

        self.validate_shape(logical_shape)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("unquantized weight storage requires a floating dtype")
        return WeightLayout(values=TensorLayout(logical_shape, dtype, 0), scales=None)

    def weight_slices(self, *, logical_shape: tuple[int, ...], selection: TensorSlice) -> WeightSlices:
        """Preserve a valid logical selection as the physical value selection."""

        self.validate_shape(logical_shape)
        selection.validate_bounds(logical_shape)
        return WeightSlices(values=selection, scales=None)
