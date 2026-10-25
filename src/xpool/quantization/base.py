"""Immutable weight-encoding declarations and derived storage geometry."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict

__all__ = ["QuantizationScheme", "TensorLayout", "TensorSlice", "WeightLayout", "WeightSlices"]


@dataclass(frozen=True, slots=True)
class TensorLayout:
    """Storage shape, element type and valid neutral initialization value.

    The description owns no Tensor. Shapes have positive integral dimensions;
    encoded scale storage may have a nonzero neutral value.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    zero_value: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("tensor layout requires a nonempty shape tuple")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError("tensor layout dimensions must be positive integers")
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError("tensor layout requires a torch dtype")
        if (
            isinstance(self.zero_value, bool)
            or not isinstance(self.zero_value, (int, float))
            or not math.isfinite(self.zero_value)
        ):
            raise ValueError("tensor layout neutral value must be finite and numeric")


@dataclass(frozen=True, slots=True)
class WeightLayout:
    """Encoded value storage and optional scale storage for one logical weight."""

    values: TensorLayout
    scales: TensorLayout | None


@dataclass(frozen=True, slots=True)
class TensorSlice:
    """A positive-step source selection along one nonnegative, zero-based axis.

    Length counts selected elements rather than their span. Bounds against the
    source shape are checked by the scheme. Zero length is valid for TP ranks
    owning an empty source suffix.
    """

    axis: int
    start: int
    length: int
    step: int = 1

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.axis, self.start, self.length)
        ):
            raise ValueError("tensor slice axis, start and length must be nonnegative integers")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step <= 0:
            raise ValueError("tensor slice step must be a positive integer")

    def validate_bounds(self, shape: tuple[int, ...]) -> None:
        """Require the final selected element to fit, allowing an empty suffix."""

        if self.axis >= len(shape):
            raise ValueError("tensor slice axis is outside its source")
        size = shape[self.axis]
        if self.start > size or (self.length and self.start + (self.length - 1) * self.step >= size):
            raise ValueError("tensor slice exceeds its source")

    def indices(self) -> tuple[slice, ...]:
        """Return basic indexing views; preceding axes remain unchanged."""

        return (slice(None),) * self.axis + (slice(self.start, self.start + self.length * self.step, self.step),)


@dataclass(frozen=True, slots=True)
class WeightSlices:
    """Physical selections corresponding to one logical weight selection."""

    values: TensorSlice
    scales: TensorSlice | None


class QuantizationScheme(BaseModel, ABC):
    """Serializable encoding declaration with allocation-free geometry methods.

    Logical weights end in ``(output, input)`` and may have leading Expert
    dimensions. Concrete schemes determine encoded storage, not model identity,
    TP placement or operator launch requirements.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @abstractmethod
    def weight_layout(self, *, logical_shape: tuple[int, ...], dtype: torch.dtype) -> WeightLayout:
        """Describe encoded storage using the effective floating weight dtype.

        Raises:
            ValueError: If shape or dtype cannot represent this encoding.
        """

    @abstractmethod
    def weight_slices(self, *, logical_shape: tuple[int, ...], selection: TensorSlice) -> WeightSlices:
        """Translate a bounded logical selection, preserving complete encoding blocks.

        Raises:
            ValueError: If the source shape, axis, bounds or alignment is invalid.
        """

    def slice_alignment(self, *, logical_shape: tuple[int, ...], axis: int) -> int:
        """Return the logical slice quantum after validating shape and axis.

        Ordinary floating axes have quantum one. Schemes override the quantum
        where encoded blocks must remain complete; kernel tile alignment is
        outside this contract.
        """

        self.validate_shape(logical_shape)
        if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis < len(logical_shape):
            raise ValueError("weight slice axis is outside its logical shape")
        return 1

    def validate_shape(self, logical_shape: tuple[int, ...]) -> None:
        """Require positive matrix geometry before encoding-specific calculations."""

        if not isinstance(logical_shape, tuple) or len(logical_shape) < 2:
            raise ValueError("logical weight shape must end in output and input dimensions")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in logical_shape):
            raise ValueError("logical weight dimensions must be positive integers")
