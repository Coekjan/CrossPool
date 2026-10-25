"""Checked integer representation at native and calibration boundaries."""

from __future__ import annotations

__all__ = ["SINT64_MAX", "UINT64_MAX", "ensure_sint64_nonnegative", "ensure_uint64"]

SINT64_MAX = (1 << 63) - 1
UINT64_MAX = (1 << 64) - 1


def ensure_uint64(*values: int) -> None:
    """Require integers in ``[0, UINT64_MAX]``, rejecting Boolean values."""

    if any(type(value) is not int or not 0 <= value <= UINT64_MAX for value in values):
        raise ValueError("values require uint64 integers")


def ensure_sint64_nonnegative(*values: int) -> None:
    """Require integers in ``[0, SINT64_MAX]``, rejecting Boolean values."""

    if any(type(value) is not int or not 0 <= value <= SINT64_MAX for value in values):
        raise ValueError("values require non-negative sint64 integers")
