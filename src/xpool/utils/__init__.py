"""Shared utility helpers for xpool."""

__all__ = ["align_up"]


def align_up(value: int, alignment: int) -> int:
    """Align one non-negative integer upward to a positive boundary."""

    if value < 0 or alignment <= 0:
        raise ValueError("alignment requires a non-negative value and positive boundary")
    return (value + alignment - 1) // alignment * alignment
