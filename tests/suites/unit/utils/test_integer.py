import pytest

from xpool.utils.integer import (
    SINT64_MAX,
    UINT64_MAX,
    ensure_sint64_nonnegative,
    ensure_uint64,
)


def test_integer_ranges_keep_signed_and_unsigned_boundaries() -> None:
    ensure_sint64_nonnegative(0, SINT64_MAX)
    ensure_uint64(0, SINT64_MAX + 1, UINT64_MAX)
    with pytest.raises(ValueError, match="sint64"):
        ensure_sint64_nonnegative(SINT64_MAX + 1)
    with pytest.raises(ValueError, match="uint64"):
        ensure_uint64(UINT64_MAX + 1)


@pytest.mark.parametrize("value", (True, -1, 1.5, "1"))
def test_integer_ranges_reject_noninteger_or_negative_values(value: object) -> None:
    with pytest.raises(ValueError):
        ensure_sint64_nonnegative(value)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        ensure_uint64(value)  # ty: ignore[invalid-argument-type]
