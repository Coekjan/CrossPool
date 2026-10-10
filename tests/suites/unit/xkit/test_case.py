import pytest
from pydantic import TypeAdapter, ValidationError

from xkit.case import CaseId


def test_case_identity_preserves_scalar_serialization_and_prefix_resolution() -> None:
    first = CaseId("550e8400-e29b-41d4-a716-446655440000")
    second = CaseId("550e8401-e29b-41d4-a716-446655440000")
    assert CaseId.resolve("550e8400", (first, second)) == first
    assert CaseId.resolve(str(second), (first, second)) == second
    assert CaseId.model_validate_json(first.model_dump_json()) == first
    assert TypeAdapter(dict[CaseId, int]).dump_json({first: 1}) == (b'{"550e8400-e29b-41d4-a716-446655440000":1}')
    with pytest.raises(ValueError, match=r"ambiguous.*550e8400.*550e8401"):
        CaseId.resolve("550e840", (first, second))
    with pytest.raises(ValueError, match="unknown"):
        CaseId.resolve("missing", (first, second))


@pytest.mark.parametrize("value", ["", "550e8400", "550E8400-E29B-41D4-A716-446655440000"])
def test_case_identity_requires_complete_canonical_spelling(value: str) -> None:
    with pytest.raises(ValidationError):
        CaseId(value)
