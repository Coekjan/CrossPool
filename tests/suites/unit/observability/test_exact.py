import json
from fractions import Fraction

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic.json_schema import JsonSchemaMode

from xpool.observability.calibration import Exact


@pytest.mark.parametrize("value", [Fraction(0), Fraction(-7, 3), Fraction(1, 3), Fraction(2**160 + 1, 3)])
def test_exact_values_round_trip_without_precision_loss(value: Fraction) -> None:
    adapter = TypeAdapter(Exact)
    document = {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    assert adapter.validate_python(value, strict=True) == value
    assert adapter.dump_python(value) == value
    assert adapter.validate_python(document, strict=True) == value
    assert adapter.validate_json(adapter.dump_json(value), strict=True) == value
    assert adapter.dump_python(value, mode="json") == document


@pytest.mark.parametrize(
    "value",
    [
        {"numerator": "1", "denominator": "0"},
        {"numerator": "1", "denominator": "-3"},
        {"numerator": "2", "denominator": "4"},
        {"numerator": "0", "denominator": "2"},
        {"numerator": "-0", "denominator": "1"},
        {"numerator": "01", "denominator": "3"},
        {"numerator": "+1", "denominator": "3"},
        {"numerator": "1.5", "denominator": "3"},
        {"numerator": "\uff11", "denominator": "3"},
        {"numerator": "1\n", "denominator": "3"},
        {"numerator": "1", "denominator": ""},
        {"numerator": 1, "denominator": "3"},
        {"numerator": "1", "denominator": True},
        {"numerator": "1"},
        {"numerator": "1", "denominator": "3", "extra": "field"},
        "1/3",
        0.5,
        [1, 3],
        None,
    ],
)
def test_invalid_rationals_raise_validation_errors(value: object) -> None:
    adapter = TypeAdapter(Exact)
    with pytest.raises(ValidationError):
        adapter.validate_python(value, strict=True)
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(value), strict=True)


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_exact_json_schema_describes_rational_object(mode: JsonSchemaMode) -> None:
    adapter = TypeAdapter(Exact)
    schema = adapter.json_schema(mode=mode)
    if "$ref" in schema:
        schema = schema["$defs"][schema["$ref"].removeprefix("#/$defs/")]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"numerator", "denominator"}
    for name in ("numerator", "denominator"):
        assert schema["properties"][name]["type"] == "string"
