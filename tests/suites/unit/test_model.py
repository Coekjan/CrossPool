"""Canonical model identities retain scalar wire representation."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from xpool.model import ModelId


def test_model_identity_ordering_and_serialization() -> None:
    model = ModelId("Qwen/Qwen3-14B")
    assert (model.namespace, model.name, model.relative_path) == ("Qwen", "Qwen3-14B", Path("Qwen/Qwen3-14B"))
    assert str(model) == "Qwen/Qwen3-14B"
    assert model.uri_encode() == "Qwen%2FQwen3-14B"
    assert model != "Qwen/Qwen3-14B"
    assert len({model, ModelId(str(model))}) == 1
    assert sorted((ModelId("a/z"), ModelId("a-/x"))) == [ModelId("a-/x"), ModelId("a/z")]
    adapter = TypeAdapter(dict[ModelId, tuple[ModelId, ...]])
    mapping: dict[ModelId, tuple[ModelId, ...]] = {model: (model,)}
    serialized = adapter.dump_json(mapping)
    assert serialized == b'{"Qwen/Qwen3-14B":["Qwen/Qwen3-14B"]}'
    assert adapter.validate_json(serialized) == mapping
    with pytest.raises(ValidationError, match="frozen"):
        model.root = "Other/Model"


@pytest.mark.parametrize(
    "value", ["model", "/model", "org/", "org/a/b", "../model", "org/.", " org/model", "组织/model"]
)
def test_model_identity_requires_canonical_components(value: str) -> None:
    with pytest.raises(ValidationError, match="namespace/name"):
        ModelId(value)
