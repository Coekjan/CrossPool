"""Explicit quality gates for public Pydantic field documentation."""

import inspect
from importlib import import_module

from pydantic import BaseModel

PUBLIC_MODEL_MODULES = (
    "xpool.config",
    "xpool.integrations.sglang.topology",
    "xpool.runtime.transport",
    "xpool.service.wire",
)


def test_public_pydantic_fields_have_descriptions() -> None:
    """Require declaration-site descriptions for public Pydantic fields."""

    missing: list[str] = []
    for module_name in PUBLIC_MODEL_MODULES:
        module = import_module(module_name)
        for name, value in inspect.getmembers(module, inspect.isclass):
            if name.startswith("_") or value.__module__ != module_name or not issubclass(value, BaseModel):
                continue
            for field_name, field in value.model_fields.items():
                if not field.description:
                    missing.append(f"{module_name}.{name}.{field_name}")

    assert missing == []
