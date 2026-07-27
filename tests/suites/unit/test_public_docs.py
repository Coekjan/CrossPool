"""Explicit quality gates for public Pydantic field documentation."""

import inspect
from importlib import import_module
from types import ModuleType

from pydantic import BaseModel

import xpool.native

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


def test_native_public_surface_has_runtime_documentation() -> None:
    """Require reflection-visible docs on the generated native API surface."""

    modules = (xpool.native, xpool.native.fabric, xpool.native.transport)
    missing: list[str] = []
    for module in modules:
        if not inspect.getdoc(module):
            missing.append(module.__name__)
        for name, value in inspect.getmembers(module):
            if name.startswith("_") or isinstance(value, ModuleType):
                continue
            qualified_name = f"{module.__name__}.{name}"
            if inspect.isclass(value):
                if not inspect.getdoc(value):
                    missing.append(qualified_name)
                for member_name, member in vars(value).items():
                    if member_name.startswith("_") or member_name in {"name", "value"}:
                        continue
                    if (inspect.isroutine(member) or inspect.isdatadescriptor(member)) and not inspect.getdoc(member):
                        missing.append(f"{qualified_name}.{member_name}")
            elif callable(value) and not inspect.getdoc(value):
                missing.append(qualified_name)

    assert missing == []
