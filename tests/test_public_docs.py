from __future__ import annotations

import dataclasses
import inspect
import re
from enum import Enum
from importlib import import_module
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

PUBLIC_MODULE_NAMES = (
    "xpool.abi",
    "xpool.config",
    "xpool.cext",
    "xpool.daemon",
    "xpool.device_agent",
    "xpool.runtime.mps",
    "xpool.integrations.sglang.adapter",
    "xpool.integrations.sglang.models.deepseek_v2",
    "xpool.integrations.sglang.plugin",
    "xpool.integrations.sglang.registry",
    "xpool.integrations.sglang.server_args",
    "xpool.integrations.sglang.shim",
    "xpool.integrations.sglang.topology",
)


def test_pydantic_fields_have_descriptions() -> None:
    missing: list[str] = []
    for cls in public_classes():
        if not issubclass(cls, BaseModel):
            continue
        for field_name, field in cls.model_fields.items():
            if field_name.startswith("_"):
                continue
            if not field.description:
                missing.append(f"{cls.__module__}.{cls.__name__}.{field_name}")

    assert missing == []


def test_dataclass_fields_are_documented_in_class_docstrings() -> None:
    missing: list[str] = []
    for cls in public_classes():
        if not dataclasses.is_dataclass(cls):
            continue
        doc = inspect.getdoc(cls) or ""
        for field in dataclasses.fields(cls):
            if field.name.startswith("_"):
                continue
            if field.name not in doc:
                missing.append(f"{cls.__module__}.{cls.__name__}.{field.name}")

    assert missing == []


def test_enum_members_are_documented_in_class_docstrings() -> None:
    missing: list[str] = []
    for cls in public_classes():
        if not issubclass(cls, Enum):
            continue
        doc = inspect.getdoc(cls) or ""
        for member_name in cls.__members__:
            if member_name not in doc:
                missing.append(f"{cls.__module__}.{cls.__name__}.{member_name}")

    assert missing == []


def test_native_abi_public_fields_have_doxygen_comments() -> None:
    header = Path("src/cext/include/abi.hpp")
    lines = header.read_text(encoding="utf-8").splitlines()
    missing: list[str] = []
    public_symbol = re.compile(
        r"^\s*(?:"
        r"(?:std::uint(?:32|64)_t)\s+(?P<field>[a-zA-Z_][a-zA-Z0-9_]*)\s*;"
        r"|(?P<enum>k[A-Z][a-zA-Z0-9_]*)\s*="
        r"|inline constexpr std::uint32_t\s+(?P<constant>k[A-Z][a-zA-Z0-9_]*)"
        r")"
    )

    for index, line in enumerate(lines):
        match = public_symbol.match(line)
        if match is None:
            continue
        name = next(value for value in match.groupdict().values() if value is not None)
        if not previous_nonblank_line_is_doxygen(lines, index):
            missing.append(f"{header}:{index + 1}:{name}")

    assert missing == []


def public_classes() -> tuple[type, ...]:
    classes: list[type] = []
    for module_name in PUBLIC_MODULE_NAMES:
        module = import_module(module_name)
        classes.extend(classes_in_module(module))
    return tuple(classes)


def classes_in_module(module: ModuleType) -> tuple[type, ...]:
    classes: list[type] = []
    for name, value in inspect.getmembers(module, inspect.isclass):
        if name.startswith("_"):
            continue
        if value.__module__ != module.__name__:
            continue
        classes.append(value)
    return tuple(classes)


def previous_nonblank_line_is_doxygen(lines: list[str], index: int) -> bool:
    cursor = index - 1
    while cursor >= 0 and not lines[cursor].strip():
        cursor -= 1
    return cursor >= 0 and lines[cursor].lstrip().startswith("///")
