"""Stable catalogue identities and single-writer TOML case authoring."""

from __future__ import annotations

import os
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Self
from uuid import UUID, uuid4

import tomli_w
from pydantic import BaseModel, ConfigDict, RootModel, field_validator

__all__ = ["CaseFamily", "CaseId", "CaseLocation", "Catalog"]


class CaseId(RootModel[str]):
    """Immutable UUIDv4 identity; prefixes select values but are never persisted."""

    model_config = ConfigDict(frozen=True, strict=True)

    @field_validator("root")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        """Require a complete lowercase, hyphenated UUIDv4 at the value boundary."""
        identity = UUID(value)
        if identity.version != 4 or str(identity) != value:
            raise ValueError("case ID must be a canonical lowercase UUIDv4")
        return value

    @classmethod
    def generate(cls) -> Self:
        """Assign a new authored identity independently of experiment conditions."""
        return cls(str(uuid4()))

    @classmethod
    def resolve(cls, prefix: str, candidates: Iterable[Self]) -> Self:
        """Resolve one nonempty canonical prefix or report unknown/ambiguous input."""
        if not prefix:
            raise ValueError("case prefix must be nonempty")
        matches = tuple(value for value in candidates if str(value).startswith(prefix))
        if not matches:
            raise ValueError(f"unknown case: {prefix!r}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous case {prefix!r}; candidates: {', '.join(str(value) for value in matches)}")
        return matches[0]

    def __str__(self) -> str:
        return self.root


@dataclass(frozen=True, slots=True)
class CaseLocation:
    """Published case's absolute catalogue path and one-based inclusive line bounds."""

    id: CaseId
    path: Path
    first_line: int
    last_line: int


class CaseFamily(StrEnum):
    """Catalogue scenario categories, independent of serving execution mode."""

    SERVING = "serving"
    TOPOLOGY = "topology"

    @property
    def table_name(self) -> str:
        """Return the TOML table containing this family's declarations."""
        return f"{self.value}_cases"


class Catalog(BaseModel, ABC):
    """Immutable loaded snapshot and single-writer authoring of its source file."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: Path

    @classmethod
    @abstractmethod
    def from_file(cls, path: Path) -> Self:
        """Validate the concrete schema and resolve declaring-file-owned paths."""

    def clone(self, prefix: str, *, family: CaseFamily) -> CaseLocation:
        """Copy a raw declaration, validating before atomic publication.

        Existing text, comments and relative references stay intact. The loaded
        snapshot is unchanged; a later load sees the appended case. Validation
        or I/O failure leaves the source file intact. The caller owns single-
        writer access rather than a concurrent-editor protocol.
        """
        path = self.path
        original = path.read_bytes().decode("utf-8")
        payload = tomllib.loads(original)
        source_id = CaseId.resolve(
            prefix,
            (CaseId(identity) for kind in CaseFamily for identity in payload.get(kind.table_name, {})),
        )
        declarations = payload.get(family.table_name, {})
        if str(source_id) not in declarations:
            raise ValueError(f"{source_id}: source case does not belong to {family.value!r}")
        identity = CaseId.generate()
        appended = tomli_w.dumps({family.table_name: {str(identity): declarations[str(source_id)]}})
        prefix_text = original + ("\n" if original.endswith("\n") else "\n\n")
        first_line = prefix_text.count("\n") + 1
        last_line = first_line + len(appended.splitlines()) - 1
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".toml", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(prefix_text + appended)
                temporary.flush()
                os.fsync(temporary.fileno())
                os.chmod(temporary_path, path.stat().st_mode)
                type(self).from_file(temporary_path)
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)
        return CaseLocation(identity, path, first_line, last_line)
