"""Canonical model identity, independent of serving and native runtimes."""

from __future__ import annotations

import re
from functools import total_ordering
from pathlib import Path
from urllib.parse import quote

from pydantic import ConfigDict, RootModel, field_validator

__all__ = ["ModelId"]


@total_ordering
class ModelId(RootModel[str]):
    """Immutable, case-sensitive model identity in ``namespace/name`` form."""

    model_config = ConfigDict(frozen=True, strict=True)

    @field_validator("root")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        """Validate the canonical spelling at the input boundary."""

        if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) is None or any(
            component in {".", ".."} for component in value.split("/")
        ):
            raise ValueError("model ID must be namespace/name with ASCII letters, digits, '_', '-' or '.'")
        return value

    @property
    def namespace(self) -> str:
        """Namespace component, preserving its canonical spelling."""

        return self.root.split("/")[0]

    @property
    def name(self) -> str:
        """Model name component, preserving its canonical spelling."""

        return self.root.split("/")[1]

    @property
    def relative_path(self) -> Path:
        """Identity path, not a resolved checkpoint location."""

        return Path(self.namespace, self.name)

    def uri_encode(self) -> str:
        """Encode the complete identity as one URI component, including its slash."""

        return quote(self.root, safe="")

    def __str__(self) -> str:
        return self.root

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ModelId):
            return NotImplemented
        return self.root < other.root
