"""Generic cross-task artifact-group contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, cast

from pydantic import JsonValue

from xtest.harness.runner.pytest_report import PytestCaseReport


@dataclass(frozen=True, slots=True)
class ArtifactGroupRef:
    """One complete cross-task artifact group declared during collection."""

    kind: str
    name: str
    expected_case_count: int

    def __post_init__(self) -> None:
        if not self.kind or not self.name:
            raise ValueError("artifact group kind and name must be nonempty")
        if (
            not isinstance(self.expected_case_count, int)
            or isinstance(self.expected_case_count, bool)
            or self.expected_case_count < 2
        ):
            raise ValueError("artifact group expected_case_count must be an integer of at least two")

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        """Strictly parse the complete collection-protocol JSON record."""

        expected = {"kind", "name", "expected_case_count"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"artifact group must contain exactly {sorted(expected)}")
        fields = cast(dict[str, object], raw)
        kind = fields["kind"]
        name = fields["name"]
        expected_case_count = fields["expected_case_count"]
        if not isinstance(kind, str) or not isinstance(name, str):
            raise ValueError("artifact group kind and name must be strings")
        if not isinstance(expected_case_count, int) or isinstance(expected_case_count, bool):
            raise ValueError("artifact group expected_case_count must be an integer")
        return cls(kind=kind, name=name, expected_case_count=expected_case_count)

    def raw(self) -> dict[str, JsonValue]:
        """Serialize the collection-protocol fields in their declared form."""

        return {"kind": self.kind, "name": self.name, "expected_case_count": self.expected_case_count}


@dataclass(frozen=True, slots=True)
class ArtifactGroupResult:
    """One adapter-classified artifact-group result."""

    name: str
    result_code: int
    detail: str | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("artifact group result name must be nonempty")
        if self.result_code not in {0, 1, 2}:
            raise ValueError("artifact group result_code must be 0, 1, or 2")


class ArtifactGroupAdapter(Protocol):
    """Component-owned evaluator injected into the generic suite runner."""

    kind: str

    def evaluate(
        self,
        group: ArtifactGroupRef,
        reports: Sequence[PytestCaseReport],
        artifact_directories: Sequence[Path],
    ) -> ArtifactGroupResult:
        """Classify one complete group after all case reports are available."""

        ...
