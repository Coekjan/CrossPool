"""Immutable strict protocol produced by isolated pytest collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Self, TypeGuard

from pydantic import JsonValue

from xkit import ResourceRequirements
from xkit.results import write_json
from xtest.harness.runner.artifact import ArtifactGroupRef

type JsonObject = dict[str, object]


def is_json_object(value: object) -> TypeGuard[JsonObject]:
    """Return whether a decoded value is a string-keyed JSON object."""

    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


class TestStage(StrEnum):
    """Ordered suite stage derived only from a canonical test path."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    MODELS = "models"

    @classmethod
    def from_path(cls, path: str) -> Self:
        """Derive the sole legal stage for one repository-relative test path."""

        parts = PurePosixPath(path).parts
        if len(parts) < 4 or parts[:2] != ("tests", "suites"):
            raise ValueError(f"test path is outside a canonical test stage: {path!r}")
        try:
            return cls(parts[2])
        except ValueError as error:
            raise ValueError(f"test path is outside a canonical test stage: {path!r}") from error


@dataclass(frozen=True, slots=True)
class CollectedTestCase:
    """One concrete pytest item and its complete scheduling metadata."""

    path: str
    nodeid: str
    stage: TestStage
    requirements: ResourceRequirements
    estimated_duration_seconds: float | None
    timeout_seconds: float
    artifact_group: ArtifactGroupRef | None

    def __post_init__(self) -> None:
        parsed_path = PurePosixPath(self.path)
        if not self.path or "\\" in self.path or parsed_path.is_absolute() or ".." in parsed_path.parts:
            raise ValueError(f"collected test path must be canonical and repository-relative: {self.path!r}")
        if self.stage is not TestStage.from_path(self.path):
            raise ValueError(f"collected test stage disagrees with path: {self.path!r}")
        if not self.nodeid.startswith(f"{self.path}::"):
            raise ValueError(f"collected test nodeid does not belong to path: {self.nodeid!r}")
        if self.stage is TestStage.UNIT and self.requirements.cuda_count != 0:
            raise ValueError("unit tests cannot require CUDA")
        if self.estimated_duration_seconds is not None and self.estimated_duration_seconds <= 0:
            raise ValueError("estimated test duration must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("test timeout must be positive")

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        """Strictly parse one collected-case JSON object."""

        expected = {
            "path",
            "nodeid",
            "stage",
            "requirements",
            "estimated_duration_seconds",
            "timeout_seconds",
            "artifact_group",
        }
        if not is_json_object(raw) or set(raw) != expected:
            raise ValueError(f"collected test case must contain exactly {sorted(expected)}")
        path = raw["path"]
        nodeid = raw["nodeid"]
        stage = raw["stage"]
        estimate = raw["estimated_duration_seconds"]
        timeout = raw["timeout_seconds"]
        artifact_group = raw["artifact_group"]
        if not isinstance(path, str) or not isinstance(nodeid, str) or not isinstance(stage, str):
            raise ValueError("collected test path, nodeid, and stage must be strings")
        if estimate is not None and (not isinstance(estimate, int | float) or isinstance(estimate, bool)):
            raise ValueError("estimated test duration must be numeric or null")
        if not isinstance(timeout, int | float) or isinstance(timeout, bool):
            raise ValueError("test timeout must be numeric")
        return cls(
            path=path,
            nodeid=nodeid,
            stage=TestStage(stage),
            requirements=ResourceRequirements.from_raw(raw["requirements"]),
            estimated_duration_seconds=float(estimate) if estimate is not None else None,
            timeout_seconds=float(timeout),
            artifact_group=ArtifactGroupRef.from_raw(artifact_group) if artifact_group is not None else None,
        )

    def raw(self) -> dict[str, JsonValue]:
        """Project this collected case to its JSON representation."""

        return {
            "path": self.path,
            "nodeid": self.nodeid,
            "stage": self.stage.value,
            "requirements": self.requirements.raw(),
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "timeout_seconds": self.timeout_seconds,
            "artifact_group": self.artifact_group.raw() if self.artifact_group is not None else None,
        }


@dataclass(frozen=True, slots=True)
class TestPlan:
    """Strict ordered result of one isolated pytest collection worker."""

    cases: tuple[CollectedTestCase, ...]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("test plan must contain at least one case")
        nodeids = tuple(case.nodeid for case in self.cases)
        if len(nodeids) != len(set(nodeids)):
            raise ValueError("test plan nodeids must be unique")
        artifact_groups: dict[tuple[str, str], tuple[int, int]] = {}
        for case in self.cases:
            if case.artifact_group is not None:
                group = case.artifact_group
                key = (group.kind, group.name)
                collected_count, expected_count = artifact_groups.get(key, (0, group.expected_case_count))
                if group.expected_case_count != expected_count:
                    raise ValueError(f"artifact group {key!r} has inconsistent expected_case_count values")
                artifact_groups[key] = (collected_count + 1, expected_count)
        incomplete = sorted(
            key
            for key, (collected_count, expected_count) in artifact_groups.items()
            if collected_count != expected_count
        )
        if incomplete:
            raise ValueError(f"artifact groups must contain every expected case: {incomplete}")

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        """Strictly parse one complete Test Plan JSON object."""

        if not is_json_object(raw) or set(raw) != {"cases"} or not isinstance(raw["cases"], list):
            raise ValueError("test plan must contain exactly one cases array")
        return cls(tuple(CollectedTestCase.from_raw(case) for case in raw["cases"]))

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read one complete Test Plan from an isolated collection worker."""

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"failed to read Test Plan {path}: {error}") from error
        return cls.from_raw(raw)

    def write(self, path: Path) -> None:
        """Atomically publish this complete Test Plan."""

        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, self.raw())

    def raw(self) -> dict[str, JsonValue]:
        """Project this Test Plan to its JSON representation."""

        return {"cases": [case.raw() for case in self.cases]}
