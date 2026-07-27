"""Immutable strict protocol produced by isolated pytest collection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Self, TypeGuard

type JsonObject = dict[str, object]


def is_json_object(value: object) -> TypeGuard[JsonObject]:
    """Return whether a decoded value is a string-keyed JSON object."""

    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


@dataclass(frozen=True, slots=True)
class TestRequirements:
    """Complete external resources required by one collected pytest item."""

    cuda_count: int
    requires_mps: bool
    requires_config: bool
    model_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.cuda_count, bool) or self.cuda_count < 0:
            raise ValueError("test requirement cuda_count must be a nonnegative integer")
        if self.requires_mps and self.cuda_count == 0:
            raise ValueError("MPS test requirements must also require CUDA")
        if self.model_ids and not self.requires_config:
            raise ValueError("model-weight test requirements must also require config")
        if any(not model_id for model_id in self.model_ids):
            raise ValueError("test requirement model IDs must be nonempty")
        if len(self.model_ids) != len(set(self.model_ids)):
            raise ValueError("test requirement model IDs must be unique")

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        """Strictly parse one requirements JSON object."""

        expected = {"cuda_count", "requires_mps", "requires_config", "model_ids"}
        if not is_json_object(raw) or set(raw) != expected:
            raise ValueError(f"test requirements must contain exactly {sorted(expected)}")
        cuda_count = raw["cuda_count"]
        requires_mps = raw["requires_mps"]
        requires_config = raw["requires_config"]
        model_ids = raw["model_ids"]
        if not isinstance(cuda_count, int) or isinstance(cuda_count, bool):
            raise ValueError("test requirement cuda_count must be an integer")
        if not isinstance(requires_mps, bool):
            raise ValueError("test requirement requires_mps must be a boolean")
        if not isinstance(requires_config, bool):
            raise ValueError("test requirement requires_config must be a boolean")
        if not isinstance(model_ids, list):
            raise ValueError("test requirement model_ids must be a string array")
        parsed_model_ids: list[str] = []
        for model_id in model_ids:
            if not isinstance(model_id, str):
                raise ValueError("test requirement model_ids must be a string array")
            parsed_model_ids.append(model_id)
        return cls(cuda_count, requires_mps, requires_config, tuple(parsed_model_ids))

    def raw(self) -> JsonObject:
        """Project these requirements to their JSON representation."""

        return {
            "cuda_count": self.cuda_count,
            "requires_mps": self.requires_mps,
            "requires_config": self.requires_config,
            "model_ids": list(self.model_ids),
        }


class TestStage(StrEnum):
    """Ordered suite stage derived only from a canonical test path."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"

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
class TokenParityGroupRef:
    """One complete cross-task token-parity group."""

    name: str
    expected_case_count: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("token parity group name must be nonempty")
        if (
            not isinstance(self.expected_case_count, int)
            or isinstance(self.expected_case_count, bool)
            or self.expected_case_count < 2
        ):
            raise ValueError("token parity group expected_case_count must be an integer of at least two")

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        """Strictly parse one parity-group JSON object."""

        expected = {"name", "expected_case_count"}
        if not is_json_object(raw) or set(raw) != expected:
            raise ValueError(f"token parity group must contain exactly {sorted(expected)}")
        name = raw["name"]
        expected_case_count = raw["expected_case_count"]
        if not isinstance(name, str):
            raise ValueError("token parity group name must be a string")
        if not isinstance(expected_case_count, int) or isinstance(expected_case_count, bool):
            raise ValueError("token parity group expected_case_count must be an integer")
        return cls(name=name, expected_case_count=expected_case_count)

    def raw(self) -> JsonObject:
        """Project this group reference to JSON."""

        return {"name": self.name, "expected_case_count": self.expected_case_count}


@dataclass(frozen=True, slots=True)
class CollectedTestCase:
    """One concrete pytest item and its complete scheduling metadata."""

    path: str
    nodeid: str
    stage: TestStage
    requirements: TestRequirements
    estimated_duration_seconds: float | None
    timeout_seconds: float
    token_parity_group: TokenParityGroupRef | None

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
            "token_parity_group",
        }
        if not is_json_object(raw) or set(raw) != expected:
            raise ValueError(f"collected test case must contain exactly {sorted(expected)}")
        path = raw["path"]
        nodeid = raw["nodeid"]
        stage = raw["stage"]
        estimate = raw["estimated_duration_seconds"]
        timeout = raw["timeout_seconds"]
        parity_group = raw["token_parity_group"]
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
            requirements=TestRequirements.from_raw(raw["requirements"]),
            estimated_duration_seconds=float(estimate) if estimate is not None else None,
            timeout_seconds=float(timeout),
            token_parity_group=TokenParityGroupRef.from_raw(parity_group) if parity_group is not None else None,
        )

    def raw(self) -> JsonObject:
        """Project this collected case to its JSON representation."""

        return {
            "path": self.path,
            "nodeid": self.nodeid,
            "stage": self.stage.value,
            "requirements": self.requirements.raw(),
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "timeout_seconds": self.timeout_seconds,
            "token_parity_group": self.token_parity_group.raw() if self.token_parity_group is not None else None,
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
        parity_groups: dict[str, tuple[int, int]] = {}
        for case in self.cases:
            if case.token_parity_group is not None:
                group = case.token_parity_group
                collected_count, expected_count = parity_groups.get(group.name, (0, group.expected_case_count))
                if group.expected_case_count != expected_count:
                    raise ValueError(f"token parity group {group.name!r} has inconsistent expected_case_count values")
                parity_groups[group.name] = (collected_count + 1, expected_count)
        incomplete = sorted(
            name
            for name, (collected_count, expected_count) in parity_groups.items()
            if collected_count != expected_count
        )
        if incomplete:
            raise ValueError(f"token parity groups must contain every expected case: {incomplete}")

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
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(self.raw(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def raw(self) -> JsonObject:
        """Project this Test Plan to its JSON representation."""

        return {"cases": [case.raw() for case in self.cases]}
