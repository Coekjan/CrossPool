"""Immutable strict protocol produced by isolated pytest collection."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass

from xkit import ResourceRequirements
from xkit.case import CaseId
from xkit.config import DeploymentConfig
from xkit.results import write_json
from xkit.serving.sglang.graph import SglangGraphMode
from xpool.model import ModelId

type PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


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


class CaseInspection(BaseModel):
    """Portable conditions projected from known typed test case values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    id: CaseId | None
    description: str
    models: tuple[ModelId, ...]
    deployment: DeploymentConfig
    graph_modes: tuple[SglangGraphMode, ...]


@dataclass(config=ConfigDict(extra="forbid", strict=True), frozen=True, slots=True)
class CollectedTestCase:
    """One concrete pytest item and its complete scheduling metadata."""

    path: str
    nodeid: str
    stage: TestStage
    requirements: ResourceRequirements
    estimated_duration_seconds: PositiveSeconds | None
    timeout_seconds: PositiveSeconds
    inspection: CaseInspection | None

    def __post_init__(self) -> None:
        parsed_path = PurePosixPath(self.path)
        if not self.path or "\\" in self.path or parsed_path.is_absolute() or ".." in parsed_path.parts:
            raise ValueError(f"collected test path must be canonical and repository-relative: {self.path!r}")
        if self.stage is not TestStage.from_path(self.path):
            raise ValueError(f"collected test stage disagrees with path: {self.path!r}")
        if not self.nodeid.startswith(f"{self.path}::"):
            raise ValueError(f"collected test nodeid does not belong to path: {self.nodeid!r}")
        if self.stage is TestStage.UNIT and self.requirements.device_count != 0:
            raise ValueError("unit tests cannot require devices")


@dataclass(config=ConfigDict(extra="forbid", strict=True), frozen=True, slots=True)
class TestPlan:
    """Strict ordered result of one isolated pytest collection worker."""

    cases: Annotated[tuple[CollectedTestCase, ...], Field(min_length=1)]
    selected_suites: tuple[str, ...]

    def __post_init__(self) -> None:
        nodeids = tuple(case.nodeid for case in self.cases)
        if len(nodeids) != len(set(nodeids)):
            raise ValueError("test plan nodeids must be unique")

    @staticmethod
    def read(path: Path) -> TestPlan:
        """Read one complete Test Plan from an isolated collection worker."""

        try:
            return TEST_PLAN_CODEC.validate_json(path.read_bytes())
        except (OSError, ValidationError) as error:
            raise ValueError(f"failed to read Test Plan {path}: {error}") from error

    def write(self, path: Path) -> None:
        """Atomically publish this complete Test Plan."""

        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, TEST_PLAN_CODEC.dump_python(self, mode="json"))


TEST_PLAN_CODEC = TypeAdapter(TestPlan)
