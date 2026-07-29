"""Cross-task token artifacts for SGLang graph-mode parity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import TypeAdapter

from tests.harness.runner.artifact import ArtifactGroupRef, ArtifactGroupResult
from tests.harness.runner.pytest_report import PytestCaseReport, PytestCaseStatus
from tests.harness.sglang.graph import SglangGraphSettings

TOKEN_PARITY_ARTIFACT_FILENAME = "token-parity.json"


class TokenParityGroupStatus(StrEnum):
    """Final cross-task token-parity group classification."""

    COMPARED = "compared"
    SKIPPED = "skipped"
    FAILED = "failed"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True, slots=True)
class TokenParityGroupResult:
    """One complete TokenParityGroup result and optional diagnostic."""

    group: str
    status: TokenParityGroupStatus
    detail: str | None

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("token parity result group must be nonempty")
        if self.status in {TokenParityGroupStatus.COMPARED, TokenParityGroupStatus.FAILED}:
            if self.detail is not None:
                raise ValueError(f"{self.status.value} token parity result cannot carry detail")
        elif not self.detail:
            raise ValueError(f"{self.status.value} token parity result requires detail")

    @property
    def result_code(self) -> int:
        """Return zero for accepted outcomes and one for group failures."""

        return 0 if self.status in {TokenParityGroupStatus.COMPARED, TokenParityGroupStatus.SKIPPED} else 1


@dataclass(frozen=True, slots=True)
class TokenOutput:
    """Deterministic token IDs returned for one configured model."""

    model_id: str
    output_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("token output model_id must be nonempty")
        if not self.output_ids or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in self.output_ids
        ):
            raise ValueError("token output must contain non-boolean integer IDs")


@dataclass(frozen=True, slots=True)
class TokenParityArtifact:
    """Versionless output evidence from one logical case and graph mode."""

    group: str
    graph_settings: SglangGraphSettings
    outputs: tuple[TokenOutput, ...]

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("token parity artifact group must be nonempty")
        model_ids = tuple(output.model_id for output in self.outputs)
        if not model_ids:
            raise ValueError("token parity artifact must contain model outputs")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("token parity artifact model IDs must be unique")

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read and validate one runner-owned artifact."""

        return TypeAdapter(cls).validate_json(path.read_bytes())

    def write(self, path: Path) -> None:
        """Exclusively create one validated runner-owned artifact."""

        with path.open("xb") as output:
            output.write(TypeAdapter(type(self)).dump_json(self, indent=2))


def assert_token_parity(artifacts: Sequence[TokenParityArtifact]) -> None:
    """Require every graph mode in one logical case to return identical tokens."""

    if len(artifacts) < 2:
        raise AssertionError("token parity requires at least two graph-mode artifacts")
    group = artifacts[0].group
    if any(artifact.group != group for artifact in artifacts):
        raise AssertionError("token parity artifacts must belong to one logical case")
    mode_ids = tuple(artifact.graph_settings.id() for artifact in artifacts)
    if len(mode_ids) != len(set(mode_ids)):
        raise AssertionError(f"token parity contains duplicate graph modes: {mode_ids}")
    eager_artifacts = tuple(
        artifact
        for artifact in artifacts
        if not artifact.graph_settings.cuda_graph and not artifact.graph_settings.piecewise_cuda_graph
    )
    if len(eager_artifacts) != 1:
        raise AssertionError("token parity requires exactly one eager reference artifact")
    eager = eager_artifacts[0]
    expected = {output.model_id: output.output_ids for output in eager.outputs}
    for artifact in artifacts:
        actual = {output.model_id: output.output_ids for output in artifact.outputs}
        if actual.keys() != expected.keys():
            raise AssertionError(
                f"token parity model IDs differ for {group} mode {artifact.graph_settings.id()}: "
                f"expected {tuple(expected)}, received {tuple(actual)}"
            )
        if actual != expected:
            raise AssertionError(
                f"token parity failed for {group} mode {artifact.graph_settings.id()}: "
                f"expected {expected}, received {actual}"
            )


def evaluate_token_parity_group(
    group: str,
    reports: Sequence[PytestCaseReport],
    artifacts: Sequence[TokenParityArtifact],
) -> TokenParityGroupResult:
    """Classify one complete group from ordinary pytest outcomes and artifacts."""

    if not reports:
        raise ValueError("token parity group evaluation requires case reports")
    if any(report.status is PytestCaseStatus.FAILED for report in reports):
        return TokenParityGroupResult(group, TokenParityGroupStatus.FAILED, None)
    skipped = tuple(report for report in reports if report.status is PytestCaseStatus.SKIPPED)
    if skipped:
        reasons = tuple(report.detail for report in skipped)
        if len(skipped) == len(reports) and len(set(reasons)) == 1:
            assert reasons[0] is not None
            return TokenParityGroupResult(group, TokenParityGroupStatus.SKIPPED, reasons[0])
        if len(skipped) == len(reports):
            return TokenParityGroupResult(
                group,
                TokenParityGroupStatus.INCONSISTENT,
                f"all cases skipped with differing reasons: {reasons}",
            )
        return TokenParityGroupResult(
            group,
            TokenParityGroupStatus.INCONSISTENT,
            "group contains both passed and skipped cases",
        )
    try:
        assert_token_parity(artifacts)
    except (AssertionError, ValueError) as error:
        return TokenParityGroupResult(group, TokenParityGroupStatus.INCONSISTENT, str(error))
    return TokenParityGroupResult(group, TokenParityGroupStatus.COMPARED, None)


class TokenParityAdapter:
    """Evaluate SGLang token-parity artifacts for the generic suite runner."""

    kind = "token_parity"

    def evaluate(
        self,
        group: ArtifactGroupRef,
        reports: Sequence[PytestCaseReport],
        artifact_directories: Sequence[Path],
    ) -> ArtifactGroupResult:
        """Classify one complete group using component-owned artifacts."""

        artifacts: tuple[TokenParityArtifact, ...] = ()
        if all(report.status is PytestCaseStatus.PASSED for report in reports):
            try:
                artifacts = tuple(
                    TokenParityArtifact.read(directory / TOKEN_PARITY_ARTIFACT_FILENAME)
                    for directory in artifact_directories
                )
            except (OSError, ValueError) as error:
                return ArtifactGroupResult(group.name, 1, f"inconsistent: {error}")
        result = evaluate_token_parity_group(group.name, reports, artifacts)
        detail = result.status.value
        if result.detail is not None:
            detail += f": {result.detail}"
        return ArtifactGroupResult(group.name, result.result_code, detail)
