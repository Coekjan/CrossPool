"""Cross-task output evidence for SGLang serving graph modes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import torch
from pydantic import TypeAdapter
from safetensors import SafetensorError
from safetensors.torch import load_file

from tests.harness.runner.artifact import ArtifactGroupRef, ArtifactGroupResult
from tests.harness.runner.pytest_report import PytestCaseReport, PytestCaseStatus
from tests.harness.sglang.serving.graph import SglangGraphMode, SglangGraphSettings

SERVING_GRAPH_ARTIFACT_FILENAME = "serving-graph.json"
PREFILL_LOGITS_ARTIFACT_FILENAME = "prefill-logits.safetensors"
PREFILL_DISTRIBUTION_KL_LIMIT = 0.05


class ServingGraphGroupStatus(StrEnum):
    """Final cross-task serving-graph group classification."""

    COMPARED = "compared"
    SKIPPED = "skipped"
    FAILED = "failed"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True, slots=True)
class ServingGraphGroupResult:
    """One complete Serving Graph group result and optional diagnostic."""

    group: str
    status: ServingGraphGroupStatus
    detail: str | None

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("serving graph result group must be nonempty")
        if self.status in {ServingGraphGroupStatus.COMPARED, ServingGraphGroupStatus.FAILED}:
            if self.detail is not None:
                raise ValueError(f"{self.status.value} serving graph result cannot carry detail")
        elif not self.detail:
            raise ValueError(f"{self.status.value} serving graph result requires detail")

    @property
    def result_code(self) -> int:
        """Return zero for accepted outcomes and one for group failures."""

        return 0 if self.status in {ServingGraphGroupStatus.COMPARED, ServingGraphGroupStatus.SKIPPED} else 1


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
class ServingGraphArtifact:
    """Versionless serving output evidence from one logical case and graph mode."""

    group: str
    graph_settings: SglangGraphSettings
    outputs: tuple[TokenOutput, ...]

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("serving graph artifact group must be nonempty")
        model_ids = tuple(output.model_id for output in self.outputs)
        if not model_ids:
            raise ValueError("serving graph artifact must contain model outputs")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("serving graph artifact model IDs must be unique")

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read and validate one runner-owned artifact."""

        return TypeAdapter(cls).validate_json(path.read_bytes())

    def write(self, path: Path) -> None:
        """Exclusively create one validated runner-owned artifact."""

        with path.open("xb") as output:
            output.write(TypeAdapter(type(self)).dump_json(self, indent=2))


def assert_serving_graph_alignment(
    artifacts: Sequence[ServingGraphArtifact],
    prefill_logits: Mapping[SglangGraphSettings, torch.Tensor],
) -> None:
    """Require decode output parity and, when present, prefill logit alignment."""

    eager_settings = SglangGraphMode.EAGER.settings()
    decode_full_settings = SglangGraphMode.DECODE_FULL.settings()
    prefill_breakable_settings = SglangGraphMode.PREFILL_BREAKABLE.settings()
    artifacts_by_settings = {artifact.graph_settings: artifact for artifact in artifacts}
    if len(artifacts_by_settings) != len(artifacts):
        raise AssertionError("serving graph alignment contains duplicate graph modes")
    if eager_settings not in artifacts_by_settings or decode_full_settings not in artifacts_by_settings:
        raise AssertionError("serving graph alignment requires exactly one Eager and one Decode Full artifact")
    unexpected_settings = artifacts_by_settings.keys() - {
        eager_settings,
        decode_full_settings,
        prefill_breakable_settings,
    }
    if unexpected_settings:
        raise AssertionError(f"serving graph alignment contains unsupported graph settings: {unexpected_settings}")

    eager = artifacts_by_settings[eager_settings]
    group = eager.group
    if any(artifact.group != group for artifact in artifacts):
        raise AssertionError("serving graph artifacts must belong to one logical case")
    eager_outputs = {output.model_id: output.output_ids for output in eager.outputs}
    for artifact in artifacts:
        model_ids = {output.model_id for output in artifact.outputs}
        if model_ids != eager_outputs.keys():
            raise AssertionError(
                f"serving graph model IDs differ for {group} mode {artifact.graph_settings.id()}: "
                f"expected {tuple(eager_outputs)}, received {tuple(sorted(model_ids))}"
            )

    full_outputs = {
        output.model_id: output.output_ids for output in artifacts_by_settings[decode_full_settings].outputs
    }
    if full_outputs != eager_outputs:
        raise AssertionError(
            f"decode output parity failed for {group}: expected {eager_outputs}, received {full_outputs}"
        )

    if prefill_breakable_settings not in artifacts_by_settings:
        if prefill_logits:
            raise AssertionError("serving graph group without Prefill Breakable must not contain prefill logits")
        return
    if len(eager_outputs) != 1:
        raise AssertionError("Prefill Breakable serving graph alignment requires exactly one model")
    if set(prefill_logits) != {eager_settings, prefill_breakable_settings}:
        raise AssertionError(
            "Prefill Breakable serving graph alignment requires exactly Eager and Prefill Breakable logits"
        )
    for settings, logits in prefill_logits.items():
        if logits.dtype is not torch.float32:
            raise AssertionError(f"prefill logits for {settings.id()} must use torch.float32")
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.numel() == 0:
            raise AssertionError(f"prefill logits for {settings.id()} must have nonempty [1, V] shape")
        if not torch.isfinite(logits).all():
            raise AssertionError(f"prefill logits for {settings.id()} must be finite")
    eager_log_probs = torch.nn.functional.log_softmax(prefill_logits[eager_settings], dim=-1)
    breakable_log_probs = torch.nn.functional.log_softmax(prefill_logits[prefill_breakable_settings], dim=-1)
    prefill_kl_divergence = torch.nn.functional.kl_div(
        breakable_log_probs,
        eager_log_probs,
        reduction="batchmean",
        log_target=True,
    )
    if prefill_kl_divergence > PREFILL_DISTRIBUTION_KL_LIMIT:
        raise AssertionError(
            f"prefill distribution KL divergence failed: observed {prefill_kl_divergence.item():.9f}, "
            f"limit {PREFILL_DISTRIBUTION_KL_LIMIT:.9f}"
        )


def evaluate_serving_graph_group(
    group: str,
    reports: Sequence[PytestCaseReport],
    artifacts: Sequence[ServingGraphArtifact],
    prefill_logits: Mapping[SglangGraphSettings, torch.Tensor],
) -> ServingGraphGroupResult:
    """Classify one complete group from ordinary pytest outcomes and artifacts."""

    if not reports:
        raise ValueError("serving graph group evaluation requires case reports")
    if any(report.status is PytestCaseStatus.FAILED for report in reports):
        return ServingGraphGroupResult(group, ServingGraphGroupStatus.FAILED, None)
    skipped = tuple(report for report in reports if report.status is PytestCaseStatus.SKIPPED)
    if skipped:
        reasons = tuple(report.detail for report in skipped)
        if len(skipped) == len(reports) and len(set(reasons)) == 1:
            assert reasons[0] is not None
            return ServingGraphGroupResult(group, ServingGraphGroupStatus.SKIPPED, reasons[0])
        if len(skipped) == len(reports):
            return ServingGraphGroupResult(
                group,
                ServingGraphGroupStatus.INCONSISTENT,
                f"all cases skipped with differing reasons: {reasons}",
            )
        return ServingGraphGroupResult(
            group,
            ServingGraphGroupStatus.INCONSISTENT,
            "group contains both passed and skipped cases",
        )
    try:
        assert_serving_graph_alignment(artifacts, prefill_logits)
    except (AssertionError, ValueError) as error:
        return ServingGraphGroupResult(group, ServingGraphGroupStatus.INCONSISTENT, str(error))
    return ServingGraphGroupResult(group, ServingGraphGroupStatus.COMPARED, None)


class ServingGraphAdapter:
    """Evaluate SGLang serving-graph artifacts for the generic suite runner."""

    kind = "serving_graph"

    def evaluate(
        self,
        group: ArtifactGroupRef,
        reports: Sequence[PytestCaseReport],
        artifact_directories: Sequence[Path],
    ) -> ArtifactGroupResult:
        """Classify one complete group using component-owned artifacts."""

        artifacts: tuple[ServingGraphArtifact, ...] = ()
        prefill_logits: dict[SglangGraphSettings, torch.Tensor] = {}
        if all(report.status is PytestCaseStatus.PASSED for report in reports):
            try:
                artifacts = tuple(
                    ServingGraphArtifact.read(directory / SERVING_GRAPH_ARTIFACT_FILENAME)
                    for directory in artifact_directories
                )
                if any(
                    artifact.graph_settings == SglangGraphMode.PREFILL_BREAKABLE.settings() for artifact in artifacts
                ):
                    for artifact, directory in zip(artifacts, artifact_directories, strict=True):
                        if artifact.graph_settings not in {
                            SglangGraphMode.EAGER.settings(),
                            SglangGraphMode.PREFILL_BREAKABLE.settings(),
                        }:
                            continue
                        tensors = load_file(directory / PREFILL_LOGITS_ARTIFACT_FILENAME, device="cpu")
                        if tuple(tensors) != ("next_token_logits",):
                            raise ValueError("prefill-logits.safetensors must contain only next_token_logits")
                        prefill_logits[artifact.graph_settings] = tensors["next_token_logits"]
            except (OSError, ValueError, SafetensorError) as error:
                return ArtifactGroupResult(group.name, 1, f"inconsistent: {error}")
        result = evaluate_serving_graph_group(group.name, reports, artifacts, prefill_logits)
        detail = result.status.value
        if result.detail is not None:
            detail += f": {result.detail}"
        return ArtifactGroupResult(group.name, result.result_code, detail)
