from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.harness.runner.artifact import ArtifactGroupRef
from tests.harness.runner.pytest_report import PytestCaseReport, PytestCaseStatus
from tests.harness.sglang.serving.alignment import (
    PREFILL_DISTRIBUTION_KL_LIMIT,
    PREFILL_LOGITS_ARTIFACT_FILENAME,
    SERVING_GRAPH_ARTIFACT_FILENAME,
    ServingGraphAdapter,
    ServingGraphArtifact,
    TokenOutput,
    assert_serving_graph_alignment,
)
from tests.harness.sglang.serving.graph import SglangGraphMode


def artifact(mode: SglangGraphMode, output_ids: tuple[int, ...] = (1, 2, 3)) -> ServingGraphArtifact:
    return ServingGraphArtifact(
        group="case",
        graph_settings=mode.settings(),
        outputs=(TokenOutput("model", output_ids),),
    )


def test_serving_graph_artifact_round_trips_and_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    expected = artifact(SglangGraphMode.EAGER)

    expected.write(path)

    assert ServingGraphArtifact.read(path) == expected
    with pytest.raises(FileExistsError):
        expected.write(path)


def test_serving_graph_alignment_compares_decode_tokens_and_prefill_distributions() -> None:
    eager_logits = torch.tensor([[0.012589254, 0.987410746]], dtype=torch.float32).log()
    piecewise_logits = torch.tensor([[0.070794578, 0.929205422]], dtype=torch.float32).log()

    assert_serving_graph_alignment(
        (
            artifact(SglangGraphMode.EAGER),
            artifact(SglangGraphMode.FULL),
            artifact(SglangGraphMode.PIECEWISE, (9, 8, 7)),
        ),
        {
            SglangGraphMode.EAGER.settings(): eager_logits,
            SglangGraphMode.PIECEWISE.settings(): piecewise_logits,
        },
    )


def test_serving_graph_alignment_rejects_prefill_distribution_divergence() -> None:
    eager_logits = torch.tensor([[0.070794578, 0.929205422]], dtype=torch.float32).log()
    piecewise_logits = torch.tensor([[0.012589254, 0.987410746]], dtype=torch.float32).log()

    with pytest.raises(
        AssertionError,
        match=r"prefill distribution KL divergence failed: observed 0\.065802\d+, limit 0\.050000000",
    ):
        assert_serving_graph_alignment(
            (
                artifact(SglangGraphMode.EAGER),
                artifact(SglangGraphMode.FULL),
                artifact(SglangGraphMode.PIECEWISE),
            ),
            {
                SglangGraphMode.EAGER.settings(): eager_logits,
                SglangGraphMode.PIECEWISE.settings(): piecewise_logits,
            },
        )


def test_serving_graph_alignment_accepts_prefill_distribution_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    monkeypatch.setattr(
        torch.nn.functional,
        "kl_div",
        lambda *args, **kwargs: torch.tensor(PREFILL_DISTRIBUTION_KL_LIMIT),
    )

    assert_serving_graph_alignment(
        (
            artifact(SglangGraphMode.EAGER),
            artifact(SglangGraphMode.FULL),
            artifact(SglangGraphMode.PIECEWISE),
        ),
        {
            SglangGraphMode.EAGER.settings(): logits,
            SglangGraphMode.PIECEWISE.settings(): logits,
        },
    )


def test_serving_graph_alignment_rejects_decode_token_mismatch() -> None:
    with pytest.raises(AssertionError, match="decode output parity failed"):
        assert_serving_graph_alignment(
            (artifact(SglangGraphMode.EAGER), artifact(SglangGraphMode.FULL, (4, 5, 6))),
            {},
        )


def test_serving_graph_alignment_requires_piecewise_prefill_logits() -> None:
    with pytest.raises(AssertionError, match="exactly Eager and Piecewise"):
        assert_serving_graph_alignment(
            (
                artifact(SglangGraphMode.EAGER),
                artifact(SglangGraphMode.FULL),
                artifact(SglangGraphMode.PIECEWISE),
            ),
            {},
        )


def test_serving_graph_adapter_reads_piecewise_group_artifacts(tmp_path: Path) -> None:
    modes = (SglangGraphMode.EAGER, SglangGraphMode.FULL, SglangGraphMode.PIECEWISE)
    directories = tuple(tmp_path / mode.value for mode in modes)
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    for mode, directory in zip(modes, directories, strict=True):
        directory.mkdir()
        artifact(mode).write(directory / SERVING_GRAPH_ARTIFACT_FILENAME)
        if mode is not SglangGraphMode.FULL:
            save_file({"next_token_logits": logits}, directory / PREFILL_LOGITS_ARTIFACT_FILENAME)
    reports = tuple(PytestCaseReport(mode.value, PytestCaseStatus.PASSED, None) for mode in modes)

    result = ServingGraphAdapter().evaluate(
        ArtifactGroupRef("serving_graph", "case", len(modes)),
        reports,
        directories,
    )

    assert result.result_code == 0
    assert result.detail == "compared"
