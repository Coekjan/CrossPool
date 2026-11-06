from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from xkit.serving.sglang.graph import SglangGraphMode
from xtest.harness.runner.artifact import ArtifactGroupRef
from xtest.harness.runner.pytest_report import PytestCaseReport, PytestCaseStatus
from xtest.harness.sglang.serving.alignment import (
    PREFILL_DISTRIBUTION_KL_LIMIT,
    PREFILL_LOGITS_ARTIFACT_FILENAME,
    SERVING_GRAPH_ARTIFACT_FILENAME,
    ServingGraphAdapter,
    ServingGraphArtifact,
    TokenOutput,
    assert_prefill_alignment,
    assert_serving_graph_alignment,
)
from xtest.harness.support.config import TEST_MODEL_ID


def artifact(mode: SglangGraphMode, output_ids: tuple[int, ...] = (1, 2, 3)) -> ServingGraphArtifact:
    return ServingGraphArtifact(
        graph_settings=mode.settings(),
        outputs=(TokenOutput(TEST_MODEL_ID, output_ids),),
    )


def test_serving_graph_artifact_round_trips_and_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    expected = artifact(SglangGraphMode.EAGER)

    expected.write(path)

    assert ServingGraphArtifact.read(path) == expected
    with pytest.raises(FileExistsError):
        expected.write(path)
    raw = json.loads(path.read_text())
    raw["outputs"][0]["output_ids"][-1] = -1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="nonnegative"):
        ServingGraphArtifact.read(path)


def test_prefill_alignment_accepts_close_distributions_with_different_argmax() -> None:
    assert_prefill_alignment(torch.tensor([[0.0, 0.001]]), torch.tensor([[0.001, 0.0]]))


def test_serving_graph_alignment_compares_decode_tokens_and_prefill_distributions() -> None:
    eager_logits = torch.tensor([[0.012589254, 0.987410746]], dtype=torch.float32).log()
    breakable_logits = torch.tensor([[0.070794578, 0.929205422]], dtype=torch.float32).log()

    assert_serving_graph_alignment(
        (
            artifact(SglangGraphMode.EAGER),
            artifact(SglangGraphMode.DECODE_FULL),
            artifact(SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE, (9, 8, 7)),
        ),
        {
            SglangGraphMode.EAGER.settings(): eager_logits,
            SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings(): breakable_logits,
        },
    )


def test_serving_graph_alignment_rejects_prefill_distribution_divergence() -> None:
    eager_logits = torch.tensor([[0.070794578, 0.929205422]], dtype=torch.float32).log()
    breakable_logits = torch.tensor([[0.012589254, 0.987410746]], dtype=torch.float32).log()

    with pytest.raises(
        AssertionError,
        match="prefill distribution KL divergence failed",
    ):
        assert_serving_graph_alignment(
            (
                artifact(SglangGraphMode.EAGER),
                artifact(SglangGraphMode.DECODE_FULL),
                artifact(SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
            ),
            {
                SglangGraphMode.EAGER.settings(): eager_logits,
                SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings(): breakable_logits,
            },
        )


def test_serving_graph_alignment_rejects_nonfinite_kl_from_finite_logits() -> None:
    logits = torch.tensor([[3e38, -3e38]], dtype=torch.float32)
    with pytest.raises(AssertionError, match="prefill distribution KL divergence failed"):
        assert_serving_graph_alignment(
            (
                artifact(SglangGraphMode.EAGER),
                artifact(SglangGraphMode.DECODE_FULL),
                artifact(SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
            ),
            {
                SglangGraphMode.EAGER.settings(): logits,
                SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings(): logits.flip(-1),
            },
        )


def test_serving_graph_alignment_rejects_mismatched_vocabulary_shapes() -> None:
    with pytest.raises(AssertionError, match="prefill logits shapes differ"):
        assert_serving_graph_alignment(
            (
                artifact(SglangGraphMode.EAGER),
                artifact(SglangGraphMode.DECODE_FULL),
                artifact(SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
            ),
            {
                SglangGraphMode.EAGER.settings(): torch.tensor([[0.0, -100.0, -100.0]]),
                SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings(): torch.tensor([[0.0]]),
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
            artifact(SglangGraphMode.DECODE_FULL),
            artifact(SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
        ),
        {
            SglangGraphMode.EAGER.settings(): logits,
            SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings(): logits,
        },
    )


def test_serving_graph_alignment_rejects_decode_token_mismatch() -> None:
    with pytest.raises(AssertionError, match="decode output parity failed"):
        assert_serving_graph_alignment(
            (artifact(SglangGraphMode.EAGER), artifact(SglangGraphMode.DECODE_FULL, (4, 5, 6))),
            {},
        )


def test_serving_graph_alignment_requires_breakable_prefill_logits() -> None:
    with pytest.raises(AssertionError, match="exactly Eager and combined-mode logits"):
        assert_serving_graph_alignment(
            (
                artifact(SglangGraphMode.EAGER),
                artifact(SglangGraphMode.DECODE_FULL),
                artifact(SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
            ),
            {},
        )


def test_serving_graph_adapter_reads_breakable_group_artifacts(tmp_path: Path) -> None:
    modes = (
        SglangGraphMode.EAGER,
        SglangGraphMode.DECODE_FULL,
        SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE,
    )
    directories = tuple(tmp_path / mode.value for mode in modes)
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    for mode, directory in zip(modes, directories, strict=True):
        directory.mkdir()
        artifact(mode).write(directory / SERVING_GRAPH_ARTIFACT_FILENAME)
        if mode is not SglangGraphMode.DECODE_FULL:
            save_file({"next_token_logits": logits}, directory / PREFILL_LOGITS_ARTIFACT_FILENAME)
    reports = tuple(PytestCaseReport(mode.value, PytestCaseStatus.PASSED, None) for mode in modes)

    result = ServingGraphAdapter().evaluate(
        ArtifactGroupRef("serving_graph", "case", len(modes)),
        reports,
        directories,
    )

    assert result.result_code == 0
    assert result.detail == "compared"
