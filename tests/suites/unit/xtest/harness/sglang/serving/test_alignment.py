from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from xkit.serving.sglang.graph import SglangGraphMode
from xtest.harness.sglang.serving.alignment import (
    DECODE_DISTRIBUTION_KL_LIMIT,
    DistributionComparison,
    compare_distributions,
)
from xtest.harness.sglang.serving.server import SglangServerResult
from xtest.harness.support.config import TEST_MODEL_ID


def result(tmp_path: Path, name: str, logprobs: torch.Tensor) -> SglangServerResult:
    path = tmp_path / f"{name}.safetensors"
    save_file({"logprobs": logprobs}, path)
    return SglangServerResult(
        model_id=TEST_MODEL_ID,
        resolved_graph_settings=SglangGraphMode.EAGER.settings(),
        request_id=name,
        input_ids=(1, 2),
        output_ids=tuple(range(logprobs.shape[0])),
        cached_tokens=0,
        inference_path=tmp_path / f"{name}.json",
        logprobs_path=path,
    )


def test_close_distributions_allow_different_argmax_and_generated_tokens(tmp_path: Path) -> None:
    reference = result(tmp_path, "reference", torch.tensor([[0.0, 0.001]]))
    actual = replace(result(tmp_path, "actual", torch.tensor([[0.001, 0.0]])), output_ids=(1,))

    comparison = compare_distributions(reference, actual, prefix_lengths=(2,))

    comparison.assert_within(DECODE_DISTRIBUTION_KL_LIMIT)


def test_one_bad_position_is_not_hidden_by_averaging(tmp_path: Path) -> None:
    reference = result(tmp_path, "reference", torch.tensor([[0.0, 0.0], [0.0, -8.0]]))
    actual = result(tmp_path, "actual", torch.tensor([[0.0, 0.0], [-8.0, 0.0]]))

    comparison = compare_distributions(reference, actual, prefix_lengths=(2, 3))

    assert comparison.kl_values[0] == 0
    with pytest.raises(AssertionError, match="prefix 3"):
        comparison.assert_within(DECODE_DISTRIBUTION_KL_LIMIT)


def test_rejects_different_conditioning(tmp_path: Path) -> None:
    reference = result(tmp_path, "reference", torch.zeros((2, 2)))
    actual = replace(result(tmp_path, "actual", torch.zeros((2, 2))), output_ids=(1, 0))
    with pytest.raises(AssertionError, match="conditioning differs at prefix 3"):
        compare_distributions(reference, actual, prefix_lengths=(2, 3))


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros((1, 2)),
        torch.zeros((2, 3)),
        torch.full((2, 2), float("nan")),
        torch.zeros((2, 2), dtype=torch.float16),
    ],
)
def test_rejects_incompatible_distribution_evidence(tmp_path: Path, values: torch.Tensor) -> None:
    reference = result(tmp_path, "reference", torch.zeros((2, 2)))
    actual = result(tmp_path, "actual", values)
    with pytest.raises(AssertionError):
        compare_distributions(reference, actual, prefix_lengths=(2, 3))


def test_accepts_inclusive_bound() -> None:
    DistributionComparison((2,), (DECODE_DISTRIBUTION_KL_LIMIT,)).assert_within(DECODE_DISTRIBUTION_KL_LIMIT)
