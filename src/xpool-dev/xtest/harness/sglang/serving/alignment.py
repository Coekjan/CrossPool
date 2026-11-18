"""Same-prefix full-vocabulary prediction-distribution comparison."""

from dataclasses import dataclass

import torch
from safetensors.torch import load_file

from xtest.harness.sglang.serving.server import SglangServerResult

PREFILL_DISTRIBUTION_KL_LIMIT = 0.05
DECODE_DISTRIBUTION_KL_LIMIT = 0.001


@dataclass(frozen=True, slots=True)
class DistributionComparison:
    """Forward KL in nats for each selected conditioning-prefix length."""

    prefix_lengths: tuple[int, ...]
    kl_values: tuple[float, ...]

    def assert_within(self, limit: float) -> None:
        """Fail on any position beyond the bound, retaining the failing prefix."""

        for prefix, divergence in zip(self.prefix_lengths, self.kl_values, strict=True):
            if divergence > limit:
                raise AssertionError(
                    f"prediction distribution KL failed at prefix {prefix}: "
                    f"observed {divergence:.9g}, limit {limit:.9g} nats"
                )


def compare_distributions(
    reference: SglangServerResult,
    actual: SglangServerResult,
    *,
    prefix_lengths: tuple[int, ...],
) -> DistributionComparison:
    """Compare selected rows with identical model, vocabulary and token prefixes.

    Raw FP32 logprobs remain unchanged. CPU FP64 normalization and reduction
    avoid cancellation when distributions are close. Missing or mismatched
    prediction evidence fails rather than broadcasting or truncating rows.
    """

    if reference.model_id != actual.model_id:
        raise AssertionError("prediction distributions belong to different models")
    if not prefix_lengths:
        raise ValueError("prediction comparison requires at least one position")
    selected = []
    for result in (reference, actual):
        if result.logprobs_path is None:
            raise AssertionError(f"missing prediction distributions for {result.request_id}")
        tensors = load_file(result.logprobs_path, device="cpu")
        logprobs = tensors["logprobs"]
        if (
            logprobs.dtype is not torch.float32
            or logprobs.ndim != 2
            or logprobs.shape[0] != len(result.prefix_lengths)
            or logprobs.shape[1] == 0
            or not torch.isfinite(logprobs).all()
        ):
            raise AssertionError(f"invalid FP32 prediction distributions for {result.request_id}")
        positions = [prefix - len(result.input_ids) for prefix in prefix_lengths]
        if any(position < 0 or position >= len(result.output_ids) for position in positions):
            raise AssertionError(f"missing prediction prefix in {result.request_id}")
        selected.append(logprobs[positions].double())
    reference_tokens = reference.input_ids + reference.output_ids
    actual_tokens = actual.input_ids + actual.output_ids
    for prefix in prefix_lengths:
        if (
            not 0 < prefix <= min(len(reference_tokens), len(actual_tokens))
            or reference_tokens[:prefix] != actual_tokens[:prefix]
        ):
            raise AssertionError(f"prediction conditioning differs at prefix {prefix}")
    if selected[0].shape != selected[1].shape:
        raise AssertionError("prediction vocabulary widths differ")
    reference_logprobs, actual_logprobs = (torch.log_softmax(values, dim=-1) for values in selected)
    divergences = (reference_logprobs.exp() * (reference_logprobs - actual_logprobs)).sum(dim=-1)
    if not torch.isfinite(divergences).all():
        raise AssertionError("prediction distribution KL must be finite")
    return DistributionComparison(prefix_lengths, tuple(divergences.tolist()))
