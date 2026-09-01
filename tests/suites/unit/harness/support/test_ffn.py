from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.harness.support.ffn import (
    FfnNumericalEvidence,
    FfnNumericalSampleEvidence,
    FfnTopologyEvidence,
    FfnTopologyMpsServerEvidence,
    FfnTopologyOutputEvidence,
    FfnTopologyProcessEvidence,
    compare_ffn_outputs,
    compare_ffn_routing,
    generate_ffn_hidden_states,
)
from xpool.fabric import FabricGenerationId


def test_hidden_state_generator_returns_deterministic_cloned_prefixes() -> None:
    first, second = generate_ffn_hidden_states(hidden_size=4, seed=17, row_counts=(1, 3))

    assert first.dtype is second.dtype is torch.bfloat16
    assert first.is_contiguous() and second.is_contiguous()
    torch.testing.assert_close(first, second[:1])
    first.zero_()
    assert second[0].count_nonzero()


def test_output_comparison_gates_aggregate_and_scale_aware_error() -> None:
    reference = torch.ones(1024, dtype=torch.bfloat16)
    reference[0] = 0
    within = reference.clone()
    within[0] = 0.03125
    passed = compare_ffn_outputs(
        reference,
        within,
        atol=0.02,
        rtol=0.02,
        normalized_rmse_limit=0.01,
        maximum_scaled_error_limit=0.10,
    )
    assert passed.passed
    assert passed.outlier_count == 1
    assert passed.normalized_rmse < passed.normalized_rmse_limit
    assert passed.maximum_scaled_error < passed.maximum_scaled_error_limit

    outside = torch.ones_like(reference)
    outside[0] = 1.125
    failed = compare_ffn_outputs(
        torch.ones_like(reference),
        outside,
        atol=0.02,
        rtol=0.02,
        normalized_rmse_limit=0.01,
        maximum_scaled_error_limit=0.10,
    )
    assert not failed.passed
    assert failed.maximum_error_flat_index == 0
    assert failed.normalized_rmse < failed.normalized_rmse_limit
    assert failed.maximum_scaled_error > failed.maximum_scaled_error_limit
    assert failed.outlier_count == 1
    assert failed.reference_at_maximum_error == 1.0


@pytest.mark.parametrize(
    ("reference", "actual", "message"),
    [
        (torch.ones((1, 2), dtype=torch.float32), torch.ones((1, 2), dtype=torch.float32), "dtype"),
        (torch.empty((0, 2), dtype=torch.bfloat16), torch.empty((0, 2), dtype=torch.bfloat16), "nonempty"),
        (
            torch.ones((1, 2), dtype=torch.bfloat16),
            torch.ones((2, 2), dtype=torch.bfloat16),
            "shape mismatch",
        ),
    ],
)
def test_output_comparison_rejects_invalid_evidence(
    reference: torch.Tensor,
    actual: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        compare_ffn_outputs(
            reference,
            actual,
            atol=0.02,
            rtol=0.02,
            normalized_rmse_limit=0.01,
            maximum_scaled_error_limit=0.10,
        )


def test_routing_comparison_is_column_order_independent() -> None:
    reference_ids = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    reference_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    actual_ids = torch.tensor([[1, 3], [2, 0]], dtype=torch.int32)
    actual_weights = torch.tensor([[0.75, 0.25], [0.40001, 0.59999]], dtype=torch.float32)

    comparison = compare_ffn_routing(
        reference_ids,
        reference_weights,
        actual_ids,
        actual_weights,
        expert_count=4,
        atol=0.0001,
        rtol=0.001,
    )

    assert comparison.passed
    assert comparison.first_mismatch_token is None


def test_routing_comparison_reports_first_semantic_mismatch() -> None:
    comparison = compare_ffn_routing(
        torch.tensor([[1, 3], [0, 2]], dtype=torch.int32),
        torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32),
        torch.tensor([[1, 2], [0, 2]], dtype=torch.int32),
        torch.tensor([[0.7, 0.3], [0.6, 0.5]], dtype=torch.float32),
        expert_count=4,
        atol=0.0001,
        rtol=0.001,
    )

    assert not comparison.passed
    assert (comparison.first_mismatch_token, comparison.first_mismatch_pair) == (0, 1)
    assert (comparison.reference_expert_id, comparison.actual_expert_id) == (3, 2)
    assert comparison.reference_weight == pytest.approx(0.3)
    assert comparison.actual_weight == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("ids", "weights", "message"),
    [
        (torch.tensor([[0, 0]], dtype=torch.int32), torch.ones((1, 2)), "duplicate"),
        (torch.tensor([[0, 4]], dtype=torch.int32), torch.ones((1, 2)), "illegal"),
        (torch.tensor([[0, 1]], dtype=torch.int64), torch.ones((1, 2)), "dtypes"),
        (torch.tensor([[0, 1]], dtype=torch.int32), torch.tensor([[1.0, float("nan")]]), "finite"),
    ],
)
def test_routing_comparison_rejects_invalid_evidence(
    ids: torch.Tensor,
    weights: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        compare_ffn_routing(
            ids,
            weights,
            ids.clone(),
            weights.clone(),
            expert_count=4,
            atol=0.0001,
            rtol=0.001,
        )


def test_numerical_evidence_writes_one_strict_artifact(tmp_path: Path) -> None:
    output = compare_ffn_outputs(
        torch.ones((1, 2), dtype=torch.bfloat16),
        torch.ones((1, 2), dtype=torch.bfloat16),
        atol=0.02,
        rtol=0.02,
        normalized_rmse_limit=0.01,
        maximum_scaled_error_limit=0.10,
    )
    evidence = FfnNumericalEvidence(
        case_id="case",
        model_id="model",
        model_spec_digest="1" * 64,
        generation=FabricGenerationId(high=1, low=2),
        samples=(FfnNumericalSampleEvidence(layer_id=0, row_count=1, output=output, routing=None),),
    )
    path = tmp_path / "ffn-numerical.json"

    evidence.write(path)

    assert FfnNumericalEvidence.model_validate_json(path.read_bytes()) == evidence


def test_topology_evidence_writes_complete_process_and_output_facts(tmp_path: Path) -> None:
    comparison = compare_ffn_outputs(
        torch.ones((1, 2), dtype=torch.bfloat16),
        torch.ones((1, 2), dtype=torch.bfloat16),
        atol=0.02,
        rtol=0.02,
        normalized_rmse_limit=0.01,
        maximum_scaled_error_limit=0.10,
    )
    evidence = FfnTopologyEvidence(
        case_id="single-rank-delivery",
        generation=FabricGenerationId(high=1, low=2),
        processes=(
            FfnTopologyProcessEvidence(name="atnagent-0", process_id=10, cuda_device=0, gpu_uuid="GPU-a"),
            FfnTopologyProcessEvidence(name="ffnagent-1", process_id=11, cuda_device=1, gpu_uuid="GPU-b"),
            FfnTopologyProcessEvidence(name="instance-0-rank-0", process_id=12, cuda_device=0, gpu_uuid="GPU-a"),
        ),
        mps_servers=(
            FfnTopologyMpsServerEvidence(
                process_id=20,
                client_process_names=("atnagent-0", "ffnagent-1", "instance-0-rank-0"),
                active_thread_percentage=50,
            ),
        ),
        outputs=(FfnTopologyOutputEvidence(request_ordinal=0, output_ranks=(0,), comparison=comparison),),
    )
    path = tmp_path / "ffn-topology.json"

    evidence.write(path)

    assert FfnTopologyEvidence.model_validate_json(path.read_bytes()) == evidence
