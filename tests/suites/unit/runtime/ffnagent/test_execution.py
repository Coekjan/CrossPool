"""Pure behavior tests for FFN execution identities and resource formulas."""

from __future__ import annotations

import pytest
import torch

from xpool.ffn import ActivationKind
from xpool.runtime.ffnagent import execution, weights


def compute_test_routed_topk(
    *,
    hidden_states: torch.Tensor,
    router_weights: weights.MoeRouterWeights,
    workspace: torch.Tensor,
    routed_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    renormalize: bool,
) -> None:
    """Stand in for one Adapter-owned Router callable in pure formula tests."""

    del hidden_states, router_weights, workspace, routed_ids, routed_weights, renormalize


def test_payload_row_capacities_cover_power_of_two_buckets_and_exact_maximum() -> None:
    assert execution.derive_payload_row_capacities(1) == (1,)
    assert execution.derive_payload_row_capacities(5) == (1, 2, 4, 5)
    assert execution.derive_payload_row_capacities(8) == (1, 2, 4, 8)
    with pytest.raises(ValueError, match="positive"):
        execution.derive_payload_row_capacities(0)


def test_dense_workspace_bytes_is_exact_and_rejects_empty_geometry() -> None:
    assert (
        execution.dense_workspace_bytes(
            payload_dtype=torch.bfloat16,
            row_capacity=4,
            local_intermediate_size=8,
        )
        == 192
    )
    signature = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        activation=ActivationKind.SILU,
    )
    assert execution.compute_workspace_bytes(signature) == 192
    with pytest.raises(ValueError, match="positive"):
        execution.dense_workspace_bytes(
            payload_dtype=torch.bfloat16,
            row_capacity=0,
            local_intermediate_size=8,
        )

    fp16_signature = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.float16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        activation=ActivationKind.SILU,
    )
    assert execution.compute_workspace_bytes(fp16_signature) == 192


def test_capture_source_shares_only_capacity_independent_weights() -> None:
    small = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        activation=ActivationKind.SILU,
    )
    large = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=8,
        hidden_size=16,
        local_intermediate_size=8,
        activation=ActivationKind.SILU,
    )

    assert execution.capture_weight_pair_key(small) == execution.capture_weight_pair_key(large)
    assert execution.control_capture_probe_storage_bytes(small) == (512, 256)
    assert execution.graph_capture_capacity_storage_bytes(small) == (128, 128, 192)


def test_moe_workspace_formulas_preserve_cache_overlap() -> None:
    assert execution.moe_alignment_workspace_shapes(
        row_capacity=4,
        effective_topk=2,
        expert_count=4,
        block_size_m=16,
    ) == (83, 6, 6)
    signature = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        expert_count=4,
        effective_topk=2,
        activation=ActivationKind.SILU,
        routed_scaling_factor=1.0,
        router=None,
    )
    _, _, selected_extent = execution.moe_workspace_layout(signature, block_size_m=16)
    assert selected_extent == 800
    assert execution.compute_workspace_bytes(signature) == 3040
    assert execution.graph_capture_capacity_storage_bytes(signature) == (128, 128, 64, 3040)

    router_owner = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        expert_count=4,
        effective_topk=2,
        activation=ActivationKind.SILU,
        routed_scaling_factor=1.8,
        router=execution.MoeRouterExecutionSignature(
            compute_routed_topk=compute_test_routed_topk,
            router_weight_dtype=torch.float32,
            routed_expert_count=4,
            router_workspace_bytes=32,
            correction_bias_present=False,
            renormalize=True,
        ),
    )
    assert execution.control_capture_probe_storage_bytes(router_owner) == (2048, 1024, 256)
    assert execution.graph_capture_capacity_storage_bytes(router_owner) == (
        128,
        128,
        64,
        execution.compute_workspace_bytes(router_owner),
        8,
    )
    with pytest.raises(ValueError, match="positive"):
        execution.moe_alignment_workspace_shapes(
            row_capacity=0,
            effective_topk=2,
            expert_count=4,
            block_size_m=16,
        )
    with pytest.raises(ValueError, match="qualified domain"):
        execution.moe_workspace_layout(signature, block_size_m=256)
