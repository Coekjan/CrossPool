"""Pure behavior tests for FFN execution identities and resource formulas."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from xpool import ffn
from xpool.ffn import RouterAlgorithm, RouterProjectionPrecision, SiluActivation
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, preparation, weights


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
        activation=SiluActivation(),
        gate_up=weights.ProjectionLayout(values=TensorLayout((16, 16), torch.bfloat16, 0), scales=None, bias=None),
        down=weights.ProjectionLayout(values=TensorLayout((16, 8), torch.bfloat16, 0), scales=None, bias=None),
    )
    assert preparation.compute_workspace_bytes(signature) == 192
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
        activation=SiluActivation(),
        gate_up=weights.ProjectionLayout(values=TensorLayout((16, 16), torch.float16, 0), scales=None, bias=None),
        down=weights.ProjectionLayout(values=TensorLayout((16, 8), torch.float16, 0), scales=None, bias=None),
    )
    assert preparation.compute_workspace_bytes(fp16_signature) == 192


def test_capture_source_shares_only_capacity_independent_weights() -> None:
    small = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        activation=SiluActivation(),
        gate_up=weights.ProjectionLayout(values=TensorLayout((16, 16), torch.bfloat16, 0), scales=None, bias=None),
        down=weights.ProjectionLayout(values=TensorLayout((16, 8), torch.bfloat16, 0), scales=None, bias=None),
    )
    large = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=8,
        hidden_size=16,
        local_intermediate_size=8,
        activation=SiluActivation(),
        gate_up=weights.ProjectionLayout(values=TensorLayout((16, 16), torch.bfloat16, 0), scales=None, bias=None),
        down=weights.ProjectionLayout(values=TensorLayout((16, 8), torch.bfloat16, 0), scales=None, bias=None),
    )

    assert execution.capture_weight_pair_key(small) == execution.capture_weight_pair_key(large)
    for activation in (
        SiluActivation(alpha=1.3),
        SiluActivation(limit=4.0),
        SiluActivation(up_offset=1.0),
    ):
        changed = replace(small, activation=activation)
        assert execution.capture_weight_pair_key(changed) != execution.capture_weight_pair_key(small)
    assert execution.control_capture_probe_storage_bytes(small) == (512, 256)
    assert preparation.graph_capture_capacity_storage_bytes(small) == (128, 128, 192)


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
        activation=SiluActivation(),
        routed_scaling_factor=1.0,
        router=None,
        operation=execution.ExpertOperationKind.SGLANG_TRITON,
        gate_up=weights.ProjectionLayout(values=TensorLayout((4, 16, 16), torch.bfloat16, 0), scales=None, bias=None),
        down=weights.ProjectionLayout(values=TensorLayout((4, 16, 8), torch.bfloat16, 0), scales=None, bias=None),
    )
    _, _, selected_extent = execution.moe_workspace_layout(signature, block_size_m=16)
    assert selected_extent == 800
    assert preparation.compute_workspace_bytes(signature) == 3040
    assert preparation.graph_capture_capacity_storage_bytes(signature) == (128, 128, 64, 3040)

    router_owner = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        expert_count=4,
        effective_topk=2,
        activation=SiluActivation(),
        routed_scaling_factor=1.8,
        router=execution.MoeRouterExecutionSignature(
            algorithm=RouterAlgorithm.SIGMOID,
            projection_precision=RouterProjectionPrecision.FP32,
            weight_rounding=ffn.TopKWeightRounding.NONE,
            routed_expert_count=4,
            router_workspace_bytes=320,
            renormalize=True,
            projection=weights.ProjectionLayout(values=TensorLayout((4, 16), torch.float32, 0), scales=None, bias=None),
            correction_bias=TensorLayout((4,), torch.float32, 0),
        ),
        operation=execution.ExpertOperationKind.SGLANG_TRITON,
        gate_up=weights.ProjectionLayout(values=TensorLayout((4, 16, 16), torch.bfloat16, 0), scales=None, bias=None),
        down=weights.ProjectionLayout(values=TensorLayout((4, 16, 8), torch.bfloat16, 0), scales=None, bias=None),
    )
    assert execution.control_capture_probe_storage_bytes(router_owner) == (2048, 1024, 256, 16)
    assert preparation.graph_capture_capacity_storage_bytes(router_owner) == (
        128,
        128,
        64,
        preparation.compute_workspace_bytes(router_owner),
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
