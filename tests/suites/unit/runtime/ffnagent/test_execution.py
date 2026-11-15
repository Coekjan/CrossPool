"""Pure behavior tests for FFN execution identities and resource formulas."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from xpool import ffn, quantization
from xpool.ffn import RouterAlgorithm, RouterProjectionPrecision, SiluActivation
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, preparation, weights
from xpool.runtime.ffnagent.operators import dense
from xpool.runtime.ffnagent.operators.activation import Activation


def test_payload_row_capacities_cover_power_of_two_buckets_and_exact_maximum() -> None:
    assert execution.derive_payload_row_capacities(1) == (1,)
    assert execution.derive_payload_row_capacities(5) == (1, 2, 4, 5)
    assert execution.derive_payload_row_capacities(8) == (1, 2, 4, 8)
    with pytest.raises(ValueError, match="positive"):
        execution.derive_payload_row_capacities(0)


def test_dense_preparation_accounts_for_gated_storage_and_ungated_aliasing() -> None:
    signature = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        activation=Activation(function=SiluActivation(), gate=ffn.GateTransform()),
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.CONCATENATED,
            values=TensorLayout((16, 16), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=None,
            values=TensorLayout((16, 8), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
    )
    assert dense.DenseExecutionPlan.prepare(signature).workspace.nbytes == 192
    ungated = replace(
        signature,
        activation=Activation(function=ffn.ReluActivation(), gate=None),
        i_proj=replace(signature.i_proj, values=TensorLayout((8, 16), torch.bfloat16, 0), gate_up_order=None),
    )
    assert dense.DenseExecutionPlan.prepare(ungated).workspace.nbytes == 64
    workspace = dense.DenseExecutionPlan.prepare(signature).workspace
    backing = torch.empty(workspace.nbytes, dtype=torch.uint8)
    projected, activated, *_ = workspace.views(backing)
    assert projected.data_ptr() == backing.data_ptr()
    assert activated.data_ptr() == backing.data_ptr() + 128
    assert projected.shape == (4, 16) and activated.shape == (4, 8)
    with pytest.raises(ValueError, match="extent"):
        workspace.views(backing[:-1])
    misaligned = torch.empty(workspace.nbytes + 1, dtype=torch.uint8)[1:]
    with pytest.raises(ValueError, match="aligned"):
        workspace.views(misaligned)

    fp16_signature = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.float16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        activation=Activation(function=SiluActivation(), gate=ffn.GateTransform()),
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.CONCATENATED,
            values=TensorLayout((16, 16), torch.float16, 0),
            scales=None,
            bias=None,
        ),
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=None,
            values=TensorLayout((16, 8), torch.float16, 0),
            scales=None,
            bias=None,
        ),
    )
    assert dense.DenseExecutionPlan.prepare(fp16_signature).workspace.nbytes == 192


def test_capture_source_shares_only_capacity_independent_weights() -> None:
    small = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        activation=Activation(function=SiluActivation(), gate=ffn.GateTransform()),
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.CONCATENATED,
            values=TensorLayout((16, 16), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=None,
            values=TensorLayout((16, 8), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
    )
    large = execution.DenseFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=8,
        hidden_size=16,
        local_intermediate_size=8,
        activation=Activation(function=SiluActivation(), gate=ffn.GateTransform()),
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.CONCATENATED,
            values=TensorLayout((16, 16), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=None,
            values=TensorLayout((16, 8), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
    )

    assert execution.capture_weight_pair_key(small) == execution.capture_weight_pair_key(large)
    for activation in (
        Activation(function=SiluActivation(alpha=1.3), gate=ffn.GateTransform()),
        Activation(function=SiluActivation(), gate=ffn.GateTransform(limit=4.0)),
        Activation(function=SiluActivation(), gate=ffn.GateTransform(up_offset=1.0)),
    ):
        changed = replace(small, activation=activation)
        assert execution.capture_weight_pair_key(changed) != execution.capture_weight_pair_key(small)
    assert execution.control_capture_probe_storage_bytes(small) == (512, 256)


def test_moe_workspace_formulas_preserve_cache_overlap() -> None:
    signature = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        expert_count=4,
        effective_topk=2,
        activation=Activation(function=SiluActivation(), gate=ffn.GateTransform()),
        routed_scaling_factor=1.0,
        router=None,
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.CONCATENATED,
            values=TensorLayout((4, 16, 16), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=None,
            values=TensorLayout((4, 16, 8), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
    )
    plan = preparation.MoeExecutionPlan.prepare(signature)
    assert plan.router is None
    backing = torch.empty(plan.workspace.nbytes, dtype=torch.uint8)
    (experts_backing,) = plan.workspace.views(backing)
    assert experts_backing.numel() == plan.experts.workspace.nbytes
    views = plan.experts.workspace.views(experts_backing)
    assert views[10].shape == (8, 8)
    assert views[11].numel() == 8 * 16 * torch.bfloat16.itemsize

    router_owner = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=4,
        hidden_size=16,
        local_intermediate_size=8,
        expert_count=4,
        effective_topk=2,
        activation=Activation(function=SiluActivation(), gate=ffn.GateTransform()),
        routed_scaling_factor=1.8,
        router=execution.MoeRouterExecutionSignature(
            algorithm=RouterAlgorithm.SIGMOID,
            projection_precision=RouterProjectionPrecision.FP32,
            weight_rounding=ffn.TopKWeightRounding.NONE,
            routed_expert_count=4,
            renormalize=True,
            projection=weights.ProjectionLayout(
                quantization=quantization.Unquantized(),
                gate_up_order=None,
                values=TensorLayout((4, 16), torch.float32, 0),
                scales=None,
                bias=None,
            ),
            correction_bias=TensorLayout((4,), torch.float32, 0),
        ),
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.CONCATENATED,
            values=TensorLayout((4, 16, 16), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=None,
            values=TensorLayout((4, 16, 8), torch.bfloat16, 0),
            scales=None,
            bias=None,
        ),
    )
    assert execution.control_capture_probe_storage_bytes(router_owner) == (2048, 1024, 256, 16)
