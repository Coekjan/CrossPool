from dataclasses import replace

import pytest
import torch

from xpool import ffn, quantization
from xpool.fabric import FfnModelPlan, InstanceFfnLayerProfile, InstanceFfnProfile, place_ffn_layer
from xpool.native.ffn import LayerKind
from xpool.quantization import Mxfp4, TensorLayout, Unquantized
from xpool.runtime.ffnagent import device_memory, execution, preparation, weights
from xpool.runtime.ffnagent.operators import moe
from xpool.runtime.ffnagent.operators.activation import Activation
from xtest.harness.support.config import TEST_MODEL_ID, install_test_config, reset_global_config, synthetic_config


@pytest.fixture
def packed_model() -> ffn.FfnModelSpec:
    return ffn.FfnModelSpec(
        model_id=TEST_MODEL_ID,
        hidden_size=2880,
        activation=ffn.SiluActivation(alpha=1.702),
        gate_transform=ffn.GateTransform(limit=7.0, up_offset=1.0),
        layers=(
            ffn.MoeFfnSpec(
                kind=LayerKind.MOE,
                layer_id=0,
                router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                router_projection_precision=ffn.RouterProjectionPrecision.PAYLOAD,
                topk_weight_rounding=ffn.TopKWeightRounding.LOGITS_DTYPE,
                expert_intermediate_size=2880,
                routed_expert_count=32,
                shared_expert_count=0,
                routed_topk=4,
                renormalize=True,
                routed_scaling_factor=1.0,
                router=ffn.ProjectionSpec(
                    quantization=Unquantized(), values_key="router.weight", bias_key="router.bias"
                ),
                routed_experts=ffn.StackedExpertSource(
                    gate_up=ffn.ProjectionSpec(
                        quantization=Mxfp4(),
                        values_key="gate_up.blocks",
                        scales_key="gate_up.scales",
                        values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                        bias_key="gate_up.bias",
                    ),
                    down=ffn.ProjectionSpec(
                        quantization=Mxfp4(),
                        values_key="down.blocks",
                        scales_key="down.scales",
                        values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                        bias_key="down.bias",
                    ),
                    gate_up_order=ffn.GateUpOrder.INTERLEAVED,
                ),
                shared_expert=None,
            ),
        ),
    )


@pytest.mark.parametrize("tp_size,width,last_length", ((4, 736, 672), (8, 384, 192), (16, 192, 0)))
def test_preparation_keeps_source_coverage_separate_from_padded_execution_width(
    packed_model: ffn.FfnModelSpec, tp_size: int, width: int, last_length: int
) -> None:
    intervals = []
    for rank in range(tp_size):
        layout = preparation.prepare_ffn_layer(
            packed_model, layer_ordinal=0, payload_dtype=torch.bfloat16, tp_size=tp_size, tp_rank=rank
        )
        assert isinstance(layout, preparation.MoeFfnLayerLayout)

        assert layout.activation == Activation(
            function=ffn.SiluActivation(alpha=1.702), gate=ffn.GateTransform(limit=7.0, up_offset=1.0)
        )
        assert layout.local_intermediate_size == width
        assert layout.i_proj.values.shape == (32, 2 * width, 1440)
        assert layout.o_proj.values.shape == (32, 2880, width // 2)
        assert layout.o_proj.scales is not None
        assert layout.o_proj.scales.shape == (32, 2880, width // 32)
        assert layout.o_proj.scales.zero_value == 127
        assert layout.o_proj.bias is not None
        assert layout.o_proj.bias.dtype is torch.float32
        if rank == 0:
            assert layout.router is not None and layout.router.bias is not None
            assert layout.router.bias.dtype is torch.bfloat16
        else:
            assert layout.router is None
        intervals.append((layout.source_start, layout.source_length))
    assert intervals == [(min(rank * width, 2880), min(width, max(0, 2880 - rank * width))) for rank in range(tp_size)]
    assert intervals[-1][1] == last_length
    assert sum(length for start, length in intervals) == 2880
    assert not torch.cuda.is_initialized()


def test_dense_preparation_retains_floating_layout_and_declared_bias() -> None:
    model = ffn.FfnModelSpec(
        model_id=TEST_MODEL_ID,
        hidden_size=16,
        activation=ffn.SiluActivation(),
        layers=(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=32,
                projections=ffn.GatedProjectionSpec(
                    gate=ffn.ProjectionSpec(quantization=Unquantized(), values_key="gate", bias_key="gate.bias"),
                    up=ffn.ProjectionSpec(quantization=Unquantized(), values_key="up"),
                    down=ffn.ProjectionSpec(quantization=Unquantized(), values_key="down"),
                ),
            ),
        ),
    )
    layout = preparation.prepare_ffn_layer(model, layer_ordinal=0, payload_dtype=torch.float16, tp_size=2, tp_rank=1)
    assert isinstance(layout, preparation.DenseFfnLayerLayout)
    assert layout.activation == Activation(function=ffn.SiluActivation(), gate=ffn.GateTransform())
    assert (layout.source_start, layout.source_length, layout.local_intermediate_size) == (16, 16, 16)
    assert layout.i_proj.allocation_bytes() == (1024, 64)
    assert layout.o_proj.allocation_bytes() == (512,)
    layer = model.layers[0]
    assert isinstance(layer, ffn.DenseFfnSpec) and isinstance(layer.projections, ffn.GatedProjectionSpec)
    ungated = layer.model_copy(
        update={"projections": ffn.UngatedProjectionSpec(up=layer.projections.up, down=layer.projections.down)}
    )
    model = model.model_copy(update={"activation": ffn.ReluActivation(), "layers": (ungated,)})
    layout = preparation.prepare_ffn_layer(model, layer_ordinal=0, payload_dtype=torch.float16, tp_size=2, tp_rank=1)
    assert isinstance(layout, preparation.DenseFfnLayerLayout)
    assert layout.activation == Activation(function=ffn.ReluActivation(), gate=None)
    assert layout.i_proj.allocation_bytes() == (512,)


def test_floating_moe_preparation_accepts_gated_relu(packed_model: ffn.FfnModelSpec) -> None:
    layer = packed_model.layers[0]
    assert isinstance(layer, ffn.MoeFfnSpec)
    source = ffn.StackedExpertSource(
        gate_up=ffn.ProjectionSpec(quantization=Unquantized(), values_key="gate_up.weight"),
        down=ffn.ProjectionSpec(quantization=Unquantized(), values_key="down.weight"),
        gate_up_order=ffn.GateUpOrder.CONCATENATED,
    )
    model = packed_model.model_copy(
        update={"activation": ffn.ReluActivation(), "layers": (layer.model_copy(update={"routed_experts": source}),)}
    )
    layout = preparation.prepare_ffn_layer(model, layer_ordinal=0, payload_dtype=torch.bfloat16, tp_size=1, tp_rank=0)
    assert isinstance(layout, preparation.MoeFfnLayerLayout)

    assert layout.activation == Activation(function=ffn.ReluActivation(), gate=packed_model.gate_transform)


def test_preparation_composes_independent_projection_encodings(packed_model: ffn.FfnModelSpec) -> None:
    layer = packed_model.layers[0]
    assert isinstance(layer, ffn.MoeFfnSpec)
    source = layer.routed_experts
    assert isinstance(source, ffn.StackedExpertSource)
    mixed = layer.model_copy(
        update={
            "routed_experts": source.model_copy(
                update={"down": ffn.ProjectionSpec(quantization=Unquantized(), values_key="down")}
            )
        }
    )
    prepared = preparation.prepare_ffn_layer(
        packed_model.model_copy(update={"layers": (mixed,)}),
        layer_ordinal=0,
        payload_dtype=torch.bfloat16,
        tp_size=1,
        tp_rank=0,
    )
    assert isinstance(prepared, preparation.MoeFfnLayerLayout)
    assert isinstance(prepared.i_proj.quantization, Mxfp4)
    assert isinstance(prepared.o_proj.quantization, Unquantized)
    invalid_router = layer.model_copy(
        update={"router": ffn.ProjectionSpec(quantization=Mxfp4(), values_key="router", scales_key="router.scales")}
    )
    prepared = preparation.prepare_ffn_layer(
        packed_model.model_copy(update={"layers": (invalid_router,)}),
        layer_ordinal=0,
        payload_dtype=torch.bfloat16,
        tp_size=1,
        tp_rank=0,
    )
    assert isinstance(prepared, preparation.MoeFfnLayerLayout)
    assert prepared.router is not None and isinstance(prepared.router.quantization, Mxfp4)


def test_packed_workspace_rejects_schedule_index_overflow_without_allocating() -> None:
    signature = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=1 << 20,
        hidden_size=64,
        local_intermediate_size=64,
        expert_count=8,
        effective_topk=4,
        activation=Activation(
            function=ffn.SiluActivation(alpha=1.702), gate=ffn.GateTransform(limit=7.0, up_offset=1.0)
        ),
        routed_scaling_factor=1.0,
        router=None,
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Mxfp4(),
            gate_up_order=ffn.GateUpOrder.INTERLEAVED,
            values=TensorLayout((8, 128, 32), torch.uint8, 0),
            scales=TensorLayout((8, 128, 2), torch.uint8, 127),
            bias=None,
        ),
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Mxfp4(),
            gate_up_order=None,
            values=TensorLayout((8, 64, 32), torch.uint8, 0),
            scales=TensorLayout((8, 64, 2), torch.uint8, 127),
            bias=None,
        ),
    )
    layout = moe.ExpertExecutionPlan.prepare(signature).workspace
    assert (
        layout.offsets[-1] + torch.Size(layout.regions[-1][1]).numel() * layout.regions[-1][0].itemsize == layout.nbytes
    )
    with pytest.raises(ValueError, match="16-bit"):
        moe.ExpertExecutionPlan.prepare(replace(signature, payload_row_capacity=(1 << 20) + 1))


@pytest.mark.usefixtures(reset_global_config.__name__)
def test_estimator_counts_encoded_padded_storages_and_router_owner(packed_model: ffn.FfnModelSpec) -> None:
    install_test_config(synthetic_config(ffn_devices=(1, 2, 3, 4)))
    profile = InstanceFfnProfile(
        payload_dtype=torch.bfloat16,
        hidden_size=packed_model.hidden_size,
        layers=tuple(
            InstanceFfnLayerProfile(layer_id=layer.layer_id, kind=layer.kind) for layer in packed_model.layers
        ),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=65,
        group_sum_complete_admitted=True,
    )
    estimator = device_memory.DeviceMemoryEstimator(
        coefficients=None,
        model_specs=(packed_model,),
        instance_profiles=(profile,),
    )
    expert_requests = (
        32 * 1472 * 1440,
        32 * 1472 * 90,
        4 * 32 * 1472,
        32 * 2880 * 368,
        32 * 2880 * 23,
        4 * 32 * 2880,
    )
    layout = preparation.prepare_ffn_layer(
        packed_model, layer_ordinal=0, payload_dtype=profile.payload_dtype, tp_size=4, tp_rank=0
    )
    plans = (
        FfnModelPlan(
            model_spec_digest=packed_model.digest(),
            layers=(place_ffn_layer(layout.layer, (0, 1, 2, 3), local_intermediate_size=736),),
        ),
    )
    for rank in (0, 3):
        prepared = preparation.prepare_ffn_layer(
            packed_model, layer_ordinal=0, payload_dtype=profile.payload_dtype, tp_size=4, tp_rank=rank
        )
        expected = (*expert_requests, 2 * 32 * 2880, 2 * 32) if rank == 0 else expert_requests
        assert prepared.weight_allocation_bytes() == expected
        point = estimator.allocation_ledger_for_model_plans(model_plans=plans, ffnagent_index=rank)[0]
        assert point.exact_resource_ledger_bytes == sum(expected)
        assert point.features.tensor_storage_allocation_count == len(expected)
        assert point.charges[0].estimated_physical_bytes == device_memory.allocator_reservation_bytes(expected)
    signatures = layout.execution_signatures(profile=profile)
    assert tuple(signature.payload_row_capacity for signature in signatures) == (1, 2, 4, 8, 16, 32, 64, 65)
    for signature in signatures:
        assert isinstance(signature, execution.MoeFfnExecutionSignature)
        assert signature.local_intermediate_size == 736

        assert signature.router is not None
        assert signature.router.weight_rounding is ffn.TopKWeightRounding.LOGITS_DTYPE
        assert execution.control_capture_probe_storage_bytes(signature) == (*expert_requests, 2 * 32 * 2880, 2 * 32)
        prepared_plan = preparation.MoeExecutionPlan.prepare(signature)
        assert prepared_plan.router is not None
        backing = torch.empty(prepared_plan.workspace.nbytes, dtype=torch.uint8)
        expert_storage, router_storage = prepared_plan.workspace.views(backing)
        assert expert_storage.numel() == prepared_plan.experts.workspace.nbytes
        assert router_storage.numel() == prepared_plan.router.workspace.nbytes
        assert preparation.prepare_execution_plans((signature,)) == (prepared_plan,)
    assert execution.capture_weight_pair_key(signatures[-2]) == execution.capture_weight_pair_key(signatures[-1])
    signature = signatures[-1]
    assert isinstance(signature, execution.MoeFfnExecutionSignature) and signature.router is not None
    unrounded = replace(signature, router=replace(signature.router, weight_rounding=ffn.TopKWeightRounding.NONE))
    assert execution.capture_weight_pair_key(unrounded) != execution.capture_weight_pair_key(signature)
    assert not torch.cuda.is_initialized()
