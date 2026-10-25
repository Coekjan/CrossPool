"""Checkpoint source reading and canonical FFN true-TP packing."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import pytest
import safetensors.torch
import torch

import xtest
from xpool import ffn
from xpool.config import XpoolConfig
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.ffn import (
    DenseFfnSpec,
    FfnModelSpec,
    MoeFfnSpec,
    PerExpertSource,
    RouterAlgorithm,
    RouterProjectionPrecision,
    SiluActivation,
)
from xpool.native.ffn import LayerKind
from xpool.quantization import Mxfp4, QuantizationSpec, TensorLayout, TensorSlice, Unquantized
from xpool.runtime.ffnagent import architecture, execution, loader, preparation, weights
from xpool.runtime.ffnagent.operators.triton_moe import TritonKernelsExperts
from xtest.harness.native.ffn.qualification import packed_moe_reference
from xtest.harness.support.config import TEST_MODEL_ID, install_test_config, reset_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def install_loader_config() -> None:
    """Install a two-reader test config without changing production defaults."""

    install_test_config(
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1], "loader": {"parallelism": 2}},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/model"}],
            }
        )
    )


def write_checkpoint(path: Path, shards: tuple[dict[str, torch.Tensor], ...]) -> None:
    """Write one indexed multi-file checkpoint for a packing test."""

    path.mkdir()
    weight_map: dict[str, str] = {}
    for index, tensors in enumerate(shards, start=1):
        shard_name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        safetensors.torch.save_file(tensors, path / shard_name)
        weight_map.update(dict.fromkeys(tensors, shard_name))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def dense_spec() -> DenseFfnSpec:
    """Return one small canonical Dense layer."""

    return DenseFfnSpec(
        kind=LayerKind.DENSE,
        layer_id=0,
        intermediate_size=6,
        projections=architecture.FfnSourceConfig({}).gated_projections("model.layers.0.mlp"),
    )


def moe_spec(router_correction_bias_key: str) -> MoeFfnSpec:
    """Return one small corrected-routing MoE layer with two shared Experts."""

    prefix = "model.layers.1.mlp"
    return MoeFfnSpec(
        kind=LayerKind.MOE,
        router_algorithm=RouterAlgorithm.SIGMOID,
        router_projection_precision=RouterProjectionPrecision.FP32,
        layer_id=1,
        expert_intermediate_size=4,
        routed_expert_count=2,
        shared_expert_count=2,
        routed_topk=2,
        renormalize=True,
        routed_scaling_factor=1.8,
        router=architecture.FfnSourceConfig({}).projection(f"{prefix}.gate"),
        router_correction_bias_key=router_correction_bias_key,
        routed_experts=PerExpertSource(
            experts=tuple(
                architecture.FfnSourceConfig({}).gated_projections(f"{prefix}.experts.{expert_id}")
                for expert_id in range(2)
            ),
        ),
        shared_expert=architecture.FfnSourceConfig({}).gated_projections(f"{prefix}.shared_experts"),
    )


def tensor_values(shape: tuple[int, ...], start: int) -> torch.Tensor:
    """Return recognizable BF16 source values."""

    return torch.arange(start, start + torch.Size(shape).numel(), dtype=torch.float32).reshape(shape).to(torch.bfloat16)


@xtest.requirements(device_count=1)
def test_projection_resources_match_layout_and_use_encoding_correct_zero_probe() -> None:
    encoded = Mxfp4().weight_layout(logical_shape=(32, 64), dtype=torch.bfloat16)
    layout = weights.ProjectionLayout(
        values=encoded.values,
        scales=encoded.scales,
        bias=TensorLayout((32,), torch.float32, 0),
    )
    device = torch.device("cuda", torch.cuda.current_device())
    loaded = layout.allocate(device)
    probe = layout.zero_probe(device)
    loaded.validate(layout)
    probe.validate(layout)
    assert loaded.scales is not None and loaded.bias is not None
    assert probe.scales is not None and probe.bias is not None
    assert tuple(tensor.nbytes for tensor in (loaded.values, loaded.scales, loaded.bias)) == layout.allocation_bytes()
    assert loaded.values.data_ptr() != probe.values.data_ptr()
    torch.testing.assert_close(probe.values, torch.zeros_like(probe.values))
    torch.testing.assert_close(probe.scales, torch.full_like(probe.scales, 127))
    torch.testing.assert_close(probe.bias, torch.zeros_like(probe.bias))
    loaded.binding_view()
    probe.binding_view()
    with pytest.raises(ValueError, match="presence disagrees"):
        loaded.validate(weights.ProjectionLayout(values=layout.values, scales=None, bias=layout.bias))
    with pytest.raises(ValueError, match="shape or dtype disagrees"):
        loaded.validate(
            weights.ProjectionLayout(
                values=TensorLayout((32, 16), torch.uint8, 0),
                scales=layout.scales,
                bias=layout.bias,
            )
        )


@xtest.requirements(device_count=1)
def test_materialize_packs_dense_and_moe_in_stable_request_order(tmp_path: Path) -> None:
    install_loader_config()
    hidden_size = 4
    dense = dense_spec()
    correction_bias_key = "model.layers.1.mlp.gate.e_score_correction_bias"
    moe = moe_spec(correction_bias_key)
    dense_tensors = {
        dense.projections.gate.values_key: tensor_values((6, hidden_size), 0),
        dense.projections.up.values_key: tensor_values((6, hidden_size), 100),
        dense.projections.down.values_key: tensor_values((hidden_size, 6), 200),
    }
    moe_tensors: dict[str, torch.Tensor] = {
        moe.router.values_key: torch.linspace(0.101, 0.909, 2 * hidden_size).reshape(2, hidden_size),
        correction_bias_key: torch.tensor([0.25, -0.5], dtype=torch.float32),
    }
    assert isinstance(moe.routed_experts, PerExpertSource)
    for expert_id, expert in enumerate(moe.routed_experts.experts):
        moe_tensors[expert.gate.values_key] = tensor_values((4, hidden_size), 400 + expert_id * 100)
        moe_tensors[expert.up.values_key] = tensor_values((4, hidden_size), 500 + expert_id * 100)
        moe_tensors[expert.down.values_key] = tensor_values((hidden_size, 4), 600 + expert_id * 100)
    shared = moe.shared_expert
    assert shared is not None
    moe_tensors[shared.gate.values_key] = tensor_values((8, hidden_size), 800)
    moe_tensors[shared.up.values_key] = tensor_values((8, hidden_size), 900)
    moe_tensors[shared.down.values_key] = tensor_values((hidden_size, 8), 1000)
    model_path = tmp_path / "model"
    write_checkpoint(model_path, (dense_tensors, moe_tensors))
    spec = FfnModelSpec(
        model_id=TEST_MODEL_ID,
        hidden_size=hidden_size,
        activation=SiluActivation(),
        layers=(dense, moe),
    )
    dense_layout = preparation.prepare_ffn_layer(
        spec,
        layer_ordinal=0,
        payload_dtype=torch.float16,
        tp_size=2,
        tp_rank=1,
    )
    moe_layout = preparation.prepare_ffn_layer(
        spec,
        layer_ordinal=1,
        payload_dtype=torch.bfloat16,
        tp_size=2,
        tp_rank=0,
    )
    assert isinstance(dense_layout, preparation.DenseFfnLayerLayout)
    assert isinstance(moe_layout, preparation.MoeFfnLayerLayout)

    materialized = loader.materialize_local_layer_weights(
        requests=(
            loader.LocalLayerWeightRequest(
                model_path=model_path,
                layer_layout=dense_layout,
            ),
            loader.LocalLayerWeightRequest(
                model_path=model_path,
                layer_layout=moe_layout,
            ),
        )
    )

    dense_weights, moe_weights = materialized
    assert isinstance(dense_weights, weights.DenseFfnWeights)
    assert isinstance(moe_weights, weights.MoeFfnWeights)
    assert dense_weights.gate_up.values.device.index == torch.cuda.current_device()
    assert dense_weights.gate_up.values.dtype is torch.float16
    torch.testing.assert_close(
        dense_weights.gate_up.values.cpu(),
        torch.cat(
            (dense_tensors[dense.projections.gate.values_key][3:], dense_tensors[dense.projections.up.values_key][3:])
        ).to(torch.float16),
    )
    torch.testing.assert_close(
        dense_weights.down.values.cpu(),
        dense_tensors[dense.projections.down.values_key][:, 3:].to(torch.float16),
    )
    assert moe_weights.router is not None
    torch.testing.assert_close(moe_weights.router.projection.values.cpu(), moe_tensors[moe.router.values_key])
    assert moe_weights.router.correction_bias is not None
    torch.testing.assert_close(
        moe_weights.router.correction_bias.cpu(),
        moe_tensors[correction_bias_key],
    )
    expected_shared_gate = moe_tensors[shared.gate.values_key].reshape(2, 4, hidden_size)[:, :2]
    torch.testing.assert_close(moe_weights.gate_up.values.cpu()[2:, :2], expected_shared_gate)
    expected_shared_down = moe_tensors[shared.down.values_key].reshape(hidden_size, 2, 4)[:, :, :2].permute(1, 0, 2)
    torch.testing.assert_close(moe_weights.down.values.cpu()[2:], expected_shared_down)

    device = torch.device("cuda", torch.cuda.current_device())
    dense_probe = weights.DenseFfnWeights.allocate(gate_up=dense_layout.gate_up, down=dense_layout.down, device=device)
    moe_probe = weights.MoeFfnWeights.allocate(
        gate_up=moe_layout.experts.gate_up,
        down=moe_layout.experts.down,
        router=moe_layout.router,
        correction_bias=moe_layout.router_correction_bias,
        device=device,
    )
    assert moe_probe.router is not None
    assert moe_probe.router.correction_bias is not None
    for source, probe in (
        (dense_weights.gate_up.values, dense_probe.gate_up.values),
        (dense_weights.down.values, dense_probe.down.values),
        (moe_weights.gate_up.values, moe_probe.gate_up.values),
        (moe_weights.down.values, moe_probe.down.values),
        (moe_weights.router.projection.values, moe_probe.router.projection.values),
        (moe_weights.router.correction_bias, moe_probe.router.correction_bias),
    ):
        assert probe.data_ptr() != source.data_ptr()
        torch.testing.assert_close(probe, torch.zeros_like(source), rtol=0, atol=0)


@xtest.requirements(device_count=1)
def test_materialize_non_router_rank_and_shape_failure(tmp_path: Path) -> None:
    install_loader_config()
    correction_bias_key = "model.layers.1.mlp.gate.e_score_correction_bias"
    layer = moe_spec(correction_bias_key)
    hidden_size = 4
    tensors: dict[str, torch.Tensor] = {}
    assert isinstance(layer.routed_experts, PerExpertSource)
    for expert_id, expert in enumerate(layer.routed_experts.experts):
        tensors[expert.gate.values_key] = tensor_values((4, hidden_size), expert_id * 100)
        tensors[expert.up.values_key] = tensor_values((4, hidden_size), 200 + expert_id * 100)
        tensors[expert.down.values_key] = tensor_values((hidden_size, 4), 400 + expert_id * 100)
    shared = layer.shared_expert
    assert shared is not None
    tensors[shared.gate.values_key] = tensor_values((8, hidden_size), 600)
    tensors[shared.up.values_key] = tensor_values((8, hidden_size), 700)
    tensors[shared.down.values_key] = tensor_values((hidden_size, 8), 800)
    tensors[layer.router.values_key] = tensor_values((2, hidden_size), 900)
    tensors[correction_bias_key] = torch.ones(3, dtype=torch.float32)
    model_path = tmp_path / "model"
    write_checkpoint(model_path, (tensors,))
    spec = FfnModelSpec(
        model_id=TEST_MODEL_ID,
        hidden_size=hidden_size,
        activation=SiluActivation(),
        layers=(layer,),
    )
    request = loader.LocalLayerWeightRequest(
        model_path=model_path,
        layer_layout=preparation.prepare_ffn_layer(
            spec,
            layer_ordinal=0,
            payload_dtype=torch.bfloat16,
            tp_rank=1,
            tp_size=2,
        ),
    )

    (materialized,) = loader.materialize_local_layer_weights(requests=(request,))
    assert isinstance(materialized, weights.MoeFfnWeights)
    assert materialized.router is None

    bad_request = loader.LocalLayerWeightRequest(
        model_path=model_path,
        layer_layout=preparation.prepare_ffn_layer(
            spec,
            layer_ordinal=0,
            payload_dtype=torch.bfloat16,
            tp_rank=0,
            tp_size=2,
        ),
    )
    with pytest.raises(RuntimeError, match=r"expected torch.float32 \(2,\), found torch.float32 \(3,\)"):
        loader.materialize_local_layer_weights(requests=(bad_request,))


@pytest.mark.parametrize(
    "encoding,source_order",
    (
        (Unquantized(), ffn.GateUpOrder.CONCATENATED),
        (Unquantized(), ffn.GateUpOrder.INTERLEAVED),
        (Mxfp4(), ffn.GateUpOrder.CONCATENATED),
        (Mxfp4(), ffn.GateUpOrder.INTERLEAVED),
        (Mxfp4(), None),
    ),
)
@xtest.requirements(device_count=1)
def test_loading_normalizes_stacked_and_separate_sources_for_the_selected_operation(
    tmp_path: Path, encoding: QuantizationSpec, source_order: ffn.GateUpOrder | None
) -> None:
    install_loader_config()
    experts, intermediate, hidden = 2, 64, 64
    encoded = encoding.weight_layout(logical_shape=(experts, intermediate, hidden), dtype=torch.bfloat16)
    gate = torch.arange(torch.Size(encoded.values.shape).numel()).reshape(encoded.values.shape).to(encoded.values.dtype)
    up = gate + 17
    gate_bias = torch.arange(experts * intermediate).reshape(experts, intermediate).bfloat16()
    up_bias = gate_bias + 128
    gate_scales = None
    up_scales = None
    if encoded.scales is not None:
        gate_scales = torch.full(encoded.scales.shape, 125, dtype=torch.uint8)
        up_scales = torch.full(encoded.scales.shape, 127, dtype=torch.uint8)
    down = gate.clone()
    down_scales = None if gate_scales is None else gate_scales.clone()
    down_bias = torch.arange(experts * hidden).reshape(experts, hidden).bfloat16()
    tensors = {"router": torch.zeros((experts, hidden), dtype=torch.bfloat16)}
    if source_order is None:
        projections = []
        for expert in range(experts):
            declarations = []
            for name, values, scales, bias in (
                ("gate", gate, gate_scales, gate_bias),
                ("up", up, up_scales, up_bias),
                ("down", down, down_scales, down_bias),
            ):
                key = f"experts.{expert}.{name}"
                tensors[key] = values[expert].contiguous()
                tensors[f"{key}.bias"] = bias[expert].contiguous()
                if scales is not None:
                    tensors[f"{key}.scales"] = scales[expert].contiguous()
                declarations.append(
                    ffn.ProjectionSpec(
                        quantization=encoding,
                        values_key=key,
                        scales_key=None if scales is None else f"{key}.scales",
                        bias_key=f"{key}.bias",
                    )
                )
            projections.append(ffn.GatedProjectionSpec(gate=declarations[0], up=declarations[1], down=declarations[2]))
        source = ffn.PerExpertSource(experts=tuple(projections))
    else:
        gate_up = (
            torch.cat((gate, up), dim=1)
            if source_order is ffn.GateUpOrder.CONCATENATED
            else torch.stack((gate, up), dim=2).flatten(1, 2)
        )
        bias = (
            torch.cat((gate_bias, up_bias), dim=1)
            if source_order is ffn.GateUpOrder.CONCATENATED
            else torch.stack((gate_bias, up_bias), dim=2).flatten(1, 2)
        )
        source_layout = ffn.ValuesSourceLayout.MATRIX
        tensors.update({"gate_up": gate_up, "gate_up.bias": bias, "down": down, "down.bias": down_bias})
        if gate_scales is not None and up_scales is not None:
            tensors["gate_up.scales"] = (
                torch.cat((gate_scales, up_scales), dim=1)
                if source_order is ffn.GateUpOrder.CONCATENATED
                else torch.stack((gate_scales, up_scales), dim=2).flatten(1, 2)
            )
            tensors["gate_up"] = gate_up.view(experts, 2 * intermediate, hidden // 32, 16)
            source_layout = ffn.ValuesSourceLayout.SCALE_BLOCKED
        if down_scales is not None:
            tensors["down.scales"] = down_scales
        source = ffn.StackedExpertSource(
            gate_up=ffn.ProjectionSpec(
                quantization=encoding,
                values_key="gate_up",
                scales_key=None if gate_scales is None else "gate_up.scales",
                bias_key="gate_up.bias",
                values_layout=source_layout,
            ),
            down=ffn.ProjectionSpec(
                quantization=encoding,
                values_key="down",
                scales_key=None if down_scales is None else "down.scales",
                bias_key="down.bias",
            ),
            gate_up_order=source_order,
        )
    layer = ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=0,
        router_algorithm=RouterAlgorithm.SOFTMAX,
        router_projection_precision=RouterProjectionPrecision.PAYLOAD,
        expert_intermediate_size=intermediate,
        routed_expert_count=experts,
        shared_expert_count=0,
        routed_topk=1,
        renormalize=True,
        routed_scaling_factor=1.0,
        router=ffn.ProjectionSpec(quantization=Unquantized(), values_key="router"),
        routed_experts=source,
        shared_expert=None,
    )
    spec = FfnModelSpec(model_id=TEST_MODEL_ID, hidden_size=hidden, activation=SiluActivation(), layers=(layer,))
    model_path = tmp_path / "model"
    write_checkpoint(model_path, (tensors,))
    layout = preparation.prepare_ffn_layer(spec, layer_ordinal=0, payload_dtype=torch.bfloat16, tp_size=2, tp_rank=1)
    (loaded,) = loader.materialize_local_layer_weights(
        requests=(loader.LocalLayerWeightRequest(model_path=model_path, layer_layout=layout),)
    )
    assert isinstance(loaded, weights.MoeFfnWeights)
    expected_gate_up = (
        torch.cat((gate[:, 32:], up[:, 32:]), dim=1)
        if isinstance(encoding, Unquantized)
        else torch.stack((gate[:, 32:], up[:, 32:]), dim=2).flatten(1, 2)
    )
    expected_bias = (
        torch.cat((gate_bias[:, 32:], up_bias[:, 32:]), dim=1)
        if isinstance(encoding, Unquantized)
        else torch.stack((gate_bias[:, 32:], up_bias[:, 32:]), dim=2).flatten(1, 2)
    )
    torch.testing.assert_close(loaded.gate_up.values.cpu(), expected_gate_up, rtol=0, atol=0)
    assert loaded.gate_up.bias is not None
    torch.testing.assert_close(loaded.gate_up.bias.cpu(), expected_bias.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.down.values.cpu(), down[:, :, 16 if isinstance(encoding, Mxfp4) else 32 :], rtol=0, atol=0
    )
    assert loaded.down.bias is not None
    torch.testing.assert_close(loaded.down.bias, torch.zeros_like(loaded.down.bias), rtol=0, atol=0)
    if gate_scales is not None and up_scales is not None:
        assert loaded.gate_up.scales is not None and loaded.down.scales is not None
        assert down_scales is not None
        expected_scales = torch.stack((gate_scales[:, 32:], up_scales[:, 32:]), dim=2).flatten(1, 2)
        torch.testing.assert_close(loaded.gate_up.scales.cpu(), expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(loaded.down.scales.cpu(), down_scales[:, :, 1:], rtol=0, atol=0)


@pytest.mark.parametrize("bias", (False, True))
@xtest.requirements(device_count=1)
def test_staged_strided_transfer_orders_initialization_without_device_tensor_allocation(bias: bool) -> None:
    shape = (2, 6) if bias else (2, 6, 8)
    dtype = torch.float32 if bias else torch.uint8
    root = torch.empty(shape, dtype=dtype, device="cuda")
    destination = root[:, 1::2]
    descriptor = loader.TensorReadDescriptor.from_key_view(
        {"weight": Path("weight.safetensors")},
        weight_key="weight",
        source_shape=tuple(destination.shape),
        selection=None,
        destination=destination,
    )
    slot = torch.empty(descriptor.staged_bytes, dtype=torch.uint8, pin_memory=True)
    expected = torch.arange(destination.numel()).to(dtype).reshape(destination.shape)
    descriptor.slot_view(slot).copy_(expected)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        root.fill_(255)
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        descriptor.copy_from(slot)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    actual = root.cpu()
    torch.testing.assert_close(actual[:, 1::2], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 0::2], torch.full_like(actual[:, 0::2], 255), rtol=0, atol=0)


@pytest.mark.parametrize("tp_size,ranks", ((4, tuple(range(4))), (8, tuple(range(8))), (16, (15,))))
@xtest.requirements(device_count=1)
def test_stacked_packed_loading_preserves_source_slices_padding_and_single_bias_contribution(
    tmp_path: Path,
    tp_size: int,
    ranks: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_loader_config()
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    experts, hidden, intermediate = 4, 64, 2880
    activation = ffn.SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0)
    layer = ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=0,
        router_algorithm=RouterAlgorithm.SOFTMAX,
        router_projection_precision=RouterProjectionPrecision.PAYLOAD,
        expert_intermediate_size=intermediate,
        routed_expert_count=experts,
        shared_expert_count=0,
        routed_topk=2,
        renormalize=True,
        routed_scaling_factor=1.0,
        router=ffn.ProjectionSpec(quantization=Unquantized(), values_key="router", bias_key="router.bias"),
        routed_experts=ffn.StackedExpertSource(
            gate_up=ffn.ProjectionSpec(
                quantization=Mxfp4(),
                values_key="gate_up",
                scales_key="gate_up.scales",
                values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                bias_key="gate_up.bias",
            ),
            down=ffn.ProjectionSpec(
                quantization=Mxfp4(),
                values_key="down",
                scales_key="down.scales",
                values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                bias_key="down.bias",
            ),
            gate_up_order=ffn.GateUpOrder.INTERLEAVED,
        ),
        shared_expert=None,
    )
    spec = FfnModelSpec(
        model_id=TEST_MODEL_ID,
        hidden_size=hidden,
        activation=activation,
        layers=(layer,),
    )
    tensors = {
        "gate_up": torch.arange(experts * 2 * intermediate * hidden // 2)
        .to(torch.uint8)
        .view(experts, 2 * intermediate, hidden // 2),
        "down": torch.arange(experts * hidden * intermediate // 2)
        .to(torch.uint8)
        .view(experts, hidden, intermediate // 2),
        "gate_up.scales": (123 + torch.arange(experts * 2 * intermediate * hidden // 32) % 5)
        .to(torch.uint8)
        .view(experts, 2 * intermediate, hidden // 32),
        "down.scales": (123 + torch.arange(experts * hidden * intermediate // 32) % 5)
        .to(torch.uint8)
        .view(experts, hidden, intermediate // 32),
        "gate_up.bias": torch.linspace(-2, 2, experts * 2 * intermediate).bfloat16().view(experts, 2 * intermediate),
        "down.bias": torch.linspace(-2, 2, experts * hidden).bfloat16().view(experts, hidden),
        "router": torch.arange(experts * hidden).bfloat16().view(experts, hidden),
        "router.bias": torch.arange(experts).bfloat16(),
    }
    stored = dict(tensors)
    stored["gate_up"] = tensors["gate_up"].view(experts, 2 * intermediate, hidden // 32, 16)
    stored["down"] = tensors["down"].view(experts, hidden, intermediate // 32, 16)
    model_path = tmp_path / "model"
    write_checkpoint(model_path, (stored,))
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(318)
    hidden_states = torch.randn((17, hidden), device=device, generator=generator, dtype=torch.bfloat16)
    ids = torch.arange(34, device=device, dtype=torch.int32).remainder(experts).view(17, 2)
    probabilities = torch.tensor((0.25, 0.75), device=device).expand(17, 2).contiguous()
    actual_output = torch.zeros_like(hidden_states, dtype=torch.float32)
    profile = InstanceFfnProfile(
        payload_dtype=torch.bfloat16,
        hidden_size=hidden,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.MOE),),
        decode_payload_row_capacity=17,
        prefill_payload_row_capacity=17,
        group_sum_complete_admitted=True,
    )
    for rank in ranks:
        layout = preparation.prepare_ffn_layer(
            spec,
            layer_ordinal=0,
            payload_dtype=torch.bfloat16,
            tp_size=tp_size,
            tp_rank=rank,
        )
        assert isinstance(layout, preparation.MoeFfnLayerLayout)
        (loaded,) = loader.materialize_local_layer_weights(
            requests=(loader.LocalLayerWeightRequest(model_path=model_path, layer_layout=layout),)
        )
        assert isinstance(loaded, weights.MoeFfnWeights)
        loaded.gate_up.validate(layout.experts.gate_up)
        loaded.down.validate(layout.experts.down)
        start, length = layout.source_start, layout.source_length
        assert loaded.gate_up.scales is not None and loaded.down.scales is not None
        assert loaded.gate_up.bias is not None and loaded.down.bias is not None
        for actual, expected in (
            (loaded.gate_up.values[:, : 2 * length].cpu(), tensors["gate_up"][:, 2 * start : 2 * (start + length)]),
            (
                loaded.gate_up.scales[:, : 2 * length].cpu(),
                tensors["gate_up.scales"][:, 2 * start : 2 * (start + length)],
            ),
            (
                loaded.gate_up.bias[:, : 2 * length].cpu(),
                tensors["gate_up.bias"][:, 2 * start : 2 * (start + length)].float(),
            ),
            (loaded.down.values[:, :, : length // 2].cpu(), tensors["down"][:, :, start // 2 : (start + length) // 2]),
            (
                loaded.down.scales[:, :, : length // 32].cpu(),
                tensors["down.scales"][:, :, start // 32 : (start + length) // 32],
            ),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for padding, neutral in (
            (loaded.gate_up.values[:, 2 * length :], 0),
            (loaded.gate_up.scales[:, 2 * length :], 127),
            (loaded.gate_up.bias[:, 2 * length :], 0),
            (loaded.down.values[:, :, length // 2 :], 0),
            (loaded.down.scales[:, :, length // 32 :], 127),
        ):
            torch.testing.assert_close(padding, torch.full_like(padding, neutral), rtol=0, atol=0)
        torch.testing.assert_close(
            loaded.down.bias.cpu(),
            tensors["down.bias"].float() if rank == 0 else torch.zeros_like(tensors["down.bias"], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        assert (loaded.router is not None) == (rank == 0)
        if loaded.router is not None:
            assert loaded.router.projection.bias is not None
            torch.testing.assert_close(loaded.router.projection.values.cpu(), tensors["router"], rtol=0, atol=0)
            torch.testing.assert_close(loaded.router.projection.bias.cpu(), tensors["router.bias"], rtol=0, atol=0)
        signature = preparation.required_execution_signatures(layer_layout=layout, profile=profile)[-1]
        assert isinstance(signature, execution.MoeFfnExecutionSignature)
        workspace = torch.empty(TritonKernelsExperts.workspace_layout(signature)[2], device=device, dtype=torch.uint8)
        operation = TritonKernelsExperts(signature, workspace)
        partial = torch.empty_like(hidden_states)
        operation.run(
            hidden_states=hidden_states,
            gate_up=loaded.gate_up,
            down=loaded.down,
            topk_ids=ids,
            topk_weights=probabilities,
            output=partial,
        )
        if length == 0:
            assert torch.count_nonzero(partial) == 0
        actual_output.add_(partial.float())

    if len(ranks) == tp_size:
        expected_output = packed_moe_reference(
            hidden_states,
            gate_up=weights.ProjectionWeights(
                tensors["gate_up"].to(device),
                tensors["gate_up.scales"].to(device),
                tensors["gate_up.bias"].float().to(device),
            ),
            down=weights.ProjectionWeights(
                tensors["down"].to(device), tensors["down.scales"].to(device), tensors["down.bias"].float().to(device)
            ),
            ids=ids,
            probabilities=probabilities,
            activation=activation,
            routed_scaling_factor=layer.routed_scaling_factor,
        ).float()
        scale = expected_output.square().mean().sqrt().clamp_min(1e-12)
        error = actual_output.to(hidden_states.dtype).float() - expected_output
        assert error.square().mean().sqrt() / scale <= 0.01
        assert (error.abs() / torch.maximum(expected_output.abs(), scale)).max() <= 0.10


@pytest.mark.parametrize(
    "source_dtype,destination_dtype",
    ((torch.uint8, torch.uint8), (torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)),
)
def test_reader_views_before_tp_slice_and_preserves_storage_encoding(
    tmp_path: Path,
    source_dtype: torch.dtype,
    destination_dtype: torch.dtype,
) -> None:
    source = torch.arange(256).to(source_dtype).view(2, 4, 2, 16)
    shard = tmp_path / "weights.safetensors"
    safetensors.torch.save_file({"weight": source}, shard)
    descriptor = loader.TensorReadDescriptor.from_key_view(
        {"weight": shard},
        weight_key="weight",
        source_shape=(2, 4, 2, 16),
        view_shape=(2, 4, 32),
        selection=TensorSlice(1, 1, 2, step=2),
        destination=torch.empty((2, 2, 32), dtype=destination_dtype),
    )
    slots: queue.Queue[torch.Tensor] = queue.Queue()
    slots.put(torch.empty(descriptor.staged_bytes, dtype=torch.uint8))
    messages: queue.Queue[loader.ReaderMessage] = queue.Queue()
    cancelled = threading.Event()
    loader.stage_shard(shard_path=shard, descriptors=(descriptor,), slots=slots, messages=messages, cancelled=cancelled)

    message = messages.get_nowait()
    assert isinstance(message, loader.StagedTensor)
    torch.testing.assert_close(
        descriptor.slot_view(message.slot),
        source.view(2, 4, 32)[:, 1:4:2].to(destination_dtype),
        rtol=0,
        atol=0,
    )
    assert isinstance(messages.get_nowait(), loader.ReaderFinished)
    assert not cancelled.is_set()


@pytest.mark.parametrize("source_dtype,destination_dtype", ((torch.uint8, torch.float32), (torch.float32, torch.uint8)))
def test_reader_rejects_numeric_conversion_of_encoded_storage(
    tmp_path: Path,
    source_dtype: torch.dtype,
    destination_dtype: torch.dtype,
) -> None:
    shard = tmp_path / "weights.safetensors"
    safetensors.torch.save_file({"weight": torch.ones((2, 32), dtype=source_dtype)}, shard)
    descriptor = loader.TensorReadDescriptor.from_key_view(
        {"weight": shard},
        weight_key="weight",
        source_shape=(2, 32),
        selection=None,
        destination=torch.empty((2, 32), dtype=destination_dtype),
    )
    slot = torch.empty(descriptor.staged_bytes, dtype=torch.uint8)
    slots: queue.Queue[torch.Tensor] = queue.Queue()
    slots.put(slot)
    messages: queue.Queue[loader.ReaderMessage] = queue.Queue()
    cancelled = threading.Event()
    loader.stage_shard(shard_path=shard, descriptors=(descriptor,), slots=slots, messages=messages, cancelled=cancelled)

    failure = messages.get_nowait()
    assert isinstance(failure, loader.ReaderFailure)
    assert isinstance(failure.error, ValueError)
    assert "incompatible dtype or shape" in str(failure.error)
    assert isinstance(messages.get_nowait(), loader.ReaderFinished)
    assert slots.get_nowait().numel() == slot.numel()
    assert cancelled.is_set()
