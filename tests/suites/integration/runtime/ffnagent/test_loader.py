"""CUDA behavior tests for canonical FFN true-TP checkpoint packing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import safetensors.torch
import torch

from tests.harness.support.config import install_test_config, reset_global_config
from xpool.config import XpoolConfig
from xpool.ffn import DenseFfnSpec, MoeFfnCheckpointKeys, MoeFfnSpec
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture, loader, weights

pytestmark = [
    pytest.mark.requires_cuda,
    pytest.mark.usefixtures(reset_global_config.__name__),
]


def install_loader_config() -> None:
    """Install a two-reader test config without changing production defaults."""

    install_test_config(
        XpoolConfig.from_mapping(
            {
                "atn": {"devices": [1]},
                "ffn": {"devices": [0], "loader": {"parallelism": 2}},
                "models": [{"id": "model", "path": "/models/model"}],
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
        checkpoint=architecture.gated_checkpoint_keys("model.layers.0.mlp"),
    )


def moe_spec() -> MoeFfnSpec:
    """Return one small corrected-routing MoE layer with two shared Experts."""

    prefix = "model.layers.1.mlp"
    return MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=1,
        expert_intermediate_size=4,
        shared_expert_count=2,
        routed_topk=2,
        renormalize=True,
        routed_scaling_factor=1.8,
        checkpoint=MoeFfnCheckpointKeys(
            router_weight_key=f"{prefix}.gate.weight",
            router_correction_bias_key=f"{prefix}.gate.e_score_correction_bias",
            routed_experts=tuple(
                architecture.gated_checkpoint_keys(f"{prefix}.experts.{expert_id}") for expert_id in range(2)
            ),
            shared_expert=architecture.gated_checkpoint_keys(f"{prefix}.shared_experts"),
        ),
    )


def tensor_values(shape: tuple[int, ...], start: int) -> torch.Tensor:
    """Return recognizable BF16 source values."""

    return torch.arange(start, start + torch.Size(shape).numel(), dtype=torch.float32).reshape(shape).to(torch.bfloat16)


def test_materialize_packs_dense_and_moe_in_stable_request_order(tmp_path: Path) -> None:
    install_loader_config()
    hidden_size = 4
    dense = dense_spec()
    moe = moe_spec()
    dense_tensors = {
        dense.checkpoint.gate_weight_key: tensor_values((6, hidden_size), 0),
        dense.checkpoint.up_weight_key: tensor_values((6, hidden_size), 100),
        dense.checkpoint.down_weight_key: tensor_values((hidden_size, 6), 200),
    }
    moe_tensors: dict[str, torch.Tensor] = {
        moe.checkpoint.router_weight_key: torch.linspace(0.101, 0.909, 2 * hidden_size).reshape(2, hidden_size),
        cast(str, moe.checkpoint.router_correction_bias_key): torch.tensor([0.25, -0.5], dtype=torch.float32),
    }
    for expert_id, expert in enumerate(moe.checkpoint.routed_experts):
        moe_tensors[expert.gate_weight_key] = tensor_values((4, hidden_size), 400 + expert_id * 100)
        moe_tensors[expert.up_weight_key] = tensor_values((4, hidden_size), 500 + expert_id * 100)
        moe_tensors[expert.down_weight_key] = tensor_values((hidden_size, 4), 600 + expert_id * 100)
    shared = moe.checkpoint.shared_expert
    assert shared is not None
    moe_tensors[shared.gate_weight_key] = tensor_values((8, hidden_size), 800)
    moe_tensors[shared.up_weight_key] = tensor_values((8, hidden_size), 900)
    moe_tensors[shared.down_weight_key] = tensor_values((hidden_size, 8), 1000)
    model_path = tmp_path / "model"
    write_checkpoint(model_path, (dense_tensors, moe_tensors))

    materialized = loader.materialize_local_layer_weights(
        requests=(
            loader.LocalLayerWeightRequest(
                model_path=model_path,
                hidden_size=hidden_size,
                payload_dtype=torch.float16,
                router_weight_dtype=None,
                layer=dense,
                tp_rank=1,
                tp_size=2,
            ),
            loader.LocalLayerWeightRequest(
                model_path=model_path,
                hidden_size=hidden_size,
                payload_dtype=torch.bfloat16,
                router_weight_dtype=torch.float32,
                layer=moe,
                tp_rank=0,
                tp_size=2,
            ),
        )
    )

    dense_weights, moe_weights = materialized
    assert isinstance(dense_weights, weights.DenseFfnWeights)
    assert isinstance(moe_weights, weights.MoeFfnWeights)
    assert dense_weights.gate_up_weight.device.index == torch.cuda.current_device()
    assert dense_weights.gate_up_weight.dtype is torch.float16
    torch.testing.assert_close(
        dense_weights.gate_up_weight.cpu(),
        torch.cat(
            (dense_tensors[dense.checkpoint.gate_weight_key][3:], dense_tensors[dense.checkpoint.up_weight_key][3:])
        ).to(torch.float16),
    )
    torch.testing.assert_close(
        dense_weights.down_weight.cpu(),
        dense_tensors[dense.checkpoint.down_weight_key][:, 3:].to(torch.float16),
    )
    assert moe_weights.router is not None
    torch.testing.assert_close(moe_weights.router.weight.cpu(), moe_tensors[moe.checkpoint.router_weight_key])
    assert moe_weights.router.correction_bias is not None
    torch.testing.assert_close(
        moe_weights.router.correction_bias.cpu(),
        moe_tensors[cast(str, moe.checkpoint.router_correction_bias_key)],
    )
    expected_shared_gate = moe_tensors[shared.gate_weight_key].reshape(2, 4, hidden_size)[:, :2]
    torch.testing.assert_close(moe_weights.expert_gate_up_weight.cpu()[2:, :2], expected_shared_gate)
    expected_shared_down = moe_tensors[shared.down_weight_key].reshape(hidden_size, 2, 4)[:, :, :2].permute(1, 0, 2)
    torch.testing.assert_close(moe_weights.expert_down_weight.cpu()[2:], expected_shared_down)


def test_materialize_non_router_rank_and_shape_failure(tmp_path: Path) -> None:
    install_loader_config()
    layer = moe_spec()
    hidden_size = 4
    tensors: dict[str, torch.Tensor] = {}
    for expert_id, expert in enumerate(layer.checkpoint.routed_experts):
        tensors[expert.gate_weight_key] = tensor_values((4, hidden_size), expert_id * 100)
        tensors[expert.up_weight_key] = tensor_values((4, hidden_size), 200 + expert_id * 100)
        tensors[expert.down_weight_key] = tensor_values((hidden_size, 4), 400 + expert_id * 100)
    shared = layer.checkpoint.shared_expert
    assert shared is not None
    tensors[shared.gate_weight_key] = tensor_values((8, hidden_size), 600)
    tensors[shared.up_weight_key] = tensor_values((8, hidden_size), 700)
    tensors[shared.down_weight_key] = tensor_values((hidden_size, 8), 800)
    tensors[layer.checkpoint.router_weight_key] = tensor_values((2, hidden_size), 900)
    tensors[cast(str, layer.checkpoint.router_correction_bias_key)] = torch.ones(3, dtype=torch.float32)
    model_path = tmp_path / "model"
    write_checkpoint(model_path, (tensors,))
    request = loader.LocalLayerWeightRequest(
        model_path=model_path,
        hidden_size=hidden_size,
        payload_dtype=torch.bfloat16,
        router_weight_dtype=torch.bfloat16,
        layer=layer,
        tp_rank=1,
        tp_size=2,
    )

    (materialized,) = loader.materialize_local_layer_weights(requests=(request,))
    assert isinstance(materialized, weights.MoeFfnWeights)
    assert materialized.router is None

    bad_request = loader.LocalLayerWeightRequest(
        model_path=model_path,
        hidden_size=hidden_size,
        payload_dtype=torch.bfloat16,
        router_weight_dtype=torch.bfloat16,
        layer=layer,
        tp_rank=0,
        tp_size=2,
    )
    with pytest.raises(RuntimeError, match=r"expected floating-point \(2,\), found torch.float32 \(3,\)"):
        loader.materialize_local_layer_weights(requests=(bad_request,))
