from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from xpool.native.ffn import LayerKind

from xpool import ffn
from xpool.quantization import Mxfp4TensorSource
from xpool.runtime.ffnagent import checkpoint
from xtest.harness.support.config import TEST_MODEL_ID


def encoded_checkpoint(expert: int = 0, *, prefix: str = "model.layers.0.mlp") -> ffn.Mxfp4GatedFfnCheckpoint:
    source = {
        "blocks_key": f"{prefix}.gate_up.blocks",
        "scales_key": f"{prefix}.gate_up.scales",
        "decoded_shape": (2, 192, 64),
        "expert_index": expert,
    }
    return ffn.Mxfp4GatedFfnCheckpoint(
        kind="mxfp4",
        gate=Mxfp4TensorSource(**source, row_part="even"),
        up=Mxfp4TensorSource(**source, row_part="odd"),
        down=Mxfp4TensorSource(
            blocks_key=f"{prefix}.down.blocks",
            scales_key=f"{prefix}.down.scales",
            decoded_shape=(2, 64, 96),
            expert_index=expert,
            row_part="all",
        ),
    )


def model_for(checkpoint: ffn.GatedFfnCheckpoint) -> ffn.FfnModelSpec:
    return ffn.FfnModelSpec(
        model_id=TEST_MODEL_ID,
        architecture_name="Qwen3ForCausalLM",
        hidden_size=64,
        activation=ffn.ActivationKind.SILU,
        layers=(ffn.DenseFfnSpec(kind=LayerKind.DENSE, layer_id=0, intermediate_size=96, checkpoint=checkpoint),),
    )


def test_floating_serialization_and_digest_retain_original_schema() -> None:
    checkpoint = ffn.GatedFfnCheckpointKeys(gate_weight_key="gate", up_weight_key="up", down_weight_key="down")
    spec = model_for(checkpoint)
    expected = {
        "model_id": str(TEST_MODEL_ID),
        "architecture_name": "Qwen3ForCausalLM",
        "hidden_size": 64,
        "activation": int(ffn.ActivationKind.SILU),
        "layers": [
            {
                "kind": int(LayerKind.DENSE),
                "layer_id": 0,
                "intermediate_size": 96,
                "checkpoint": {"gate_weight_key": "gate", "up_weight_key": "up", "down_weight_key": "down"},
            }
        ],
    }
    assert spec.model_dump(mode="json") == expected
    assert (
        spec.digest()
        == hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    assert ffn.FfnModelSpec.model_validate_json(spec.model_dump_json()) == spec


def test_encoded_roundtrip_digest_and_physical_key_order() -> None:
    spec = model_for(encoded_checkpoint())
    assert ffn.FfnModelSpec.model_validate_json(spec.model_dump_json()) == spec
    assert spec.digest() != model_for(encoded_checkpoint(1)).digest()
    assert ffn.checkpoint_keys_for_layer(spec.layers[0]) == (
        "model.layers.0.mlp.gate_up.blocks",
        "model.layers.0.mlp.gate_up.scales",
        "model.layers.0.mlp.down.blocks",
        "model.layers.0.mlp.down.scales",
    )


def test_matrix_views_can_share_a_pair_without_an_expert_index() -> None:
    source = encoded_checkpoint().model_dump()
    for projection in ("gate", "up", "down"):
        source[projection]["decoded_shape"] = source[projection]["decoded_shape"][1:]
        source[projection]["expert_index"] = None
    spec = model_for(ffn.Mxfp4GatedFfnCheckpoint.model_validate(source))
    assert len(ffn.checkpoint_keys_for_layer(spec.layers[0])) == 4


@pytest.mark.parametrize("difference", ("missing", "extra"))
def test_encoded_namespace_coverage_remains_exact(difference: str) -> None:
    spec = model_for(encoded_checkpoint())
    keys = dict.fromkeys(ffn.checkpoint_keys_for_layer(spec.layers[0]), Path("checkpoint.safetensors"))
    if difference == "missing":
        del keys["model.layers.0.mlp.down.scales"]
    else:
        keys["model.layers.0.mlp.unowned"] = Path("checkpoint.safetensors")
    with pytest.raises(ValueError, match=difference):
        checkpoint.validate_ffn_coverage(spec, keys, allow_trailing_ffn_layers=False)


def test_moe_distinct_experts_share_physical_sources() -> None:
    layer = ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=0,
        expert_intermediate_size=96,
        shared_expert_count=0,
        routed_topk=2,
        renormalize=True,
        routed_scaling_factor=1,
        checkpoint=ffn.MoeFfnCheckpointKeys(
            router_weight_key="model.layers.0.mlp.router",
            router_correction_bias_key="model.layers.0.mlp.correction",
            routed_experts=(encoded_checkpoint(0), encoded_checkpoint(1)),
            shared_expert=None,
        ),
    )
    ffn.FfnModelSpec(
        model_id=TEST_MODEL_ID,
        architecture_name="Qwen3MoeForCausalLM",
        hidden_size=64,
        activation=ffn.ActivationKind.SILU,
        layers=(layer,),
    )
    assert ffn.checkpoint_keys_for_layer(layer)[:2] == ("model.layers.0.mlp.router", "model.layers.0.mlp.correction")
    assert len(ffn.checkpoint_keys_for_layer(layer)) == 6


@pytest.mark.parametrize("failure", ("duplicate", "all", "shape", "partner", "role", "floating", "cross_layer"))
def test_encoded_aliases_fail_before_key_deduplication(failure: str) -> None:
    original = encoded_checkpoint()
    payload = original.model_dump()
    if failure == "duplicate":
        payload["up"] = payload["gate"]
    elif failure == "all":
        payload["up"]["row_part"] = "all"
    elif failure == "shape":
        payload["up"]["decoded_shape"] = (2, 194, 64)
    elif failure == "partner":
        payload["up"]["scales_key"] = "other"
    elif failure == "role":
        payload["down"]["blocks_key"] = original.gate.scales_key
    altered = ffn.Mxfp4GatedFfnCheckpoint.model_validate(payload)
    layer = ffn.DenseFfnSpec(kind=LayerKind.DENSE, layer_id=0, intermediate_size=96, checkpoint=altered)
    layers = (layer,)
    if failure == "floating":
        floating = ffn.GatedFfnCheckpointKeys(
            gate_weight_key=original.gate.blocks_key,
            up_weight_key="up",
            down_weight_key="down",
        )
        layers += (ffn.DenseFfnSpec(kind=LayerKind.DENSE, layer_id=1, intermediate_size=96, checkpoint=floating),)
    elif failure == "cross_layer":
        layers += (ffn.DenseFfnSpec(kind=LayerKind.DENSE, layer_id=1, intermediate_size=96, checkpoint=original),)
    with pytest.raises(ValidationError):
        ffn.FfnModelSpec(
            model_id=TEST_MODEL_ID,
            architecture_name="Qwen3ForCausalLM",
            hidden_size=64,
            activation=ffn.ActivationKind.SILU,
            layers=layers,
        )


@pytest.mark.parametrize(
    "change", ({"kind": "other"}, {"kind": None}, {"down": "float-key"}, {"gate_weight_key": "extra"})
)
def test_encoded_variants_do_not_fall_through_to_floating(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ffn.DenseFfnSpec.model_validate(
            {
                "kind": LayerKind.DENSE,
                "layer_id": 0,
                "intermediate_size": 96,
                "checkpoint": {**encoded_checkpoint().model_dump(), **change},
            }
        )


def test_duplicate_floating_keys_and_layer_ids_remain_rejected() -> None:
    keys = ffn.GatedFfnCheckpointKeys(gate_weight_key="same", up_weight_key="same", down_weight_key="down")
    with pytest.raises(ValidationError, match="globally unique"):
        model_for(keys)
    spec = model_for(encoded_checkpoint())
    with pytest.raises(ValidationError, match="layer ids"):
        ffn.FfnModelSpec.model_validate({**spec.model_dump(), "layers": (spec.layers[0], spec.layers[0])})
