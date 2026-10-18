from __future__ import annotations

import pytest
import torch

from xpool import ffn
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.runtime.ffnagent import architecture, execution
from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter
from xtest.harness.support.config import TEST_MODEL_ID


@pytest.mark.parametrize(("layers", "experts"), ((24, 32), (36, 128)))
def test_compile_both_gpt_oss_geometries_and_resource_ledger(layers: int, experts: int) -> None:
    config = {
        "model_type": "gpt_oss",
        "torch_dtype": "bfloat16",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": layers,
        "num_local_experts": experts,
        "num_experts_per_tok": 4,
        "swiglu_limit": 7,
        "quantization_config": {"quant_method": "mxfp4"},
    }
    spec = GptOssAdapter.compile(model_id=TEST_MODEL_ID, model_config=architecture.FfnSourceConfig(config))
    assert spec.activation is ffn.ActivationKind.SWIGLU_OAI
    assert spec.activation_alpha == 1.702 and spec.activation_limit == 7
    assert len(spec.layers) == layers
    layer = spec.layers[0]
    assert isinstance(layer, ffn.MoeFfnSpec) and layer.routed_expert_count == experts
    assert len(ffn.checkpoint_keys_for_layer(layer)) == 8
    profile = InstanceFfnProfile(
        payload_dtype=torch.bfloat16,
        hidden_size=2880,
        layers=tuple(InstanceFfnLayerProfile(layer_id=entry.layer_id, kind=entry.kind) for entry in spec.layers),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=32,
        group_sum_complete_admitted=False,
    )
    owner = execution.required_execution_signatures(
        model_spec=spec, profile=profile, layer_ordinal=0, tp_size=2, tp_rank=0
    )
    peer = execution.required_execution_signatures(
        model_spec=spec, profile=profile, layer_ordinal=0, tp_size=2, tp_rank=1
    )
    assert len(owner) == len(peer)
    assert isinstance(owner[0], execution.MoeFfnExecutionSignature)
    assert isinstance(peer[0], execution.MoeFfnExecutionSignature)
    owner_bytes = execution.control_capture_probe_storage_bytes(owner[0])
    peer_bytes = execution.control_capture_probe_storage_bytes(peer[0])
    packed = 3 * experts * 2880 * 1440 * 17 // 32
    gate_bias = experts * 2 * 1440 * 2
    assert sum(peer_bytes) == packed + gate_bias
    assert sum(owner_bytes) == packed + gate_bias + experts * 2880 * 2 * 2 + experts * 2
    with pytest.raises(ValueError, match="32-value blocks"):
        execution.required_execution_signatures(model_spec=spec, profile=profile, layer_ordinal=0, tp_size=4, tp_rank=0)


@pytest.mark.parametrize(
    "patch",
    (
        {"quantization_config": {"quant_method": "nvfp4"}},
        {"torch_dtype": "float16"},
        {"hidden_size": 2881},
        {"swiglu_limit": 0},
    ),
)
def test_gpt_oss_rejects_unsupported_quantization_and_geometry(patch: dict[str, object]) -> None:
    config = {
        "model_type": "gpt_oss",
        "torch_dtype": "bfloat16",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 24,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "swiglu_limit": 7,
        "quantization_config": {"quant_method": "mxfp4"},
        **patch,
    }
    with pytest.raises(ValueError):
        GptOssAdapter.compile(model_id=TEST_MODEL_ID, model_config=architecture.FfnSourceConfig(config))
