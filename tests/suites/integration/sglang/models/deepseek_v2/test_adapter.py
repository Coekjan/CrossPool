from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.plugins.hook_registry import HookType

import xpool.config
from tests.harness.support.config import reset_global_config
from tests.harness.support.sglang.deepseek import deepseek_config, install_adapter_config
from tests.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture
from tests.harness.support.sglang.runtime import published_sglang_config
from xpool.integrations.sglang.adapter import (
    SglangInstanceRankBinding,
    SglangInstanceRankRuntime,
    filter_decoder_ffn_weights,
)
from xpool.integrations.sglang.models.deepseek_v2 import (
    DeepseekV2ShimAdapter,
    XpoolDeepseekV2MLP,
    XpoolDeepseekV2MoE,
)
from xpool.integrations.sglang.topology import SglangAttentionKind, SglangModelMetadata

pytestmark = pytest.mark.usefixtures(
    reset_global_config.__name__, install_adapter_config.__name__, published_sglang_config.__name__
)


def test_deepseek_adapter_declares_its_sglang_hooks() -> None:
    hooks = {(spec.target, spec.kind): spec.handler for spec in DeepseekV2ShimAdapter().hooks()}

    assert hooks[("sglang.srt.models.deepseek_v2.DeepseekV2MLP", HookType.REPLACE)] is XpoolDeepseekV2MLP
    assert hooks[("sglang.srt.models.deepseek_v2.DeepseekV2MoE", HookType.REPLACE)] is XpoolDeepseekV2MoE
    assert ("sglang.srt.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights", HookType.AROUND) in hooks


def test_deepseek_adapter_matches_only_deepseek_v2_architecture() -> None:
    adapter = DeepseekV2ShimAdapter()

    assert adapter.matches(runner_with_architecture("DeepseekV2ForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("DeepseekV3ForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Qwen2ForCausalLM").as_model_runner())


def test_deepseek_ffn_weight_filter_skips_mlp_subtree() -> None:
    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.empty(1)),
        ("model.layers.0.mlp.gate_proj.weight", torch.empty(1)),
        ("model.layers.1.mlp.experts.0.down_proj.weight", torch.empty(1)),
        ("model.layers.2.mlp.shared_experts.gate_up_proj.weight", torch.empty(1)),
        ("model.norm.weight", torch.empty(1)),
    ]

    kept = [name for name, tensor in filter_decoder_ffn_weights(weights)]

    assert kept == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    ]


def test_deepseek_loaded_model_validation_requires_integer_layer_count() -> None:
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    config = deepseek_config()
    setattr(config, "num_hidden_layers", None)
    runner.model = loaded_model(DeepseekV2ForCausalLM, config, [])

    with pytest.raises(RuntimeError, match="integer num_hidden_layers"):
        DeepseekV2ShimAdapter().validate_after_load(runner.as_model_runner())


def test_deepseek_loaded_model_rejects_non_full_mlp_boundary() -> None:
    model = loaded_model(
        DeepseekV2ForCausalLM,
        deepseek_config(),
        [
            FakeDecoderLayer(
                XpoolDeepseekV2MLP(
                    hidden_size=2048,
                    intermediate_size=8192,
                    hidden_act="silu",
                    prefix="model.layers.0.mlp",
                ),
                mlp_mode=ScatterMode.TP_ATTN_FULL,
                allow_reduce_scatter=True,
            ),
            FakeDecoderLayer(
                XpoolDeepseekV2MoE(
                    config=deepseek_config(),
                    layer_id=1,
                    prefix="model.layers.1.mlp",
                ),
                mlp_mode=ScatterMode.TP_ATTN_FULL,
                allow_reduce_scatter=True,
            ),
        ],
    )
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model = model

    with pytest.raises(RuntimeError, match=r"requires ScatterMode\.FULL"):
        DeepseekV2ShimAdapter().validate_after_load(runner.as_model_runner())


def test_deepseek_model_binding_resolves_instance_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "synthetic-deepseek-v2"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
{
  "model_type": "deepseek_v2",
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "intermediate_size": 8192,
  "moe_intermediate_size": 8192
}
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "xpool.toml"
    config_path.write_text(
        f"""
[daemon]
host = "127.0.0.1"
port = 9810

[scheduler]
atn_concurrency = 1
ffn_concurrency = 1

[atn]
devices = [0]

[ffn]
devices = [1]

[[models]]
id = "test/deepseek-v2"
path = "{model_path}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    monkeypatch.setattr(
        SglangModelMetadata,
        "load",
        lambda config_path, *, model_id: SglangModelMetadata(
            model_id=model_id,
            family=model_id,
            hidden_size=2048,
            num_atn_heads=16,
            num_key_value_heads=2,
            atn_kind=SglangAttentionKind.GQA,
            raw_config_path=config_path,
        ),
    )
    monkeypatch.setattr(xpool.config, "global_config", None)
    xpool.config.init_global_config()
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model_config.model_path = str(model_path)

    binding = SglangInstanceRankBinding.resolve(
        runner.as_model_runner(),
        supports_dp_attention=True,
    )
    SglangInstanceRankRuntime.attach(runner.as_model_runner(), binding)

    assert binding.instance_id == "test/deepseek-v2"
    assert runner.xpool_runtime is not None
    assert runner.xpool_runtime.binding == binding
