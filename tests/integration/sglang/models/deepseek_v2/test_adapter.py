from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch
from sglang.srt.plugins.hook_registry import HookType
from torch import nn

import xpool.config as config_module
from tests.harness.sglang.deepseek import (
    pytest,
)
from tests.harness.sglang.fakes import runner_with_architecture
from xpool.integrations.sglang import topology as sglang_topology
from xpool.integrations.sglang.adapter import attach_model_binding
from xpool.integrations.sglang.models.deepseek_v2 import (
    DeepseekV2Adapter,
    XpoolDeepseekV2MLP,
    XpoolDeepseekV2MoE,
    filter_ffn_weights,
)
from xpool.integrations.sglang.topology import AtnKind, SglangModelMetadata


def test_deepseek_adapter_declares_its_sglang_hooks() -> None:
    hooks = {(spec.target, spec.kind): spec.handler for spec in DeepseekV2Adapter().hooks()}

    assert hooks[("sglang.srt.models.deepseek_v2.DeepseekV2MLP", HookType.REPLACE)] is XpoolDeepseekV2MLP
    assert hooks[("sglang.srt.models.deepseek_v2.DeepseekV2MoE", HookType.REPLACE)] is XpoolDeepseekV2MoE
    assert ("sglang.srt.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights", HookType.AROUND) in hooks


def test_deepseek_adapter_matches_only_deepseek_v2_architecture() -> None:
    adapter = DeepseekV2Adapter()

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

    kept = [name for name, tensor in filter_ffn_weights(weights)]

    assert kept == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    ]


def test_around_load_weights_filters_ffn_weights_before_original() -> None:
    """The AROUND hook must drop mlp weights, then hand the rest to SGLang verbatim."""

    from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

    from xpool.integrations.sglang.models.deepseek_v2 import around_load_weights

    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.empty(1)),
        ("model.layers.0.mlp.gate_proj.weight", torch.empty(1)),
        ("model.norm.weight", torch.empty(1)),
    ]
    seen: list[str] = []

    def original(
        model: DeepseekV2ForCausalLM,
        filtered: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        seen.extend(name for name, tensor in filtered)

    around_load_weights(original, cast(DeepseekV2ForCausalLM, object()), iter(weights))

    assert seen == ["model.layers.0.self_attn.q_proj.weight", "model.norm.weight"]


def test_deepseek_loaded_model_validation_requires_integer_layer_count() -> None:
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    setattr(runner.model_config.hf_config, "num_hidden_layers", None)
    runner.model = nn.Module()

    with pytest.raises(RuntimeError, match="integer num_hidden_layers"):
        DeepseekV2Adapter().validate_after_load(runner.as_model_runner())


def test_deepseek_model_binding_resolves_instance_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "DeepSeek-V2-Lite-Chat"
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

[devices]
atn_cuda_devices = [0]
ffn_cuda_devices = [1]

[[models]]
id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
path = "{model_path}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    monkeypatch.setattr(
        sglang_topology,
        "load_sglang_model_metadata",
        lambda config_path, *, model_id: SglangModelMetadata(
            family=model_id,
            hidden_size=2048,
            num_atn_heads=16,
            num_key_value_heads=2,
            atn_kind=AtnKind.GQA,
            physical_kv_lanes=2,
        ),
    )
    monkeypatch.setattr(config_module, "global_config", None)
    config_module.init_global_config()
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model_config.model_path = str(model_path)

    binding = attach_model_binding(runner.as_model_runner())

    assert binding.instance_id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
    assert runner.xpool_model_binding == binding
