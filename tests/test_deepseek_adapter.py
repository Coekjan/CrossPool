from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.server_args import ServerArgs
from sglang_fakes import server_args as make_server_args
from torch import nn
from transformers import PretrainedConfig

from xpool.integrations.sglang.adapter import XpoolModelBinding, bind_model_instance, inject_shim_identity
from xpool.integrations.sglang.models.deepseek_v2 import (
    DeepseekV2Adapter,
    XpoolDeepseekV2MLP,
    XpoolDeepseekV2MoE,
    filter_ffn_weights,
)
from xpool.integrations.sglang.shim import FfnLayerKind, FfnShimModule, ShimUnavailableError, iter_ffn_shims


def test_ffn_shim_module_has_no_parameters() -> None:
    shim = FfnShimModule(
        model_architecture="DeepseekV2ForCausalLM",
        layer_id=3,
        hidden_size=2048,
        layer_kind=FfnLayerKind.SPARSE,
    )

    assert list(shim.parameters()) == []
    assert "layer_id=3" in shim.extra_repr()


def test_deepseek_dense_shim_preserves_mlp_isinstance_only() -> None:
    dense = XpoolDeepseekV2MLP(
        hidden_size=2048,
        intermediate_size=8192,
        hidden_act="silu",
        prefix="model.layers.3.mlp",
    )

    assert isinstance(dense, DeepseekV2MLP)
    assert not isinstance(dense, DeepseekV2MoE)
    assert list(dense.parameters()) == []
    assert dense.layer_id == 3
    assert dense.layer_kind is FfnLayerKind.DENSE


def test_deepseek_dense_shim_requires_layer_prefix() -> None:
    with pytest.raises(ShimUnavailableError, match="cannot derive layer id"):
        XpoolDeepseekV2MLP(
            hidden_size=2048,
            intermediate_size=8192,
            hidden_act="silu",
            prefix="model.decoder.mlp",
        )


def test_deepseek_moe_shim_preserves_moe_isinstance_and_minimal_attrs() -> None:
    moe = XpoolDeepseekV2MoE(
        config=deepseek_config(),
        layer_id=4,
        prefix="model.layers.4.mlp",
    )

    assert isinstance(moe, DeepseekV2MoE)
    assert not isinstance(moe, DeepseekV2MLP)
    assert list(moe.parameters()) == []
    assert moe.layer_id == 4
    assert moe.layer_kind is FfnLayerKind.SPARSE
    assert moe.experts.moe_runner_config.inplace is True
    assert moe.get_moe_weights() == []


def test_shim_forward_accepts_sglang_decoder_layer_call_contract() -> None:
    """Shim forward() must accept the positional args the decoder layer passes.

    ``DeepseekV2DecoderLayer.forward`` calls its FFN module as
    ``self.mlp(hidden_states, forward_batch, should_allreduce_fusion,
    use_reduce_scatter, gemm_output_zero_allocator)``: five positional args. The real
    ``DeepseekV2MoE.forward`` also declares trailing ``input_ids``/``input_ids_global``
    kwargs, but those are only consumed by internal MoE sub-paths that the shim replaces
    wholesale, so the decoder layer never passes them. Binding the decoder-layer call to
    the shim signature guards the real integration contract against SGLang drift without
    needing a GPU; a full forward-signature equality check would be too strict.
    """

    sentinel = object()
    decoder_call_args = (sentinel, sentinel, False, False, None)

    for shim_class in (XpoolDeepseekV2MLP, XpoolDeepseekV2MoE):
        signature = inspect.signature(shim_class.forward)
        bound = signature.bind(sentinel, *decoder_call_args)  # sentinel as ``self``
        bound.apply_defaults()
        assert tuple(bound.arguments)[1:] == (
            "hidden_states",
            "forward_batch",
            "should_allreduce_fusion",
            "use_reduce_scatter",
            "gemm_output_zero_allocator",
        )


def test_shim_forward_rejects_unsupported_sglang_fusion_paths() -> None:
    """Shim must fail closed on SGLang FFN fusion paths the native ABI does not cover."""

    shim = XpoolDeepseekV2MLP(
        hidden_size=2048,
        intermediate_size=8192,
        hidden_act="silu",
        prefix="model.layers.0.mlp",
    )
    hidden_states = torch.empty((1, 2048), dtype=torch.bfloat16)

    with pytest.raises(ShimUnavailableError, match="all-reduce fusion"):
        shim(hidden_states, should_allreduce_fusion=True)
    with pytest.raises(ShimUnavailableError, match="reduce-scatter"):
        shim(hidden_states, use_reduce_scatter=True)


def test_deepseek_adapter_declares_its_sglang_hooks() -> None:
    hooks = {(spec.target, spec.kind): spec.handler for spec in DeepseekV2Adapter().hooks()}

    assert hooks[("sglang.srt.models.deepseek_v2.DeepseekV2MLP", HookType.REPLACE)] is XpoolDeepseekV2MLP
    assert hooks[("sglang.srt.models.deepseek_v2.DeepseekV2MoE", HookType.REPLACE)] is XpoolDeepseekV2MoE
    assert ("sglang.srt.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights", HookType.AROUND) in hooks


def test_deepseek_adapter_matches_only_deepseek_v2_architecture() -> None:
    adapter = DeepseekV2Adapter()

    assert adapter.matches(as_model_runner(runner_with_architecture("DeepseekV2ForCausalLM")))
    assert not adapter.matches(as_model_runner(runner_with_architecture("DeepseekV3ForCausalLM")))
    assert not adapter.matches(as_model_runner(runner_with_architecture("Qwen2ForCausalLM")))


def test_deepseek_ffn_weight_filter_skips_mlp_subtree() -> None:
    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.empty(1)),
        ("model.layers.0.mlp.gate_proj.weight", torch.empty(1)),
        ("model.layers.1.mlp.experts.0.down_proj.weight", torch.empty(1)),
        ("model.layers.2.mlp.shared_experts.gate_up_proj.weight", torch.empty(1)),
        ("model.norm.weight", torch.empty(1)),
    ]

    kept = [name for name, _tensor in filter_ffn_weights(weights)]

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
        _model: DeepseekV2ForCausalLM,
        filtered: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        seen.extend(name for name, _tensor in filtered)

    around_load_weights(original, cast(DeepseekV2ForCausalLM, object()), iter(weights))

    assert seen == ["model.layers.0.self_attn.q_proj.weight", "model.norm.weight"]


def test_deepseek_loaded_model_validation_counts_xpool_shims() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList(
        [
            XpoolDeepseekV2MLP(
                hidden_size=2048,
                intermediate_size=8192,
                hidden_act="silu",
                prefix="model.layers.0.mlp",
            ),
            XpoolDeepseekV2MoE(
                config=deepseek_config(),
                layer_id=1,
                prefix="model.layers.1.mlp",
            ),
        ]
    )
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model = model

    DeepseekV2Adapter().validate_after_load(as_model_runner(runner))

    assert runner.xpool_ffn_shim_count == 2
    assert [shim.layer_id for shim in iter_ffn_shims(model)] == [0, 1]


def test_deepseek_loaded_model_validation_requires_shims() -> None:
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model = nn.Module()

    with pytest.raises(RuntimeError, match="produced no FFN shim"):
        DeepseekV2Adapter().validate_after_load(as_model_runner(runner))


def test_deepseek_loaded_model_validation_requires_integer_layer_count() -> None:
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    setattr(runner.model_config.hf_config, "num_hidden_layers", None)
    runner.model = nn.Module()

    with pytest.raises(RuntimeError, match="integer num_hidden_layers"):
        DeepseekV2Adapter().validate_after_load(as_model_runner(runner))


def test_deepseek_model_binding_resolves_instance_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "DeepSeek-V2-Lite-Chat"
    model_path.mkdir()
    config_path = tmp_path / "xpool.toml"
    config_path.write_text(
        f"""
[daemon]
host = "127.0.0.1"
port = 9810

[scheduler]
attention_concurrency = 1
transport_concurrency = 1

[devices]
attention_cuda_devices = [0]
ffn_cuda_devices = [1]

[[models]]
id = "deepseek-v2-lite-chat"
path = "{model_path}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model_config.model_path = str(model_path)

    binding = bind_model_instance(as_model_runner(runner))

    assert binding.instance_id == "deepseek-v2-lite-chat"
    assert runner.xpool_model_binding == binding


def test_inject_shim_identity_binds_loaded_deepseek_shims() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList(
        [
            XpoolDeepseekV2MLP(
                hidden_size=2048,
                intermediate_size=8192,
                hidden_act="silu",
                prefix="model.layers.0.mlp",
            ),
            XpoolDeepseekV2MoE(
                config=deepseek_config(),
                layer_id=1,
                prefix="model.layers.1.mlp",
            ),
        ]
    )
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model = model
    binding = XpoolModelBinding(
        instance_id="deepseek-v2-lite-chat",
        model_path=Path("/models/deepseek-v2-lite-chat"),
        instance_index=2,
        model_index=3,
        sglang_tp_size=1,
        sglang_dp_size=1,
    )

    inject_shim_identity(as_model_runner(runner), binding)

    assert [(shim.instance_index, shim.model_index) for shim in iter_ffn_shims(model)] == [(2, 3), (2, 3)]


def runner_with_architecture(
    architecture: str,
    *,
    server_args: ServerArgs | None = None,
) -> "_FakeModelRunner":
    hf_config = PretrainedConfig(architectures=[architecture])
    setattr(hf_config, "num_hidden_layers", 2)
    return _FakeModelRunner(
        model_config=_FakeModelConfig(hf_config=hf_config),
        server_args=server_args or make_server_args(),
    )


def as_model_runner(runner: "_FakeModelRunner") -> ModelRunner:
    return cast(ModelRunner, runner)


def deepseek_config() -> PretrainedConfig:
    config = PretrainedConfig(architectures=["DeepseekV2ForCausalLM"])
    setattr(config, "num_hidden_layers", 2)
    setattr(config, "hidden_size", 2048)
    setattr(config, "hidden_act", "silu")
    setattr(config, "n_shared_experts", 1)
    return config


@dataclass(slots=True)
class _FakeModelConfig:
    hf_config: PretrainedConfig | None
    model_path: str = "/tmp/xpool/deepseek-v2-lite-chat"


@dataclass
class _FakeModelRunner:
    model_config: _FakeModelConfig
    server_args: ServerArgs
    model: nn.Module | None = None
    xpool_ffn_shim_count: int = 0
    xpool_model_binding: XpoolModelBinding | None = None
