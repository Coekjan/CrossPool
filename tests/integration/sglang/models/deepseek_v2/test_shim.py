from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.srt.layers.dp_attention import DpPaddingMode as SglangDpPaddingMode
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from torch import nn

import xpool.config as config_module
import xpool.integrations.sglang.shim as shim_module
from tests.harness.sglang.deepseek import (
    FfnLayerKind,
    FfnShimModule,
    SglangForwardMode,
    XpoolConfig,
    bound_shim,
    decode_forward_batch,
    deepseek_config,
    pytest,
)
from tests.harness.sglang.fakes import runner_with_architecture
from xpool.config import init_global_config
from xpool.integrations.sglang.adapter import XpoolModelBinding, inject_shim_identity
from xpool.integrations.sglang.models.deepseek_v2 import DeepseekV2Adapter, XpoolDeepseekV2MLP, XpoolDeepseekV2MoE
from xpool.integrations.sglang.shim import ShimUnavailableError, iter_ffn_shims


def test_ffn_shim_module_has_no_parameters() -> None:
    shim = FfnShimModule(
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


def test_deepseek_dense_shim_rejects_auxiliary_layer_prefix() -> None:
    with pytest.raises(ShimUnavailableError, match="cannot derive layer id"):
        XpoolDeepseekV2MLP(
            hidden_size=2048,
            intermediate_size=8192,
            hidden_act="silu",
            prefix="model.draft.layers.0.mlp",
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


def test_shim_forward_rejects_unsupported_allreduce_fusion_path() -> None:
    """Shim must fail closed on SGLang FFN fusion paths the native ABI does not cover."""

    shim = XpoolDeepseekV2MLP(
        hidden_size=2048,
        intermediate_size=8192,
        hidden_act="silu",
        prefix="model.layers.0.mlp",
    )
    hidden_states = torch.zeros((1, 2048), dtype=torch.bfloat16)

    with pytest.raises(ShimUnavailableError, match="all-reduce fusion"):
        shim(hidden_states, should_allreduce_fusion=True)


def test_shim_forward_rejects_reduce_scatter_until_transport_supports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shim must fail closed on SGLang reduce-scatter output requests."""

    shim = bound_shim()
    hidden_states = torch.zeros((1, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(shim_module, "validate_hidden_states", lambda *args, **kwargs: None)

    with pytest.raises(ShimUnavailableError, match="reduce-scatter"):
        shim(hidden_states, decode_forward_batch(), use_reduce_scatter=True)


def test_shim_forward_accepts_idle_forward_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        instance_index: int,
        sglang_rank: int,
        layer_id: int,
        forward_mode: int,
        *metadata: object,
    ) -> torch.Tensor:
        assert forward_mode == 4
        return hidden_states.clone()

    shim = XpoolDeepseekV2MLP(
        hidden_size=2048,
        intermediate_size=8192,
        hidden_act="silu",
        prefix="model.layers.0.mlp",
    )
    shim.bind_identity(instance_index=0, sglang_rank=0, model_architecture="DeepseekV2ForCausalLM")
    hidden_states = torch.empty((0, 2048), dtype=torch.bfloat16)
    forward_batch = type("FakeForwardBatch", (), {"forward_mode": SglangForwardMode.IDLE})()
    monkeypatch.setattr(
        torch.ops, "xpool", SimpleNamespace(instance=SimpleNamespace(ffn_shim=fake_ffn_shim)), raising=False
    )

    assert torch.equal(shim(hidden_states, forward_batch), hidden_states)


def test_shim_forward_rejects_integer_forward_mode() -> None:
    shim = XpoolDeepseekV2MLP(
        hidden_size=2048,
        intermediate_size=8192,
        hidden_act="silu",
        prefix="model.layers.0.mlp",
    )
    shim.bind_identity(instance_index=0, sglang_rank=0, model_architecture="DeepseekV2ForCausalLM")
    hidden_states = torch.empty((0, 2048), dtype=torch.bfloat16)
    forward_batch = type("FakeForwardBatch", (), {"forward_mode": int(SglangForwardMode.DECODE)})()

    with pytest.raises(ShimUnavailableError, match="forward mode"):
        shim(hidden_states, forward_batch)


def test_shim_forward_reports_missing_native_op(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = bound_shim()
    hidden_states = torch.empty((1, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(torch.ops, "xpool", SimpleNamespace(instance=SimpleNamespace()), raising=False)
    monkeypatch.setattr(shim_module, "validate_hidden_states", lambda *args, **kwargs: None)

    with pytest.raises(AttributeError, match="ffn_shim"):
        shim(hidden_states, decode_forward_batch())


def test_shim_forward_preserves_native_runtime_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        instance_index: int,
        sglang_rank: int,
        layer_id: int,
        forward_mode: int,
        *metadata: object,
    ) -> torch.Tensor:
        assert forward_mode == 2
        raise RuntimeError("native detail")

    shim = bound_shim()
    hidden_states = torch.empty((1, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(
        torch.ops, "xpool", SimpleNamespace(instance=SimpleNamespace(ffn_shim=fake_ffn_shim)), raising=False
    )
    monkeypatch.setattr(shim_module, "validate_hidden_states", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="native detail"):
        shim(hidden_states, decode_forward_batch())


def test_shim_forward_passes_structured_native_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        instance_index: int,
        sglang_rank: int,
        layer_id: int,
        forward_mode: int,
        *metadata: object,
    ) -> torch.Tensor:
        assert global_num_tokens_gpu is None
        assert instance_index == 2
        assert sglang_rank == 0
        assert layer_id == 4
        assert forward_mode == 2
        assert len(metadata) == 7
        return hidden_states + 1

    monkeypatch.setattr(config_module, "global_config", None)
    init_global_config(
        config=XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={"XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "1"},
        )
    )
    shim = bound_shim(layer_id=4, instance_index=2)
    hidden_states = torch.zeros((1, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(
        torch.ops, "xpool", SimpleNamespace(instance=SimpleNamespace(ffn_shim=fake_ffn_shim)), raising=False
    )
    monkeypatch.setattr(shim_module, "validate_hidden_states", lambda *args, **kwargs: None)

    output = shim(hidden_states, decode_forward_batch())

    assert torch.equal(output, hidden_states + 1)


@pytest.mark.parametrize("padding_mode", [SglangDpPaddingMode.MAX_LEN, SglangDpPaddingMode.SUM_LEN])
def test_shim_forward_normalizes_single_rank_dp_metadata(
    monkeypatch: pytest.MonkeyPatch,
    padding_mode: SglangDpPaddingMode,
) -> None:
    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        instance_index: int,
        sglang_rank: int,
        layer_id: int,
        forward_mode: int,
        collective_policy: int,
        dp_padding_mode: int,
        global_dp_buffer_len: int,
        *topology: object,
    ) -> torch.Tensor:
        assert global_num_tokens_gpu is None
        assert dp_padding_mode == 0
        assert global_dp_buffer_len == hidden_states.shape[0]
        return hidden_states.clone()

    hidden_states = torch.zeros((3, 2048), dtype=torch.bfloat16)
    forward_batch = type(
        "FakeForwardBatch",
        (),
        {
            "forward_mode": SglangForwardMode.DECODE,
            "dp_padding_mode": padding_mode,
            "global_dp_buffer_len": 99,
            "global_num_tokens_gpu": torch.tensor([3], dtype=torch.int32),
        },
    )()
    monkeypatch.setattr(
        torch.ops, "xpool", SimpleNamespace(instance=SimpleNamespace(ffn_shim=fake_ffn_shim)), raising=False
    )
    monkeypatch.setattr(shim_module, "validate_hidden_states", lambda *args, **kwargs: None)

    output = bound_shim()(hidden_states, forward_batch)

    assert torch.equal(output, hidden_states)


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

    DeepseekV2Adapter().validate_after_load(runner.as_model_runner())

    assert runner.xpool_ffn_shim_count == 2
    assert [shim.layer_id for shim in iter_ffn_shims(model)] == [0, 1]


def test_deepseek_loaded_model_validation_requires_shims() -> None:
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model = nn.Module()

    with pytest.raises(RuntimeError, match="produced no FFN shim"):
        DeepseekV2Adapter().validate_after_load(runner.as_model_runner())


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
        instance_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        model_path=Path("/models/deepseek-ai/DeepSeek-V2-Lite-Chat"),
        instance_index=2,
        sglang_rank=3,
        cuda_device=6,
        sglang_tp_size=1,
        sglang_dp_size=1,
        sglang_base_gpu_id=0,
        sglang_gpu_id_step=2,
        enable_dp_atn=False,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )

    inject_shim_identity(runner.as_model_runner(), binding)

    assert [shim.instance_index for shim in iter_ffn_shims(model)] == [2, 2]
    assert [shim.sglang_rank for shim in iter_ffn_shims(model)] == [3, 3]
    assert [shim.model_architecture for shim in iter_ffn_shims(model)] == [
        "DeepseekV2ForCausalLM",
        "DeepseekV2ForCausalLM",
    ]
