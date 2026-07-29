from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from sglang.srt.layers import dp_attention
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV2MLP, DeepseekV2MoE
from torch import nn

import xpool.config
import xpool.ops
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.sglang.deepseek import (
    bound_shim,
    decode_forward_batch,
    deepseek_config,
    install_adapter_config,
)
from tests.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture
from xpool.abi import DpPaddingMode, FfnResultHandoff
from xpool.config import XpoolConfig
from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.adapter import XpoolModelBinding
from xpool.integrations.sglang.models.deepseek_v2 import DeepseekV2Adapter, XpoolDeepseekV2MLP, XpoolDeepseekV2MoE
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError, iter_ffn_shims
from xpool.transport import FfnRequestMetadata

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, install_adapter_config.__name__)


def test_ffn_shim_module_reports_identity() -> None:
    shim = FfnShimModule(
        layer_id=3,
        hidden_size=2048,
        layer_kind=FfnLayerKind.SPARSE,
    )

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


@pytest.mark.parametrize("prefix", ["model.decoder.mlp", "model.draft.layers.0.mlp"])
def test_deepseek_dense_shim_rejects_invalid_layer_prefix(prefix: str) -> None:
    with pytest.raises(ShimUnavailableError, match="cannot derive layer id"):
        XpoolDeepseekV2MLP(
            hidden_size=2048,
            intermediate_size=8192,
            hidden_act="silu",
            prefix=prefix,
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


def test_shim_forward_maps_sglang_reduce_scatter_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SGLang's request-time decision is the sole handoff policy source."""

    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        request_metadata: FfnRequestMetadata,
    ) -> torch.Tensor:
        assert global_num_tokens_gpu is None
        assert request_metadata.result_handoff is FfnResultHandoff.REDUCE_SCATTER_INPUT
        return hidden_states.clone()

    shim = bound_shim()
    hidden_states = torch.zeros((1, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(xpool.ops, "ffn_shim", fake_ffn_shim)

    output = shim(hidden_states, decode_forward_batch(), use_reduce_scatter=True)

    assert torch.equal(output, hidden_states)


def test_shim_forward_accepts_idle_forward_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        request_metadata: FfnRequestMetadata,
    ) -> torch.Tensor:
        assert request_metadata.forward_mode == 4
        assert request_metadata.layer_ordinal == 0
        return hidden_states.clone()

    shim = XpoolDeepseekV2MLP(
        hidden_size=2048,
        intermediate_size=8192,
        hidden_act="silu",
        prefix="model.layers.0.mlp",
    )
    shim.bind_runtime(
        layer_ordinal=0,
        model_architecture="DeepseekV2ForCausalLM",
    )
    hidden_states = torch.empty((0, 2048), dtype=torch.bfloat16)
    forward_batch = type("FakeForwardBatch", (), {"forward_mode": forward_batch_info.ForwardMode.IDLE})()
    monkeypatch.setattr(xpool.ops, "ffn_shim", fake_ffn_shim)

    assert torch.equal(shim(hidden_states, forward_batch), hidden_states)


def test_shim_forward_rejects_integer_forward_mode() -> None:
    shim = XpoolDeepseekV2MLP(
        hidden_size=2048,
        intermediate_size=8192,
        hidden_act="silu",
        prefix="model.layers.0.mlp",
    )
    shim.bind_runtime(
        layer_ordinal=0,
        model_architecture="DeepseekV2ForCausalLM",
    )
    hidden_states = torch.empty((0, 2048), dtype=torch.bfloat16)
    forward_batch = type(
        "FakeForwardBatch",
        (),
        {"forward_mode": int(forward_batch_info.ForwardMode.DECODE)},
    )()

    with pytest.raises(ShimUnavailableError, match="forward mode"):
        shim(hidden_states, forward_batch)


def test_shim_forward_preserves_native_runtime_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        request_metadata: FfnRequestMetadata,
    ) -> torch.Tensor:
        assert request_metadata.forward_mode == 2
        assert request_metadata.layer_ordinal == 0
        raise RuntimeError("native detail")

    shim = bound_shim()
    hidden_states = torch.empty((1, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(xpool.ops, "ffn_shim", fake_ffn_shim)

    with pytest.raises(RuntimeError, match="native detail"):
        shim(hidden_states, decode_forward_batch())


def test_shim_forward_passes_structured_native_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        request_metadata: FfnRequestMetadata,
    ) -> torch.Tensor:
        assert global_num_tokens_gpu is None
        assert request_metadata.layer_ordinal == 0
        assert request_metadata.forward_mode == 2
        return hidden_states + 1

    monkeypatch.setattr(xpool.config, "global_config", None)
    install_test_config(
        config=XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={"XPOOL_DEBUG_LOOPBACK_ENABLE": "1", "XPOOL_DEBUG_LOOPBACK_SITE": "instance"},
        )
    )
    shim = bound_shim(layer_id=4)
    hidden_states = torch.zeros((1, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(xpool.ops, "ffn_shim", fake_ffn_shim)

    output = shim(hidden_states, decode_forward_batch())

    assert torch.equal(output, hidden_states + 1)


@pytest.mark.parametrize(
    "padding_mode",
    [dp_attention.DpPaddingMode.MAX_LEN, dp_attention.DpPaddingMode.SUM_LEN],
)
def test_shim_forward_normalizes_single_rank_dp_metadata(
    monkeypatch: pytest.MonkeyPatch,
    padding_mode: dp_attention.DpPaddingMode,
) -> None:
    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        request_metadata: FfnRequestMetadata,
    ) -> torch.Tensor:
        assert global_num_tokens_gpu is None
        assert request_metadata.layer_ordinal == 0
        assert request_metadata.dp_padding_mode is DpPaddingMode.NONE
        return hidden_states.clone()

    hidden_states = torch.zeros((3, 2048), dtype=torch.bfloat16)
    forward_batch = type(
        "FakeForwardBatch",
        (),
        {
            "forward_mode": forward_batch_info.ForwardMode.DECODE,
            "dp_padding_mode": padding_mode,
            "global_num_tokens_gpu": torch.tensor([3], dtype=torch.int32),
        },
    )()
    monkeypatch.setattr(xpool.ops, "ffn_shim", fake_ffn_shim)

    output = bound_shim()(hidden_states, forward_batch)

    assert torch.equal(output, hidden_states)


@pytest.mark.parametrize(
    ("sglang_padding_mode", "request_padding_mode"),
    [
        (dp_attention.DpPaddingMode.MAX_LEN, DpPaddingMode.MAX_LEN),
        (dp_attention.DpPaddingMode.SUM_LEN, DpPaddingMode.SUM_LEN),
    ],
)
def test_shim_forward_publishes_attention_dp_token_counts(
    monkeypatch: pytest.MonkeyPatch,
    sglang_padding_mode: dp_attention.DpPaddingMode,
    request_padding_mode: DpPaddingMode,
) -> None:
    token_counts = torch.tensor([3, 2], dtype=torch.int32)

    def fake_ffn_shim(
        hidden_states: torch.Tensor,
        global_num_tokens_gpu: torch.Tensor | None,
        request_metadata: FfnRequestMetadata,
    ) -> torch.Tensor:
        assert global_num_tokens_gpu is token_counts
        assert request_metadata.dp_padding_mode is request_padding_mode
        return hidden_states.clone()

    forward_batch = type(
        "FakeForwardBatch",
        (),
        {
            "forward_mode": forward_batch_info.ForwardMode.DECODE,
            "dp_padding_mode": sglang_padding_mode,
            "global_num_tokens_gpu": token_counts,
        },
    )()
    hidden_states = torch.zeros((3, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(xpool.ops, "ffn_shim", fake_ffn_shim)

    output = bound_shim(atn_dp_size=2)(hidden_states, forward_batch)

    assert torch.equal(output, hidden_states)


def test_shim_forward_rejects_missing_attention_dp_token_counts() -> None:
    forward_batch = type(
        "FakeForwardBatch",
        (),
        {
            "forward_mode": forward_batch_info.ForwardMode.DECODE,
            "dp_padding_mode": dp_attention.DpPaddingMode.MAX_LEN,
            "global_num_tokens_gpu": None,
        },
    )()
    hidden_states = torch.zeros((3, 2048), dtype=torch.bfloat16)

    with pytest.raises(ShimUnavailableError, match="global_num_tokens_gpu"):
        bound_shim(atn_dp_size=2)(hidden_states, forward_batch)


def test_deepseek_loaded_model_validation_counts_xpool_shims() -> None:
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
                allow_reduce_scatter=True,
            ),
            FakeDecoderLayer(
                XpoolDeepseekV2MoE(
                    config=deepseek_config(),
                    layer_id=1,
                    prefix="model.layers.1.mlp",
                ),
                allow_reduce_scatter=True,
            ),
        ],
    )
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model = model

    DeepseekV2Adapter().validate_after_load(runner.as_model_runner())

    assert runner.xpool_ffn_shim_count == 2
    assert [shim.layer_id for shim in iter_ffn_shims(model)] == [0, 1]


def test_deepseek_loaded_model_validation_requires_shims() -> None:
    runner = runner_with_architecture("DeepseekV2ForCausalLM")
    runner.model = loaded_model(DeepseekV2ForCausalLM, deepseek_config(), [])

    with pytest.raises(RuntimeError, match="produced no FFN shim"):
        DeepseekV2Adapter().validate_after_load(runner.as_model_runner())


def test_bind_shim_runtime_binds_loaded_deepseek_shims() -> None:
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
        instance_id="test/model",
        model_path=Path("/models/test/model"),
        instance_index=2,
        worker_rank=3,
        cuda_device=6,
        worker_world_size=1,
        sglang_base_gpu_id=0,
        sglang_gpu_id_step=2,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )

    binding.bind_shim_runtime(runner.as_model_runner())

    assert [shim.layer_ordinal for shim in iter_ffn_shims(model)] == [0, 1]
    assert [shim.model_architecture for shim in iter_ffn_shims(model)] == [
        "DeepseekV2ForCausalLM",
        "DeepseekV2ForCausalLM",
    ]
