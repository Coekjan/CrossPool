from __future__ import annotations

from pathlib import Path

import pytest
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_exec, publish
from transformers import Qwen3Config

import xpool.integrations.sglang.hooks.lifecycle
from tests.harness.support.config import reset_global_config
from tests.harness.support.sglang.fakes import FakeModelRunner, ServerArgs, server_args
from tests.harness.support.sglang.plugin import FakeAdapter, reset_plugin_required_hook_targets
from tests.harness.support.sglang.runtime import published_sglang_config
from xpool.integrations.sglang.server_args import validate_sglang_server_args

pytestmark = pytest.mark.usefixtures(
    reset_global_config.__name__, reset_plugin_required_hook_targets.__name__, published_sglang_config.__name__
)


def test_model_runner_hook_rejects_server_args_before_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner(server_args=server_args(enable_two_batch_overlap=True))
    get_context().override("test", enable_two_batch_overlap=True)
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="Two-Batch Overlap"):
        xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
            (adapter,), original, runner.as_model_runner()
        )


def test_global_server_arg_gate_allows_default_sglang_features() -> None:
    validate_sglang_server_args()


def test_gate_reads_graph_modes_resolved_from_real_model_config(tmp_path: Path) -> None:
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=2,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    args = ServerArgs(
        model_path=str(tmp_path), device="cpu", cuda_graph_backend_decode="full", cuda_graph_backend_prefill="breakable"
    )
    publish(args, role="test")

    assert args.cuda_graph_config is None
    assert isinstance(get_exec().graph.cuda_graph_config, CudaGraphConfig)
    validate_sglang_server_args()


def test_global_server_arg_gate_allows_dp_atn_when_parallel_policy_matches() -> None:
    get_context().override("test", tp_size=2, dp_size=2, enable_dp_attention=True)
    validate_sglang_server_args()


def test_global_server_arg_gate_allows_grpc_beside_http() -> None:
    get_context().override("test", grpc_port=30001)
    validate_sglang_server_args()


@pytest.mark.parametrize(
    ("override", "label"),
    [
        ({"enable_two_batch_overlap": True}, "Two-Batch Overlap"),
        ({"enable_layernorm_sp": True}, "LayerNorm Sequence Parallelism"),
        ({"cpu_offload_gb": 1}, "SGLang CPU Offload"),
        ({"enable_unified_memory": True}, "Unified Memory KV Allocator"),
        ({"enable_page_major_kv_layout": True}, "Page-Major KV Layout"),
        ({"disable_radix_cache": True}, "Disabled Radix Cache"),
        ({"enable_lmcache": True}, "LMCache"),
        ({"enable_flexkv": True}, "FlexKV"),
        ({"radix_cache_backend": "custom"}, "Custom Radix Cache Backend"),
        ({"enable_lora": True}, "LoRA"),
        ({"enable_torch_compile": True}, "Torch Compile"),
        (
            {
                "cuda_graph_config": CudaGraphConfig(
                    decode=PhaseConfig(backend="breakable"),
                    prefill=PhaseConfig(backend="disabled"),
                )
            },
            "Decode CUDA Graph Backend",
        ),
        ({"attn_cp_size": 2}, "Attention Context Parallelism"),
        ({"enable_prefill_cp": True}, "Prefill Context Parallelism"),
        (
            {
                "cuda_graph_config": CudaGraphConfig(
                    decode=PhaseConfig(backend="disabled"),
                    prefill=PhaseConfig(backend="tc_piecewise"),
                )
            },
            "Prefill CUDA Graph Backend",
        ),
        ({"enable_waterfill": True}, "DeepEP Waterfill"),
        ({"speculative_algorithm": "EAGLE"}, "Speculative Decoding"),
        ({"speculative_moe_a2a_backend": "deepep"}, "Speculative MoE A2A Backend"),
        ({"startup_weight_load_mode": "overlap"}, "Startup Weight Loading"),
        ({"enable_mixed_chunk": True}, "Mixed Chunked Prefill"),
        ({"disaggregation_mode": "prefill"}, "PD Disaggregation"),
        ({"dllm_algorithm": "next_block"}, "Diffusion LLM"),
        ({"enable_pdmux": True}, "PD Multiplexing"),
        ({"grpc_mode": True}, "gRPC-only Serving"),
        ({"smg_grpc_mode": True}, "gRPC-only Serving"),
        ({"ssl_keyfile": "/tmp/key.pem"}, "TLS Serving"),
        ({"ssl_certfile": "/tmp/cert.pem"}, "TLS Serving"),
        ({"ssl_ca_certs": "/tmp/ca.pem"}, "TLS Serving"),
        ({"ssl_keyfile_password": "secret"}, "TLS Serving"),
        ({"enable_ssl_refresh": True}, "TLS Serving"),
    ],
)
def test_global_server_arg_gate_rejects_unsupported_features(
    override: dict[str, object],
    label: str,
) -> None:
    get_context().override("test", **override)

    with pytest.raises(RuntimeError, match=label):
        validate_sglang_server_args()


@pytest.mark.parametrize(
    ("name", "value", "label"),
    [
        ("SGLANG_USE_HND_KVCACHE", "1", "HND KV Layout"),
        ("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE", "1", "C\\+\\+ Radix Tree"),
        ("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", "rust", "Unified Radix TreeCore Backend"),
    ],
)
def test_global_server_arg_gate_rejects_incompatible_kv_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    label: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=label):
        validate_sglang_server_args()
