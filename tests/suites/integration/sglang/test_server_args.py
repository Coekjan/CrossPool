from __future__ import annotations

import pytest
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.model_runner import ModelRunner

import xpool.integrations.sglang.plugin
from tests.harness.support.config import reset_global_config
from tests.harness.support.sglang.fakes import FakeModelRunner, server_args
from tests.harness.support.sglang.plugin import FakeAdapter, reset_plugin_required_hook_targets
from xpool.integrations.sglang.server_args import validate_sglang_server_args

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, reset_plugin_required_hook_targets.__name__)


def test_model_runner_hook_rejects_server_args_before_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner(server_args=server_args(enable_two_batch_overlap=True))
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="Two-Batch Overlap"):
        xpool.integrations.sglang.plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_global_server_arg_gate_allows_default_sglang_features() -> None:
    validate_sglang_server_args(server_args())


def test_global_server_arg_gate_allows_dp_atn_when_parallel_policy_matches() -> None:
    args = server_args(tp_size=2, dp_size=2, enable_dp_attention=True)

    validate_sglang_server_args(args)


def test_global_server_arg_gate_allows_grpc_beside_http() -> None:
    validate_sglang_server_args(server_args(grpc_port=30001))


@pytest.mark.parametrize(
    ("override", "label"),
    [
        ({"enable_two_batch_overlap": True}, "Two-Batch Overlap"),
        ({"cpu_offload_gb": 1}, "SGLang CPU Offload"),
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
    args = server_args(**override)

    with pytest.raises(RuntimeError, match=label):
        validate_sglang_server_args(args)
