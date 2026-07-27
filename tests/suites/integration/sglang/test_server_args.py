from __future__ import annotations

import pytest
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs

import xpool.integrations.sglang.plugin
from tests.harness.sglang.fakes import FakeModelRunner, server_args
from tests.harness.sglang.plugin import FakeAdapter, reset_plugin_required_hook_targets
from xpool.integrations.sglang.server_args import validate_sglang_server_args

pytestmark = pytest.mark.usefixtures(reset_plugin_required_hook_targets.__name__)


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
    validate_sglang_server_args(server_args(piecewise_cuda_graph_compiler="eager"))


def test_global_server_arg_gate_allows_resolved_sglang_defaults() -> None:
    args = ServerArgs(model_path="dummy")

    validate_sglang_server_args(args)


def test_global_server_arg_gate_allows_dp_atn_when_parallel_policy_matches() -> None:
    args = server_args(tp_size=2, dp_size=2, enable_dp_attention=True)

    validate_sglang_server_args(args)


@pytest.mark.parametrize(
    ("override", "label"),
    [
        ({"enable_two_batch_overlap": True}, "Two-Batch Overlap"),
        ({"cpu_offload_gb": 1}, "SGLang CPU Offload"),
        ({"enable_lora": True}, "LoRA"),
        ({"enable_torch_compile": True}, "Torch Compile"),
        ({"piecewise_cuda_graph_compiler": "inductor"}, "Piecewise CUDA Graph Compiler"),
        ({"attn_cp_size": 2}, "Attention Context Parallelism"),
        ({"enforce_piecewise_cuda_graph": True}, "Piecewise CUDA Graph Enforcement"),
        ({"enable_mixed_chunk": True}, "Mixed Chunked Prefill"),
        ({"disaggregation_mode": "prefill"}, "PD Disaggregation"),
        ({"dllm_algorithm": "next_block"}, "Diffusion LLM"),
        ({"enable_pdmux": True}, "PD Multiplexing"),
    ],
)
def test_global_server_arg_gate_rejects_unsupported_features(
    override: dict[str, object],
    label: str,
) -> None:
    args = server_args(**override)

    with pytest.raises(RuntimeError, match=label):
        validate_sglang_server_args(args)
