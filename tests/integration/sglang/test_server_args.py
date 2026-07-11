from __future__ import annotations

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs

from tests.harness.sglang.fakes import FakeModelConfig, FakeModelRunner, server_args
from tests.harness.sglang.plugin import (
    FakeAdapter,
    ModelRunner,
    Path,
    configure_xpool_model,
    pytest,
    sglang_plugin,
)
from xpool.integrations.sglang.server_args import SGLANG_SERVER_ARG_RULES, validate_sglang_server_args

UNSUPPORTED_FORWARD_MODE_RULE_LABEL_BY_NAME = {
    "MIXED": "Mixed Chunked Prefill",
    "TARGET_VERIFY": "Speculative Decoding",
    "DRAFT_EXTEND": "Speculative Decoding",
    "DRAFT_EXTEND_V2": "Speculative Decoding",
    "PREBUILT": "PD Disaggregation",
    "SPLIT_PREFILL": "PD Multiplexing",
    "DLLM_EXTEND": "Diffusion LLM",
}


def test_every_unsupported_sglang_forward_mode_has_server_arg_gate_label() -> None:
    """Fail when pinned SGLang adds a forward mode without an xpool gate."""

    labels = {rule.label for rule in SGLANG_SERVER_ARG_RULES}
    unsupported_modes = set(ForwardMode.__members__) - {"DECODE", "EXTEND", "IDLE"}

    assert set(UNSUPPORTED_FORWARD_MODE_RULE_LABEL_BY_NAME) == unsupported_modes
    assert set(UNSUPPORTED_FORWARD_MODE_RULE_LABEL_BY_NAME.values()).issubset(labels)


def test_model_runner_hook_rejects_server_args_before_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner(server_args=server_args(enable_two_batch_overlap=True))
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="Two-Batch Overlap"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_requires_server_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner(
        model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")),
        server_args=None,
    )
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match=r"requires ModelRunner\.server_args"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_global_server_arg_gate_rejects_unsupported_sglang_features() -> None:
    args = server_args(enable_two_batch_overlap=True, cpu_offload_gb=1)

    with pytest.raises(RuntimeError, match="Two-Batch Overlap"):
        validate_sglang_server_args(args)


def test_global_server_arg_gate_allows_default_sglang_features() -> None:
    validate_sglang_server_args(server_args())
    validate_sglang_server_args(server_args(piecewise_cuda_graph_compiler="eager"))


def test_global_server_arg_gate_allows_resolved_sglang_defaults() -> None:
    args = ServerArgs(model_path="dummy")

    validate_sglang_server_args(args)


def test_global_server_arg_gate_allows_dp_atn_when_parallel_policy_matches() -> None:
    args = server_args(tp_size=2, dp_size=2, enable_dp_attention=True)

    validate_sglang_server_args(args)


def test_global_server_arg_gate_rejects_lora_when_enabled() -> None:
    args = server_args(enable_lora=True)

    with pytest.raises(RuntimeError, match="LoRA"):
        validate_sglang_server_args(args)


def test_global_server_arg_gate_rejects_compile_paths() -> None:
    with pytest.raises(RuntimeError, match="Torch Compile"):
        validate_sglang_server_args(server_args(enable_torch_compile=True))
    with pytest.raises(RuntimeError, match="Piecewise CUDA Graph Compiler"):
        validate_sglang_server_args(server_args(piecewise_cuda_graph_compiler="inductor"))


@pytest.mark.parametrize(
    ("override", "label"),
    [
        ({"enable_mixed_chunk": True}, "Mixed Chunked Prefill"),
        ({"disaggregation_mode": "prefill"}, "PD Disaggregation"),
        ({"dllm_algorithm": "next_block"}, "Diffusion LLM"),
        ({"enable_pdmux": True}, "PD Multiplexing"),
    ],
)
def test_global_server_arg_gate_rejects_composite_forward_modes(
    override: dict[str, object],
    label: str,
) -> None:
    args = server_args(**override)

    with pytest.raises(RuntimeError, match=label):
        validate_sglang_server_args(args)
