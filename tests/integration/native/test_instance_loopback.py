from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode as SglangForwardMode

import xpool.config as config_module
from tests.harness.native.loopback import expected_loopback_rotation, run_isolated_native_case
from xpool.abi import DebugLoopbackSite, DebugOptions, RuntimeRole
from xpool.config import XpoolConfig, init_global_config
from xpool.integrations.sglang.shim import FfnLayerKind, FfnShimModule

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.timeout(180),
]


def install_instance_loopback_config() -> None:
    """Initialize an isolated process for instance-loopback execution."""

    config_module.global_config = None
    torch.ops.xpool.init(
        torch.cuda.current_device(),
        int(RuntimeRole.INSTANCE),
        DebugOptions.create(loopback_site=DebugLoopbackSite.INSTANCE).raw,
    )
    init_global_config(
        config=XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={"XPOOL_DEBUG_LOOPBACK_ENABLE": "1", "XPOOL_DEBUG_LOOPBACK_SITE": "instance"},
        )
    )


def isolated_case_instance_loopback_eager_rotates_hidden_pairs(dtype_name: str) -> None:
    install_instance_loopback_config()
    dtype = getattr(torch, dtype_name)
    shim = loopback_shim(hidden_size=4)
    hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]],
        device="cuda",
        dtype=dtype,
    )

    output = shim(hidden_states, forward_batch())

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device
    assert torch.allclose(output.float(), expected_loopback_rotation(hidden_states).float(), atol=2e-2, rtol=2e-2)


def isolated_case_instance_loopback_rejects_odd_hidden_size() -> None:
    install_instance_loopback_config()
    shim = loopback_shim(hidden_size=3)
    hidden_states = torch.ones((2, 3), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="even hidden size"):
        shim(hidden_states, forward_batch())


def isolated_case_instance_loopback_cuda_graph_replay_rotates_updated_inputs() -> None:
    install_instance_loopback_config()
    shim = loopback_shim(hidden_size=4)
    static_input = torch.empty((2, 4), device="cuda", dtype=torch.float32)
    first_input = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]],
        device="cuda",
    )
    second_input = torch.tensor(
        [[10.0, 4.0, -2.0, 6.0], [5.0, -1.0, 7.0, 9.0]],
        device="cuda",
    )

    static_input.copy_(first_input)
    shim(static_input, forward_batch())
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = shim(static_input, forward_batch())

    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(static_output, expected_loopback_rotation(first_input), atol=1e-5, rtol=1e-5)
    output_storage_ptr = static_output.untyped_storage().data_ptr()

    static_input.copy_(second_input)
    graph.replay()
    torch.cuda.synchronize()
    assert static_output.untyped_storage().data_ptr() == output_storage_ptr
    assert torch.allclose(static_output, expected_loopback_rotation(second_input), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        pytest.param(torch.bfloat16, marks=pytest.mark.requires_cuda(bf16=True)),
    ],
)
def test_instance_loopback_eager_rotates_hidden_pairs(dtype: torch.dtype) -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_instance_loopback_eager_rotates_hidden_pairs",
        str(dtype).removeprefix("torch."),
    )


def test_instance_loopback_rejects_odd_hidden_size() -> None:
    run_isolated_native_case(Path(__file__), "isolated_case_instance_loopback_rejects_odd_hidden_size")


def test_instance_loopback_cuda_graph_replay_rotates_updated_inputs() -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_instance_loopback_cuda_graph_replay_rotates_updated_inputs",
    )


def loopback_shim(*, hidden_size: int) -> FfnShimModule:
    """Return one bound direct-loopback FFN shim."""

    shim = FfnShimModule(layer_id=7, hidden_size=hidden_size, layer_kind=FfnLayerKind.DENSE)
    shim.bind_identity(instance_index=0, sglang_rank=0, model_architecture="loopback")
    return shim


def forward_batch(mode: SglangForwardMode = SglangForwardMode.DECODE) -> object:
    """Return minimal SGLang forward metadata for direct shim tests."""

    return type("FakeForwardBatch", (), {"forward_mode": mode})()
