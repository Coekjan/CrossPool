from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sglang.srt.model_executor import forward_batch_info

import xpool.config
import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.debug import native_debug_options
from tests.harness.support.config import install_test_config
from tests.harness.support.native.loopback import expected_loopback_rotation
from xpool.config import LoopbackSite, XpoolConfig
from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.shim import FfnShimModule
from xpool.runtime import RuntimeRole

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.timeout(180),
]


def install_instance_loopback_config() -> None:
    """Initialize an isolated process for instance-loopback execution."""

    xpool.config.global_config = None
    xpool.native.initialize(
        RuntimeRole.INSTANCE,
        cuda_device=torch.cuda.current_device(),
        debug_options=native_debug_options(loopback_site=LoopbackSite.INSTANCE),
    )
    install_test_config(
        XpoolConfig.from_mapping(
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
        torch.bfloat16,
    ],
)
def test_instance_loopback_eager_rotates_hidden_pairs(dtype: torch.dtype, tmp_path: Path) -> None:
    run_native_case(
        isolated_case_instance_loopback_eager_rotates_hidden_pairs,
        str(dtype).removeprefix("torch."),
        workdir=tmp_path / "case",
    )


def test_instance_loopback_rejects_odd_hidden_size(tmp_path: Path) -> None:
    run_native_case(isolated_case_instance_loopback_rejects_odd_hidden_size, workdir=tmp_path / "case")


def test_instance_loopback_cuda_graph_replay_rotates_updated_inputs(tmp_path: Path) -> None:
    run_native_case(
        isolated_case_instance_loopback_cuda_graph_replay_rotates_updated_inputs,
        workdir=tmp_path / "case",
    )


def loopback_shim(*, hidden_size: int) -> FfnShimModule:
    """Return one bound direct-loopback FFN shim."""

    shim = FfnShimModule(layer_id=7, hidden_size=hidden_size, layer_kind=FfnLayerKind.DENSE)
    shim.bind_runtime(layer_ordinal=0, model_architecture="loopback")
    return shim


def forward_batch(
    mode: forward_batch_info.ForwardMode = forward_batch_info.ForwardMode.DECODE,
) -> object:
    """Return minimal SGLang forward metadata for direct shim tests."""

    return type("FakeForwardBatch", (), {"forward_mode": mode})()
