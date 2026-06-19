from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode as SglangForwardMode

import xpool.config as config_module
from xpool.cext import NativeLoadError, ensure_xpool_ops_loaded
from xpool.config import XpoolConfig, init_global_config
from xpool.integrations.sglang.shim import FfnLayerKind, FfnShimModule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for native loopback shim tests",
)


@pytest.fixture(scope="module", autouse=True)
def load_cext() -> None:
    try:
        ensure_xpool_ops_loaded()
    except NativeLoadError as exc:
        pytest.skip(f"xpool native ops are not installed: {exc}")


@pytest.fixture(autouse=True)
def enable_loopback(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(config_module, "_global_config", None)
    init_global_config(
        config=XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={"XPOOL_DEBUG_ENABLE_SHIM_LOOPBACK": "1"},
        )
    )
    yield
    monkeypatch.setattr(config_module, "_global_config", None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ffn_shim_loopback_eager_rotates_hidden_pairs(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
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
    assert torch.allclose(output.float(), expected_rotation(hidden_states).float(), atol=2e-2, rtol=2e-2)


def test_ffn_shim_loopback_rejects_odd_hidden_size() -> None:
    shim = loopback_shim(hidden_size=3)
    hidden_states = torch.ones((2, 3), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="even hidden size"):
        shim(hidden_states, forward_batch())


def test_ffn_shim_loopback_cuda_graph_replay_rotates_updated_inputs() -> None:
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
    assert torch.allclose(static_output, expected_rotation(first_input), atol=1e-5, rtol=1e-5)
    output_storage_ptr = static_output.untyped_storage().data_ptr()

    static_input.copy_(second_input)
    graph.replay()
    torch.cuda.synchronize()
    assert static_output.untyped_storage().data_ptr() == output_storage_ptr
    assert torch.allclose(static_output, expected_rotation(second_input), atol=1e-5, rtol=1e-5)


def loopback_shim(*, hidden_size: int) -> FfnShimModule:
    shim = FfnShimModule(layer_id=7, hidden_size=hidden_size, layer_kind=FfnLayerKind.DENSE)
    shim.bind_identity(instance_index=0, model_index=0, model_architecture="loopback")
    return shim


def forward_batch(mode: SglangForwardMode = SglangForwardMode.DECODE) -> object:
    return type("FakeForwardBatch", (), {"forward_mode": mode})()


def expected_rotation(hidden_states: torch.Tensor) -> torch.Tensor:
    x_values = hidden_states.float()[..., 0::2]
    y_values = hidden_states.float()[..., 1::2]
    output = torch.empty_like(hidden_states.float())
    output[..., 0::2] = (x_values - y_values) / math.sqrt(2.0)
    output[..., 1::2] = (x_values + y_values) / math.sqrt(2.0)
    return output.to(hidden_states.dtype)
