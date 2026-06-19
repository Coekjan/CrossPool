from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    enable_piecewise_cuda_graph_compile,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode as SglangForwardMode
from torch import nn

import xpool.config as config_module
from xpool.abi import ABI_VERSION
from xpool.cext import NativeLoadError, ensure_xpool_ops_loaded
from xpool.config import XpoolConfig, init_global_config
from xpool.integrations.sglang.shim import FfnLayerKind, FfnShimModule


@pytest.fixture(scope="module", autouse=True)
def load_native_ops() -> None:
    try:
        ensure_xpool_ops_loaded()
    except NativeLoadError as exc:
        pytest.skip(f"xpool native ops are not installed: {exc}")


@pytest.fixture(autouse=True)
def reset_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(config_module, "_global_config", None)
    yield
    monkeypatch.setattr(config_module, "_global_config", None)


def test_ffn_shim_is_unimplemented() -> None:
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="not implemented"):
        torch.ops.xpool.ffn_shim(hidden_states, 1, 2, 3, 2)


def test_ffn_shim_loopback_meta_dispatch_preserves_shape_dtype_and_device() -> None:
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)

    output = torch.ops.xpool.ffn_shim_loopback(hidden_states, 1, 2, 3, 2)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


def test_native_abi_version_matches_python() -> None:
    assert int(torch.ops.xpool.abi_version()) == ABI_VERSION


def test_ffn_shim_loopback_forward_is_fullgraph_traceable_with_meta_dispatch() -> None:
    install_global_config(debug_loopback=True)
    shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=FfnLayerKind.DENSE)
    shim.bind_identity(instance_index=1, model_index=2, model_architecture="trace-smoke")
    forward_batch = type("FakeForwardBatch", (), {"forward_mode": SglangForwardMode.DECODE})()
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)

    def call_shim(tensor: torch.Tensor) -> torch.Tensor:
        return shim(tensor, forward_batch)

    compiled = torch.compile(call_shim, fullgraph=True, backend="eager")
    output = compiled(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


def test_ffn_shim_loopback_matches_sglang_piecewise_compile_context() -> None:
    install_global_config(debug_loopback=True)
    shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=FfnLayerKind.DENSE)
    shim.bind_identity(instance_index=1, model_index=2, model_architecture="pcg-smoke")
    forward_batch = type("FakeForwardBatch", (), {"forward_mode": SglangForwardMode.EXTEND})()
    wrapper = ShimWrapper(shim=shim, forward_batch=forward_batch)
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)

    def backend_factory(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[object],
    ) -> Callable[..., object]:
        return graph_module.forward

    with enable_piecewise_cuda_graph():
        install_torch_compiled(
            wrapper,
            dynamic_arg_dims={},
            backend_factory=backend_factory,
            fullgraph=True,
        )
        with enable_piecewise_cuda_graph_compile():
            output = wrapper(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


class ShimWrapper(nn.Module):
    """Minimal module matching SGLang PCG's install_torch_compiled entry point."""

    def __init__(self, *, shim: FfnShimModule, forward_batch: object) -> None:
        """Store a shim and closed-over SGLang forward-batch object.

        Args:
            shim: Bound FFN shim module to invoke.
            forward_batch: SGLang-like forward metadata used by the shim.

        Side Effects:
            Initializes ``nn.Module``.
        """

        super().__init__()
        self.shim = shim
        self.forward_batch = forward_batch

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward hidden states through the contained shim.

        Args:
            hidden_states: Meta or CUDA hidden-state tensor.

        Returns:
            Shim output tensor.
        """

        return self.shim(hidden_states, self.forward_batch)


def install_global_config(*, debug_loopback: bool) -> None:
    """Install a minimal global config for shim routing tests.

    Args:
        debug_loopback: Whether to enable the debug loopback source.

    Side Effects:
        Replaces the process-global xpool config.
    """

    env = {"XPOOL_DEBUG_ENABLE_SHIM_LOOPBACK": "1"} if debug_loopback else {}
    init_global_config(
        config=XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env=env,
        ),
    )
