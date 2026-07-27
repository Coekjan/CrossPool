"""Pinned SGLang compile integration for the xpool FFN shim."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager

import pytest
import torch
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    enable_piecewise_cuda_graph_compile,
)
from sglang.srt.model_executor import forward_batch_info
from torch import nn

import xpool.native
from tests.harness.config import install_test_config
from tests.harness.native.transport import transport_arena_handles
from xpool.config import XpoolConfig
from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.shim import FfnShimModule
from xpool.runtime import RuntimeRole

pytestmark = pytest.mark.requires_cuda()


class ShimGraphModule(nn.Module):
    """Minimal module matching SGLang's piecewise compiler entry point."""

    def __init__(self, shim: FfnShimModule, forward_batch: object) -> None:
        super().__init__()
        self.shim = shim
        self.forward_batch = forward_batch

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shim(hidden_states, self.forward_batch)


@contextmanager
def attached_shim_runtime() -> Generator[None, None, None]:
    """Install one Instance runtime, Transport attachment, and shim config."""

    xpool.native.initialize(RuntimeRole.INSTANCE, torch.cuda.current_device(), None)
    xpool.native.transport.detach_arena()
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )
    )
    with transport_arena_handles() as create_arena:
        handle = create_arena(instance_index=1, instance_rank=0)
        xpool.native.transport.attach_arena(1, 0, handle)
        try:
            yield
        finally:
            xpool.native.transport.detach_arena()


def test_ffn_shim_is_fullgraph_traceable_with_meta_dispatch(reset_global_config: None) -> None:
    with attached_shim_runtime():
        shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=FfnLayerKind.DENSE)
        shim.bind_runtime(layer_ordinal=0, model_architecture="trace-smoke")
        forward_batch = type("FakeForwardBatch", (), {"forward_mode": forward_batch_info.ForwardMode.DECODE})()
        hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)

        def call_shim(tensor: torch.Tensor) -> torch.Tensor:
            return shim(tensor, forward_batch)

        output = torch.compile(call_shim, fullgraph=True, backend="eager")(hidden_states)
        assert output.shape == hidden_states.shape
        assert output.dtype == hidden_states.dtype
        assert output.device.type == "meta"


def test_ffn_shim_piecewise_compile_reuses_dynamic_token_dimension(reset_global_config: None) -> None:
    with attached_shim_runtime():
        shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=FfnLayerKind.DENSE)
        shim.bind_runtime(layer_ordinal=0, model_architecture="pcg-dynamic")
        forward_batch = type("FakeForwardBatch", (), {"forward_mode": forward_batch_info.ForwardMode.EXTEND})()
        module = ShimGraphModule(shim, forward_batch)
        compile_count = 0

        def backend_factory(
            graph_module: torch.fx.GraphModule,
            example_inputs: list[object],
        ) -> Callable[..., object]:
            nonlocal compile_count
            compile_count += 1
            return graph_module.forward

        with enable_piecewise_cuda_graph():
            install_torch_compiled(
                module,
                dynamic_arg_dims={"hidden_states": 0},
                backend_factory=backend_factory,
                fullgraph=True,
            )
            with enable_piecewise_cuda_graph_compile():
                first = module(torch.empty((4, 4), device="meta", dtype=torch.float32))
                second = module(torch.empty((8, 4), device="meta", dtype=torch.float32))

        assert first.shape == (4, 4)
        assert second.shape == (8, 4)
        assert compile_count == 1
