"""Pinned SGLang compile integration for the xpool FFN shim."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    enable_piecewise_cuda_graph_compile,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from torch import nn

from tests.harness.support.config import reset_global_config
from tests.harness.support.sglang.fakes import forward_batch
from xpool.integrations.sglang.shim import FfnShimModule
from xpool.native.ffn import LayerKind

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.usefixtures(reset_global_config.__name__),
]


class ShimGraphModule(nn.Module):
    """Minimal module matching SGLang's piecewise compiler entry point."""

    def __init__(self, shim: FfnShimModule, forward_batch: ForwardBatch) -> None:
        super().__init__()
        self.shim = shim
        self.forward_batch = forward_batch

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shim(hidden_states, self.forward_batch)


def test_ffn_shim_is_fullgraph_traceable_with_meta_dispatch() -> None:
    shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=LayerKind.DENSE)
    shim.bind_runtime(layer_ordinal=0, model_architecture="trace-smoke")
    batch = forward_batch(ForwardMode.DECODE)
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)

    def call_shim(tensor: torch.Tensor) -> torch.Tensor:
        return shim(tensor, batch)

    output = torch.compile(call_shim, fullgraph=True, backend="eager")(hidden_states)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


def test_ffn_shim_piecewise_compile_reuses_dynamic_token_dimension() -> None:
    shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=LayerKind.DENSE)
    shim.bind_runtime(layer_ordinal=0, model_architecture="pcg-dynamic")
    module = ShimGraphModule(shim, forward_batch(ForwardMode.EXTEND))
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
