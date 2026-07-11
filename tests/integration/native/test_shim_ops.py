from __future__ import annotations

from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    enable_piecewise_cuda_graph_compile,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode as SglangForwardMode

from tests.harness.native.ops import (
    Callable,
    FfnShimModule,
    ShimWrapper,
    attach_transport_arena_for_test,
    ffn_shim_native,
    install_global_config,
    pytest,
    torch,
)
from xpool.integrations.sglang.shim import FfnLayerKind

pytestmark = pytest.mark.requires_cuda()


def test_ffn_shim_requires_attached_transport_arena() -> None:
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="no attached instance transport arena"):
        ffn_shim_native(hidden_states, instance_index=1, rank=0)


def test_ffn_shim_meta_dispatch_uses_attached_transport_arena(native_arena_handle: Callable[..., str]) -> None:
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)
    attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=0)

    output = ffn_shim_native(hidden_states, instance_index=1, rank=0)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


def test_ffn_shim_native_meta_dispatch_preserves_shape_dtype_and_device(
    native_arena_handle: Callable[..., str],
) -> None:
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)
    attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=0)

    output = torch.ops.xpool.instance.ffn_shim(hidden_states, None, 1, 0, 3, 2, 1, 0, 2, 0, 1, 0, 1)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


def test_ffn_shim_forward_is_fullgraph_traceable_with_meta_dispatch(native_arena_handle: Callable[..., str]) -> None:
    install_global_config(debug_loopback=False)
    attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=0)
    shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=FfnLayerKind.DENSE)
    shim.bind_identity(instance_index=1, sglang_rank=0, model_architecture="trace-smoke")
    forward_batch = type("FakeForwardBatch", (), {"forward_mode": SglangForwardMode.DECODE})()
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)

    def call_shim(tensor: torch.Tensor) -> torch.Tensor:
        return shim(tensor, forward_batch)

    compiled = torch.compile(call_shim, fullgraph=True, backend="eager")
    output = compiled(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


def test_ffn_shim_matches_sglang_piecewise_compile_context(native_arena_handle: Callable[..., str]) -> None:
    install_global_config(debug_loopback=False)
    attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=0)
    shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=FfnLayerKind.DENSE)
    shim.bind_identity(instance_index=1, sglang_rank=0, model_architecture="pcg-smoke")
    forward_batch = type("FakeForwardBatch", (), {"forward_mode": SglangForwardMode.EXTEND})()
    wrapper = ShimWrapper(shim=shim, forward_batch=forward_batch)
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)

    def backend_factory(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[object],
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


def test_ffn_shim_piecewise_compile_reuses_dynamic_token_dimension(native_arena_handle: Callable[..., str]) -> None:
    install_global_config(debug_loopback=False)
    attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=0)
    shim = FfnShimModule(layer_id=7, hidden_size=4, layer_kind=FfnLayerKind.DENSE)
    shim.bind_identity(instance_index=1, sglang_rank=0, model_architecture="pcg-dynamic")
    forward_batch = type("FakeForwardBatch", (), {"forward_mode": SglangForwardMode.EXTEND})()
    wrapper = ShimWrapper(shim=shim, forward_batch=forward_batch)
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
            wrapper,
            dynamic_arg_dims={"hidden_states": 0},
            backend_factory=backend_factory,
            fullgraph=True,
        )
        with enable_piecewise_cuda_graph_compile():
            first_output = wrapper(torch.empty((4, 4), device="meta", dtype=torch.float32))
            second_output = wrapper(torch.empty((8, 4), device="meta", dtype=torch.float32))

    assert first_output.shape == (4, 4)
    assert second_output.shape == (8, 4)
    assert compile_count == 1
