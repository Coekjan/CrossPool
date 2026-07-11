from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
import torch
from torch import nn

from tests.harness.native.transport import devagent_arena_process
from xpool.abi import DebugOption, RuntimeRole
from xpool.config import XpoolConfig, init_global_config
from xpool.integrations.sglang.shim import FfnShimModule

pytestmark = [
    pytest.mark.requires_cuda(),
]


def uses_native_ops_fixtures(request: pytest.FixtureRequest) -> bool:
    """Return whether a module uses the direct native-op setup."""

    return request.module.__name__.rsplit(".", 1)[-1] in {"test_arena_ops", "test_runtime_ops", "test_shim_ops"}


@pytest.fixture(scope="module", autouse=True)
def load_native_ops(request: pytest.FixtureRequest) -> None:
    if not uses_native_ops_fixtures(request):
        return
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption(0)))


@pytest.fixture(autouse=True)
def reset_native_runtime(
    request: pytest.FixtureRequest,
    reset_global_config: None,
) -> Iterator[None]:
    if not uses_native_ops_fixtures(request):
        yield
        return
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption(0)))
    torch.ops.xpool.instance.detach_transport_arena(1, 0)
    torch.ops.xpool.instance.detach_transport_arena(1, 1)
    yield
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption(0)))
    torch.ops.xpool.instance.detach_transport_arena(1, 0)
    torch.ops.xpool.instance.detach_transport_arena(1, 1)


@pytest.fixture
def native_arena_handle() -> Iterator[Callable[..., str]]:
    with ExitStack() as stack:

        def create(
            *,
            max_tokens: int = 8,
            hidden_size: int = 4,
            element_size: int = 4,
            atn_dp_size: int = 1,
            launch_kernel: bool = False,
        ) -> str:
            return stack.enter_context(
                devagent_arena_process(
                    cuda_device=torch.cuda.current_device(),
                    max_tokens=max_tokens,
                    hidden_size=hidden_size,
                    element_size=element_size,
                    atn_dp_size=atn_dp_size,
                    launch_kernel=launch_kernel,
                )
            )

        yield create


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

    env = {"XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "1"} if debug_loopback else {}
    init_global_config(
        config=XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env=env,
        ),
    )
    debug_options = DebugOption.SHIM_LOOPBACK if debug_loopback else DebugOption(0)
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(debug_options))


def attach_transport_arena_for_test(
    native_arena_handle: Callable[..., str],
    *,
    instance_index: int,
    rank: int,
) -> str:
    handle = native_arena_handle()
    torch.ops.xpool.instance.attach_transport_arena(instance_index, rank, handle)
    return handle


def assert_fake_transport_arena_usable(*, instance_index: int, rank: int) -> None:
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)
    output = ffn_shim_native(hidden_states, instance_index=instance_index, rank=rank)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


def ffn_shim_native(hidden_states: torch.Tensor, *, instance_index: int, rank: int) -> torch.Tensor:
    return torch.ops.xpool.instance.ffn_shim(
        hidden_states,
        None,
        instance_index,
        rank,
        3,
        2,
        1,
        0,
        int(hidden_states.shape[0]),
        0,
        1,
        0,
        1,
    )
