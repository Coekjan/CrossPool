from __future__ import annotations

from tests.harness.native.ops import (
    DebugOption,
    RuntimeRole,
    pytest,
    torch,
)
from xpool.abi import ABI_VERSION


def test_native_abi_version_matches_python() -> None:
    assert int(torch.ops.xpool.abi_version()) == ABI_VERSION


def test_devagent_ops_reject_instance_role_process() -> None:
    with pytest.raises(RuntimeError, match="requires runtime role devagent"):
        torch.ops.xpool.devagent.create_transport_arena(torch.cuda.current_device(), 8, 4, 4, 1)


pytestmark = pytest.mark.requires_cuda()


def test_xpool_init_is_idempotent_for_same_role_and_debug_options() -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption(0)))
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption(0)))


def test_xpool_init_rejects_role_switch() -> None:
    with pytest.raises(RuntimeError, match="requires runtime role devagent"):
        torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.DEVAGENT), int(DebugOption(0)))


def test_xpool_init_rejects_debug_option_switch() -> None:
    with pytest.raises(RuntimeError, match="debug options differ"):
        torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption.SHIM_LOOPBACK))


def test_xpool_init_rejects_negative_cuda_device() -> None:
    with pytest.raises(RuntimeError, match="non-negative CUDA device"):
        torch.ops.xpool.init(-1, int(RuntimeRole.INSTANCE), int(DebugOption(0)))


def test_xpool_init_rejects_invalid_runtime_role() -> None:
    with pytest.raises(RuntimeError, match="invalid runtime role"):
        torch.ops.xpool.init(torch.cuda.current_device(), 99, int(DebugOption(0)))


def test_xpool_init_rejects_unknown_debug_option_bits() -> None:
    with pytest.raises(RuntimeError, match="unknown debug option bits"):
        torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), 1 << 8)


def test_xpool_init_rejects_mutually_exclusive_debug_options() -> None:
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        torch.ops.xpool.init(
            torch.cuda.current_device(),
            int(RuntimeRole.INSTANCE),
            int(DebugOption.SHIM_LOOPBACK | DebugOption.TRANSPORT_LOOPBACK),
        )
