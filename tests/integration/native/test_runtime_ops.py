from __future__ import annotations

from tests.harness.native.ops import (
    DebugLoopbackSite,
    DebugOptions,
    RuntimeRole,
    pytest,
    torch,
)
from xpool.abi import ABI_VERSION


def test_native_abi_version_matches_python() -> None:
    assert int(torch.ops.xpool.abi_version()) == ABI_VERSION


def test_atnagent_ops_reject_instance_role_process() -> None:
    with pytest.raises(RuntimeError, match="requires runtime role atnagent"):
        torch.ops.xpool.atnagent.create_transport_arena(torch.cuda.current_device(), 8, 4, 4, 1)


pytestmark = pytest.mark.requires_cuda()


def test_xpool_init_is_idempotent_for_same_role_and_debug_options() -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), DebugOptions(0).raw)
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), DebugOptions(0).raw)


def test_xpool_init_rejects_role_switch() -> None:
    with pytest.raises(RuntimeError, match="requires runtime role atnagent"):
        torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.ATNAGENT), DebugOptions(0).raw)


def test_xpool_init_rejects_debug_option_switch() -> None:
    with pytest.raises(RuntimeError, match="debug options differ"):
        torch.ops.xpool.init(
            torch.cuda.current_device(),
            int(RuntimeRole.INSTANCE),
            DebugOptions.create(loopback_site=DebugLoopbackSite.INSTANCE).raw,
        )


def test_xpool_init_rejects_negative_cuda_device() -> None:
    with pytest.raises(RuntimeError, match="non-negative CUDA device"):
        torch.ops.xpool.init(-1, int(RuntimeRole.INSTANCE), DebugOptions(0).raw)


def test_xpool_init_rejects_invalid_runtime_role() -> None:
    with pytest.raises(RuntimeError, match="invalid runtime role"):
        torch.ops.xpool.init(torch.cuda.current_device(), 99, DebugOptions(0).raw)


def test_xpool_init_rejects_unknown_debug_option_bits() -> None:
    with pytest.raises(RuntimeError, match="unknown option flags"):
        torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), 1 << 34)


def test_xpool_init_rejects_reserved_debug_option_bits() -> None:
    with pytest.raises(RuntimeError, match="reserved option bits"):
        torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), 1 << 2)


def test_xpool_init_rejects_loopback_option_without_site() -> None:
    with pytest.raises(RuntimeError, match="option and site"):
        torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), 1 << 32)
