from __future__ import annotations

from tests.harness.native.ops import (
    Callable,
    assert_fake_transport_arena_usable,
    attach_transport_arena_for_test,
    devagent_arena_process,
    ffn_shim_native,
    pytest,
    torch,
)

pytestmark = pytest.mark.requires_cuda()


def test_detach_instance_transport_arena_removes_only_requested_rank(native_arena_handle: Callable[..., str]) -> None:
    attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=0)
    attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=1)

    torch.ops.xpool.instance.detach_transport_arena(1, 0)

    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="no attached instance transport arena"):
        ffn_shim_native(hidden_states, instance_index=1, rank=0)
    assert_fake_transport_arena_usable(instance_index=1, rank=1)


def test_attach_instance_transport_arena_accepts_identical_reattach(native_arena_handle: Callable[..., str]) -> None:
    handle = attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=0)
    torch.ops.xpool.instance.attach_transport_arena(1, 0, handle)

    assert_fake_transport_arena_usable(instance_index=1, rank=0)


def test_attach_instance_transport_arena_rejects_different_arena_without_detach(
    native_arena_handle: Callable[..., str],
) -> None:
    attach_transport_arena_for_test(native_arena_handle, instance_index=1, rank=0)
    different_handle = native_arena_handle()

    with pytest.raises(RuntimeError, match="already attached with different arena"):
        torch.ops.xpool.instance.attach_transport_arena(1, 0, different_handle)

    assert_fake_transport_arena_usable(instance_index=1, rank=0)


def test_attach_instance_transport_arena_rejects_invalid_handle() -> None:
    with pytest.raises(RuntimeError, match="unexpected byte length"):
        torch.ops.xpool.instance.attach_transport_arena(1, 0, "00" * 63)


def test_transport_arena_geometry_overflow_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="overflows int64"):
        with devagent_arena_process(
            cuda_device=torch.cuda.current_device(),
            max_tokens=2**32,
            hidden_size=2**31,
            element_size=4,
            atn_dp_size=1,
            launch_kernel=False,
        ):
            pass
