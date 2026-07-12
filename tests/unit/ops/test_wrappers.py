from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import xpool.ops
from xpool.abi import (
    DebugLoopbackSite,
    DebugOptions,
    DpPaddingMode,
    FfnCollectivePolicy,
    FfnRequestMetadata,
    RuntimeRole,
    TransportArenaHandle,
    XPoolForwardMode,
)


def test_ffn_shim_expands_keyword_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    dp_tokens = torch.tensor([2])

    def fake_ffn_shim(*args: object) -> torch.Tensor:
        calls.append(args)
        return hidden_states.clone()

    monkeypatch.setattr(
        xpool.ops.torch.ops,
        "xpool",
        SimpleNamespace(instance=SimpleNamespace(ffn_shim=fake_ffn_shim)),
        raising=False,
    )
    request_metadata = FfnRequestMetadata(
        instance_index=3,
        layer_id=7,
        forward_mode=int(XPoolForwardMode.DECODE),
        collective_policy=int(FfnCollectivePolicy.ATN_TP_PARTIAL),
        dp_padding_mode=int(DpPaddingMode.MAX_LEN),
        global_dp_buffer_len=16,
        atn_tp_rank=1,
        atn_tp_size=4,
        atn_dp_rank=2,
        atn_dp_size=3,
    )
    output = xpool.ops.instance.ffn_shim(hidden_states, dp_tokens, request_metadata, 2)

    assert torch.equal(output, hidden_states)
    assert calls == [
        (
            hidden_states,
            dp_tokens,
            3,
            2,
            7,
            int(XPoolForwardMode.DECODE),
            int(FfnCollectivePolicy.ATN_TP_PARTIAL),
            int(DpPaddingMode.MAX_LEN),
            16,
            1,
            4,
            2,
            3,
        )
    ]


def test_init_delegates_runtime_role_to_torch_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        xpool.ops.torch.ops,
        "xpool",
        SimpleNamespace(init=lambda *args: calls.append(("init", *args))),
        raising=False,
    )
    monkeypatch.setattr(
        xpool.ops,
        "get_global_config",
        lambda: SimpleNamespace(
            debug=SimpleNamespace(
                loopback=SimpleNamespace(enable=True, site=xpool.ops.LoopbackSite.ATNAGENT),
                transport_observer=SimpleNamespace(enable=False),
            )
        ),
    )

    xpool.ops.init(3, RuntimeRole.INSTANCE)
    xpool.ops.init(4, RuntimeRole.ATNAGENT)

    assert calls == [
        (
            "init",
            3,
            int(RuntimeRole.INSTANCE),
            DebugOptions.create(loopback_site=DebugLoopbackSite.ATNAGENT).raw,
        ),
        (
            "init",
            4,
            int(RuntimeRole.ATNAGENT),
            DebugOptions.create(loopback_site=DebugLoopbackSite.ATNAGENT).raw,
        ),
    ]


def test_atnagent_ops_delegate_to_torch_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    handle = "00" * 64

    def fake_create_transport_arena(*args: object) -> str:
        calls.append(("create", *args))
        return handle

    def fake_launch_transport_kernel(*args: object) -> None:
        calls.append(("launch", *args))

    def fake_destroy_transport_arena(*args: object) -> tuple[int, int, list[list[int]]]:
        calls.append(("destroy", *args))
        return 0, 0, []

    monkeypatch.setattr(
        xpool.ops.torch.ops,
        "xpool",
        SimpleNamespace(
            atnagent=SimpleNamespace(
                create_transport_arena=fake_create_transport_arena,
                launch_transport_kernel=fake_launch_transport_kernel,
                destroy_transport_arena=fake_destroy_transport_arena,
            )
        ),
        raising=False,
    )

    arena = TransportArenaHandle(handle)
    assert xpool.ops.atnagent.create_transport_arena(0, 8, 4, 2, 1) == arena
    xpool.ops.atnagent.launch_transport_kernel(arena)
    assert xpool.ops.atnagent.destroy_transport_arena(arena).records == ()

    assert calls == [
        ("create", 0, 8, 4, 2, 1),
        ("launch", "00" * 64),
        ("destroy", "00" * 64),
    ]


def test_destroy_transport_arena_decodes_observer_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_record = list(range(1, 16))
    monkeypatch.setattr(
        xpool.ops.torch.ops,
        "xpool",
        SimpleNamespace(
            atnagent=SimpleNamespace(
                destroy_transport_arena=lambda handle: (7, 2, [raw_record]),
            )
        ),
        raising=False,
    )

    snapshot = xpool.ops.atnagent.destroy_transport_arena(TransportArenaHandle("00" * 64))

    assert snapshot.sequence == 7
    assert snapshot.dropped == 2
    assert len(snapshot.records) == 1
    assert snapshot.records[0].trace_id == 1
    assert snapshot.records[0].slot_recycled == 15


def test_instance_ops_delegate_to_torch_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    handle = TransportArenaHandle(handle="00" * 64)

    monkeypatch.setattr(
        xpool.ops.torch.ops,
        "xpool",
        SimpleNamespace(
            instance=SimpleNamespace(
                attach_transport_arena=lambda *args: calls.append(("attach", *args)),
                detach_transport_arena=lambda *args: calls.append(("detach", *args)),
            )
        ),
        raising=False,
    )

    xpool.ops.instance.attach_transport_arena(7, 3, handle)
    xpool.ops.instance.detach_transport_arena(7, 3)

    assert calls == [
        ("attach", 7, 3, "00" * 64),
        ("detach", 7, 3),
    ]


@pytest.mark.parametrize("handle", ["00" * 63, "00" * 63 + "0g", "AA" * 64])
def test_transport_arena_handle_rejects_invalid_handle(handle: str) -> None:
    with pytest.raises(ValueError, match="transport arena handle"):
        TransportArenaHandle(handle=handle)
