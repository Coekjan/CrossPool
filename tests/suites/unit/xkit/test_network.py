"""Generic TCP endpoint reservation behavior."""

from __future__ import annotations

import errno

import pytest

import xkit.network
from xkit.network import (
    TcpEndpointAllocationError,
    TcpEndpointConflict,
    TcpEndpointReservation,
    TcpEndpointUnreachable,
    TcpPortSpace,
)


def test_port_space_parses_linux_policy_and_rejects_ineligible_ports() -> None:
    port_space = TcpPortSpace.parse("1024\n", "32768 60999\n", "2000,3000-3002\n")

    assert port_space.unprivileged_port_start == 1024
    assert port_space.ephemeral == range(32_768, 61_000)
    assert port_space.administratively_reserved == frozenset({2_000, 3_000, 3_001, 3_002})
    assert port_space.is_eligible(1_024)
    assert not port_space.is_eligible(1_023)
    assert not port_space.is_eligible(2_000)
    assert not port_space.is_eligible(32_768)
    assert port_space.is_eligible(61_000)


@pytest.mark.parametrize(
    ("unprivileged", "ephemeral", "reserved"),
    [
        ("", "32768 60999", ""),
        ("1024", "32768", ""),
        ("1024", "60999 32768", ""),
        ("1024", "32768 60999", "2000-1000"),
        ("1024", "32768 60999", "70000"),
        ("1024", "32768 60999", "1000,,1002"),
    ],
)
def test_port_space_rejects_malformed_linux_policy(
    unprivileged: str,
    ephemeral: str,
    reserved: str,
) -> None:
    with pytest.raises(ValueError, match="invalid Linux TCP port-space configuration"):
        TcpPortSpace.parse(unprivileged, ephemeral, reserved)


def test_port_space_candidates_visit_each_eligible_port_once_from_random_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xkit.network.secrets, "randbelow", lambda count: 1)
    port_space = TcpPortSpace(
        unprivileged_port_start=65_530,
        ephemeral=range(65_532, 65_534),
        administratively_reserved=frozenset({65_535}),
    )

    candidates = tuple(port_space.candidates())
    assert candidates[0] == 65_531
    assert len(candidates) == len(set(candidates)) == 3
    assert set(candidates) == {65_530, 65_531, 65_534}


def test_endpoint_reservation_counts_collisions_and_unreachable_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port_space = TcpPortSpace(65_533, range(1, 65_533), frozenset())
    monkeypatch.setattr(xkit.network.secrets, "randbelow", lambda count: 0)

    def reserve_exact(
        cls: type[TcpEndpointReservation], host: str, port: int, *, deadline: float | None = None
    ) -> TcpEndpointReservation:
        if port == 65_533:
            raise OSError(errno.EADDRINUSE, "occupied")
        if port == 65_534:
            raise TcpEndpointUnreachable((host, port))
        raise OSError(errno.EADDRINUSE, "occupied")

    monkeypatch.setattr(TcpEndpointReservation, "reserve_exact", classmethod(reserve_exact))

    with pytest.raises(TcpEndpointAllocationError) as error:
        TcpEndpointReservation.reserve("127.0.0.1", port_space=port_space)

    assert error.value.collision_count == 2
    assert error.value.unreachable_count == 1
    assert isinstance(error.value.__cause__, OSError)


def test_endpoint_conflict_preserves_ordered_addresses() -> None:
    addresses = (("127.0.0.1", 20_001), ("127.0.0.1", 20_002))

    conflict = TcpEndpointConflict(addresses)

    assert conflict.addresses == addresses
    assert str(conflict).endswith("127.0.0.1:20001, 127.0.0.1:20002")
