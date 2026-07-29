"""Generic TCP endpoint reservation behavior."""

from __future__ import annotations

import errno
import socket

import pytest

import tests.harness.runner.network
from tests.harness.runner.network import (
    TcpEndpointAllocationError,
    TcpEndpointConflict,
    TcpEndpointReservation,
    TcpEndpointUnreachable,
    TcpPortSpace,
)


def test_endpoint_reservation_holds_port_until_spawn_release() -> None:
    port_space = TcpPortSpace.local()
    reservation = TcpEndpointReservation.reserve("127.0.0.1", port_space=port_space)
    competing = TcpEndpointReservation.reserve("127.0.0.1", port_space=port_space)
    try:
        assert reservation.port != competing.port
        assert port_space.is_eligible(reservation.port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            with pytest.raises(OSError):
                listener.bind((reservation.host, reservation.port))
        reservation.release_for_spawn()
        with pytest.raises(RuntimeError, match="already released"):
            reservation.release_for_spawn()
    finally:
        reservation.close()
        competing.close()


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
    monkeypatch.setattr(tests.harness.runner.network.secrets, "randbelow", lambda count: 1)
    port_space = TcpPortSpace(
        unprivileged_port_start=65_530,
        ephemeral=range(65_532, 65_534),
        administratively_reserved=frozenset({65_535}),
    )

    assert tuple(port_space.candidates()) == (65_531, 65_534, 65_530)


def test_exact_endpoint_reservation_releases_owned_listener() -> None:
    selected = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    port = selected.port
    selected.close()

    exact = TcpEndpointReservation.reserve_exact("127.0.0.1", port)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            with pytest.raises(OSError):
                listener.bind((exact.host, exact.port))
    finally:
        exact.close()


def test_endpoint_reservation_reacquires_released_listener() -> None:
    reservation = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    try:
        with pytest.raises(RuntimeError, match="already reserved"):
            reservation.reacquire()
        reservation.release_for_spawn()
        reservation.reacquire()
        assert reservation.listener is not None
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
            with pytest.raises(OSError) as error:
                competitor.bind(reservation.address)
            assert error.value.errno == errno.EADDRINUSE
    finally:
        reservation.close()


def test_endpoint_reservation_remains_released_after_reacquire_conflict() -> None:
    reservation = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    reservation.release_for_spawn()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
        competitor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        competitor.bind(reservation.address)
        competitor.listen()
        with pytest.raises(OSError) as error:
            reservation.reacquire()
        assert error.value.errno == errno.EADDRINUSE
        assert reservation.listener is None
    reservation.close()


def test_endpoint_conflict_preserves_ordered_addresses() -> None:
    addresses = (("127.0.0.1", 20_001), ("127.0.0.1", 20_002))

    conflict = TcpEndpointConflict(addresses)

    assert conflict.addresses == addresses
    assert str(conflict).endswith("127.0.0.1:20001, 127.0.0.1:20002")


def test_endpoint_reservation_fails_after_finite_candidate_exhaustion() -> None:
    occupied = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    port_space = TcpPortSpace(
        unprivileged_port_start=occupied.port,
        ephemeral=range(occupied.port + 1, 65_536),
        administratively_reserved=frozenset(),
    )
    try:
        with pytest.raises(TcpEndpointAllocationError, match="no eligible TCP endpoint"):
            TcpEndpointReservation.reserve("127.0.0.1", port_space=port_space)
    finally:
        occupied.close()


def test_endpoint_reservation_counts_collisions_and_unreachable_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port_space = TcpPortSpace(65_533, range(1, 65_533), frozenset())
    monkeypatch.setattr(tests.harness.runner.network.secrets, "randbelow", lambda count: 0)

    def reserve_exact(cls: type[TcpEndpointReservation], host: str, port: int) -> TcpEndpointReservation:
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
