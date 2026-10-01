"""Generic TCP endpoint reservation behavior."""

from __future__ import annotations

import errno
import socket

import pytest

from xkit.network import (
    TcpEndpointAllocationError,
    TcpEndpointReservation,
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
