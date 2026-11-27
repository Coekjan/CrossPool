"""Generic TCP endpoint reservation behavior."""

from __future__ import annotations

import errno
import socket
from asyncio import CancelledError

import pytest

import xkit.network
from xkit.network import (
    TcpEndpointAllocationError,
    TcpEndpointReservation,
    TcpEndpointUnreachable,
    TcpPortSpace,
)


def test_endpoint_port_lock_excludes_competitors_during_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    port_space = TcpPortSpace.local()
    reservation = TcpEndpointReservation.reserve("127.0.0.1", port_space=port_space)
    competing = TcpEndpointReservation.reserve("127.0.0.1", port_space=port_space)
    try:
        assert reservation.listener is not None
        assert reservation.listener.family == socket.AF_INET
        assert reservation.port != competing.port
        assert port_space.is_eligible(reservation.port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            with pytest.raises(OSError):
                listener.bind((reservation.host, reservation.port))
        reservation.release_for_spawn()
        with monkeypatch.context() as patch:
            patch.setattr(
                xkit.network,
                "create_qualified_tcp_listener",
                lambda *args, **kwargs: pytest.fail("contender probed an already reserved port"),
            )
            with pytest.raises(OSError) as error:
                TcpEndpointReservation.reserve_exact("127.0.0.2", reservation.port)
            assert error.value.errno == errno.EADDRINUSE
        with pytest.raises(RuntimeError, match="already released"):
            reservation.release_for_spawn()
    finally:
        reservation.close()
        competing.close()


def test_unreachable_endpoint_closes_owned_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    listener = xkit.network.create_tcp_listener(("0.0.0.0", 0))
    address = ("0.0.0.0", listener.getsockname()[1])

    def refuse_connection(client: socket.socket, address: tuple[str, int]) -> None:
        raise ConnectionRefusedError(errno.ECONNREFUSED, "injected connection refusal")

    monkeypatch.setattr(xkit.network, "create_tcp_listener", lambda address: listener)
    monkeypatch.setattr(socket.socket, "connect", refuse_connection)
    try:
        with pytest.raises(TcpEndpointUnreachable) as error:
            xkit.network.create_qualified_tcp_listener(address)
        assert isinstance(error.value.__cause__, ConnectionRefusedError)
        assert listener.fileno() == -1
    finally:
        listener.close()


def test_cancelled_endpoint_probe_closes_owned_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    def cancel_connection(client: socket.socket, address: tuple[str, int]) -> None:
        raise CancelledError

    with xkit.network.create_tcp_listener(("0.0.0.0", 0)) as listener:
        address = ("0.0.0.0", listener.getsockname()[1])
        monkeypatch.setattr(xkit.network, "create_tcp_listener", lambda address: listener)
        monkeypatch.setattr(socket.socket, "connect", cancel_connection)
        with pytest.raises(CancelledError):
            xkit.network.create_qualified_tcp_listener(address)
        assert listener.fileno() == -1


def test_exact_endpoint_reservation_releases_owned_listener() -> None:
    selected = TcpEndpointReservation.reserve("localhost", port_space=TcpPortSpace.local())
    port = selected.port
    selected.close()

    exact = TcpEndpointReservation.reserve_exact("localhost", port)
    try:
        assert exact.address == ("127.0.0.1", port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            with pytest.raises(OSError):
                listener.bind((exact.host, exact.port))
    finally:
        exact.close()


def test_endpoint_reservation_reacquires_released_listener() -> None:
    reservation = TcpEndpointReservation.reserve("localhost", port_space=TcpPortSpace.local())
    try:
        with pytest.raises(RuntimeError, match="already reserved"):
            reservation.reacquire()
        reservation.release_for_spawn()
        reservation.reacquire()
        assert reservation.host == "127.0.0.1"
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
