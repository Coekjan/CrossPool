from __future__ import annotations

import errno
import socket

import pytest
from sglang.srt.server_args import DP_ATTENTION_HANDSHAKE_PORT_DELTA, ZMQ_TCP_PORT_DELTA

from tests.harness.runner.network import TcpEndpointReservation, TcpPortSpace
from tests.harness.sglang.endpoints import (
    SglangEndpointFamily,
    SglangEndpointFamilyLease,
    reserve_namespace_lock,
)


def test_endpoint_family_projects_pinned_sglang_dp_ports() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 2)

    assert family.ports == (
        20_000,
        21_000,
        22_000,
        20_000 + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
        *(20_000 + ZMQ_TCP_PORT_DELTA + offset for offset in range(7)),
    )


def test_endpoint_family_dp_one_owns_exact_root_endpoints() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)

    assert family.ports == (20_000, 21_000, 22_000)


def test_endpoint_family_rejects_duplicate_root_endpoints() -> None:
    with pytest.raises(ValueError, match="duplicate ports"):
        SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 20_000, 1)


def test_endpoint_family_lease_reserves_and_releases_every_tcp_port() -> None:
    port_space = TcpPortSpace.local()
    lease = SglangEndpointFamilyLease.acquire("127.0.0.1", dp_size=2, port_space=port_space)
    try:
        assert all(port_space.is_eligible(port) for port in lease.family.ports)
        for port in lease.family.ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
                with pytest.raises(OSError) as error:
                    competitor.bind((lease.family.host, port))
                assert error.value.errno == errno.EADDRINUSE

        lease.release_tcp_for_spawn()
        for port in lease.family.ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
                competitor.bind((lease.family.host, port))
    finally:
        lease.close()


def test_endpoint_namespace_lock_survives_tcp_release() -> None:
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=1,
        port_space=TcpPortSpace.local(),
    )
    try:
        lease.release_tcp_for_spawn()
        with pytest.raises(OSError) as error:
            reserve_namespace_lock(lease.family.host, lease.family.http_port)
        assert error.value.errno == errno.EADDRINUSE
    finally:
        lease.close()


def test_endpoint_family_reacquires_all_released_ports_and_reports_conflicts() -> None:
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=2,
        port_space=TcpPortSpace.local(),
    )
    occupied_ports = (lease.family.ports[1], lease.family.ports[-1])
    competitors: list[socket.socket] = []
    try:
        lease.release_tcp_for_spawn()
        for port in occupied_ports:
            competitor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            competitor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            competitor.bind((lease.family.host, port))
            competitor.listen()
            competitors.append(competitor)

        assert lease.reacquire_tcp() == occupied_ports
        for reservation in lease.tcp_reservations:
            if reservation.port in occupied_ports:
                assert reservation.listener is None
            else:
                assert reservation.listener is not None
    finally:
        for competitor in competitors:
            competitor.close()
        lease.close()


def test_endpoint_family_reacquire_propagates_non_conflict_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=1,
        port_space=TcpPortSpace.local(),
    )
    failed_port = lease.family.nccl_port
    original_reacquire = TcpEndpointReservation.reacquire

    def reacquire(reservation: TcpEndpointReservation) -> None:
        if reservation.port == failed_port:
            raise OSError(errno.EACCES, "permission denied")
        original_reacquire(reservation)

    monkeypatch.setattr(TcpEndpointReservation, "reacquire", reacquire)
    try:
        with pytest.raises(RuntimeError, match="were not released"):
            lease.reacquire_tcp()
        lease.release_tcp_for_spawn()
        with pytest.raises(OSError) as error:
            lease.reacquire_tcp()
        assert error.value.errno == errno.EACCES
    finally:
        lease.close()


def test_endpoint_family_acquisition_exhausts_each_http_candidate_once() -> None:
    occupied = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    port_space = TcpPortSpace(
        unprivileged_port_start=occupied.port,
        ephemeral=range(occupied.port + 1, 65_536),
        administratively_reserved=frozenset(),
    )
    try:
        with pytest.raises(RuntimeError, match="no eligible SGLang endpoint family"):
            SglangEndpointFamilyLease.acquire("127.0.0.1", dp_size=1, port_space=port_space)
    finally:
        occupied.close()
