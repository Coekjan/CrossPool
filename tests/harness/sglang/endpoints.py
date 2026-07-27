"""Atomic ownership of every TCP endpoint derived by one SGLang server."""

from __future__ import annotations

import errno
import hashlib
import socket
from dataclasses import dataclass
from typing import Self

from sglang.srt.server_args import DP_ATTENTION_HANDSHAKE_PORT_DELTA, ZMQ_TCP_PORT_DELTA

from tests.harness.network import TcpEndpointReservation, TcpPortSpace

SGLANG_ZMQ_PORT_OFFSETS = range(7)


@dataclass(frozen=True, slots=True)
class SglangEndpointFamily:
    """Immutable TCP endpoint family derived by pinned SGLang."""

    host: str
    http_port: int
    nccl_port: int
    grpc_port: int
    dp_size: int

    @property
    def ports(self) -> tuple[int, ...]:
        """Return every TCP port SGLang may bind for this topology."""

        if self.dp_size == 1:
            return (self.http_port, self.nccl_port, self.grpc_port)
        zmq_port = self.http_port + ZMQ_TCP_PORT_DELTA
        if zmq_port > 65_535:
            zmq_port = self.http_port - ZMQ_TCP_PORT_DELTA
        return (
            self.http_port,
            self.nccl_port,
            self.grpc_port,
            self.http_port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            *(zmq_port + offset for offset in SGLANG_ZMQ_PORT_OFFSETS),
        )

    def __post_init__(self) -> None:
        if self.dp_size <= 0:
            raise ValueError("SGLang endpoint family dp_size must be positive")
        if any(port <= 0 or port > 65_535 for port in self.ports):
            raise ValueError(f"SGLang endpoint family contains an invalid port: {self.ports}")
        if len(self.ports) != len(set(self.ports)):
            raise ValueError(f"SGLang endpoint family contains duplicate ports: {self.ports}")


@dataclass(slots=True)
class SglangEndpointFamilyLease:
    """Own real listeners and namespace locks for one complete endpoint family."""

    family: SglangEndpointFamily
    tcp_reservations: tuple[TcpEndpointReservation, ...]
    lock_sockets: tuple[socket.socket, ...]
    tcp_released: bool = False
    closed: bool = False

    @classmethod
    def acquire(cls, host: str, *, dp_size: int, port_space: TcpPortSpace) -> Self:
        """Atomically reserve one conflict-free endpoint family."""

        if dp_size <= 0:
            raise ValueError("SGLang endpoint family dp_size must be positive")
        last_collision: OSError | None = None
        for http_port in port_space.candidates():
            reservations: list[TcpEndpointReservation] = []
            locks: list[socket.socket] = []
            try:
                http = TcpEndpointReservation.reserve_exact(host, http_port)
                reservations.append(http)
                nccl = TcpEndpointReservation.reserve(host, port_space=port_space)
                reservations.append(nccl)
                grpc = TcpEndpointReservation.reserve(host, port_space=port_space)
                reservations.append(grpc)
                family = SglangEndpointFamily(host, http.port, nccl.port, grpc.port, dp_size)
                if not all(port_space.is_eligible(port) for port in family.ports):
                    raise ValueError("SGLang endpoint family contains an ineligible port")
                reserved_ports = {http.port, nccl.port, grpc.port}
                for port in family.ports:
                    if port not in reserved_ports:
                        reservations.append(TcpEndpointReservation.reserve_exact(host, port))
                        reserved_ports.add(port)
                locks.extend(reserve_namespace_lock(host, port) for port in family.ports)
                return cls(family, tuple(reservations), tuple(locks))
            except OSError as error:
                for reservation in reservations:
                    reservation.close()
                for lock in locks:
                    lock.close()
                if error.errno != errno.EADDRINUSE:
                    raise
                last_collision = error
            except ValueError:
                for reservation in reservations:
                    reservation.close()
                for lock in locks:
                    lock.close()
            except BaseException:
                for reservation in reservations:
                    reservation.close()
                for lock in locks:
                    lock.close()
                raise
        raise RuntimeError("no eligible SGLang endpoint family is available") from last_collision

    def release_tcp_for_spawn(self) -> None:
        """Release all real listeners exactly once while retaining namespace locks."""

        if self.tcp_released:
            raise RuntimeError("SGLang endpoint family TCP reservations were already released")
        for reservation in self.tcp_reservations:
            reservation.release_for_spawn()
        self.tcp_released = True

    def close(self) -> None:
        """Release every unconsumed listener and namespace lock idempotently."""

        if self.closed:
            return
        self.closed = True
        for reservation in self.tcp_reservations:
            reservation.close()
        for lock in self.lock_sockets:
            lock.close()


def reserve_namespace_lock(host: str, port: int) -> socket.socket:
    """Reserve one Linux abstract-socket lock shared by xpool test runners."""

    identity = hashlib.sha256(f"{host}:{port}".encode()).hexdigest()
    lock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        lock.set_inheritable(False)
        lock.bind(f"\0xpool-test-endpoint-{identity}")
    except BaseException:
        lock.close()
        raise
    return lock
