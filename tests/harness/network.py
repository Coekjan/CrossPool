"""Exclusive TCP endpoint reservations shared by test harnesses."""

from __future__ import annotations

import errno
import secrets
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True, slots=True)
class TcpPortSpace:
    """Immutable explicit-bind policy of one Linux network namespace."""

    unprivileged_port_start: int
    ephemeral: range
    administratively_reserved: frozenset[int]

    @classmethod
    def parse(
        cls,
        unprivileged_port_start: str,
        ephemeral: str,
        administratively_reserved: str,
    ) -> Self:
        """Parse Linux TCP port policy values read from ``/proc/sys``."""

        try:
            unprivileged_tokens = unprivileged_port_start.split()
            if len(unprivileged_tokens) != 1:
                raise ValueError
            unprivileged_start = int(unprivileged_tokens[0])

            ephemeral_tokens = ephemeral.split()
            if len(ephemeral_tokens) != 2:
                raise ValueError
            ephemeral_start, ephemeral_stop = map(int, ephemeral_tokens)

            reserved: set[int] = set()
            reserved_text = administratively_reserved.strip()
            if reserved_text:
                for item in reserved_text.split(","):
                    bounds = item.strip().split("-")
                    if len(bounds) == 1:
                        start = stop = int(bounds[0])
                    elif len(bounds) == 2:
                        start, stop = map(int, bounds)
                    else:
                        raise ValueError
                    if start <= 0 or stop > 65_535 or start > stop:
                        raise ValueError
                    reserved.update(range(start, stop + 1))
        except ValueError as error:
            raise ValueError("invalid Linux TCP port-space configuration") from error

        if (
            unprivileged_start < 0
            or unprivileged_start > 65_535
            or ephemeral_start <= 0
            or ephemeral_stop > 65_535
            or ephemeral_start > ephemeral_stop
        ):
            raise ValueError("invalid Linux TCP port-space configuration")
        return cls(
            unprivileged_port_start=unprivileged_start,
            ephemeral=range(ephemeral_start, ephemeral_stop + 1),
            administratively_reserved=frozenset(reserved),
        )

    @classmethod
    def local(cls) -> Self:
        """Read the explicit-bind policy of the current Linux network namespace."""

        return cls.parse(
            Path("/proc/sys/net/ipv4/ip_unprivileged_port_start").read_text(encoding="utf-8"),
            Path("/proc/sys/net/ipv4/ip_local_port_range").read_text(encoding="utf-8"),
            Path("/proc/sys/net/ipv4/ip_local_reserved_ports").read_text(encoding="utf-8"),
        )

    def is_eligible(self, port: int) -> bool:
        """Return whether ``port`` is eligible for a delayed explicit bind."""

        return (
            1 <= port <= 65_535
            and port >= self.unprivileged_port_start
            and port not in self.ephemeral
            and port not in self.administratively_reserved
        )

    def candidates(self) -> Iterator[int]:
        """Visit every eligible port once from a cryptographically random start."""

        eligible = tuple(port for port in range(1, 65_536) if self.is_eligible(port))
        if not eligible:
            return
        start = secrets.randbelow(len(eligible))
        yield from eligible[start:]
        yield from eligible[:start]


@dataclass(slots=True)
class TcpEndpointReservation:
    """Own one bound TCP listener until immediately before process spawn."""

    address: tuple[str, int]
    listener: socket.socket | None

    @classmethod
    def reserve(cls, host: str, *, port_space: TcpPortSpace) -> Self:
        """Reserve the first available endpoint in one finite candidate traversal."""

        last_collision: OSError | None = None
        for port in port_space.candidates():
            try:
                return cls.reserve_exact(host, port)
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
                last_collision = error
        raise RuntimeError("no eligible TCP endpoint is available") from last_collision

    @classmethod
    def reserve_exact(cls, host: str, port: int) -> Self:
        """Reserve one exact TCP endpoint."""

        if port <= 0 or port > 65_535:
            raise ValueError(f"TCP endpoint port must be in 1..65535, got {port}")
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        try:
            listener.set_inheritable(False)
            listener.bind((host, port))
            listener.listen()
        except BaseException:
            listener.close()
            raise
        return cls(address=(host, port), listener=listener)

    @property
    def host(self) -> str:
        """Return the reserved bind host."""

        return self.address[0]

    @property
    def port(self) -> int:
        """Return the reserved TCP port."""

        return self.address[1]

    def release_for_spawn(self) -> None:
        """Release the listener exactly once immediately before process spawn."""

        if self.listener is None:
            raise RuntimeError(f"TCP endpoint {self.host}:{self.port} was already released")
        self.listener.close()
        self.listener = None

    def close(self) -> None:
        """Release an unconsumed listener idempotently."""

        if self.listener is not None:
            self.listener.close()
            self.listener = None
