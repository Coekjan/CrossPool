"""Control-plane error types shared by daemon and clients."""

from __future__ import annotations

from http import HTTPStatus
from typing import Literal

type XpoolClientErrorKind = Literal["transport", "status", "protocol"]
type XpoolDaemonErrorKind = Literal["conflict", "not_found", "not_ready"]


class XpoolClientError(RuntimeError):
    """Client-local daemon API error."""

    kind: XpoolClientErrorKind
    status_code: HTTPStatus | None

    def __init__(
        self,
        kind: XpoolClientErrorKind,
        message: str,
        *,
        status_code: HTTPStatus | None = None,
    ) -> None:
        """Create an error for local transport, HTTP, or protocol failures."""

        super().__init__(message)
        self.kind = kind
        self.status_code = status_code

    @property
    def is_recoverable(self) -> bool:
        """Return whether retrying the request may succeed without local repair."""

        return self.kind == "transport" or self.status_code in {
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.GATEWAY_TIMEOUT,
        }


class XpoolDaemonError(RuntimeError):
    """Daemon-domain error converted to HTTP or raised by daemon clients."""

    kind: XpoolDaemonErrorKind
    status_code: HTTPStatus | None

    def __init__(
        self,
        kind: XpoolDaemonErrorKind,
        message: str,
        *,
        status_code: HTTPStatus | None = None,
    ) -> None:
        """Create a daemon-domain error with a general kind and message."""

        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code

    @property
    def is_recoverable(self) -> bool:
        """Return whether the daemon explicitly reported transient unavailability."""

        return self.kind == "not_ready"
