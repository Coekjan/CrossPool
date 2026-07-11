"""Typed daemon HTTP client used by xpool runtime participants."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from http import HTTPStatus
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from xpool.abi import TransportArenaHandle
from xpool.config import get_global_config
from xpool.service.errors import XpoolClientError, XpoolDaemonError
from xpool.service.wire import (
    ControlPlaneWarning,
    DevagentRegistration,
    DevagentTransportArenaBinding,
    DevagentTransportArenaDrainResponse,
    DevagentTransportArenaUpsertRequest,
    HeartbeatResponse,
    InstanceRegistration,
    ProcessHeartbeat,
    ProcessRef,
    ReadinessScope,
    ReadinessSnapshot,
    TransportArenaHandleRecord,
    XpoolDaemonErrorDetail,
)

__all__ = ["XpoolClient"]

DAEMON_HTTP_TIMEOUT_S = 5.0
DEVAGENT_TRANSPORT_DRAIN_TIMEOUT_S = 20.0
DAEMON_HEALTH_RETRY_ATTEMPTS = 3
DAEMON_HEALTH_RETRY_DELAY_S = 0.5
logger = logging.getLogger(__name__)


def daemon_error_detail(response: httpx.Response) -> XpoolDaemonErrorDetail | None:
    """Decode a structured daemon error detail when one is present."""

    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if not isinstance(detail, dict):
        return None
    try:
        return XpoolDaemonErrorDetail.model_validate(detail)
    except ValidationError:
        return None


def error_from_status(response: httpx.Response) -> XpoolClientError | XpoolDaemonError:
    """Translate one failed HTTP response into an xpool client-domain error."""

    try:
        status_code = HTTPStatus(response.status_code)
        status_text = f"{status_code.value} {status_code.phrase}"
    except ValueError:
        status_code = None
        status_text = str(response.status_code)
    daemon_error = daemon_error_detail(response)
    if daemon_error is None:
        return XpoolClientError(
            "status",
            f"xpool daemon returned HTTP {status_text}",
            status_code=status_code,
        )
    return XpoolDaemonError(
        daemon_error.kind,
        daemon_error.message,
        status_code=status_code,
    )


class XpoolClient:
    """Synchronous client for the xpool daemon control-plane API."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        timeout_s: float = DAEMON_HTTP_TIMEOUT_S,
    ) -> None:
        """Create a daemon API client from process-global configuration.

        Args:
            http_client: Optional preconfigured client used by test harnesses.
                The xpool client assumes ownership and closes it with
                :meth:`close`.
            timeout_s: Per-request HTTP timeout in seconds.

        Raises:
            XpoolClientError: If the daemon health route remains unreachable or
                non-OK after bounded retries.
        """

        if http_client is None:
            config = get_global_config()
            daemon_host = f"[{config.daemon.host}]" if ":" in config.daemon.host else config.daemon.host
            http_client = httpx.Client(
                base_url=f"http://{daemon_host}:{config.daemon.port}",
                timeout=timeout_s,
            )
        self.http_client = http_client
        try:
            health_error: XpoolClientError | XpoolDaemonError | None = None
            for attempt in range(DAEMON_HEALTH_RETRY_ATTEMPTS):
                try:
                    self.health()
                    health_error = None
                    break
                except (XpoolClientError, XpoolDaemonError) as exc:
                    health_error = exc
                    if not exc.is_recoverable:
                        raise
                    message = str(exc)
                logger.warning(
                    "daemon health check failed on attempt %s/%s: %s",
                    attempt + 1,
                    DAEMON_HEALTH_RETRY_ATTEMPTS,
                    message,
                )
                if attempt + 1 < DAEMON_HEALTH_RETRY_ATTEMPTS:
                    time.sleep(DAEMON_HEALTH_RETRY_DELAY_S)
            else:
                if health_error is None:
                    raise RuntimeError("daemon health retry loop completed without an error")
                raise health_error
        except Exception:
            self.http_client.close()
            raise

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self.http_client.close()

    def log_warnings(self, warnings: list[ControlPlaneWarning]) -> None:
        """Log warnings returned by a daemon control-plane operation."""

        for warning in warnings:
            logger.warning(
                "xpool daemon warning [%s cuda_device=%s]: %s",
                warning.kind,
                warning.cuda_device,
                warning.message,
            )

    def request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: dict[str, int | str | list[str]] | None = None,
        json: object | None = None,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        """Issue one daemon request and translate transport or status failures."""

        try:
            if timeout_s is None:
                response = self.http_client.request(method, path, params=params, json=json)
            else:
                response = self.http_client.request(method, path, params=params, json=json, timeout=timeout_s)
        except httpx.HTTPError as exc:
            raise XpoolClientError("transport", f"xpool daemon request failed: {exc}") from exc
        if response.is_error:
            raise error_from_status(response)
        return response

    def decode_json(self, response: httpx.Response, context: str) -> object:
        """Decode a daemon JSON response or raise a protocol error."""

        try:
            return response.json()
        except ValueError as exc:
            raise XpoolClientError("protocol", f"xpool daemon returned invalid JSON for {context}") from exc

    def decode_model[M: BaseModel](self, response: httpx.Response, model: type[M], context: str) -> M:
        """Validate a daemon response against one Pydantic wire model."""

        try:
            return model.model_validate(self.decode_json(response, context))
        except ValidationError as exc:
            raise XpoolClientError("protocol", f"xpool daemon returned invalid {context} response") from exc

    def health(self) -> HTTPStatus:
        """Check daemon liveness.

        Returns:
            ``HTTPStatus.OK`` when the daemon health route responds.

        Raises:
            XpoolClientError: If the daemon is unreachable or returns an
                unstructured non-OK status.
            XpoolDaemonError: If the daemon returns a structured daemon-domain error.
        """

        self.request("GET", "/health")
        return HTTPStatus.OK

    def readiness(self, scopes: Sequence[ReadinessScope] = ()) -> ReadinessSnapshot:
        """Return daemon readiness details for selected participant scopes.

        Args:
            scopes: Participant scopes to query. An empty sequence selects all
                scopes on the daemon.
        """

        params: dict[str, int | str | list[str]] | None = (
            None if not scopes else {"scope": [scope.value for scope in scopes]}
        )
        response = self.request("GET", "/ready", params=params)
        return self.decode_model(response, ReadinessSnapshot, "readiness")

    def check_config(self) -> None:
        """Require the daemon to accept this client's effective config.

        Raises:
            XpoolClientError: If the daemon cannot be reached or returns an
                invalid error response.
            XpoolDaemonError: If the daemon rejects the effective config.
        """

        self.request(
            "POST",
            "/config/check",
            json=get_global_config().model_dump(mode="json"),
        )

    def register_devagent(self, registration: DevagentRegistration) -> None:
        """Register one devagent process with the daemon.

        Args:
            registration: Devagent registration payload to send.

        Raises:
            XpoolClientError: If config validation or registration cannot reach
                the daemon or receives an invalid response.
            XpoolDaemonError: If the daemon rejects the effective config or
                registration.
        """

        self.check_config()
        self.request("POST", "/devagent/register", json=registration.model_dump(mode="json"))

    def heartbeat_devagent(self, cuda_device: int, heartbeat: ProcessHeartbeat) -> HeartbeatResponse:
        """Refresh one devagent heartbeat and log daemon warnings."""

        response = self.request("POST", f"/devagent/{cuda_device}/heartbeat", json=heartbeat.model_dump(mode="json"))
        heartbeat_response = self.decode_model(response, HeartbeatResponse, "devagent heartbeat")
        self.log_warnings(heartbeat_response.warnings)
        return heartbeat_response

    def upsert_devagent_transport_arenas(
        self,
        cuda_device: int,
        bindings: list[DevagentTransportArenaBinding],
        *,
        publisher: ProcessRef,
    ) -> None:
        """Merge devagent transport arena bindings into the daemon.

        Args:
            cuda_device: CUDA device owned by the publishing devagent.
            bindings: Transport arena bindings to upsert.
            publisher: Process identity for the publishing devagent.
        """

        self.request(
            "POST",
            f"/devagent/{cuda_device}/transport-arenas",
            json=DevagentTransportArenaUpsertRequest(
                publisher=publisher,
                bindings=bindings,
            ).model_dump(mode="json"),
        )

    def drain_devagent_transport_arenas(
        self,
        cuda_device: int,
        *,
        publisher: ProcessRef,
    ) -> DevagentTransportArenaDrainResponse:
        """Drain one devagent's transport arenas and return active leases."""

        response = self.request(
            "POST",
            f"/devagent/{cuda_device}/transport-arenas/drain",
            json=publisher.model_dump(mode="json"),
            timeout_s=DEVAGENT_TRANSPORT_DRAIN_TIMEOUT_S,
        )
        return self.decode_model(response, DevagentTransportArenaDrainResponse, "devagent transport arena drain")

    def list_instances(self) -> list[InstanceRegistration]:
        """Return instance-rank registrations from the daemon."""

        response = self.request("GET", "/instances")
        payload = self.decode_json(response, "instance list")
        if not isinstance(payload, list):
            raise XpoolClientError("protocol", "xpool daemon returned invalid instance list response")
        try:
            return [InstanceRegistration.model_validate(item) for item in payload]
        except (TypeError, ValidationError) as exc:
            raise XpoolClientError("protocol", "xpool daemon returned invalid instance list response") from exc

    def register_instance(self, registration: InstanceRegistration) -> None:
        """Register one instance-rank process with the daemon.

        Args:
            registration: Instance-rank registration payload to send.

        Raises:
            XpoolClientError: If config validation or registration cannot reach
                the daemon or receives an invalid response.
            XpoolDaemonError: If the daemon rejects the effective config or
                registration.
        """

        self.check_config()
        self.request("POST", "/instance/register", json=registration.model_dump(mode="json"))

    def deregister_instance(self, instance_id: str, *, rank: int, owner: ProcessRef) -> None:
        """Remove one instance-rank registration owned by this process.

        Args:
            instance_id: Instance id whose registration should be removed.
            rank: Rank-local process index to deregister.
            owner: Process identity that must match the current registration.
        """

        instance_path = quote(instance_id, safe="/")
        self.request(
            "POST",
            f"/instance/{instance_path}/deregister",
            params={"rank": rank},
            json=owner.model_dump(mode="json"),
        )

    def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: ProcessHeartbeat) -> HeartbeatResponse:
        """Refresh one instance-rank heartbeat and log daemon warnings."""

        instance_path = quote(instance_id, safe="/")
        response = self.request(
            "POST",
            f"/instance/{instance_path}/heartbeat",
            params={"rank": rank},
            json=heartbeat.model_dump(mode="json"),
        )
        heartbeat_response = self.decode_model(response, HeartbeatResponse, "instance heartbeat")
        self.log_warnings(heartbeat_response.warnings)
        return heartbeat_response

    def acquire_instance_transport_arena(
        self,
        instance_id: str,
        *,
        rank: int,
        owner: ProcessRef,
    ) -> TransportArenaHandle:
        """Attempt one daemon-brokered transport arena acquisition.

        Args:
            instance_id: Instance id whose arena should be fetched.
            rank: Local instance rank whose transport arena should be fetched.
            owner: Process identity for the acquiring instance rank.
        Returns:
            Transport arena handle for the requested instance rank.

        Raises:
            XpoolClientError: If the daemon request fails with a local error.
            XpoolDaemonError: If the daemon rejects the acquisition.
        """

        instance_path = quote(instance_id, safe="/")
        response = self.request(
            "POST",
            f"/instance/{instance_path}/transport-arena/acquire",
            params={"rank": rank},
            json=owner.model_dump(mode="json"),
        )
        return self.decode_model(response, TransportArenaHandleRecord, "transport arena handle").to_handle()
