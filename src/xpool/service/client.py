"""Typed daemon HTTP client used by CrossPool runtime participants."""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from xpool.config import get_global_config
from xpool.fabric import FabricGenerationId, FabricPlan
from xpool.service.errors import XpoolClientError, XpoolDaemonError
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    AtnAgentTransportArenaUpsertRequest,
    AtnAgentTransportLeaseQuiesceResponse,
    FabricParticipantReport,
    FabricQuiesceRequest,
    FfnAgentRegistration,
    HeartbeatResponse,
    InstanceRankInitializedPublication,
    InstanceRankRegistration,
    KvControlChannelRef,
    ProcessRef,
    ReadinessSnapshot,
    XpoolDaemonErrorDetail,
)
from xpool.transport import TransportArenaHandle

__all__ = ["XpoolClient"]

DAEMON_HTTP_TIMEOUT_S = 5.0
ATNAGENT_TRANSPORT_LEASE_QUIESCE_TIMEOUT_S = 20.0
DAEMON_HEALTH_RETRY_ATTEMPTS = 3
DAEMON_HEALTH_RETRY_DELAY_S = 0.5
logger = logging.getLogger(__name__)


class XpoolClient:
    """Synchronous client for the CrossPool daemon control-plane API."""

    def __init__(
        self,
        *,
        timeout_s: float = DAEMON_HTTP_TIMEOUT_S,
    ) -> None:
        """Create a daemon API client from process-global configuration.

        Args:
            timeout_s: Per-request HTTP timeout in seconds.

        Raises:
            XpoolClientError: If the daemon health route remains unreachable or
                non-OK after bounded retries.
            XpoolDaemonError: If the daemon reports an unrecoverable health
                failure.
        """

        config = get_global_config()
        daemon_host = f"[{config.daemon.host}]" if ":" in config.daemon.host else config.daemon.host
        self.http_client = httpx.Client(
            base_url=f"http://{daemon_host}:{config.daemon.port}",
            timeout=timeout_s,
        )
        try:
            self.wait_for_health()
        except Exception:
            self.http_client.close()
            raise

    def wait_for_health(self) -> None:
        """Require daemon health within the bounded retry policy."""

        for attempt in range(DAEMON_HEALTH_RETRY_ATTEMPTS):
            try:
                self.health()
                return
            except (XpoolClientError, XpoolDaemonError) as error:
                if not error.is_recoverable:
                    raise
                logger.debug(
                    "daemon health check failed on attempt %s/%s: %s",
                    attempt + 1,
                    DAEMON_HEALTH_RETRY_ATTEMPTS,
                    error,
                )
                if attempt + 1 == DAEMON_HEALTH_RETRY_ATTEMPTS:
                    raise
                time.sleep(DAEMON_HEALTH_RETRY_DELAY_S)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self.http_client.close()

    @staticmethod
    def error_from_response(response: httpx.Response) -> XpoolClientError | XpoolDaemonError:
        """Translate one failed HTTP response into a CrossPool client-domain error."""

        try:
            status_code = HTTPStatus(response.status_code)
            status_text = f"{status_code.value} {status_code.phrase}"
        except ValueError:
            status_code = None
            status_text = str(response.status_code)
        try:
            payload = response.json()
            detail = payload.get("detail") if isinstance(payload, dict) else None
            daemon_error = XpoolDaemonErrorDetail.model_validate(detail) if isinstance(detail, dict) else None
        except (ValueError, ValidationError):
            daemon_error = None
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
            raise self.error_from_response(response)
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

    def readiness(self) -> ReadinessSnapshot:
        """Return the single global daemon readiness snapshot."""

        response = self.request("GET", "/ready")
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

    def fabric_plan(self) -> FabricPlan:
        """Return the immutable allocation and execution plan for the generation."""

        response = self.request("GET", "/fabric/plan")
        return self.decode_model(response, FabricPlan, "fabric plan")

    def kv_control_channel(self, generation: FabricGenerationId) -> KvControlChannelRef:
        """Return the native KV control channel for one current generation."""

        response = self.request("GET", f"/kv/control-channel/{quote(generation.format(), safe='')}")
        return self.decode_model(response, KvControlChannelRef, "kv control channel")

    def request_fabric_quiesce(self, request: FabricQuiesceRequest) -> None:
        """Ask the daemon to stop admission for one retained generation."""

        self.request("POST", "/fabric/quiesce", json=request.model_dump(mode="json"))

    def report_fabric_participant(self, report: FabricParticipantReport) -> None:
        """Commit one self-contained Fabric participant report."""

        self.request("POST", "/fabric/participant-reports", json=report.model_dump(mode="json"))

    def register_atnagent(self, registration: AtnAgentRegistration) -> None:
        """Register one AtnAgent process with the daemon.

        Args:
            registration: AtnAgent registration payload to send.

        Raises:
            XpoolClientError: If config validation or registration cannot reach
                the daemon or receives an invalid response.
            XpoolDaemonError: If the daemon rejects the effective config or
                registration.
        """

        self.check_config()
        self.request("POST", "/atnagent/register", json=registration.model_dump(mode="json"))

    def register_ffnagent(self, registration: FfnAgentRegistration) -> None:
        """Register one FfnAgent process with the daemon.

        Args:
            registration: FfnAgent registration payload to send.

        Raises:
            XpoolClientError: If config validation or registration cannot reach
                the daemon or receives an invalid response.
            XpoolDaemonError: If the daemon rejects the registration.
        """

        self.check_config()
        self.request("POST", "/ffnagent/register", json=registration.model_dump(mode="json"))

    def heartbeat_ffnagent(self, cuda_device: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh one FfnAgent heartbeat and return daemon warnings."""

        response = self.request("POST", f"/ffnagent/{cuda_device}/heartbeat", json=heartbeat.model_dump(mode="json"))
        return self.decode_model(response, HeartbeatResponse, "ffnagent heartbeat")

    def heartbeat_atnagent(self, cuda_device: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh one AtnAgent heartbeat and return daemon warnings.

        Args:
            cuda_device: CUDA device owned by the registered AtnAgent.
            heartbeat: Process identity used to prove registration ownership.

        Returns:
            Validated heartbeat response from the daemon.

        Raises:
            XpoolClientError: If the request fails or the response is invalid.
            XpoolDaemonError: If the daemon rejects the heartbeat.

        """

        response = self.request("POST", f"/atnagent/{cuda_device}/heartbeat", json=heartbeat.model_dump(mode="json"))
        return self.decode_model(response, HeartbeatResponse, "atnagent heartbeat")

    def upsert_atnagent_transport_arenas(
        self,
        cuda_device: int,
        bindings: list[AtnAgentTransportArenaBinding],
        *,
        publisher: ProcessRef,
    ) -> None:
        """Merge AtnAgent transport arena bindings into the daemon.

        Args:
            cuda_device: CUDA device owned by the publishing AtnAgent.
            bindings: Transport arena bindings to upsert.
            publisher: Process identity for the publishing AtnAgent.

        Raises:
            XpoolClientError: If the request fails.
            XpoolDaemonError: If the daemon rejects the publication.

        Side Effects:
            Adds or replaces daemon-brokered arena bindings owned by the
            publisher.
        """

        self.request(
            "POST",
            f"/atnagent/{cuda_device}/transport-arenas",
            json=AtnAgentTransportArenaUpsertRequest(
                publisher=publisher,
                bindings=bindings,
            ).model_dump(mode="json"),
        )

    def quiesce_atnagent_transport_leases(
        self,
        cuda_device: int,
        *,
        publisher: ProcessRef,
    ) -> AtnAgentTransportLeaseQuiesceResponse:
        """Close one AtnAgent's lease admission and return active leases.

        Args:
            cuda_device: CUDA device owned by the publishing AtnAgent.
            publisher: Process identity that owns the arena generation.

        Returns:
            Instance ranks that still hold fresh arena leases.

        Raises:
            XpoolClientError: If the bounded quiesce request fails or its
                response is invalid.
            XpoolDaemonError: If the daemon rejects the quiesce request.

        Side Effects:
            Closes lease admission for the arena generation and may stop
            processes that retain stale leases.
        """

        response = self.request(
            "POST",
            f"/atnagent/{cuda_device}/transport-leases/quiesce",
            json=publisher.model_dump(mode="json"),
            timeout_s=ATNAGENT_TRANSPORT_LEASE_QUIESCE_TIMEOUT_S,
        )
        return self.decode_model(
            response,
            AtnAgentTransportLeaseQuiesceResponse,
            "atnagent transport lease quiesce",
        )

    def list_instances(self) -> list[InstanceRankRegistration]:
        """Return instance-rank registrations from the daemon."""

        response = self.request("GET", "/instances")
        payload = self.decode_json(response, "instance list")
        if not isinstance(payload, list):
            raise XpoolClientError("protocol", "xpool daemon returned invalid instance list response")
        try:
            return [InstanceRankRegistration.model_validate(item) for item in payload]
        except (TypeError, ValidationError) as exc:
            raise XpoolClientError("protocol", "xpool daemon returned invalid instance list response") from exc

    def register_instance(self, registration: InstanceRankRegistration) -> None:
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

    def publish_instance_initialized(
        self,
        instance_id: str,
        *,
        rank: int,
        publication: InstanceRankInitializedPublication,
    ) -> None:
        """Publish one SGLang rank's post-initialize startup barrier."""

        instance_path = quote(instance_id, safe="/")
        self.request(
            "POST",
            f"/instance/{instance_path}/initialized",
            params={"rank": rank},
            json=publication.model_dump(mode="json"),
        )

    def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh one instance-rank heartbeat and return daemon warnings."""

        instance_path = quote(instance_id, safe="/")
        response = self.request(
            "POST",
            f"/instance/{instance_path}/heartbeat",
            params={"rank": rank},
            json=heartbeat.model_dump(mode="json"),
        )
        return self.decode_model(response, HeartbeatResponse, "instance heartbeat")

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
        try:
            return TypeAdapter(TransportArenaHandle).validate_python(
                self.decode_json(response, "transport arena handle")
            )
        except ValidationError as exc:
            raise XpoolClientError("protocol", "xpool daemon returned invalid transport arena handle") from exc
