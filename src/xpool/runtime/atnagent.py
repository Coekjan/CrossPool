"""Attention-side transport arena ownership and publication."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import xpool.native
from xpool.abi import ABI_VERSION
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationPhase, FabricParticipantPhase
from xpool.runtime import RuntimeRole
from xpool.runtime.agent import AGENT_SHUTDOWN_POLL_INTERVAL_S, Agent, AgentError, AgentHeartbeat, logger
from xpool.service.client import XpoolClient, XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    HeartbeatResponse,
    InstanceRegistration,
    ProcessRef,
)
from xpool.transport import TransportArenaHandle

__all__ = ["AtnAgent", "AtnTransportCatalog", "AtnTransportEntry"]

TRANSPORT_PUBLICATION_RECOVERY_S = 60.0
TRANSPORT_SHUTDOWN_DEADLINE_S = 60.0


@dataclass(slots=True)
class AtnTransportEntry:
    """One native arena and its current daemon publication state.

    Attributes:
        instance_id: Configured instance consuming this rank-local arena.
        registration: Latest matching instance registration and geometry.
        handle: Native CUDA IPC arena handle owned by this process.
        published_epoch: AtnAgent registration epoch that accepted this handle.
    """

    instance_id: str
    registration: InstanceRegistration
    handle: TransportArenaHandle
    published_epoch: int | None = None

    def binding(self) -> AtnAgentTransportArenaBinding:
        """Return this entry's daemon publication value."""

        return AtnAgentTransportArenaBinding(
            instance_id=self.instance_id,
            rank=self.registration.rank,
            handle=self.handle,
        )

    def accept_registration(self, registration: InstanceRegistration) -> None:
        """Retain a replacement owner only when native geometry is unchanged."""

        if self.registration.transport != registration.transport:
            raise AgentError(
                f"transport attributes changed for instance {self.instance_id} rank "
                f"{self.registration.rank}; runtime hot resize is unsupported"
            )
        self.registration = registration


class AtnTransportCatalog:
    """Own all rank-local transport arenas for one AtnAgent process."""

    def __init__(
        self,
        *,
        client: XpoolClient,
        cuda_device: int,
        local_rank: int,
        publisher: ProcessRef,
    ) -> None:
        """Create an empty catalog bound to one daemon registration slot."""

        self.client = client
        self.cuda_device = cuda_device
        self.local_rank = local_rank
        self.publisher = publisher
        self.entries: dict[str, AtnTransportEntry] = {}
        self.publication_deadline: float | None = None

    @property
    def resources(self) -> tuple[AtnTransportEntry, ...]:
        """Return a stable snapshot of currently owned entries."""

        return tuple(self.entries.values())

    @property
    def published(self) -> bool:
        """Return whether any owned handle was published to the daemon."""

        return any(entry.published_epoch is not None for entry in self.entries.values())

    def local_registrations(self) -> dict[str, InstanceRegistration]:
        """Return configured instance registrations for this local ATN rank."""

        return {
            registration.instance_id: registration
            for registration in self.client.list_instances()
            if registration.rank == self.local_rank and registration.instance_id in get_global_config().instance_by_id
        }

    def retry_or_raise(self, error: Exception) -> bool:
        """Apply the bounded recovery policy for list and publication failures."""

        deadline = self.publication_deadline or (time.monotonic() + TRANSPORT_PUBLICATION_RECOVERY_S)
        self.publication_deadline = deadline
        if time.monotonic() >= deadline:
            raise AgentError("atnagent transport arenas were not published before the recovery deadline") from error
        logger.warning("waiting to publish atnagent transport arenas: %s", error)
        return False

    def reconcile(self, registration_epoch: int) -> bool:
        """Create and publish every currently registered local instance arena.

        Args:
            registration_epoch: Monotonic epoch of the current AtnAgent daemon
                registration. Handles are republished once per epoch.

        Returns:
            True when every configured instance has a current-epoch publication.

        Raises:
            AgentError: If geometry changes, native creation fails, or bounded
                recoverable publication retries expire.
        """

        try:
            registrations = self.local_registrations()
        except (XpoolDaemonError, XpoolClientError) as error:
            if not error.is_recoverable:
                raise AgentError(f"atnagent instance-list reconcile failed: {error}") from error
            return self.retry_or_raise(error)

        for instance_id, entry in self.entries.items():
            registration = registrations.get(instance_id)
            if registration is not None:
                entry.accept_registration(registration)

        created: list[AtnTransportEntry] = []
        try:
            for instance_index, instance in enumerate(get_global_config().instances):
                registration = registrations.get(instance.id)
                if registration is None or instance.id in self.entries:
                    continue
                handle = TransportArenaHandle(
                    handle=xpool.native.transport.create_arena(
                        instance_index,
                        registration.rank,
                        registration.transport.max_tokens,
                        registration.transport.hidden_size,
                        int(registration.workload.dtype),
                        registration.transport.atn_tp_rank,
                        registration.transport.atn_tp_size,
                        registration.transport.atn_dp_rank,
                        registration.transport.atn_dp_size,
                    )
                )
                created.append(AtnTransportEntry(instance_id=instance.id, registration=registration, handle=handle))
        except Exception as error:
            try:
                self.rollback(created)
            except Exception as cleanup_error:
                raise AgentError("failed to release partially created transport arenas") from cleanup_error
            raise AgentError(f"failed to create transport arenas for CUDA device {self.cuda_device}") from error

        self.entries.update((entry.instance_id, entry) for entry in created)
        publishable = [
            entry
            for entry in self.entries.values()
            if entry.instance_id in registrations and entry.published_epoch != registration_epoch
        ]
        if publishable:
            try:
                self.client.upsert_atnagent_transport_arenas(
                    self.cuda_device,
                    [entry.binding() for entry in publishable],
                    publisher=self.publisher,
                )
            except (XpoolDaemonError, XpoolClientError) as error:
                if not error.is_recoverable:
                    raise AgentError(f"atnagent transport arena upsert failed: {error}") from error
                return self.retry_or_raise(error)
            for entry in publishable:
                entry.published_epoch = registration_epoch
            self.publication_deadline = None

        configured = frozenset(get_global_config().instance_by_id)
        complete = configured == frozenset(registrations) and all(
            entry.published_epoch == registration_epoch for entry in self.entries.values()
        )
        if not complete:
            missing = configured - frozenset(registrations)
            if missing:
                logger.info(
                    "waiting for local instance registrations on CUDA device %s rank %s; missing instances: %s",
                    self.cuda_device,
                    self.local_rank,
                    ", ".join(sorted(missing)),
                )
        return complete

    def activate(self) -> None:
        """Launch the sole Resident over the immutable local arena set."""

        try:
            xpool.native.transport.activate()
        except Exception as error:
            raise AgentError(f"transport Resident launch failed on CUDA device {self.cuda_device}") from error

    def check_health(self) -> None:
        """Reject unexpected completion of the process-wide Transport Resident."""

        try:
            xpool.native.transport.check_health()
        except Exception as error:
            raise AgentError(f"transport Resident failed on CUDA device {self.cuda_device}") from error

    def quiesce_leases(self) -> None:
        """Close lease admission and wait for every live Instance owner."""

        deadline = time.monotonic() + TRANSPORT_SHUTDOWN_DEADLINE_S
        while time.monotonic() < deadline:
            try:
                response = self.client.quiesce_atnagent_transport_leases(
                    self.cuda_device,
                    publisher=self.publisher,
                )
            except (XpoolClientError, XpoolDaemonError) as error:
                if not error.is_recoverable:
                    raise AgentError("failed to quiesce AtnAgent transport leases") from error
                logger.warning("failed to quiesce AtnAgent transport leases: %s", error)
                time.sleep(AGENT_SHUTDOWN_POLL_INTERVAL_S)
                continue
            if not response.in_use:
                return
            logger.info(
                "waiting for transport arena leases on rank %s: %s",
                self.local_rank,
                ", ".join(f"{entry.instance_id}:{entry.rank}" for entry in response.in_use),
            )
            time.sleep(AGENT_SHUTDOWN_POLL_INTERVAL_S)
        raise AgentError("timed out quiescing transport leases; native arenas remain allocated")

    def drain(self) -> None:
        """Drain the sole process-wide Resident against one deadline."""

        if not self.entries:
            return
        xpool.native.transport.drain_async()
        deadline = time.monotonic() + TRANSPORT_SHUTDOWN_DEADLINE_S
        while time.monotonic() < deadline:
            if not xpool.native.transport.drain_pending():
                return
            if time.monotonic() < deadline:
                time.sleep(AGENT_SHUTDOWN_POLL_INTERVAL_S)
        raise AgentError(f"timed out draining Transport Resident on CUDA device {self.cuda_device}")

    def rollback(self, entries: Sequence[AtnTransportEntry]) -> None:
        """Destroy newly created Dormant arenas before Resident activation."""

        if not entries:
            return
        xpool.native.transport.destroy_arenas([entry.handle.handle for entry in entries])

    def quiesce(self) -> None:
        """Stop arena leases and drain every local transport arena."""

        if self.published:
            self.quiesce_leases()
        self.drain()

    def close(self) -> None:
        """Destroy and forget every previously drained Transport arena."""

        resources = self.resources
        if resources:
            xpool.native.transport.destroy_arenas([entry.handle.handle for entry in resources])
        self.entries.clear()


class AtnAgent(Agent):
    """Attention-side agent owning one rank-local transport catalog."""

    def __init__(self, *, cuda_device: int) -> None:
        """Create an AtnAgent for one configured attention CUDA device."""

        super().__init__(cuda_device=cuda_device, runtime_role=RuntimeRole.ATNAGENT)
        self.registration = AtnAgentRegistration(
            cuda_device=cuda_device,
            abi_version=ABI_VERSION,
            pid=self.proc_id.pid,
        )
        try:
            self.local_rank = get_global_config().devices.atn_cuda_devices.index(cuda_device)
        except ValueError as error:
            raise AgentError(f"CUDA device {cuda_device} has no local instance-rank arenas") from error
        self.registration_epoch = 0
        self.catalog = AtnTransportCatalog(
            client=self.client,
            cuda_device=cuda_device,
            local_rank=self.local_rank,
            publisher=self.process_ref,
        )
        self.heartbeat_worker = AgentHeartbeat(agent=self)

    def register(self) -> None:
        """Register this AtnAgent and begin a new publication epoch."""

        try:
            self.client.register_atnagent(self.registration)
        except (XpoolDaemonError, XpoolClientError) as error:
            self.registered = False
            if not error.is_recoverable:
                raise AgentError(f"AtnAgent registration received unrecoverable daemon error: {error}") from error
            logger.warning("AtnAgent registration failed: %s", error)
            return
        self.registered = True
        self.registration_epoch += 1

    def send_heartbeat(self) -> HeartbeatResponse:
        """Publish this AtnAgent's heartbeat."""

        return self.client.heartbeat_atnagent(self.cuda_device, self.process_ref)

    def prepare_fabric(self) -> bool:
        """Reconcile all local arenas before collective Fabric join."""

        return self.catalog.reconcile(self.registration_epoch)

    def activate_fabric_plan(self) -> None:
        """Launch transport kernels after collective Fabric join."""

        self.catalog.activate()

    def quiesce_fabric(self) -> None:
        """Drain local leases and transport kernels before Fabric drain."""

        self.catalog.quiesce()

    def poll_fabric_health(self) -> None:
        """Check both Fabric and the process-wide Transport Resident."""

        super().poll_fabric_health()
        report = self.participant_report
        if (
            report is not None
            and report.phase is FabricParticipantPhase.ACTIVE
            and report.invocation_failure is None
            and self.fabric_phase in {FabricGenerationPhase.JOINING, FabricGenerationPhase.EXECUTABLE}
        ):
            try:
                self.catalog.check_health()
            except AgentError as error:
                self.report_local_protocol_failure(str(error))
                raise

    def close_role(self) -> None:
        """Release every local transport arena."""

        if self.registered and self.catalog.published:
            self.catalog.quiesce_leases()
        self.catalog.close()
