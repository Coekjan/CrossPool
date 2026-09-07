"""Attention-side transport arena ownership and publication."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationPhase, FabricParticipantPhase
from xpool.native import ABI_VERSION, RuntimeRole
from xpool.runtime.agent import AGENT_SHUTDOWN_POLL_INTERVAL_S, Agent, AgentError, AgentHeartbeat
from xpool.service.client import XpoolClient, XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    HeartbeatResponse,
    InstanceRankRegistration,
    ProcessRef,
)
from xpool.transport import TransportArenaHandle

__all__ = ["AtnAgent", "AtnAgentTransportArenaState", "AtnAgentTransportRuntime"]

logger = logging.getLogger(__name__)

TRANSPORT_PUBLICATION_RECOVERY_S = 60.0
TRANSPORT_SHUTDOWN_DEADLINE_S = 60.0


@dataclass(slots=True)
class AtnAgentTransportArenaState:
    """One native arena and its current daemon publication state.

    Attributes:
        instance_id: Configured instance consuming this rank-local arena.
        registration: Latest matching instance registration and geometry.
        handle: Native CUDA IPC arena handle owned by this process.
        published_epoch: AtnAgent registration epoch that accepted this handle.
    """

    instance_id: str
    registration: InstanceRankRegistration
    handle: TransportArenaHandle
    published_epoch: int | None = None

    def binding(self) -> AtnAgentTransportArenaBinding:
        """Return this entry's daemon publication value."""

        return AtnAgentTransportArenaBinding(
            instance_id=self.instance_id,
            rank=self.registration.rank,
            handle=self.handle,
        )

    def accept_registration(self, registration: InstanceRankRegistration) -> None:
        """Retain a replacement owner only when native geometry is unchanged."""

        if self.registration.transport != registration.transport:
            raise AgentError(
                f"transport attributes changed for instance {self.instance_id} rank "
                f"{self.registration.rank}; runtime hot resize is unsupported"
            )
        self.registration = registration


class AtnAgentTransportRuntime:
    """Own all rank-local transport arenas for one AtnAgent process."""

    def __init__(
        self,
        *,
        client: XpoolClient,
        cuda_device: int,
        local_rank: int,
        publisher: ProcessRef,
    ) -> None:
        """Create an empty runtime bound to one daemon registration slot."""

        self.client = client
        self.cuda_device = cuda_device
        self.local_rank = local_rank
        self.publisher = publisher
        self.entries: dict[str, AtnAgentTransportArenaState] = {}
        self.publication_deadline: float | None = None

    @property
    def resources(self) -> tuple[AtnAgentTransportArenaState, ...]:
        """Return a stable snapshot of currently owned entries."""

        return tuple(self.entries.values())

    @property
    def published(self) -> bool:
        """Return whether any owned handle was published to the daemon."""

        return any(entry.published_epoch is not None for entry in self.entries.values())

    def local_registrations(self, instance_ranks: Mapping[str, int]) -> dict[str, InstanceRankRegistration]:
        """Return registrations assigned to this AtnAgent by the Fabric Plan."""

        return {
            registration.instance_id: registration
            for registration in self.client.list_instances()
            if registration.rank == instance_ranks.get(registration.instance_id)
        }

    def retry_or_raise(self, error: Exception) -> bool:
        """Apply the bounded recovery policy for list and publication failures."""

        deadline = self.publication_deadline or (time.monotonic() + TRANSPORT_PUBLICATION_RECOVERY_S)
        self.publication_deadline = deadline
        if time.monotonic() >= deadline:
            raise AgentError("atnagent transport arenas were not published before the recovery deadline") from error
        logger.debug("waiting to publish transport arenas: %s", error)
        return False

    def prepare(self, registration_epoch: int, instance_ranks: Mapping[str, int]) -> bool:
        """Create and publish every currently registered local instance arena.

        Args:
            registration_epoch: Monotonic epoch of the current AtnAgent daemon
                registration. Handles are republished once per epoch.
            instance_ranks: Instance ranks assigned to this AtnAgent by the
                current Fabric Plan.

        Returns:
            True when every configured instance has a current-epoch publication.

        Raises:
            AgentError: If geometry changes, native creation fails, or bounded
                recoverable publication retries expire.
        """

        try:
            registrations = self.local_registrations(instance_ranks)
        except (XpoolDaemonError, XpoolClientError) as error:
            if not error.is_recoverable:
                raise AgentError(f"atnagent instance-list reconcile failed: {error}") from error
            return self.retry_or_raise(error)

        for instance_id, entry in self.entries.items():
            registration = registrations.get(instance_id)
            if registration is not None:
                entry.accept_registration(registration)

        created: list[AtnAgentTransportArenaState] = []
        try:
            for instance_index, instance in enumerate(get_global_config().instances):
                registration = registrations.get(instance.id)
                if registration is None or instance.id in self.entries:
                    continue
                handle = TransportArenaHandle(
                    handle=xpool.native.transport.create_arena(
                        instance_index,
                        registration.rank,
                        registration.transport.payload_row_capacity,
                        registration.transport.hidden_size,
                        registration.ffn_profile.payload_dtype,
                        registration.transport.atn_tp_rank,
                        registration.transport.atn_tp_size,
                        registration.transport.atn_dp_rank,
                        registration.transport.atn_dp_size,
                    )
                )
                created.append(
                    AtnAgentTransportArenaState(
                        instance_id=instance.id,
                        registration=registration,
                        handle=handle,
                    )
                )
        except Exception as error:
            try:
                self.rollback(created)
            except Exception as cleanup_error:
                error.add_note(f"transport arena rollback also failed: {type(cleanup_error).__name__}: {cleanup_error}")
                raise AgentError(
                    f"failed to create transport arenas for CUDA device {self.cuda_device} and release partial arenas"
                ) from error
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

        configured = frozenset(instance_ranks)
        complete = configured == frozenset(registrations) and all(
            entry.published_epoch == registration_epoch for entry in self.entries.values()
        )
        if not complete:
            missing = configured - frozenset(registrations)
            if missing:
                logger.debug(
                    "waiting for local instance registrations device=%s rank=%s missing=%s",
                    self.cuda_device,
                    self.local_rank,
                    ", ".join(sorted(missing)),
                )
        return complete

    def activate(self) -> None:
        """Launch the sole Resident over the immutable local arena set."""

        if not self.entries:
            return
        try:
            xpool.native.transport.activate()
        except Exception as error:
            raise AgentError(f"transport Resident launch failed on CUDA device {self.cuda_device}") from error

    def check_health(self) -> None:
        """Reject unexpected completion of the process-wide Transport Resident."""

        if not self.entries:
            return
        try:
            xpool.native.transport.check_health()
        except Exception as error:
            raise AgentError(f"transport Resident failed on CUDA device {self.cuda_device}") from error

    def quiesce_leases(self) -> None:
        """Close lease admission and wait for every live Instance-rank owner."""

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
                logger.debug("failed to quiesce transport leases: %s", error)
                time.sleep(AGENT_SHUTDOWN_POLL_INTERVAL_S)
                continue
            if not response.in_use:
                return
            logger.debug(
                "waiting for transport arena leases rank=%s in_use=%s",
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

    def rollback(self, entries: Sequence[AtnAgentTransportArenaState]) -> None:
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
    """Attention-side agent owning one rank-local transport runtime."""

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
        self.transport = AtnAgentTransportRuntime(
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
            logger.debug("registration failed device=%s detail=%s", self.cuda_device, error)
            return
        self.registered = True
        self.registration_epoch += 1
        logger.info("registered device=%s pid=%s", self.cuda_device, self.proc_id.pid)

    def send_heartbeat(self) -> HeartbeatResponse:
        """Publish this AtnAgent's heartbeat."""

        return self.client.heartbeat_atnagent(self.cuda_device, self.process_ref)

    def prepare_fabric_join(self) -> bool:
        """Reconcile all local arenas before collective Fabric join."""

        if self.fabric_plan is None:
            raise AgentError("AtnAgent cannot prepare transport before receiving a Fabric Plan")
        instance_ranks = {
            instance_plan.instance_id: rank
            for instance_plan in self.fabric_plan.instance_plans
            for rank, atnagent_index in enumerate(instance_plan.instance_rank_topology.atnagent_indices)
            if atnagent_index == self.local_rank
        }
        prepared = self.transport.prepare(self.registration_epoch, instance_ranks)
        if prepared:
            logger.info(
                "transport prepared device=%s rank=%s instance_count=%s",
                self.cuda_device,
                self.local_rank,
                len(instance_ranks),
            )
        return prepared

    def prepare_fabric_execution(self) -> None:
        """Require no AtnAgent resource binding between join and activation."""

    def activate_fabric(self) -> None:
        """Launch transport kernels after collective Fabric join."""

        self.transport.activate()
        self.transport.check_health()

    def quiesce_fabric(self) -> None:
        """Drain local leases and transport kernels before Fabric drain."""

        self.transport.quiesce()

    def poll_fabric_health(self) -> None:
        """Check both Fabric and the process-wide Transport Resident."""

        super().poll_fabric_health()
        report = self.participant_report
        if (
            report is not None
            and report.phase is FabricParticipantPhase.ACTIVE
            and report.invocation_failure is None
            and self.fabric_phase in {FabricGenerationPhase.ACTIVATING, FabricGenerationPhase.EXECUTABLE}
        ):
            try:
                self.transport.check_health()
            except AgentError as error:
                self.report_local_control_failure(str(error))
                raise

    def close_role(self) -> None:
        """Release every local transport arena."""

        if self.registered and self.transport.published:
            self.transport.quiesce_leases()
        self.transport.close()
