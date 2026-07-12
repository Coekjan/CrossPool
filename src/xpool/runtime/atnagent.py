"""Attention-side atnagent transport arena lifecycle."""

from __future__ import annotations

import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import xpool.ops
from xpool.abi import ABI_VERSION, RuntimeRole, TransportArenaHandle, TransportTraceSnapshot
from xpool.config import get_global_config
from xpool.runtime.agent import (
    AGENT_CONTROL_INTERVAL_S,
    AGENT_SHUTDOWN_POLL_INTERVAL_S,
    Agent,
    AgentError,
    AgentHeartbeat,
    logger,
)
from xpool.service.client import XpoolClientError, XpoolDaemonError
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    InstanceRegistration,
    TransportArenaHandleRecord,
)
from xpool.utils.sighandler import sighandle

__all__ = ["AtnAgent"]

TRANSPORT_PUBLICATION_RECOVERY_S = 60.0
TRANSPORT_SHUTDOWN_DEADLINE_S = 60.0


@dataclass(frozen=True, slots=True)
class AtnArenaResource:
    """Transport arena resource owned by one ATN atnagent for one instance.

    Attributes:
        instance_id: Configured instance id that will consume this arena.
        registration: Instance registration whose transport attributes define
            this arena. The resource rejects later geometry changes because the
            native arena layout cannot be resized in place.
        handle: CUDA IPC handle for the native transport arena.
    """

    instance_id: str
    registration: InstanceRegistration
    handle: TransportArenaHandle

    def binding(self) -> AtnAgentTransportArenaBinding:
        """Return the daemon binding that publishes this arena."""

        return AtnAgentTransportArenaBinding(
            instance_id=self.instance_id,
            rank=self.registration.rank,
            handle=TransportArenaHandleRecord.from_handle(self.handle),
        )

    def destroy(self) -> TransportTraceSnapshot:
        """Destroy the native CUDA IPC arena represented by this resource.

        Returns:
            ABI-defined trace snapshot copied during native destruction. Its
            records are empty when transport observation was disabled.
        """

        return xpool.ops.atnagent.destroy_transport_arena(self.handle)

    def validate_registration(self, registration: InstanceRegistration) -> None:
        """Reject runtime transport-attribute changes for this resource."""

        if self.registration.transport == registration.transport:
            return
        message = (
            f"transport attributes changed for instance {self.instance_id} rank {self.registration.rank}; "
            "runtime hot resize is unsupported"
        )
        raise AgentError(message)


@dataclass(frozen=True, slots=True)
class AtnAgentState(ABC):
    """Base state for the whole ATN atnagent arena lifecycle.

    Attributes:
        resources: Native arenas owned by this state. States before arena
            creation carry an empty tuple.
        published_instance_ids: Instances published to the current daemon
            registration generation.
        previously_published: Whether any owned resource was published before,
            including before daemon registration recovery.
    """

    resources: tuple[AtnArenaResource, ...] = ()
    published_instance_ids: frozenset[str] = frozenset()
    previously_published: bool = False

    def step(self, atnagent: AtnAgent) -> AtnAgentState:
        """Recover registration if needed, then perform the state transition."""

        should_recover_registration = not atnagent.registered
        if atnagent.registered:
            should_recover_registration = atnagent.heartbeat_worker.consume_registration_missing()
        if should_recover_registration:
            self.recover_registration_if_missing(atnagent)
        if not atnagent.registered:
            return self
        if should_recover_registration:
            return self.registration_recover(atnagent)
        return self.advance(atnagent)

    @abstractmethod
    def advance(self, atnagent: AtnAgent) -> AtnAgentState:
        """Perform this state's legal side effects and return the next state."""

        raise NotImplementedError

    def recover_registration_if_missing(self, atnagent: AtnAgent) -> None:
        """Recover the daemon atnagent registration when heartbeat marked it stale."""

        atnagent.heartbeat_worker.stop()
        atnagent.registered = False
        atnagent.register()
        if atnagent.registered:
            atnagent.heartbeat_worker.start()
            atnagent.heartbeat_worker.raise_if_failed()
            if atnagent.heartbeat_worker.consume_registration_missing():
                atnagent.heartbeat_worker.stop()
                atnagent.registered = False

    def registration_recover(self, atnagent: AtnAgent) -> AtnAgentState:
        """Return the state that follows successful daemon registration recovery."""

        return self

    def local_instance_registrations(self, atnagent: AtnAgent) -> dict[str, InstanceRegistration]:
        """Return daemon registrations for this ATN atnagent's local rank."""

        if not atnagent.registered:
            raise AgentError("atnagent must be registered before listing local instance registrations")
        return {
            registration.instance_id: registration
            for registration in atnagent.client.list_instances()
            if registration.rank == atnagent.local_rank
            and registration.instance_id in get_global_config().instance_by_id
        }


@dataclass(frozen=True, slots=True)
class AtnInitialized(AtnAgentState):
    """Initial ATN atnagent state before daemon registration succeeds."""

    def advance(self, atnagent: AtnAgent) -> AtnAgentState:
        """Move to registered state after daemon registration succeeds."""

        return AtnRegistered()

    def registration_recover(self, atnagent: AtnAgent) -> AtnAgentState:
        """Enter registered state after the initial daemon registration."""

        return AtnRegistered()


@dataclass(frozen=True, slots=True)
class AtnRegistered(AtnAgentState):
    """ATN atnagent state that reconciles registered instances incrementally."""

    def advance(self, atnagent: AtnAgent) -> AtnAgentState:
        """Create or select the next publishable local arena batch."""

        try:
            registrations_by_instance = self.local_instance_registrations(atnagent)
        except (XpoolDaemonError, XpoolClientError) as exc:
            if not exc.is_recoverable:
                raise AgentError(
                    f"atnagent instance-list reconcile received unrecoverable daemon error: {exc}"
                ) from exc
            logger.warning("failed to reconcile atnagent transport arenas: %s", exc)
            return self

        resources_by_instance = {resource.instance_id: resource for resource in self.resources}
        publication_resources = [
            resource
            for resource in self.resources
            if resource.instance_id in registrations_by_instance
            and resource.instance_id not in self.published_instance_ids
        ]
        for resource in self.resources:
            registration = registrations_by_instance.get(resource.instance_id)
            if registration is not None:
                resource.validate_registration(registration)

        created_resources: list[AtnArenaResource] = []
        try:
            for instance in get_global_config().instances:
                registration = registrations_by_instance.get(instance.id)
                if registration is None or instance.id in resources_by_instance:
                    continue
                handle = xpool.ops.atnagent.create_transport_arena(
                    atnagent.cuda_device,
                    registration.transport.max_tokens,
                    registration.transport.hidden_size,
                    registration.transport.element_size,
                    registration.transport.atn_dp_size,
                )
                resource = AtnArenaResource(
                    instance_id=instance.id,
                    registration=registration,
                    handle=handle,
                )
                created_resources.append(resource)
                publication_resources.append(resource)
        except Exception as exc:
            for resource in created_resources:
                try:
                    resource.destroy()
                except Exception as cleanup_exc:
                    raise AgentError(
                        "failed to destroy partially created ATN transport arenas after create failure"
                    ) from cleanup_exc
            raise AgentError(f"failed to create transport arenas for CUDA device {atnagent.cuda_device}") from exc

        resources = (*self.resources, *created_resources)
        if created_resources:
            return AtnArenaCreated(
                resources=resources,
                published_instance_ids=self.published_instance_ids,
                previously_published=self.previously_published,
                created_resources=tuple(created_resources),
                publication_resources=tuple(publication_resources),
            )
        if publication_resources:
            return AtnKernelLaunched(
                resources=resources,
                published_instance_ids=self.published_instance_ids,
                previously_published=self.previously_published,
                publication_resources=tuple(publication_resources),
            )

        missing = frozenset(get_global_config().instance_by_id) - set(registrations_by_instance)
        if missing:
            logger.info(
                "waiting for local instance registrations on CUDA device %s rank %s; missing instances: %s",
                atnagent.cuda_device,
                atnagent.local_rank,
                ", ".join(sorted(missing)),
            )
            return self

        return self

    def registration_recover(self, atnagent: AtnAgent) -> AtnAgentState:
        """Reset current-generation publications after daemon re-registration."""

        return AtnRegistered(
            resources=self.resources,
            previously_published=self.previously_published or bool(self.published_instance_ids),
        )


@dataclass(frozen=True, slots=True)
class AtnArenaCreated(AtnAgentState):
    """ATN atnagent state after one native arena batch is created.

    Attributes:
        created_resources: Newly allocated resources whose resident kernels
            have not yet been launched.
        publication_resources: Running or newly created resources to publish
            after the new kernels launch.
    """

    created_resources: tuple[AtnArenaResource, ...] = ()
    publication_resources: tuple[AtnArenaResource, ...] = ()

    def advance(self, atnagent: AtnAgent) -> AtnAgentState:
        """Launch persistent kernels for the newly created arena batch."""

        for resource in self.created_resources:
            try:
                xpool.ops.atnagent.launch_transport_kernel(resource.handle)
            except Exception as exc:
                raise AgentError(
                    "atnagent transport kernel launch failed for "
                    f"instance {resource.instance_id} rank {resource.registration.rank}"
                ) from exc
        return AtnKernelLaunched(
            resources=self.resources,
            published_instance_ids=self.published_instance_ids,
            previously_published=self.previously_published,
            publication_resources=self.publication_resources,
        )

    def registration_recover(self, atnagent: AtnAgent) -> AtnAgentState:
        """Retain created resources while resetting daemon publications."""

        return AtnArenaCreated(
            resources=self.resources,
            previously_published=self.previously_published or bool(self.published_instance_ids),
            created_resources=self.created_resources,
            publication_resources=self.resources,
        )


@dataclass(frozen=True, slots=True)
class AtnKernelLaunched(AtnAgentState):
    """ATN atnagent state after a persistent-kernel batch is running.

    Attributes:
        publication_deadline: Monotonic deadline for a recoverable publication
            attempt, or ``None`` before the first failed attempt.
        publication_resources: Resources eligible for the current publication
            attempt.
    """

    publication_resources: tuple[AtnArenaResource, ...] = ()
    publication_deadline: float | None = None

    def advance(self, atnagent: AtnAgent) -> AtnAgentState:
        """Publish the currently registered subset of the launched batch."""

        try:
            registrations_by_instance = self.local_instance_registrations(atnagent)
        except (XpoolDaemonError, XpoolClientError) as exc:
            if not exc.is_recoverable:
                raise AgentError(f"atnagent instance-list reconcile failed after kernel launch: {exc}") from exc
            return self.retry_or_raise(exc)
        publishable: list[AtnArenaResource] = []
        for resource in self.publication_resources:
            registration = registrations_by_instance.get(resource.instance_id)
            if registration is None:
                continue
            resource.validate_registration(registration)
            publishable.append(resource)
        if not publishable:
            return AtnRegistered(
                resources=self.resources,
                published_instance_ids=self.published_instance_ids,
                previously_published=self.previously_published,
            )
        try:
            atnagent.client.upsert_atnagent_transport_arenas(
                atnagent.cuda_device,
                [resource.binding() for resource in publishable],
                publisher=atnagent.process_ref,
            )
        except (XpoolDaemonError, XpoolClientError) as exc:
            if not exc.is_recoverable:
                raise AgentError(f"atnagent transport arena upsert failed: {exc}") from exc
            return self.retry_or_raise(exc)
        published_instance_ids = self.published_instance_ids | frozenset(
            resource.instance_id for resource in publishable
        )
        if published_instance_ids == frozenset(get_global_config().instance_by_id):
            return AtnArenaPublished(
                resources=self.resources,
                published_instance_ids=published_instance_ids,
                previously_published=True,
            )
        return AtnRegistered(
            resources=self.resources,
            published_instance_ids=published_instance_ids,
            previously_published=True,
        )

    def retry_or_raise(self, exc: Exception) -> AtnKernelLaunched:
        deadline = self.publication_deadline or (time.monotonic() + TRANSPORT_PUBLICATION_RECOVERY_S)
        if time.monotonic() >= deadline:
            raise AgentError("atnagent transport arenas were not republished before the recovery deadline") from exc
        logger.warning("waiting to publish atnagent transport arenas: %s", exc)
        return AtnKernelLaunched(
            resources=self.resources,
            published_instance_ids=self.published_instance_ids,
            previously_published=self.previously_published,
            publication_resources=self.publication_resources,
            publication_deadline=deadline,
        )

    def registration_recover(self, atnagent: AtnAgent) -> AtnAgentState:
        """Republish every launched resource after daemon re-registration."""

        return AtnKernelLaunched(
            resources=self.resources,
            previously_published=self.previously_published or bool(self.published_instance_ids),
            publication_resources=self.resources,
            publication_deadline=time.monotonic() + TRANSPORT_PUBLICATION_RECOVERY_S,
        )


@dataclass(frozen=True, slots=True)
class AtnArenaPublished(AtnAgentState):
    """ATN atnagent state after daemon publication succeeds."""

    def advance(self, atnagent: AtnAgent) -> AtnAgentState:
        """Keep the published arenas stable without polling daemon state."""

        return self

    def registration_recover(self, atnagent: AtnAgent) -> AtnAgentState:
        """Republish original handles incrementally after daemon recovery."""

        return AtnKernelLaunched(
            resources=self.resources,
            previously_published=True,
            publication_resources=self.resources,
            publication_deadline=time.monotonic() + TRANSPORT_PUBLICATION_RECOVERY_S,
        )


class AtnAgent(Agent):
    """Attention-side atnagent that publishes local transport arenas.

    Attributes:
        local_rank: Rank mapped to this process's configured ATN CUDA device.
        state: Aggregate owner of all native arenas and publication progress.
        heartbeat_worker: Background daemon-registration heartbeat owner.
    """

    local_rank: int
    state: AtnAgentState
    heartbeat_worker: AgentHeartbeat

    def __init__(self, *, cuda_device: int) -> None:
        """Create an attention atnagent and derive its local instance rank."""

        super().__init__(cuda_device=cuda_device, runtime_role=RuntimeRole.ATNAGENT)
        self.registration = AtnAgentRegistration(
            cuda_device=cuda_device,
            abi_version=ABI_VERSION,
            pid=self.proc_id.pid,
        )
        config = get_global_config()
        try:
            self.local_rank = config.devices.atn_cuda_devices.index(cuda_device)
        except ValueError as exc:
            raise AgentError(f"CUDA device {cuda_device} has no local instance-rank arenas") from exc
        self.state = AtnInitialized()
        self.heartbeat_worker = AgentHeartbeat(
            cuda_device=cuda_device,
            heartbeat=self.heartbeat_payload,
            sender=lambda device, heartbeat: self.client.heartbeat_atnagent(device, heartbeat),
        )

    def register(self) -> None:
        """Register this AtnAgent and update local registration state."""

        try:
            self.client.register_atnagent(self.registration)
        except (XpoolDaemonError, XpoolClientError) as exc:
            self.registered = False
            if not exc.is_recoverable:
                raise AgentError(f"AtnAgent registration received unrecoverable daemon error: {exc}") from exc
            logger.warning("AtnAgent registration failed: %s", exc)
            return
        self.registered = True

    def run(self) -> None:
        """Run the attention-side resident lifecycle until interrupted."""

        with sighandle(signal.SIGTERM, signal.default_int_handler):
            try:
                while True:
                    self.heartbeat_worker.raise_if_failed()
                    self.advance_state()
                    time.sleep(AGENT_CONTROL_INTERVAL_S)
            except KeyboardInterrupt:
                return
            finally:
                self.heartbeat_worker.close()
                if self.registered or self.state.resources:
                    self.shutdown()
                self.client.close()

    def advance_state(self) -> None:
        """Advance the state machine by one state transition."""

        self.state = self.state.step(self)

    def shutdown(self) -> None:
        """Drain arenas, wait for daemon leases, and destroy native arenas."""

        should_drain = (
            isinstance(self.state, AtnArenaPublished)
            or self.state.previously_published
            or bool(self.state.published_instance_ids)
        )
        if self.registered and should_drain:
            self.wait_for_transport_arena_leases_to_drain()
        for resource in self.state.resources:
            resource.destroy()
        self.state = AtnInitialized()

    def wait_for_transport_arena_leases_to_drain(self) -> None:
        """Wait until daemon reports no live lease owners for local arenas."""

        if not self.registered:
            raise AgentError("atnagent must be registered before draining transport arenas")
        deadline = time.monotonic() + TRANSPORT_SHUTDOWN_DEADLINE_S
        while time.monotonic() < deadline:
            try:
                response = self.client.drain_atnagent_transport_arenas(
                    self.cuda_device,
                    publisher=self.process_ref,
                )
            except (XpoolClientError, XpoolDaemonError) as exc:
                if not exc.is_recoverable:
                    raise AgentError("failed to drain atnagent transport arenas") from exc
                logger.warning("failed to drain atnagent transport arenas before destroying arenas: %s", exc)
                time.sleep(AGENT_SHUTDOWN_POLL_INTERVAL_S)
                continue
            if not response.in_use:
                return
            in_use = ", ".join(f"{entry.instance_id}:{entry.rank}" for entry in response.in_use)
            logger.info(
                "waiting for transport arena leases to drain before destroying CUDA IPC arenas on rank %s; "
                "in-use instances: %s",
                self.local_rank,
                in_use,
            )
            time.sleep(AGENT_SHUTDOWN_POLL_INTERVAL_S)
        raise AgentError("timed out draining transport arena leases; native arenas remain allocated")
