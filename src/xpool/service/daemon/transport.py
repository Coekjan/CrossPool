"""Transport publication and lease state for the daemon control plane."""

from __future__ import annotations

from dataclasses import dataclass, field

from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.daemon.registration import InstanceRankId
from xpool.service.errors import XpoolDaemonError
from xpool.transport import TransportArenaHandle
from xpool.utils.procs import ProcUniqId


@dataclass(slots=True)
class TransportArenaLease:
    """One Instance rank's lease on an exact AtnAgent publication."""

    cuda_device: int
    handle: TransportArenaHandle
    publisher: ProcUniqId
    termination_requested: bool = False


@dataclass(frozen=True, slots=True)
class TransportArenaPublication:
    """One AtnAgent-published handle and its immutable geometry."""

    instance: InstanceRankId
    handle: TransportArenaHandle
    transport: InstanceTransportAttributes


@dataclass(slots=True)
class AtnAgentTransportState:
    """Transport state belonging to one exact AtnAgent process generation."""

    owner: ProcUniqId
    publications: dict[InstanceRankId, TransportArenaPublication] = field(default_factory=dict)
    accepting_leases: bool = True


class TransportBroker:
    """Caller-synchronized Transport publication and lease registry."""

    __slots__ = ("atnagents", "leases")

    def __init__(self) -> None:
        """Create an empty broker."""

        self.atnagents: dict[int, AtnAgentTransportState] = {}
        self.leases: dict[InstanceRankId, TransportArenaLease] = {}

    def install_atnagent(self, cuda_device: int, owner: ProcUniqId) -> None:
        """Install or retain transport state for an exact owner generation."""

        current = self.atnagents.get(cuda_device)
        if current is None or current.owner != owner:
            self.atnagents[cuda_device] = AtnAgentTransportState(owner=owner)

    def publish(
        self,
        cuda_device: int,
        owner: ProcUniqId,
        publications: list[TransportArenaPublication],
    ) -> None:
        """Commit validated publications for the current owner."""

        if not publications:
            raise XpoolDaemonError("conflict", "atnagent transport arena upsert must not be empty")
        state = self.require_owner(cuda_device, owner)
        if not state.accepting_leases:
            raise XpoolDaemonError("not_ready", "atnagent transport leases are quiescing")
        by_handle = {publication.handle.handle: instance for instance, publication in state.publications.items()}
        for publication in publications:
            existing = state.publications.get(publication.instance)
            if existing is not None and existing != publication:
                raise XpoolDaemonError("conflict", "published transport arena cannot be replaced in place")
            handle_owner = by_handle.get(publication.handle.handle)
            if handle_owner is not None and handle_owner != publication.instance:
                raise XpoolDaemonError("conflict", "atnagent transport arenas contain duplicate arena handle")
        state.publications.update({publication.instance: publication for publication in publications})

    def quiesce(self, cuda_device: int, owner: ProcUniqId) -> None:
        """Close publication and lease admission for one owner generation."""

        self.require_owner(cuda_device, owner).accepting_leases = False

    def publication(
        self,
        cuda_device: int,
        instance: InstanceRankId,
        owner: ProcUniqId,
    ) -> TransportArenaPublication:
        """Return one acquisition-eligible publication."""

        state = self.require_owner(cuda_device, owner)
        if not state.accepting_leases:
            raise XpoolDaemonError("not_ready", "local attention atnagent transport leases are quiescing")
        publication = state.publications.get(instance)
        if publication is None:
            raise XpoolDaemonError(
                "not_ready", "local attention atnagent has no transport arena handle for instance rank"
            )
        return publication

    def acquire(
        self,
        instance: InstanceRankId,
        lease: TransportArenaLease,
        *,
        last_seen_at: float,
        now: float,
        heartbeat_timeout_s: float,
    ) -> None:
        """Acquire an exact publication, replacing only an abandoned lease."""

        existing = self.leases.get(instance)
        if existing is not None:
            if (
                existing.cuda_device == lease.cuda_device
                and existing.handle == lease.handle
                and existing.publisher == lease.publisher
            ):
                return
            if now - last_seen_at <= heartbeat_timeout_s:
                raise XpoolDaemonError("conflict", "instance rank already holds different transport arena handle")
        self.leases[instance] = lease

    def remove_instance(self, instance: InstanceRankId) -> TransportArenaLease | None:
        """Release and return daemon lease state when an Instance deregisters."""

        return self.leases.pop(instance, None)

    def leased_instances(self, cuda_device: int, publisher: ProcUniqId | None = None) -> list[InstanceRankId]:
        """Return Instance ranks leasing from a device and optional owner."""

        return [
            instance
            for instance, lease in self.leases.items()
            if lease.cuda_device == cuda_device and (publisher is None or lease.publisher == publisher)
        ]

    def mark_termination_requested(self, instances: list[InstanceRankId]) -> list[InstanceRankId]:
        """Mark leases and return the subset requiring a first termination request."""

        requested: list[InstanceRankId] = []
        for instance in instances:
            lease = self.leases.get(instance)
            if lease is not None and not lease.termination_requested:
                lease.termination_requested = True
                requested.append(instance)
        return requested

    def termination_requested(self, instance: InstanceRankId) -> bool:
        """Return whether lease quiesce requested this Instance's termination."""

        lease = self.leases.get(instance)
        return lease is not None and lease.termination_requested

    def is_quiescing(self, cuda_device: int) -> bool:
        """Return whether the current owner has closed lease admission."""

        state = self.atnagents.get(cuda_device)
        return state is not None and not state.accepting_leases

    def published_for(self, cuda_device: int, instance: InstanceRankId) -> TransportArenaPublication | None:
        """Return a publication without changing acquisition state."""

        state = self.atnagents.get(cuda_device)
        return None if state is None else state.publications.get(instance)

    def require_owner(self, cuda_device: int, owner: ProcUniqId) -> AtnAgentTransportState:
        """Require transport state to belong to the expected process generation."""

        state = self.atnagents.get(cuda_device)
        if state is None or state.owner != owner:
            raise XpoolDaemonError("not_ready", "atnagent registration changed during transport operation")
        return state
