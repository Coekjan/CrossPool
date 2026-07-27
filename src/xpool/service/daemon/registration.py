"""Process registration records for the daemon control plane."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass

import psutil

from xpool.fabric import FfnWorkload
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    AtnAgentRegistration,
    FfnAgentRegistration,
    InstanceRegistration,
    ProcessRef,
    ReadinessStatus,
)
from xpool.utils.procs import ProcUniqId

HEARTBEAT_WARNING_WATERMARK_S = 15.0


class CommonRegistration(ABC):
    """Common process identity and heartbeat state stored by the daemon."""

    __slots__ = ("alive", "last_seen_at", "proc")

    abi_version: int
    alive: bool
    last_seen_at: float
    proc: ProcUniqId

    def __init__(self, pid: int, *, now: float) -> None:
        """Capture the daemon-side identity for a live client process."""

        try:
            self.proc = ProcUniqId(pid)
        except (psutil.Error, ValueError) as exc:
            raise XpoolDaemonError("conflict", f"registering pid {pid} is not live") from exc
        self.alive = True
        self.last_seen_at = now

    def readiness_status(self, now: float) -> ReadinessStatus:
        """Return the cached liveness and heartbeat status."""

        if not self.alive:
            return ReadinessStatus.OFFLINE
        if now - self.last_seen_at > HEARTBEAT_WARNING_WATERMARK_S:
            return ReadinessStatus.STALE
        return ReadinessStatus.ONLINE

    def require_online(self, now: float, *, context: str) -> None:
        """Require cached process liveness and a fresh heartbeat."""

        match self.readiness_status(now):
            case ReadinessStatus.ONLINE:
                return
            case ReadinessStatus.OFFLINE:
                raise XpoolDaemonError("not_ready", f"{context} process is not live")
            case ReadinessStatus.STALE:
                raise XpoolDaemonError("not_ready", f"{context} heartbeat is stale")

    def validate_process_ref(self, process: ProcessRef, *, context: str) -> None:
        """Validate ABI, pid, and live process identity outside the domain lock."""

        if process.abi_version != self.abi_version:
            raise XpoolDaemonError("conflict", f"{context} ABI version does not match registration ABI")
        if process.pid != self.proc.pid:
            raise XpoolDaemonError("conflict", f"{context} pid does not match registration pid")
        if not self.proc.is_alive():
            raise XpoolDaemonError("conflict", f"{context} process identity no longer matches registration")

    @abstractmethod
    def key(self) -> Hashable:
        """Return the unique key in the registration table."""

    def conflicts_with(self, candidate: CommonRegistration) -> bool:
        """Return whether two live registrations have different owners."""

        if type(candidate) is not type(self):
            raise TypeError(f"expected {type(self).__name__}, got {type(candidate).__name__}")
        return self.proc != candidate.proc


@dataclass(frozen=True, order=True, slots=True)
class InstanceRankId:
    """Daemon-local identity of one configured Instance rank."""

    instance_id: str
    rank: int


class InstanceRegistrationState(CommonRegistration):
    """One live Instance rank registration."""

    __slots__ = ("abi_version", "instance", "transport", "workload")

    abi_version: int
    instance: InstanceRankId
    transport: InstanceTransportAttributes
    workload: FfnWorkload

    def __init__(
        self,
        *,
        instance: InstanceRankId,
        abi_version: int,
        pid: int,
        transport: InstanceTransportAttributes,
        workload: FfnWorkload,
        now: float,
    ) -> None:
        """Create one registration from the Instance's declared contract."""

        super().__init__(pid, now=now)
        self.abi_version = abi_version
        self.instance = instance
        self.transport = transport
        self.workload = workload

    def key(self) -> InstanceRankId:
        """Return the configured Instance rank key."""

        return self.instance

    def conflicts_with(self, candidate: CommonRegistration) -> bool:
        """Return whether owner or declared contracts differ."""

        if not isinstance(candidate, InstanceRegistrationState):
            raise TypeError(f"expected InstanceRegistrationState, got {type(candidate).__name__}")
        return (
            super().conflicts_with(candidate)
            or self.instance != candidate.instance
            or self.abi_version != candidate.abi_version
            or self.transport != candidate.transport
            or self.workload != candidate.workload
        )


class AtnAgentRegistrationState(CommonRegistration):
    """One live AtnAgent registration."""

    __slots__ = ("abi_version", "cuda_device")

    abi_version: int
    cuda_device: int

    def __init__(self, *, cuda_device: int, abi_version: int, pid: int, now: float) -> None:
        """Create one AtnAgent registration."""

        super().__init__(pid, now=now)
        self.abi_version = abi_version
        self.cuda_device = cuda_device

    def key(self) -> int:
        """Return the owned CUDA device."""

        return self.cuda_device

    def conflicts_with(self, candidate: CommonRegistration) -> bool:
        """Return whether owner, device, or ABI differs."""

        if not isinstance(candidate, AtnAgentRegistrationState):
            raise TypeError(f"expected AtnAgentRegistrationState, got {type(candidate).__name__}")
        return (
            super().conflicts_with(candidate)
            or self.cuda_device != candidate.cuda_device
            or self.abi_version != candidate.abi_version
        )


class FfnAgentRegistrationState(CommonRegistration):
    """One live FfnAgent registration."""

    __slots__ = ("abi_version", "cuda_device")

    abi_version: int
    cuda_device: int

    def __init__(self, *, cuda_device: int, abi_version: int, pid: int, now: float) -> None:
        """Create one FfnAgent registration."""

        super().__init__(pid, now=now)
        self.abi_version = abi_version
        self.cuda_device = cuda_device

    def key(self) -> int:
        """Return the owned CUDA device."""

        return self.cuda_device

    def conflicts_with(self, candidate: CommonRegistration) -> bool:
        """Return whether owner, device, or ABI differs."""

        if not isinstance(candidate, FfnAgentRegistrationState):
            raise TypeError(f"expected FfnAgentRegistrationState, got {type(candidate).__name__}")
        return (
            super().conflicts_with(candidate)
            or self.cuda_device != candidate.cuda_device
            or self.abi_version != candidate.abi_version
        )


class RegistrationTable[R: CommonRegistration]:
    """Caller-synchronized keyed registration table owned by ``ControlPlane``."""

    __slots__ = ("registrations",)

    def __init__(self) -> None:
        """Create an empty table."""

        self.registrations: dict[Hashable, R] = {}

    def query(self, key: Hashable) -> R | None:
        """Return a registration while the caller holds the domain lock."""

        return self.registrations.get(key)

    def install_snapshot(self, registration: R, existing: R | None, *, existing_alive: bool) -> bool:
        """Commit a candidate after its prior owner's liveness was probed.

        Returns:
            Whether the process identity installed for the key changed.
        """

        key = registration.key()
        if self.registrations.get(key) is not existing:
            raise XpoolDaemonError("not_ready", "registration changed during liveness validation")
        if existing is None or not existing_alive:
            self.registrations[key] = registration
            return existing is None or existing.proc != registration.proc
        if existing.conflicts_with(registration):
            raise XpoolDaemonError("conflict", "registration is already owned by another live process")
        existing.last_seen_at = registration.last_seen_at
        return False

    def commit_heartbeat(self, key: Hashable, registration: R, *, now: float) -> None:
        """Commit a validated heartbeat if the registration is unchanged."""

        if self.registrations.get(key) is not registration:
            raise XpoolDaemonError("not_ready", "registration changed during heartbeat")
        registration.last_seen_at = now

    def remove_snapshot(self, key: Hashable, registration: R) -> None:
        """Remove a validated owner if the registration is unchanged."""

        if self.registrations.get(key) is not registration:
            raise XpoolDaemonError("not_ready", "registration changed during deregistration")
        del self.registrations[key]

    def values(self) -> list[R]:
        """Return a snapshot while the caller holds the domain lock."""

        return list(self.registrations.values())


class RegistrationBook:
    """Caller-synchronized process registrations composed by ``ControlPlane``."""

    __slots__ = ("atnagents", "ffnagents", "instances")

    def __init__(self) -> None:
        """Create empty role-specific registration tables."""

        self.instances = RegistrationTable[InstanceRegistrationState]()
        self.atnagents = RegistrationTable[AtnAgentRegistrationState]()
        self.ffnagents = RegistrationTable[FfnAgentRegistrationState]()

    def all_values(self) -> list[CommonRegistration]:
        """Return every registration in stable role order."""

        return [*self.atnagents.values(), *self.ffnagents.values(), *self.instances.values()]

    def instance_views(self) -> list[InstanceRegistration]:
        """Project Instance records to wire views."""

        return [
            InstanceRegistration(
                pid=registration.proc.pid,
                instance_id=registration.instance.instance_id,
                rank=registration.instance.rank,
                abi_version=registration.abi_version,
                transport=registration.transport,
                workload=registration.workload,
            )
            for registration in self.instances.values()
        ]

    def atnagent_views(self) -> list[AtnAgentRegistration]:
        """Project AtnAgent records to wire views."""

        return [
            AtnAgentRegistration(
                pid=registration.proc.pid,
                cuda_device=registration.cuda_device,
                abi_version=registration.abi_version,
            )
            for registration in self.atnagents.values()
        ]

    def ffnagent_views(self) -> list[FfnAgentRegistration]:
        """Project FfnAgent records to wire views."""

        return [
            FfnAgentRegistration(
                pid=registration.proc.pid,
                cuda_device=registration.cuda_device,
                abi_version=registration.abi_version,
            )
            for registration in self.ffnagents.values()
        ]
