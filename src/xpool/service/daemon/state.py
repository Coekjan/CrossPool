"""Host control-plane for xpool system."""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import psutil

from xpool.abi import ABI_VERSION
from xpool.config import DeviceRole, XpoolConfig, get_global_config
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.daemon.mps import MpsProbeResult, MpsStatusProvider, probe_mps_controller
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    ControlPlaneWarning,
    DevagentRegistration,
    DevagentTransportArenaBinding,
    DevagentTransportArenaDrainResponse,
    HeartbeatResponse,
    InstanceRankRef,
    InstanceRegistration,
    ProcessHeartbeat,
    ProcessRef,
    ReadinessDevagent,
    ReadinessInstance,
    ReadinessScope,
    ReadinessSnapshot,
    ReadinessStatus,
    TransportArenaHandleRecord,
)
from xpool.utils.procs import ProcUniqId

HEARTBEAT_WARNING_WATERMARK_S = 15.0
TRANSPORT_ARENA_LEASE_HEARTBEAT_TIMEOUT_S = 2.0 * HEARTBEAT_WARNING_WATERMARK_S
TRANSPORT_DRAIN_TERM_GRACE_S = HEARTBEAT_WARNING_WATERMARK_S
DEVAGENT_REPLACEMENT_TIMEOUT_S = 60.0
GLOBAL_WARNING_CACHE_S = 1.0
MPS_READINESS_CACHE_S = 1.0
logger = logging.getLogger(__name__)


class CommonRegistration(ABC):
    """Common process registration state stored by the daemon."""

    __slots__ = ("last_seen_at", "proc")

    abi_version: int
    last_seen_at: float
    proc: ProcUniqId

    def __init__(self, pid: int, *, now: float) -> None:
        """Capture the daemon-side process identity for a client-supplied pid."""

        try:
            self.proc = ProcUniqId(pid)
        except (psutil.Error, ValueError) as exc:
            raise XpoolDaemonError("conflict", f"registering pid {pid} is not live") from exc
        self.last_seen_at = now

    def readiness_status(self, now: float) -> ReadinessStatus:
        """Return the daemon-observed liveness status for this registration."""

        if not self.proc.is_alive():
            return ReadinessStatus.OFFLINE
        if now - self.last_seen_at > HEARTBEAT_WARNING_WATERMARK_S:
            return ReadinessStatus.STALE
        return ReadinessStatus.ONLINE

    def require_online(self, now: float, *, context: str) -> None:
        """Require that this registration is live and within the heartbeat watermark."""

        match self.readiness_status(now):
            case ReadinessStatus.ONLINE:
                return
            case ReadinessStatus.OFFLINE:
                raise XpoolDaemonError("not_ready", f"{context} process is not live")
            case ReadinessStatus.STALE:
                raise XpoolDaemonError("not_ready", f"{context} heartbeat is stale")

    def validate_process_ref(self, process: ProcessRef, *, context: str) -> None:
        """Validate that a client-supplied process reference owns this registration."""

        if process.abi_version != self.abi_version:
            raise XpoolDaemonError("conflict", f"{context} ABI version does not match registration ABI")
        if process.pid != self.proc.pid:
            raise XpoolDaemonError("conflict", f"{context} pid does not match registration pid")
        if not self.proc.is_alive():
            raise XpoolDaemonError("conflict", f"{context} process identity no longer matches registration")

    def heartbeat(self, heartbeat: ProcessHeartbeat, *, now: float) -> None:
        """Refresh this registration's heartbeat after validating process identity."""

        self.validate_process_ref(heartbeat, context="heartbeat")
        self.last_seen_at = now

    @abstractmethod
    def key(self) -> Hashable:
        """Return the daemon registry key for this registration."""

        raise NotImplementedError

    def conflicts_with(self, candidate: CommonRegistration) -> bool:
        """Return whether two live registrations conflict."""

        if type(candidate) is not type(self):
            raise TypeError(f"expected {type(self).__name__}, got {type(candidate).__name__}")
        return self.proc != candidate.proc


@dataclass(frozen=True, order=True, slots=True)
class InstanceUniqId:
    """Daemon-local unique id for one instance rank slot.

    Attributes:
        instance_id: Instance id from the resolved xpool config.
        rank: Rank-local process index within the instance.
    """

    instance_id: str
    rank: int


@dataclass(slots=True)
class TransportArenaLease:
    """Daemon-owned record for one acquired transport arena handle.

    Attributes:
        cuda_device: Local attention devagent CUDA device that owns the handle.
        handle: Opaque transport arena handle record acquired by the instance rank.
        publisher: Exact devagent process generation that published the arena.
        termination_requested: Whether daemon has already requested owner
            process termination during devagent arena drain.
    """

    cuda_device: int
    handle: TransportArenaHandleRecord
    publisher: ProcUniqId
    termination_requested: bool = False


@dataclass(frozen=True, slots=True)
class TransportArenaPublication:
    """Daemon-internal binding between one rank, its handle, and transport contract.

    Attributes:
        instance: Exact configured instance-rank slot that may acquire the
            publication.
        handle: Opaque CUDA IPC arena handle returned to the matching instance
            rank while the publisher generation remains active.
        transport: Immutable geometry captured when the arena was created and
            compared with later instance registrations before acquisition.
    """

    instance: InstanceUniqId
    handle: TransportArenaHandleRecord
    transport: InstanceTransportAttributes


class InstanceRegistrationState(CommonRegistration):
    """One live instance rank registration stored by the daemon.

    Attributes:
        instance: Daemon-local unique id for this registered instance rank.
        abi_version: xpool descriptor ABI version used by the registering rank.
        transport: Transport attributes declared by the registering rank.
        proc: Process unique id captured by the daemon for the registering rank.
    """

    __slots__ = ("abi_version", "instance", "transport", "transport_arena_lease")

    abi_version: int
    instance: InstanceUniqId
    transport: InstanceTransportAttributes
    transport_arena_lease: TransportArenaLease | None

    def __init__(
        self,
        *,
        instance: InstanceUniqId,
        abi_version: int,
        pid: int,
        transport: InstanceTransportAttributes,
        now: float,
    ) -> None:
        """Create one instance rank registration from a client-supplied pid."""

        super().__init__(pid, now=now)
        self.abi_version = abi_version
        self.instance = instance
        self.transport = transport
        self.transport_arena_lease = None

    def key(self) -> InstanceUniqId:
        """Return the instance-rank registry key for this registration."""

        return self.instance

    def conflicts_with(self, candidate: CommonRegistration) -> bool:
        """Return whether two live rank registrations conflict."""

        if not isinstance(candidate, InstanceRegistrationState):
            raise TypeError(f"expected InstanceRegistrationState, got {type(candidate).__name__}")
        conflict = super().conflicts_with(candidate)
        return (
            conflict
            or self.instance != candidate.instance
            or self.abi_version != candidate.abi_version
            or self.transport != candidate.transport
        )


class DevagentRegistrationState(CommonRegistration):
    """One live devagent registration stored by the daemon.

    Attributes:
        cuda_device: CUDA device index owned by the registering devagent.
        abi_version: xpool descriptor ABI version used by the registering devagent.
        transport_arenas: Transport arena bindings published by this devagent, or ``None`` before first publication.
        transport_arenas_terminating: Whether the devagent is draining its published transport arenas.
        proc: Process unique id captured by the daemon for the registering devagent.
    """

    __slots__ = ("abi_version", "cuda_device", "transport_arenas", "transport_arenas_terminating")

    abi_version: int
    cuda_device: int
    transport_arenas: dict[InstanceUniqId, TransportArenaPublication] | None
    transport_arenas_terminating: bool

    def __init__(self, *, cuda_device: int, abi_version: int, pid: int, now: float) -> None:
        """Create one devagent registration from a client-supplied pid."""

        super().__init__(pid, now=now)
        self.abi_version = abi_version
        self.cuda_device = cuda_device
        self.transport_arenas = None
        self.transport_arenas_terminating = False

    def key(self) -> int:
        """Return the CUDA device registry key for this registration."""

        return self.cuda_device

    def conflicts_with(self, candidate: CommonRegistration) -> bool:
        """Return whether two live devagent registrations conflict."""

        if not isinstance(candidate, DevagentRegistrationState):
            raise TypeError(f"expected DevagentRegistrationState, got {type(candidate).__name__}")
        conflict = super().conflicts_with(candidate)
        return conflict or self.cuda_device != candidate.cuda_device or self.abi_version != candidate.abi_version


class RegistrationRegistry[R: CommonRegistration]:
    """Thread-safe registry for daemon process registrations."""

    __slots__ = ("lock", "registrations")

    def __init__(self) -> None:
        """Create a registry with its own process-registration lock."""

        self.lock = threading.Lock()
        self.registrations: dict[Hashable, R] = {}

    def install(self, registration: R) -> None:
        """Install a registration unless a different live process already owns the key."""

        key = registration.key()
        with self.lock:
            existing = self.registrations.get(key)
        existing_alive = False if existing is None else existing.proc.is_alive()
        self.install_snapshot(registration, existing, existing_alive=existing_alive)

    def install_snapshot(
        self,
        registration: R,
        existing: R | None,
        *,
        existing_alive: bool,
    ) -> None:
        """Install using a liveness result obtained outside daemon locks."""

        key = registration.key()
        with self.lock:
            if self.registrations.get(key) is not existing:
                raise XpoolDaemonError("not_ready", "registration changed during liveness validation")
            if existing is None or not existing_alive:
                self.drop(key)
                self.registrations[key] = registration
                return
            if existing.conflicts_with(registration):
                raise XpoolDaemonError("conflict", "registration is already owned by another live process")
            existing.last_seen_at = registration.last_seen_at

    def query(self, key: Hashable) -> R | None:
        """Return a registration by key."""

        with self.lock:
            return self.registrations.get(key)

    def heartbeat(
        self,
        key: Hashable,
        heartbeat: ProcessHeartbeat,
        *,
        now: float,
    ) -> None:
        """Refresh one registration heartbeat."""

        with self.lock:
            registration = self.registrations.get(key)
        if registration is None:
            raise XpoolDaemonError("not_ready", "registration is not registered")
        registration.validate_process_ref(heartbeat, context="heartbeat")
        with self.lock:
            if self.registrations.get(key) is not registration:
                raise XpoolDaemonError("not_ready", "registration changed during heartbeat")
            registration.last_seen_at = now

    def remove_owned(self, key: Hashable, owner: ProcessRef) -> None:
        """Remove a registration after validating the owning process reference."""

        with self.lock:
            registration = self.registrations.get(key)
        if registration is None:
            raise XpoolDaemonError("not_found", "registration is not registered")
        registration.validate_process_ref(owner, context="deregister")
        with self.lock:
            if self.registrations.get(key) is not registration:
                raise XpoolDaemonError("not_ready", "registration changed during deregistration")
            self.drop(key)

    def items(self) -> list[tuple[Hashable, R]]:
        """Return a read-only snapshot of registration items."""

        with self.lock:
            return list(self.registrations.items())

    def values(self) -> list[R]:
        """Return a read-only snapshot of registration values."""

        with self.lock:
            return list(self.registrations.values())

    def drop(self, key: Hashable) -> None:
        """Remove one registration while the caller owns the registry lock."""

        self.registrations.pop(key, None)


class InstanceRegistry(RegistrationRegistry[InstanceRegistrationState]):
    """Registry for instance-rank process registrations."""

    def rank_registration(
        self,
        instance_id: str,
        rank: int,
    ) -> InstanceRegistrationState | None:
        """Return an instance-rank registration."""

        return self.query(InstanceUniqId(instance_id=instance_id, rank=rank))

    def views(self) -> list[InstanceRegistration]:
        """Return instance registrations as wire views."""

        return [
            InstanceRegistration(
                pid=registration.proc.pid,
                instance_id=registration.instance.instance_id,
                rank=registration.instance.rank,
                abi_version=registration.abi_version,
                transport=registration.transport,
            )
            for registration in self.values()
        ]

    def acquire_transport_arena_lease(
        self,
        instance: InstanceUniqId,
        *,
        registration: InstanceRegistrationState,
        lease: TransportArenaLease,
        now: float,
    ) -> None:
        """Commit a lease after ownership and liveness were validated."""

        with self.lock:
            if self.registrations.get(instance) is not registration:
                raise XpoolDaemonError("not_ready", "instance registration changed during arena acquisition")
            existing = registration.transport_arena_lease
            if existing is not None:
                if (
                    existing.cuda_device == lease.cuda_device
                    and existing.handle == lease.handle
                    and existing.publisher == lease.publisher
                ):
                    return
                if now - registration.last_seen_at <= TRANSPORT_ARENA_LEASE_HEARTBEAT_TIMEOUT_S:
                    raise XpoolDaemonError("conflict", "instance rank already holds different transport arena handle")
            registration.transport_arena_lease = lease

    def terminate_transport_arena_leases(
        self,
        cuda_device: int,
        *,
        term_grace_s: float,
    ) -> list[InstanceRankRef]:
        """Terminate every live process tree leasing from one devagent."""

        with self.lock:
            candidates = list(self.registrations.values())
        lease_owners = [
            (registration, lease)
            for registration in candidates
            if (lease := registration.transport_arena_lease) is not None
            and lease.cuda_device == cuda_device
            and registration.proc.is_alive()
        ]
        with self.lock:
            lease_owners = [
                (registration, lease)
                for registration, lease in lease_owners
                if self.registrations.get(registration.instance) is registration
                and registration.transport_arena_lease is lease
            ]
            procs_to_terminate = [
                registration.proc for registration, lease in lease_owners if not lease.termination_requested
            ]
            for registration, lease in lease_owners:
                lease.termination_requested = True
        if procs_to_terminate:
            with ThreadPoolExecutor(max_workers=len(procs_to_terminate)) as executor:
                futures = [
                    executor.submit(proc.terminate_tree, term_grace_s=term_grace_s) for proc in procs_to_terminate
                ]
                for future in futures:
                    future.result()
        registrations = self.values()
        return [
            InstanceRankRef(
                pid=registration.proc.pid,
                abi_version=registration.abi_version,
                instance_id=registration.instance.instance_id,
                rank=registration.instance.rank,
            )
            for registration in registrations
            if (lease := registration.transport_arena_lease) is not None
            and lease.cuda_device == cuda_device
            and registration.proc.is_alive()
        ]

    def lease_owners_for_generation(
        self,
        cuda_device: int,
        publisher: ProcUniqId,
    ) -> list[InstanceRegistrationState]:
        """Return every registration leased from one devagent generation."""

        with self.lock:
            return [
                registration
                for registration in self.registrations.values()
                if registration.transport_arena_lease is not None
                and registration.transport_arena_lease.cuda_device == cuda_device
                and registration.transport_arena_lease.publisher == publisher
            ]

    def remove_dead(self, registrations: list[InstanceRegistrationState]) -> None:
        """Remove registrations from a terminated generation after process death."""

        dead = [registration for registration in registrations if not registration.proc.is_alive()]
        with self.lock:
            for registration in dead:
                if self.registrations.get(registration.instance) is registration:
                    self.drop(registration.instance)


class DevagentRegistry(RegistrationRegistry[DevagentRegistrationState]):
    """Registry for devagent process registrations and their transport arenas."""

    def upsert_transport_arenas(
        self,
        cuda_device: int,
        publications: list[TransportArenaPublication],
        *,
        registration: DevagentRegistrationState,
    ) -> None:
        """Commit publications after owner and liveness validation."""

        if not publications:
            raise XpoolDaemonError("conflict", "devagent transport arena upsert must not be empty")
        with self.lock:
            if self.registrations.get(cuda_device) is not registration:
                raise XpoolDaemonError("not_ready", "devagent registration changed during arena publication")
            if registration.transport_arenas_terminating:
                raise XpoolDaemonError("not_ready", "devagent transport arenas are terminating")
            publication_by_instance = dict(registration.transport_arenas or {})
            publication_by_instance.update({publication.instance: publication for publication in publications})
            registration.transport_arenas = publication_by_instance

    def drain_transport_arenas(
        self,
        cuda_device: int,
        *,
        registration: DevagentRegistrationState,
    ) -> None:
        """Commit arena termination after owner and liveness validation."""

        with self.lock:
            if self.registrations.get(cuda_device) is not registration:
                raise XpoolDaemonError("not_ready", "devagent registration changed during arena drain")
            registration.transport_arenas_terminating = True

    def transport_arena(
        self,
        cuda_device: int,
        instance: InstanceUniqId,
        *,
        registration: DevagentRegistrationState,
    ) -> tuple[TransportArenaPublication, ProcUniqId]:
        """Return a publication after devagent liveness was validated."""

        with self.lock:
            if self.registrations.get(cuda_device) is not registration:
                raise XpoolDaemonError("not_ready", "devagent registration changed during arena acquisition")
            if registration.transport_arenas_terminating:
                raise XpoolDaemonError("not_ready", "local attention devagent transport arenas are terminating")
            if registration.transport_arenas is None:
                raise XpoolDaemonError("not_ready", "local attention devagent transport arenas are not published")
            publication = registration.transport_arenas.get(instance)
            if publication is not None:
                return publication, registration.proc
        raise XpoolDaemonError("not_ready", "local attention devagent has no transport arena handle for instance rank")

    def views(self) -> list[DevagentRegistration]:
        """Return devagent registrations as wire views."""

        return [
            DevagentRegistration(
                pid=registration.proc.pid,
                cuda_device=registration.cuda_device,
                abi_version=registration.abi_version,
            )
            for registration in self.values()
        ]


@dataclass(slots=True)
class XpoolDaemonState:
    """Mutable daemon state shared by FastAPI route handlers.

    Attributes:
        started_at: Unix timestamp recorded when the daemon state was created.
        instance_registrations: Instance-rank process registration registry.
        devagent_registrations: Devagent process registration and transport arena handle registry.
        transport_lock: State-level lock for cross-registry transport acquire
            and drain invariants.
        warning_cache_lock: Lock protecting the bounded-heartbeat warning cache.
        warning_cache_at: Monotonic timestamp of the cached warning snapshot.
        warning_cache: Warning snapshot reused for one cache interval to avoid
            repeated process probes on every heartbeat.
    """

    mps_status_provider: MpsStatusProvider = probe_mps_controller
    started_at: float = field(default_factory=time.time)
    instance_registrations: InstanceRegistry = field(default_factory=InstanceRegistry)
    devagent_registrations: DevagentRegistry = field(default_factory=DevagentRegistry)
    transport_lock: threading.Lock = field(default_factory=threading.Lock)
    warning_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    warning_cache_at: float = field(default=float("-inf"))
    warning_cache: tuple[ControlPlaneWarning, ...] = ()
    mps_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    mps_cache_at: float = field(default=float("-inf"))
    mps_cache_result: MpsProbeResult | None = None

    def mps_readiness_status(self) -> ReadinessStatus:
        """Return cached MPS controller readiness and refresh it when stale.

        Returns:
            Online when the configured controller responds, otherwise offline.

        Side Effects:
            Runs a bounded MPS controller probe at most once per cache interval
            and logs status transitions with diagnostic context.
        """

        now = time.monotonic()
        with self.mps_cache_lock:
            if self.mps_cache_result is not None and now - self.mps_cache_at < MPS_READINESS_CACHE_S:
                return ReadinessStatus.ONLINE if self.mps_cache_result.online else ReadinessStatus.OFFLINE
            previous = self.mps_cache_result
            result = self.mps_status_provider()
            self.mps_cache_at = now
            self.mps_cache_result = result
        if previous is None or previous.online != result.online:
            log = logger.info if result.online else logger.warning
            log("CUDA MPS readiness changed to %s: %s", "online" if result.online else "offline", result.diagnostic)
        return ReadinessStatus.ONLINE if result.online else ReadinessStatus.OFFLINE

    def check_config(self, client_config: XpoolConfig) -> None:
        """Require a runtime participant's effective config to match the daemon.

        Args:
            client_config: Validated effective config submitted by a runtime
                participant before registration.

        Raises:
            XpoolDaemonError: If any effective config value differs from the
                daemon process-global config.
        """

        differences: list[str] = []
        missing = object()

        def display(value: object) -> str:
            return "<missing>" if value is missing else json.dumps(value, sort_keys=True)

        def compare(client_value: object, daemon_value: object, path: str) -> None:
            if isinstance(client_value, dict) and isinstance(daemon_value, dict):
                for key in sorted(client_value.keys() | daemon_value.keys()):
                    child_path = f"{path}.{key}" if path else str(key)
                    compare(client_value.get(key, missing), daemon_value.get(key, missing), child_path)
                return
            if isinstance(client_value, list) and isinstance(daemon_value, list):
                for index in range(max(len(client_value), len(daemon_value))):
                    compare(
                        client_value[index] if index < len(client_value) else missing,
                        daemon_value[index] if index < len(daemon_value) else missing,
                        f"{path}[{index}]",
                    )
                return
            if client_value is not missing and daemon_value is not missing:
                if type(client_value) is type(daemon_value) and client_value == daemon_value:
                    return
            differences.append(f"- {path}: client={display(client_value)}, daemon={display(daemon_value)}")

        compare(
            client_config.model_dump(mode="json"),
            get_global_config().model_dump(mode="json"),
            "",
        )
        if differences:
            raise XpoolDaemonError(
                "conflict",
                "client xpool config differs from daemon config:\n" + "\n".join(differences),
            )

    def global_warnings(self, now: float) -> list[ControlPlaneWarning]:
        """Return warnings visible to any heartbeat sender."""

        with self.warning_cache_lock:
            if now - self.warning_cache_at < GLOBAL_WARNING_CACHE_S:
                return list(self.warning_cache)
        warnings: list[ControlPlaneWarning] = []
        devagent_by_dev = dict(self.devagent_registrations.items())
        for devagent in get_global_config().devagents:
            if devagent.role is not DeviceRole.ATN:
                continue
            registration = devagent_by_dev.get(devagent.cuda_device)
            if registration is None or registration.readiness_status(now) is not ReadinessStatus.ONLINE:
                warnings.append(
                    ControlPlaneWarning(
                        kind="stale_devagent",
                        cuda_device=devagent.cuda_device,
                        message=f"devagent on CUDA device {devagent.cuda_device} is not heartbeating",
                    )
                )
            elif registration.transport_arenas_terminating:
                warnings.append(
                    ControlPlaneWarning(
                        kind="terminating_devagent",
                        cuda_device=devagent.cuda_device,
                        message=f"devagent on CUDA device {devagent.cuda_device} is terminating",
                    )
                )

        instance_by_uid = dict(self.instance_registrations.items())
        for instance in get_global_config().instances:
            for rank, cuda_device in enumerate(get_global_config().devices.atn_cuda_devices):
                registration = instance_by_uid.get(InstanceUniqId(instance_id=instance.id, rank=rank))
                if registration is None or registration.readiness_status(now) is not ReadinessStatus.ONLINE:
                    warnings.append(
                        ControlPlaneWarning(
                            kind="stale_instance",
                            cuda_device=cuda_device,
                            message=(
                                f"instance {instance.id} rank {rank} on CUDA device {cuda_device} is not heartbeating"
                            ),
                        )
                    )
        with self.warning_cache_lock:
            self.warning_cache_at = now
            self.warning_cache = tuple(warnings)
        return warnings

    def register_devagent(self, registration: DevagentRegistrationState) -> None:
        """Install a devagent registration after draining a dead generation.

        Args:
            registration: Candidate devagent process registration.

        Raises:
            XpoolDaemonError: If the device or ABI is invalid, a conflicting
                generation remains live, or old arena users outlive the bounded
                replacement cleanup.

        Side Effects:
            Concurrently terminates live instance processes leasing arenas from
            a dead prior generation and removes their registrations after death.
        """

        if registration.cuda_device not in get_global_config().cuda_devices:
            raise XpoolDaemonError("not_found", "unknown devagent")
        if registration.abi_version != ABI_VERSION:
            raise XpoolDaemonError("conflict", "devagent ABI version does not match daemon ABI")
        existing = self.devagent_registrations.query(registration.cuda_device)
        if existing is not None and existing.proc != registration.proc and not existing.proc.is_alive():
            owners = self.instance_registrations.lease_owners_for_generation(
                registration.cuda_device,
                existing.proc,
            )
            live_owners = [owner for owner in owners if owner.proc.is_alive()]
            if live_owners:
                deadline = time.monotonic() + DEVAGENT_REPLACEMENT_TIMEOUT_S
                with ThreadPoolExecutor(max_workers=len(live_owners)) as executor:
                    futures = [
                        executor.submit(owner.proc.terminate_tree, term_grace_s=TRANSPORT_DRAIN_TERM_GRACE_S)
                        for owner in live_owners
                    ]
                    for future in futures:
                        future.result()
                while any(owner.proc.is_alive() for owner in live_owners) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if any(owner.proc.is_alive() for owner in live_owners):
                    raise XpoolDaemonError("not_ready", "previous devagent generation still has live arena users")
                self.instance_registrations.remove_dead(live_owners)
            if self.devagent_registrations.query(registration.cuda_device) is not existing:
                raise XpoolDaemonError("not_ready", "devagent registration changed during generation cleanup")
        self.devagent_registrations.install(registration)

    def register_instance(self, registration: InstanceRegistrationState) -> None:
        """Install an instance registration unless a different live one exists."""

        instance_id = registration.instance.instance_id
        rank = registration.instance.rank
        if registration.abi_version != ABI_VERSION:
            raise XpoolDaemonError("conflict", "instance ABI version does not match daemon ABI")
        if instance_id not in get_global_config().instance_by_id:
            raise XpoolDaemonError("not_found", "unknown instance")
        self.validate_instance_rank(rank)
        transport = registration.transport
        if transport.atn_dp_size != 1 or transport.atn_dp_rank != 0:
            raise XpoolDaemonError("conflict", "instance transport DP must be rank 0 of size 1")
        if transport.atn_tp_size != get_global_config().atn_world_size or transport.atn_tp_rank != rank:
            raise XpoolDaemonError(
                "conflict",
                "instance transport TP rank/size does not match configured ATN rank placement",
            )
        existing = self.instance_registrations.query(registration.instance)
        existing_alive = False if existing is None else existing.proc.is_alive()
        with self.transport_lock:
            if self.instance_registrations.query(registration.instance) is not existing:
                raise XpoolDaemonError("not_ready", "instance registration changed during validation")
            for peer in self.instance_registrations.values():
                if peer.instance.instance_id != instance_id or peer.instance == registration.instance:
                    continue
                peer_contract = (
                    peer.transport.element_size,
                    peer.transport.hidden_size,
                    peer.transport.max_tokens,
                    peer.transport.atn_tp_size,
                    peer.transport.atn_dp_size,
                )
                candidate_contract = (
                    transport.element_size,
                    transport.hidden_size,
                    transport.max_tokens,
                    transport.atn_tp_size,
                    transport.atn_dp_size,
                )
                if peer_contract != candidate_contract:
                    raise XpoolDaemonError("conflict", "instance transport attributes disagree across ranks")
            self.instance_registrations.install_snapshot(
                registration,
                existing,
                existing_alive=existing_alive,
            )

    def deregister_instance(self, instance_id: str, rank: int, owner: ProcessRef) -> None:
        """Remove an instance-rank registration when the owner process requests it."""

        if instance_id not in get_global_config().instance_by_id:
            raise XpoolDaemonError("not_found", "unknown instance")
        self.validate_instance_rank(rank)
        self.instance_registrations.remove_owned(
            InstanceUniqId(instance_id=instance_id, rank=rank),
            owner,
        )

    def heartbeat_devagent(self, cuda_device: int, heartbeat: ProcessHeartbeat) -> HeartbeatResponse:
        """Refresh a devagent heartbeat and return global daemon warnings."""

        now = time.monotonic()
        self.devagent_registrations.heartbeat(
            cuda_device,
            heartbeat,
            now=now,
        )
        return HeartbeatResponse(warnings=self.global_warnings(now))

    def heartbeat_instance(self, instance_id: str, rank: int, heartbeat: ProcessHeartbeat) -> HeartbeatResponse:
        """Refresh an instance-rank heartbeat and return global daemon warnings."""

        now = time.monotonic()
        self.instance_registrations.heartbeat(
            InstanceUniqId(instance_id=instance_id, rank=rank),
            heartbeat,
            now=now,
        )
        return HeartbeatResponse(warnings=self.global_warnings(now))

    def upsert_devagent_transport_arenas(
        self,
        cuda_device: int,
        bindings: list[DevagentTransportArenaBinding],
        publisher: ProcessRef,
    ) -> None:
        """Upsert transport arenas if the owning devagent registration is live."""

        now = time.monotonic()
        devagent = get_global_config().devagent_by_cuda_device.get(cuda_device)
        if devagent is None:
            raise XpoolDaemonError("not_found", "unknown devagent")
        if devagent.role is not DeviceRole.ATN:
            raise XpoolDaemonError(
                "conflict",
                "only attention devagents may upsert transport arenas",
            )
        registration = self.devagent_registrations.query(cuda_device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "devagent must register before upserting transport arenas")
        registration.validate_process_ref(publisher, context="transport arena publisher")
        registration.require_online(now, context="local attention devagent")
        with self.transport_lock:
            if self.devagent_registrations.query(cuda_device) is not registration:
                raise XpoolDaemonError("not_ready", "devagent registration changed during arena publication")
            publications = self.validate_devagent_transport_arenas(cuda_device, bindings)
            existing_publications = registration.transport_arenas or {}
            existing_instance_by_handle = {
                publication.handle.handle: instance for instance, publication in existing_publications.items()
            }
            for publication in publications:
                existing = existing_publications.get(publication.instance)
                if existing is not None and existing != publication:
                    raise XpoolDaemonError("conflict", "published transport arena cannot be replaced in place")
                handle_owner = existing_instance_by_handle.get(publication.handle.handle)
                if handle_owner is not None and handle_owner != publication.instance:
                    raise XpoolDaemonError("conflict", "devagent transport arenas contain duplicate arena handle")
            self.devagent_registrations.upsert_transport_arenas(
                cuda_device,
                publications,
                registration=registration,
            )

    def drain_devagent_transport_arenas(
        self,
        cuda_device: int,
        publisher: ProcessRef,
    ) -> DevagentTransportArenaDrainResponse:
        """Mark transport arenas terminating and terminate every live lease owner."""

        now = time.monotonic()
        devagent = get_global_config().devagent_by_cuda_device.get(cuda_device)
        if devagent is None:
            raise XpoolDaemonError("not_found", "unknown devagent")
        if devagent.role is not DeviceRole.ATN:
            raise XpoolDaemonError(
                "conflict",
                "only attention devagents may drain transport arenas",
            )
        registration = self.devagent_registrations.query(cuda_device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "devagent must register before draining transport arenas")
        registration.validate_process_ref(publisher, context="transport arena drain")
        if registration.readiness_status(now) is ReadinessStatus.OFFLINE:
            raise XpoolDaemonError("not_ready", "local attention devagent process is not live")
        with self.transport_lock:
            self.devagent_registrations.drain_transport_arenas(
                cuda_device,
                registration=registration,
            )
        return DevagentTransportArenaDrainResponse(
            in_use=self.instance_registrations.terminate_transport_arena_leases(
                cuda_device,
                term_grace_s=TRANSPORT_DRAIN_TERM_GRACE_S,
            )
        )

    def validate_devagent_transport_arenas(
        self,
        cuda_device: int,
        bindings: list[DevagentTransportArenaBinding],
    ) -> list[TransportArenaPublication]:
        """Validate and normalize transport arenas published by one devagent."""

        atn_cuda_devices = get_global_config().devices.atn_cuda_devices
        seen: set[InstanceUniqId] = set()
        seen_handles: set[str] = set()
        publications: list[TransportArenaPublication] = []
        for binding in bindings:
            handle = binding.handle
            if handle.handle in seen_handles:
                raise XpoolDaemonError("conflict", "devagent transport arenas contain duplicate arena handle")
            seen_handles.add(handle.handle)
            if binding.instance_id not in get_global_config().instance_by_id:
                raise XpoolDaemonError("conflict", "devagent transport arena handle references unknown instance")
            self.validate_instance_rank(binding.rank)
            expected_cuda_device = atn_cuda_devices[binding.rank]
            if expected_cuda_device != cuda_device:
                raise XpoolDaemonError(
                    "conflict",
                    (
                        f"devagent transport arena handle rank {binding.rank} belongs to CUDA device "
                        f"{expected_cuda_device}, "
                        f"not {cuda_device}"
                    ),
                )
            instance = InstanceUniqId(instance_id=binding.instance_id, rank=binding.rank)
            if instance in seen:
                raise XpoolDaemonError("conflict", "devagent transport arenas contain duplicate instance-rank handle")
            registration = self.instance_registrations.rank_registration(
                binding.instance_id,
                binding.rank,
            )
            if registration is None:
                raise XpoolDaemonError(
                    "not_ready",
                    "devagent transport arena handle references an instance rank that is not registered",
                )
            publications.append(
                TransportArenaPublication(
                    instance=instance,
                    handle=handle,
                    transport=registration.transport,
                )
            )
            seen.add(instance)
        return publications

    def readiness_snapshot(
        self,
        selected_scopes: tuple[ReadinessScope, ...] = (ReadinessScope.ATN, ReadinessScope.FFN),
    ) -> ReadinessSnapshot:
        """Return daemon readiness for selected participant scopes."""

        now = time.monotonic()
        mps_status = self.mps_readiness_status()
        atn_cuda_devices = get_global_config().devices.atn_cuda_devices
        devagent_by_dev = dict(self.devagent_registrations.items())
        instance_by_uid = dict(self.instance_registrations.items())

        selected_roles = {
            role
            for role, scope in ((DeviceRole.ATN, ReadinessScope.ATN), (DeviceRole.FFN, ReadinessScope.FFN))
            if scope in selected_scopes
        }
        devagents = [
            ReadinessDevagent(
                pid=None if registration is None else registration.proc.pid,
                cuda_device=devagent.cuda_device,
                role=devagent.role,
                status=ReadinessStatus.OFFLINE if registration is None else registration.readiness_status(now),
            )
            for devagent in get_global_config().devagents
            if devagent.role in selected_roles
            for registration in (devagent_by_dev.get(devagent.cuda_device),)
        ]

        instances = [
            ReadinessInstance(
                pid=None if registration is None else registration.proc.pid,
                instance_id=instance.id,
                cuda_device=cuda_device,
                rank=rank,
                status=ReadinessStatus.OFFLINE if registration is None else registration.readiness_status(now),
            )
            for instance in get_global_config().instances
            if ReadinessScope.ATN in selected_scopes
            for rank, cuda_device in enumerate(atn_cuda_devices)
            for registration in (instance_by_uid.get(InstanceUniqId(instance_id=instance.id, rank=rank)),)
        ]
        atn_devagents_online = all(
            entry.status == ReadinessStatus.ONLINE for entry in devagents if entry.role is DeviceRole.ATN
        )
        ffn_devagents_online = all(
            entry.status == ReadinessStatus.ONLINE for entry in devagents if entry.role is DeviceRole.FFN
        )
        instances_online = all(entry.status == ReadinessStatus.ONLINE for entry in instances)
        arenas_ready = True
        if ReadinessScope.ATN in selected_scopes:
            for instance in get_global_config().instances:
                for rank, cuda_device in enumerate(atn_cuda_devices):
                    instance_uid = InstanceUniqId(instance_id=instance.id, rank=rank)
                    instance_registration = instance_by_uid.get(instance_uid)
                    devagent_registration = devagent_by_dev.get(cuda_device)
                    publication = (
                        None
                        if devagent_registration is None or devagent_registration.transport_arenas is None
                        else devagent_registration.transport_arenas.get(instance_uid)
                    )
                    if (
                        instance_registration is None
                        or devagent_registration is None
                        or devagent_registration.transport_arenas_terminating
                        or publication is None
                        or publication.transport != instance_registration.transport
                    ):
                        arenas_ready = False
                        break
                if not arenas_ready:
                    break
        participant_scopes = {
            scope: (
                atn_devagents_online and instances_online and arenas_ready
                if scope is ReadinessScope.ATN
                else ffn_devagents_online
            )
            for scope in selected_scopes
        }
        scopes = {scope: ready and mps_status is ReadinessStatus.ONLINE for scope, ready in participant_scopes.items()}
        return ReadinessSnapshot(
            ready=all(scopes.values()),
            mps_status=mps_status,
            scopes=scopes,
            cuda_devices=get_global_config().cuda_devices,
            devagents=devagents,
            instances=instances,
        )

    def acquire_instance_transport_arena(
        self,
        instance_id: str,
        *,
        rank: int,
        owner: ProcessRef,
    ) -> TransportArenaHandleRecord:
        """Acquire a daemon-brokered transport arena handle for one registered instance rank."""

        instance = get_global_config().instance_by_id.get(instance_id)
        if instance is None:
            raise XpoolDaemonError("not_found", "unknown instance")

        atn_cuda_devices = get_global_config().devices.atn_cuda_devices
        self.validate_instance_rank(rank)
        instance_uid = InstanceUniqId(instance_id=instance_id, rank=rank)
        cuda_device = atn_cuda_devices[rank]
        now = time.monotonic()
        registration = self.instance_registrations.rank_registration(instance_id, rank)
        if registration is None:
            raise XpoolDaemonError("not_ready", "instance rank is not registered")
        registration.validate_process_ref(owner, context="transport arena handle acquirer")
        registration.require_online(now, context="instance rank")
        devagent_registration = self.devagent_registrations.query(cuda_device)
        if devagent_registration is None:
            raise XpoolDaemonError("not_ready", "local attention devagent is not registered")
        devagent_registration.require_online(now, context="local attention devagent")
        with self.transport_lock:
            if self.instance_registrations.rank_registration(instance_id, rank) is not registration:
                raise XpoolDaemonError("not_ready", "instance registration changed during arena acquisition")
            publication, publisher = self.devagent_registrations.transport_arena(
                cuda_device,
                instance_uid,
                registration=devagent_registration,
            )
            if publication.transport != registration.transport:
                raise XpoolDaemonError("not_ready", "published transport arena geometry is stale")
            self.instance_registrations.acquire_transport_arena_lease(
                instance_uid,
                registration=registration,
                lease=TransportArenaLease(
                    cuda_device=cuda_device,
                    handle=publication.handle,
                    publisher=publisher,
                ),
                now=now,
            )
        return publication.handle

    def validate_instance_rank(self, rank: int) -> None:
        """Reject an instance rank outside the configured ATN world."""

        if rank < 0 or rank >= get_global_config().atn_world_size:
            raise XpoolDaemonError(
                "conflict",
                f"instance rank {rank} is outside atn_world_size={get_global_config().atn_world_size}",
            )
