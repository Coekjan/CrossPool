"""Authoritative orchestration for the daemon control plane."""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, sleep

import psutil

import xpool.native
from xpool import ffn
from xpool.config import FfnSchedulingPolicy, get_global_config
from xpool.fabric import (
    FabricGenerationId,
    FabricGenerationPhase,
    FabricParticipantPhase,
    FabricPlan,
    FabricRole,
    FabricUid,
    FifoSchedulerPolicy,
    RandomSchedulerPolicy,
)
from xpool.model import ModelId
from xpool.native import ABI_VERSION
from xpool.service.daemon.fabric import FabricController, FabricGenerationState, FabricMembership
from xpool.service.daemon.ffn_placement import place_ffn_models
from xpool.service.daemon.kv import KvCapacityPolicy
from xpool.service.daemon.readiness import ControlPlaneProjection
from xpool.service.daemon.registration import (
    HEARTBEAT_WARNING_WATERMARK_S,
    AtnAgentRegistrationState,
    FfnAgentRegistrationState,
    InstanceRankId,
    InstanceRankRegistrationState,
    RegistrationBook,
)
from xpool.service.daemon.transport import (
    TransportArenaLease,
    TransportArenaPublication,
    TransportBroker,
)
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    AgentStartupAdmission,
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    AtnAgentTransportLeaseQuiesceResponse,
    ConfigCheckRequest,
    ControlPlaneWarning,
    FabricInstanceRankOwnerFailure,
    FabricOwnerFailureReason,
    FabricParticipantReport,
    FabricPeOwnerFailure,
    FabricQuiesceRequest,
    FfnAgentRegistration,
    HeartbeatResponse,
    InstanceRankInitializedPublication,
    InstanceRankRef,
    InstanceRankRegistration,
    KvControlChannelRef,
    MpsClientTermination,
    ProcessRef,
    ReadinessSnapshot,
    ReadinessStatus,
    ServingListener,
)
from xpool.transport import TransportArenaHandle
from xpool.utils.device import normalize_environment, visible_uuids
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S, MpsEndpoint, MpsProbeResult, MpsScope
from xpool.utils.procs import ProcUniqId
from xpool.utils.sighandler import defer_signal_exceptions

TRANSPORT_ARENA_LEASE_HEARTBEAT_TIMEOUT_S = 2.0 * HEARTBEAT_WARNING_WATERMARK_S
FABRIC_PHASE_TIMEOUT_S = {
    FabricGenerationPhase.JOINING: 60.0,
    FabricGenerationPhase.QUIESCING: 60.0,
    FabricGenerationPhase.DRAINING: 60.0,
    FabricGenerationPhase.FINALIZING: 60.0,
}
INSTANCE_STARTUP_TIMEOUT_S = 600.0
GLOBAL_WARNING_CACHE_S = 1.0
MPS_READINESS_CACHE_S = 1.0
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServingStartupState:
    """Generation-scoped Instance listeners and serving-health confirmation."""

    generation: FabricGenerationId | None = None
    listeners: dict[ModelId, ServingListener] = field(default_factory=dict)
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class ServingHealthTargets:
    """Immutable ordered listener snapshot for one Fabric generation."""

    generation: FabricGenerationId
    listeners: tuple[tuple[ModelId, ServingListener], ...]


class ControlPlane:
    """Authoritative daemon control plane shared by FastAPI route handlers.

    Attributes:
        started_at: Unix timestamp recorded when the daemon state was created.
        registrations: Process identities and declared registration contracts.
        transport_broker: Transport publications and Instance-rank leases.
        fabric_controller: Installed Fabric generation state.
        serving_startup: Generation-scoped Instance listeners and one-time
            serving-health confirmation.
        fabric_plan_formation_lock: Serializes expensive generation formation
            without blocking ordinary domain reads and writes.
        lock: Reentrant domain lock protecting every mutable control-plane
            record and cross-module invariant.
        warning_cache_at: Monotonic timestamp of the cached warning snapshot.
        warning_cache: Warning snapshot reused for one cache interval to avoid
            repeated process probes on every heartbeat.
    """

    def __init__(self) -> None:
        """Create an empty control plane with one reentrant domain lock."""

        self.started_at = time.time()
        self.lock = threading.RLock()
        self.fabric_plan_formation_lock = threading.Lock()
        self.registrations = RegistrationBook()
        self.transport_broker = TransportBroker()
        self.fabric_controller = FabricController()
        self.kv_capacity_policy: KvCapacityPolicy | None = None
        self.serving_startup = ServingStartupState()
        self.membership_revision = 0
        self.warning_cache_at = float("-inf")
        self.warning_cache: tuple[ControlPlaneWarning, ...] = ()
        self.mps_cache_at = float("-inf")
        self.mps_cache_result: MpsProbeResult | None = None
        self.mps_scope: MpsScope | None = None
        self.device_uuids: tuple[str, ...] | None = None
        self.admission_closed = False
        self.cleanup_lock = threading.Lock()
        self.cleanup_deadline: float | None = None
        self.closed = False

    def start(self) -> None:
        """Resolve placement and start the retained attention-only controller.

        The application retains this control plane before startup.
        Partial startup leaves all resources on
        this owner for ``close``. Startup and close serialize on the same lock,
        while a signal can immediately close admission and record its deadline.

        Raises:
            ValueError: Configured placement exceeds original visibility.
            RuntimeError: Startup was already attempted or controller startup fails.
            OSError: Device observation or controller creation fails.
            TimeoutError: A bounded startup operation expires.

        Side Effects:
            Observes physical device visibility and
            launches the foreground MPS controller over attention devices only.
        """

        with self.cleanup_lock:
            if self.closed or self.device_uuids is not None:
                raise RuntimeError("daemon resource startup was already attempted")
            if self.admission_closed:
                return
            config = get_global_config()
            normalize_environment()
            self.device_uuids = visible_uuids()
            devices = config.devices
            if any(device >= len(self.device_uuids) for device in devices):
                raise ValueError("daemon placement exceeds original device visibility")
            if self.admission_closed:
                return
            with defer_signal_exceptions():
                self.mps_scope = MpsScope(
                    MpsEndpoint(tuple(self.device_uuids[device] for device in config.atn.devices))
                )
            if self.admission_closed:
                return
            try:
                self.mps_scope.start()
            except InterruptedError:
                # The daemon signal handler records retirement instead of
                # throwing. The scope stops creating resources at that boundary.
                if not self.admission_closed:
                    raise

    def admit_agent_startup(self, request: AgentStartupAdmission) -> None:
        """Retain a configured Agent before any device initialization.

        Configured role/device and exact process identity are validated at
        admission; formal registration must retain this same identity. Partial
        construction remains owned until the admitted process exits.

        Raises:
            XpoolDaemonError: ABI, configured placement or process identity
                conflicts, or startup/controller admission is unavailable.
            OSError: Ownership-critical process observation fails.
        """

        with self.lock:
            scope = self.mps_scope
            visibility = self.device_uuids
            if self.admission_closed or scope is None or visibility is None:
                raise XpoolDaemonError("not_ready", "Agent startup admission is unavailable")
            config = get_global_config()
            devices = config.atn.devices if request.role is FabricRole.ATNAGENT else config.ffn.devices
            if request.abi_version != ABI_VERSION:
                raise XpoolDaemonError("conflict", "Agent startup ABI version does not match daemon ABI")
            if request.device not in devices:
                raise XpoolDaemonError("conflict", "Agent startup role placement does not match daemon")
            fabric = self.fabric_controller.generation
            if fabric is not None and not any(
                placement.role is request.role
                and placement.device == request.device
                and fabric.agent_owners[pe].pid == request.pid
                and fabric.agent_owners[pe].create_time == request.create_time
                for pe, placement in enumerate(fabric.plan.pe_placements)
            ):
                raise XpoolDaemonError("not_ready", "retained Fabric generation forbids new Agent startup")
        try:
            identity = ProcUniqId(request.pid)
        except psutil.NoSuchProcess as error:
            raise XpoolDaemonError("conflict", "Agent startup process is absent") from error
        if identity.create_time != request.create_time or not identity.is_alive():
            raise XpoolDaemonError("conflict", "Agent startup process identity does not match")
        if psutil.Process(identity.pid).uids().real != os.getuid():
            raise XpoolDaemonError("conflict", "Agent startup belongs to another user")
        namespace = Path(f"/proc/{identity.pid}/ns/pid").stat()
        own_namespace = Path("/proc/self/ns/pid").stat()
        if (namespace.st_dev, namespace.st_ino) != (own_namespace.st_dev, own_namespace.st_ino):
            raise XpoolDaemonError("conflict", "Agent startup requires the daemon's PID namespace")
        if scope.probe().online is not True:
            raise XpoolDaemonError("not_ready", "owned MPS controller is unavailable for Agent startup")
        with self.lock:
            if self.admission_closed or self.mps_scope is not scope or self.fabric_controller.generation is not fabric:
                raise XpoolDaemonError("not_ready", "Agent startup admission changed during validation")
            key = (request.role, request.device)
            existing = self.registrations.agent_startups.get(key)
            if existing is not None and existing != identity and existing.is_alive():
                raise XpoolDaemonError("conflict", "Agent startup device is owned by another live process")
            self.registrations.agent_startups[key] = identity

    def terminate_serving_client(self, request: MpsClientTermination) -> None:
        """Terminate contexts of an exact client of the retained MPS scope.

        This retirement operation remains available after startup admission
        closes. It does not signal clients, stop MPS or finalize
        Fabric. The serving owner retains responsibility for subsequent exit.

        Args:
            request: Target identity, matching ABI and monotonic cleanup bound.

        Raises:
            XpoolDaemonError: Target/ABI conflicts or owned MPS termination
                cannot be confirmed before the applicable cleanup deadline.
            OSError: Process-domain observation cannot establish ownership.
        """

        with self.lock:
            scope = self.mps_scope
            if scope is None or self.closed:
                raise XpoolDaemonError("not_ready", "owned MPS client termination is unavailable")
            if request.abi_version != ABI_VERSION:
                raise XpoolDaemonError("conflict", "MPS termination ABI version does not match daemon ABI")
            try:
                identity = ProcUniqId(request.pid)
                if identity.create_time != request.create_time or not identity.is_alive():
                    raise XpoolDaemonError("conflict", "MPS termination target identity does not match")
            except (psutil.NoSuchProcess, ProcessLookupError) as error:
                raise XpoolDaemonError("conflict", "MPS termination target is absent") from error
            deadline = request.deadline
            if self.cleanup_deadline is not None:
                deadline = min(deadline, self.cleanup_deadline)
        try:
            scope.terminate_client(identity, deadline=deadline)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, psutil.Error) as error:
            raise XpoolDaemonError("not_ready", f"owned MPS client termination is unconfirmed: {error}") from error

    def refresh_mps_status(self, now: float) -> None:
        """Refresh MPS readiness outside the domain lock when its cache is stale."""

        with self.lock:
            if self.mps_cache_result is not None and now - self.mps_cache_at < MPS_READINESS_CACHE_S:
                return
            previous = self.mps_cache_result
            scope = self.mps_scope
        try:
            result = (
                MpsProbeResult(False, None, "owned MPS controller has not started") if scope is None else scope.probe()
            )
        except subprocess.TimeoutExpired as error:
            result = MpsProbeResult(False, None, f"MPS management command exceeded {error.timeout}s")
        except TimeoutError:
            if scope is None or scope.cleanup_deadline is None or monotonic() < scope.cleanup_deadline:
                raise
            result = MpsProbeResult(None, None, "MPS observation unavailable after cleanup deadline")
        with self.lock:
            self.mps_cache_at = now
            self.mps_cache_result = result
        if previous is None or previous.online != result.online:
            log = logger.warning if result.online is False else logger.info
            status = "unavailable" if result.online is None else "online" if result.online else "offline"
            log("mps readiness changed status=%s detail=%s", status, result.diagnostic)

    def check_config(self, request: ConfigCheckRequest) -> None:
        """Require agreement on effective configuration and ordered deployment UUIDs.

        Args:
            request: Validated effective config and visibility submitted by a runtime
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
            request.config.model_dump(mode="json"),
            get_global_config().model_dump(mode="json"),
            "",
        )
        compare(request.visible_devices, list(visible_uuids()), "visible_devices")
        if differences:
            raise XpoolDaemonError(
                "conflict",
                "client xpool config differs from daemon config:\n" + "\n".join(differences),
            )

    def list_atnagents(self) -> list[AtnAgentRegistration]:
        """Return the current AtnAgent registration wire views."""

        with self.lock:
            return self.registrations.atnagent_views()

    def list_ffnagents(self) -> list[FfnAgentRegistration]:
        """Return the current FfnAgent registration wire views."""

        with self.lock:
            return self.registrations.ffnagent_views()

    def list_instances(self) -> list[InstanceRankRegistration]:
        """Return the current Instance-rank registration wire views."""

        with self.lock:
            return self.registrations.instance_views()

    def global_warnings(self, now: float) -> list[ControlPlaneWarning]:
        """Return daemon-owned warnings and log device-scoped state edges.

        Returned warning details remain per contributor. Logging aggregates
        them so the first warning for a ``(kind, device)`` condition emits its
        entry edge and removal of the final matching warning emits its clear
        edge.
        """

        with self.lock:
            if now - self.warning_cache_at < GLOBAL_WARNING_CACHE_S:
                return list(self.warning_cache)
            projection = ControlPlaneProjection.capture(
                config=get_global_config(),
                now=now,
                mps_online=None if self.mps_cache_result is None else self.mps_cache_result.online,
                registrations=self.registrations,
                transport=self.transport_broker,
                fabric=self.fabric_controller,
            )
            previous_warnings = self.warning_cache
            self.warning_cache_at = now
            self.warning_cache = projection.warnings
            warnings = projection.warnings
        previous_by_key = {(warning.kind, warning.device) for warning in previous_warnings}
        current_by_key = {(warning.kind, warning.device) for warning in warnings}
        for kind, device in sorted(current_by_key - previous_by_key):
            logger.warning("global warning entered kind=%s device=%s", kind, device)
        for kind, device in sorted(previous_by_key - current_by_key):
            logger.info("global warning cleared kind=%s device=%s", kind, device)
        return list(warnings)

    def retire_terminal_generation(self) -> None:
        """Retire terminal generation state after every old owner has exited."""

        with self.lock:
            fabric = self.fabric_controller.generation
            if fabric is None or fabric.phase is not FabricGenerationPhase.STOPPED:
                return
            owners = tuple({*fabric.agent_owners.values(), *fabric.instance_owners.values()})
        any_alive = any(owner.is_alive() for owner in owners)
        with self.lock:
            if self.fabric_controller.generation is fabric and not any_alive:
                policy = self.kv_capacity_policy
                if policy is not None:
                    policy.close()
                    self.kv_capacity_policy = None
                self.fabric_controller.generation = None
                self.serving_startup = ServingStartupState()

    def register_atnagent(self, registration: AtnAgentRegistrationState) -> None:
        """Install an AtnAgent registration after its prior arena users retire.

        Args:
            registration: Candidate atnagent process registration.

        Raises:
            XpoolDaemonError: If the device or ABI is invalid, a conflicting
                generation remains live, or prior arena users remain live.

        """

        self.retire_terminal_generation()
        if registration.device not in get_global_config().atnagent_by_device:
            raise XpoolDaemonError("not_found", "unknown AtnAgent")
        if registration.abi_version != ABI_VERSION:
            raise XpoolDaemonError("conflict", "atnagent ABI version does not match daemon ABI")
        with self.lock:
            if self.registrations.agent_startups.get((FabricRole.ATNAGENT, registration.device)) != registration.proc:
                if self.admission_closed:
                    raise XpoolDaemonError("not_ready", "participant admission is closed")
                raise XpoolDaemonError("conflict", "AtnAgent registration requires its admitted startup identity")
            existing = self.registrations.atnagents.query(registration.device)
            fabric = self.fabric_controller.generation
            if fabric is not None:
                pe = next(
                    pe
                    for pe, item in enumerate(fabric.plan.pe_placements)
                    if item.role is FabricRole.ATNAGENT and item.device == registration.device
                )
                if fabric.agent_owners[pe] != registration.proc:
                    fabric.record_owner_failure(
                        FabricPeOwnerFailure(
                            role="atnagent",
                            pe=pe,
                            reason=FabricOwnerFailureReason.REPLACED,
                        )
                    )
                    self.fabric_controller.abort(now=monotonic())
                    raise XpoolDaemonError("conflict", "retained Fabric generation forbids AtnAgent replacement")
        existing_alive = False if existing is None else existing.proc.is_alive()
        if existing is not None and existing.proc != registration.proc and not existing_alive:
            with self.lock:
                leased_instances = self.transport_broker.leased_instances(registration.device, existing.proc)
                owners = [
                    owner
                    for instance in leased_instances
                    if (owner := self.registrations.instances.query(instance)) is not None
                ]
            if any(owner.proc.is_alive() for owner in owners):
                raise XpoolDaemonError("not_ready", "previous AtnAgent still has live arena users")
            with self.lock:
                if self.registrations.atnagents.query(registration.device) is not existing:
                    raise XpoolDaemonError("not_ready", "atnagent registration changed during generation cleanup")
        with self.lock:
            if self.registrations.agent_startups.get((FabricRole.ATNAGENT, registration.device)) != registration.proc:
                raise XpoolDaemonError("conflict", "AtnAgent startup ownership changed during registration")
            changed = self.registrations.atnagents.install_snapshot(
                registration,
                existing,
                existing_alive=existing_alive,
            )
            installed = self.registrations.atnagents.query(registration.device)
            if installed is None:
                raise RuntimeError("AtnAgent registration disappeared during installation")
            self.transport_broker.install_atnagent(registration.device, installed.proc)
            if changed:
                self.membership_revision += 1
                logger.info(
                    "registration accepted role=atnagent device=%s pid=%s",
                    registration.device,
                    registration.proc.pid,
                )
        self.ensure_fabric_plan()

    def register_ffnagent(
        self,
        registration: FfnAgentRegistrationState,
        model_specs: tuple[ffn.FfnModelSpec, ...],
    ) -> None:
        """Install a configured FfnAgent process registration.

        Args:
            registration: Candidate FfnAgent process registration.
            model_specs: Complete ordered Model Specs declared by the process.

        Raises:
            XpoolDaemonError: If placement or ABI is invalid or a different
                live process owns the configured device.
        """

        self.retire_terminal_generation()
        if registration.device not in get_global_config().ffnagent_by_device:
            raise XpoolDaemonError("not_found", "unknown FfnAgent")
        if registration.abi_version != ABI_VERSION:
            raise XpoolDaemonError("conflict", "ffnagent ABI version does not match daemon ABI")
        with self.lock:
            if self.registrations.agent_startups.get((FabricRole.FFNAGENT, registration.device)) != registration.proc:
                if self.admission_closed:
                    raise XpoolDaemonError("not_ready", "participant admission is closed")
                raise XpoolDaemonError("conflict", "FfnAgent registration requires its admitted startup identity")
            existing = self.registrations.ffnagents.query(registration.device)
            fabric = self.fabric_controller.generation
            if fabric is not None:
                pe = next(
                    pe
                    for pe, item in enumerate(fabric.plan.pe_placements)
                    if item.role is FabricRole.FFNAGENT and item.device == registration.device
                )
                if fabric.agent_owners[pe] != registration.proc:
                    fabric.record_owner_failure(
                        FabricPeOwnerFailure(
                            role="ffnagent",
                            pe=pe,
                            reason=FabricOwnerFailureReason.REPLACED,
                        )
                    )
                    self.fabric_controller.abort(now=monotonic())
                    raise XpoolDaemonError("conflict", "retained Fabric generation forbids FfnAgent replacement")
        existing_alive = False if existing is None else existing.proc.is_alive()
        with self.lock:
            if self.registrations.agent_startups.get((FabricRole.FFNAGENT, registration.device)) != registration.proc:
                raise XpoolDaemonError("conflict", "FfnAgent startup ownership changed during registration")
            canonical_specs = self.registrations.ffn_model_specs
            if canonical_specs is not None and model_specs != canonical_specs:
                raise XpoolDaemonError("conflict", "FfnAgent Model Specs disagree with the canonical declaration")
            changed = self.registrations.ffnagents.install_snapshot(
                registration,
                existing,
                existing_alive=existing_alive,
            )
            if canonical_specs is None:
                self.registrations.ffn_model_specs = model_specs
            if changed:
                self.membership_revision += 1
                logger.info(
                    "registration accepted role=ffnagent device=%s pid=%s",
                    registration.device,
                    registration.proc.pid,
                )
        self.ensure_fabric_plan()

    def register_instance(self, registration: InstanceRankRegistrationState) -> None:
        """Install a live Instance worker with matching physical placement.

        Ordered attention UUIDs must match the retained endpoint. Registration
        commit shares the domain lock with closing participant admission.
        """

        self.retire_terminal_generation()
        model_id = registration.instance.model_id
        rank = registration.instance.rank
        if registration.abi_version != ABI_VERSION:
            raise XpoolDaemonError("conflict", "instance ABI version does not match daemon ABI")
        self.validate_instance_rank(model_id, rank)
        transport = registration.transport
        config = get_global_config()
        model = config.model_by_id[model_id]
        if (transport.atn_tp_size, transport.atn_dp_size) != (
            config.atn_tp_size_of(model_id),
            model.atn_dp_size,
        ):
            raise XpoolDaemonError(
                "conflict",
                "instance transport TP-by-DP topology disagrees with configured model geometry",
            )
        if transport.atn_dp_rank * transport.atn_tp_size + transport.atn_tp_rank != rank:
            raise XpoolDaemonError("conflict", "instance transport coordinates do not use TP-fastest rank order")
        with self.lock:
            if self.admission_closed:
                raise XpoolDaemonError("not_ready", "participant admission is closed")
            existing = self.registrations.instances.query(registration.instance)
            fabric = self.fabric_controller.generation
            if fabric is not None:
                expected_owner = fabric.instance_owners.get(registration.instance)
                if expected_owner != registration.proc:
                    fabric.record_owner_failure(
                        FabricInstanceRankOwnerFailure(
                            model_id=model_id,
                            rank=rank,
                            reason=FabricOwnerFailureReason.REPLACED,
                        )
                    )
                    self.fabric_controller.quiesce(now=monotonic())
                    raise XpoolDaemonError("conflict", "retained Fabric generation forbids Instance-rank replacement")
        existing_alive = False if existing is None else existing.proc.is_alive()
        with self.lock:
            if self.admission_closed:
                raise XpoolDaemonError("not_ready", "participant admission is closed")
            if self.registrations.instances.query(registration.instance) is not existing:
                raise XpoolDaemonError("not_ready", "instance registration changed during validation")
            if self.mps_scope is None:
                raise XpoolDaemonError("not_ready", "owned attention MPS scope is unavailable")
            for peer in self.registrations.instances.values():
                if peer.instance.model_id != model_id or peer.instance == registration.instance:
                    continue
                peer_contract = (
                    peer.transport.hidden_size,
                    peer.transport.payload_row_capacity,
                    peer.transport.atn_tp_size,
                    peer.transport.atn_dp_size,
                )
                candidate_contract = (
                    transport.hidden_size,
                    transport.payload_row_capacity,
                    transport.atn_tp_size,
                    transport.atn_dp_size,
                )
                if peer_contract != candidate_contract:
                    raise XpoolDaemonError("conflict", "instance transport attributes disagree across ranks")
                if peer.ffn_profile != registration.ffn_profile:
                    raise XpoolDaemonError("conflict", "FFN ffn_profile disagrees across instance ranks")
                if peer.transport.atn_dp_rank == transport.atn_dp_rank and peer.kv_capacity != registration.kv_capacity:
                    raise XpoolDaemonError("conflict", "kv capacity geometry disagrees across instance ranks")
            changed = self.registrations.instances.install_snapshot(
                registration,
                existing,
                existing_alive=existing_alive,
            )
            if changed:
                self.transport_broker.remove_instance(registration.instance)
                self.membership_revision += 1
                logger.info(
                    "registration accepted role=instance instance=%s rank=%s pid=%s device=%s",
                    registration.instance.model_id,
                    registration.instance.rank,
                    registration.proc.pid,
                    get_global_config().atn.devices[registration.instance.rank],
                )
        self.ensure_fabric_plan()

    def deregister_instance(self, model_id: ModelId, rank: int, owner: ProcessRef) -> None:
        """Remove an instance-rank registration when the owner process requests it."""

        self.validate_instance_rank(model_id, rank)
        instance = InstanceRankId(model_id=model_id, rank=rank)
        with self.lock:
            registration = self.registrations.instances.query(instance)
        if registration is None:
            raise XpoolDaemonError("not_found", "registration is not registered")
        registration.validate_process_ref(owner, context="deregister")
        with self.lock:
            self.registrations.instances.remove_snapshot(instance, registration)
            lease = self.transport_broker.remove_instance(instance)
            self.membership_revision += 1
            self.fabric_controller.instance_departed(
                instance,
                FabricInstanceRankOwnerFailure(
                    model_id=model_id,
                    rank=rank,
                    reason=FabricOwnerFailureReason.EXITED,
                ),
                leases_quiescing=lease is not None and self.transport_broker.is_quiescing(lease.device),
                now=monotonic(),
            )
        logger.info(
            "registration removed role=instance instance=%s rank=%s pid=%s",
            model_id,
            rank,
            registration.proc.pid,
        )

    def ensure_fabric_plan(self) -> FabricPlan | None:
        """Create a Fabric plan from one complete membership snapshot.

        Native UID creation runs outside the domain lock. The candidate is
        committed only if its membership revision still matches.
        """

        with self.fabric_plan_formation_lock:
            config = get_global_config()
            # Capture: freeze complete membership while the registration revision
            # and current-generation check are protected by the domain lock.
            with self.lock:
                if self.fabric_controller.generation is not None:
                    return self.fabric_controller.generation.plan
                if self.admission_closed:
                    return None
                membership = FabricMembership.capture(config, self.registrations, self.membership_revision)

            # Validate: reject incomplete or dead membership without holding the
            # domain lock or creating generation resources.
            if membership is None or not all(owner.is_alive() for owner in membership.owners()):
                return None

            # Plan: materialize the scheduler and immutable plan without
            # holding the domain lock.
            plan_started_at = monotonic()
            if config.scheduler.ffn_policy is FfnSchedulingPolicy.FIFO:
                scheduler = FifoSchedulerPolicy()
            else:
                seed = config.scheduler.ffn_random_seed
                while seed is None or seed == 0:
                    seed = secrets.randbits(64)
                scheduler = RandomSchedulerPolicy(seed=seed)
            model_plans = place_ffn_models(
                model_specs=membership.model_specs,
                instance_plans=membership.instance_plans,
                ffnagent_free_memory_bytes=membership.ffnagent_free_memory_bytes,
            )
            if not all(owner.is_alive() for owner in membership.owners()):
                return None
            plan = FabricPlan(
                generation=FabricGenerationId.create(),
                uid=FabricUid(value=xpool.native.fabric.create_uid()),
                pe_placements=membership.pe_placements,
                executor_lane_count=config.scheduler.ffn_concurrency,
                scheduler=scheduler,
                model_plans=model_plans,
                instance_plans=membership.instance_plans,
            )
            policy = KvCapacityPolicy.create(config, plan.generation)
            # Commit: install only if no generation appeared and the captured
            # membership revision still describes the authoritative registration set.
            with self.lock:
                if self.fabric_controller.generation is not None:
                    policy.close()
                    return self.fabric_controller.generation.plan
                if self.admission_closed or self.membership_revision != membership.revision:
                    policy.close()
                    return None
                installed_plan = self.fabric_controller.install(
                    FabricGenerationState(
                        plan=plan,
                        phase=FabricGenerationPhase.PREPARING_JOIN,
                        phase_started_at=monotonic(),
                        invocation_failure=None,
                        owner_failure=None,
                        control_failure=None,
                        agent_owners=dict(membership.agent_owners),
                        instance_owners=dict(membership.instance_owners),
                    )
                )
                self.kv_capacity_policy = policy
                self.serving_startup = ServingStartupState(generation=installed_plan.generation)
            logger.info(
                "fabric plan installed generation=%s model_count=%s pe_count=%s elapsed=%.3fs",
                installed_plan.generation.format(),
                len(installed_plan.model_plans),
                len(installed_plan.pe_placements),
                monotonic() - plan_started_at,
            )
            return installed_plan

    def require_fabric_plan(self) -> FabricPlan:
        """Return the installed immutable Fabric plan without forming one."""

        with self.lock:
            return self.fabric_controller.require_plan()

    def kv_control_channel(self, generation: FabricGenerationId) -> KvControlChannelRef:
        """Return the KV control channel for the retained Fabric generation."""

        with self.lock:
            fabric = self.fabric_controller.generation
            policy = self.kv_capacity_policy
            if fabric is None or policy is None or fabric.plan.generation != generation:
                raise XpoolDaemonError("not_found", "kv control channel generation is not retained")
            return policy.channel_ref

    def step_kv_capacity(self) -> None:
        """Advance one Generation-scoped KV policy transition."""

        with self.lock:
            fabric = self.fabric_controller.generation
            policy = self.kv_capacity_policy
            if fabric is None or policy is None:
                return
            if fabric.phase is not FabricGenerationPhase.EXECUTABLE:
                return
            try:
                policy.step(self.registrations, fabric)
            except Exception as error:
                fabric.record_control_failure(f"kv capacity policy failed: {error}")
                self.fabric_controller.abort(now=monotonic())
                raise

    def begin_close(self, *, deadline: float | None = None) -> float:
        """Seal admission and retain the first retirement trigger's deadline.

        This synchronous transition is shared by signals, background failures
        and direct close. It leaves control reports available while the blocking
        cleanup operation runs. Subsequent triggers preserve the same budget.

        Args:
            deadline: Absolute monotonic deadline, or ``None`` to start the
                production cleanup budget at this trigger.

        Returns:
            The daemon-owned absolute monotonic cleanup deadline.
        """

        with self.lock:
            self.admission_closed = True
            if self.cleanup_deadline is None:
                self.cleanup_deadline = monotonic() + MPS_CLEANUP_TIMEOUT_S if deadline is None else deadline
            if self.mps_scope is not None and self.mps_scope.cleanup_deadline is None:
                self.mps_scope.cleanup_deadline = self.cleanup_deadline
            return self.cleanup_deadline

    def close(self, *, deadline: float | None = None) -> None:
        """Retire known participants and actual clients before releasing the controller.

        Concurrent callers join the same cleanup operation. The first caller
        retains its absolute monotonic deadline, including any preceding wait
        after a signal. Expiry or failed observation keeps this owner alive;
        automatic management stops, while confirmed later retirement can still
        complete cleanup. Control routes and background coordination must remain
        usable until this operation returns.

        Args:
            deadline: First trigger's absolute monotonic deadline. Direct close
                starts the production cleanup budget when none is supplied.

        Side Effects:
            Closes admission, waits for known Instance owners, requests
            Agent/Fabric quiesce, stops only the retained MPS scope, closes KV
            state after verified process exit.
            Neither ordinary waiting nor expiry signals serving processes.
        """

        deadline = self.begin_close(deadline=deadline)
        with self.cleanup_lock:
            if self.closed:
                return
            with self.lock:
                scope = self.mps_scope
            logger.info("shutdown started pid=%s", os.getpid())
            last_diagnostic: tuple[type[Exception], str] | None = None
            expiry_reported = False
            notified_agents: set[ProcUniqId] = set()
            while True:
                now = monotonic()
                if now >= deadline and not expiry_reported:
                    logger.error(
                        "shutdown deadline expired; retaining ownership; manual resolution required mps_directory=%s",
                        None if scope is None else scope.endpoint.directory,
                    )
                    expiry_reported = True
                try:
                    with self.lock:
                        fabric = self.fabric_controller.generation
                        owners = tuple(
                            {
                                *(registration.proc for registration in self.registrations.all_values()),
                                *self.registrations.agent_startups.values(),
                                *(() if fabric is None else fabric.agent_owners.values()),
                                *(() if fabric is None else fabric.instance_owners.values()),
                            }
                        )
                        scope = self.mps_scope
                        instance_owners = {
                            *(registration.proc for registration in self.registrations.instances.values()),
                            *(() if fabric is None else fabric.instance_owners.values()),
                        }
                        pre_join_agents = (
                            (*self.registrations.atnagents.values(), *self.registrations.ffnagents.values())
                            if fabric is None
                            else ()
                        )
                    # Agent.run installs its handlers before formal registration.
                    # Already-admitted startup may publish that safe pre-join
                    # boundary during close; new startup remains sealed.
                    if now < deadline:
                        for registration in pre_join_agents:
                            if registration.proc not in notified_agents:
                                registration.proc.send_signal(signal.SIGTERM)
                                notified_agents.add(registration.proc)
                    if not any(owner.is_alive() for owner in instance_owners):
                        with self.lock:
                            if fabric is not None and self.fabric_controller.generation is fabric and now < deadline:
                                self.fabric_controller.quiesce(now=now)
                        if not any(owner.is_alive() for owner in owners):
                            # After expiry, only a controller already exited can
                            # complete housekeeping; no new command gets a budget.
                            if scope is not None:
                                if now >= deadline and scope.controller is not None and scope.controller.poll() is None:
                                    sleep(0.1)
                                    continue
                                scope.stop(deadline=deadline)
                            with self.lock:
                                if self.kv_capacity_policy is not None:
                                    self.kv_capacity_policy.close()
                                    self.kv_capacity_policy = None
                                self.closed = True
                            logger.info("shutdown completed pid=%s", os.getpid())
                            return
                    last_diagnostic = None
                except Exception as error:
                    diagnostic = (type(error), str(error))
                    if diagnostic != last_diagnostic:
                        logger.error("shutdown incomplete; retaining ownership detail=%s", error)
                        last_diagnostic = diagnostic
                sleep(0.1)

    def request_fabric_quiesce(self, request: FabricQuiesceRequest) -> None:
        """Authenticate an Agent owner and stop generation admission."""

        with self.lock:
            fabric = self.fabric_controller.generation
            if fabric is None or request.generation != fabric.plan.generation:
                raise XpoolDaemonError("conflict", "fabric quiesce generation is not retained")
            owner_matches = 0
            for pe, placement in enumerate(fabric.plan.pe_placements):
                registration = (
                    self.registrations.atnagents.query(placement.device)
                    if placement.role is FabricRole.ATNAGENT
                    else self.registrations.ffnagents.query(placement.device)
                )
                if (
                    registration is not None
                    and registration.proc == fabric.agent_owners[pe]
                    and registration.proc.pid == request.owner.pid
                    and registration.abi_version == request.owner.abi_version
                ):
                    owner_matches += 1
            if owner_matches != 1:
                raise XpoolDaemonError("conflict", "fabric quiesce owner is not a current Agent participant")
            self.fabric_controller.quiesce(now=monotonic())

    def record_fabric_participant(
        self,
        report: FabricParticipantReport,
    ) -> None:
        """Commit one owner-validated participant report."""

        with self.lock:
            now = monotonic()
            fabric = self.fabric_controller.generation
            if fabric is None:
                raise XpoolDaemonError("conflict", "participant reported a retired Fabric generation")
            if report.pe < 0 or report.pe >= len(fabric.plan.pe_placements):
                self.fabric_controller.reject_report("participant reported an unknown Fabric PE", now=now)
            placement = fabric.plan.pe_placements[report.pe]
            registration = (
                self.registrations.atnagents.query(placement.device)
                if placement.role is FabricRole.ATNAGENT
                else self.registrations.ffnagents.query(placement.device)
            )
            try:
                if registration is None or registration.proc != fabric.agent_owners[report.pe]:
                    raise XpoolDaemonError("conflict", "Fabric participant registration does not match its plan owner")
                registration.validate_process_ref(report.owner, context="Fabric participant report")
            except XpoolDaemonError:
                self.fabric_controller.reject_report("Fabric participant report owner does not match its PE", now=now)
            self.fabric_controller.record_participant(report, now=now)

    def publish_instance_initialized(
        self,
        model_id: ModelId,
        *,
        rank: int,
        publication: InstanceRankInitializedPublication,
    ) -> None:
        """Record one Instance Rank's scheduler construction and listener."""

        self.validate_instance_rank(model_id, rank)
        instance = InstanceRankId(model_id=model_id, rank=rank)
        with self.lock:
            registration = self.registrations.instances.query(instance)
        if registration is None:
            raise XpoolDaemonError("not_ready", "instance rank must register before initialization barrier")
        registration.validate_process_ref(publication.owner, context="instance initialized")
        with self.lock:
            if self.fabric_controller.generation is None:
                raise XpoolDaemonError("not_ready", "fabric generation retired during initialization")
            if self.registrations.instances.query(instance) is not registration:
                raise XpoolDaemonError("not_ready", "instance registration changed during initialization")
            listener = self.serving_startup.listeners.get(model_id)
            if listener is not None and listener != publication.serving_listener:
                raise XpoolDaemonError("conflict", "serving listener disagrees across instance ranks")
            self.fabric_controller.record_initialized(
                registration.instance,
                registration.proc,
                generation=publication.generation,
            )
            self.serving_startup.listeners.setdefault(model_id, publication.serving_listener)

    def heartbeat_response(self, now: float) -> HeartbeatResponse:
        """Build the unified heartbeat response from authoritative state."""

        with self.lock:
            fabric = self.fabric_controller.generation
            generation = None if fabric is None else fabric.plan.generation
            fabric_phase = None if fabric is None else fabric.phase
        return HeartbeatResponse(
            warnings=self.global_warnings(now),
            generation=generation,
            fabric_phase=fabric_phase,
        )

    def heartbeat_atnagent(self, device: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh a atnagent heartbeat and return global daemon warnings."""

        now = monotonic()
        with self.lock:
            registration = self.registrations.atnagents.query(device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "registration is not registered")
        registration.validate_process_ref(heartbeat, context="heartbeat")
        with self.lock:
            self.registrations.atnagents.commit_heartbeat(device, registration, now=now)
        return self.heartbeat_response(now)

    def heartbeat_ffnagent(self, device: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh an FfnAgent heartbeat and return global daemon warnings."""

        now = monotonic()
        with self.lock:
            registration = self.registrations.ffnagents.query(device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "registration is not registered")
        registration.validate_process_ref(heartbeat, context="heartbeat")
        with self.lock:
            self.registrations.ffnagents.commit_heartbeat(device, registration, now=now)
        return self.heartbeat_response(now)

    def heartbeat_instance(self, model_id: ModelId, rank: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh an instance-rank heartbeat and return global daemon warnings."""

        now = monotonic()
        instance = InstanceRankId(model_id=model_id, rank=rank)
        with self.lock:
            registration = self.registrations.instances.query(instance)
        if registration is None:
            raise XpoolDaemonError("not_ready", "registration is not registered")
        registration.validate_process_ref(heartbeat, context="heartbeat")
        with self.lock:
            self.registrations.instances.commit_heartbeat(instance, registration, now=now)
        return self.heartbeat_response(now)

    def watchdog(self) -> None:
        """Detect owner loss, enforce barriers, and retry fail-stop cleanup."""

        now = monotonic()
        self.refresh_mps_status(now)

        # Snapshot identities under the lock, but perform process-liveness I/O
        # outside it so a slow kernel/process query cannot block daemon control.
        with self.lock:
            registrations = tuple(self.registrations.all_values())
            fabric_snapshot = self.fabric_controller.generation
            generation_owners = (
                ()
                if fabric_snapshot is None
                else tuple({*fabric_snapshot.agent_owners.values(), *fabric_snapshot.instance_owners.values()})
            )
        liveness = {registration.proc: registration.proc.is_alive() for registration in registrations}
        owner_liveness = {
            owner: liveness[owner] if owner in liveness else owner.is_alive() for owner in generation_owners
        }
        with self.lock:
            for registration in registrations:
                registration.alive = liveness[registration.proc]

        # Reconcile only the generation captured above. A replacement makes
        # this watchdog iteration stale and therefore harmless.
        with self.lock:
            fabric = self.fabric_controller.generation
            if fabric is None or fabric is not fabric_snapshot:
                return
            for pe, placement in enumerate(fabric.plan.pe_placements):
                participant = fabric.participants.get(pe)
                if participant is not None and participant.phase is FabricParticipantPhase.FINALIZED:
                    continue
                owner = fabric.agent_owners[pe]
                registration = (
                    self.registrations.atnagents.query(placement.device)
                    if placement.role is FabricRole.ATNAGENT
                    else self.registrations.ffnagents.query(placement.device)
                )
                reason = None
                if registration is None or not owner_liveness[owner]:
                    reason = FabricOwnerFailureReason.EXITED
                elif registration.proc != owner:
                    reason = FabricOwnerFailureReason.REPLACED
                elif registration.readiness_status(now) is not ReadinessStatus.ONLINE:
                    reason = FabricOwnerFailureReason.STALE
                if reason is not None:
                    fabric.record_owner_failure(
                        FabricPeOwnerFailure(
                            role=placement.role.value,
                            pe=pe,
                            reason=reason,
                        )
                    )
                    self.fabric_controller.abort(now=now)
                    break

            if fabric.phase not in {
                FabricGenerationPhase.DRAINING,
                FabricGenerationPhase.FINALIZING,
                FabricGenerationPhase.ABORTING,
                FabricGenerationPhase.STOPPED,
            }:
                for instance, owner in fabric.instance_owners.items():
                    registration = self.registrations.instances.query(instance)
                    reason = None
                    if registration is None or not owner_liveness[owner]:
                        reason = FabricOwnerFailureReason.EXITED
                    elif registration.proc != owner:
                        reason = FabricOwnerFailureReason.REPLACED
                    elif registration.readiness_status(now) is not ReadinessStatus.ONLINE:
                        reason = FabricOwnerFailureReason.STALE
                    if reason is not None:
                        lease = self.transport_broker.leases.get(instance)
                        if (
                            fabric.phase is FabricGenerationPhase.QUIESCING
                            and lease is not None
                            and self.transport_broker.is_quiescing(lease.device)
                        ):
                            continue
                        fabric.record_owner_failure(
                            FabricInstanceRankOwnerFailure(
                                model_id=instance.model_id,
                                rank=instance.rank,
                                reason=reason,
                            )
                        )
                        self.fabric_controller.quiesce(now=now)
                        break

            expected_initialized = len(get_global_config().instances) * get_global_config().atn_world_size
            if (
                fabric.phase is FabricGenerationPhase.EXECUTABLE
                and len(fabric.initialized_instances) < expected_initialized
                and now - fabric.phase_started_at > INSTANCE_STARTUP_TIMEOUT_S
            ):
                fabric.record_control_failure("SGLang initialization barrier timed out")
                self.fabric_controller.abort(now=now)
            elif (
                phase_timeout := FABRIC_PHASE_TIMEOUT_S.get(fabric.phase)
            ) is not None and now - fabric.phase_started_at > phase_timeout:
                phase = fabric.phase
                fabric.record_control_failure(f"Fabric {phase.value} transition timed out")
                self.fabric_controller.abort(now=now)

            if fabric.phase is FabricGenerationPhase.ABORTING:
                instances_retired = not any(owner_liveness[owner] for owner in fabric.instance_owners.values())
                agents_drained = all(
                    not owner_liveness[owner]
                    or (
                        (report := fabric.participants.get(pe)) is not None
                        and report.phase in {FabricParticipantPhase.DRAINED, FabricParticipantPhase.FINALIZED}
                    )
                    for pe, owner in fabric.agent_owners.items()
                )
                if instances_retired and agents_drained:
                    # STOPPED permits failed owner exit, not generation reclamation.
                    fabric.transition(FabricGenerationPhase.STOPPED, now=now)
        self.retire_terminal_generation()

    def upsert_atnagent_transport_arenas(
        self,
        device: int,
        bindings: list[AtnAgentTransportArenaBinding],
        publisher: ProcessRef,
    ) -> None:
        """Upsert transport arenas if the owning atnagent registration is live."""

        now = monotonic()
        if device not in get_global_config().atnagent_by_device:
            raise XpoolDaemonError("not_found", "unknown AtnAgent")
        with self.lock:
            registration = self.registrations.atnagents.query(device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "atnagent must register before upserting transport arenas")
        registration.validate_process_ref(publisher, context="transport arena publisher")
        registration.require_online(now, context="local attention atnagent")
        with self.lock:
            if self.registrations.atnagents.query(device) is not registration:
                raise XpoolDaemonError("not_ready", "atnagent registration changed during arena publication")
            publications = self.validate_atnagent_transport_arenas(device, bindings)
            self.transport_broker.publish(device, registration.proc, publications)

    def quiesce_atnagent_transport_leases(
        self,
        device: int,
        publisher: ProcessRef,
    ) -> AtnAgentTransportLeaseQuiesceResponse:
        """Close lease admission and report clients that still own arenas."""

        now = monotonic()
        if device not in get_global_config().atnagent_by_device:
            raise XpoolDaemonError("not_found", "unknown AtnAgent")
        with self.lock:
            registration = self.registrations.atnagents.query(device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "atnagent must register before quiescing transport leases")
        registration.validate_process_ref(publisher, context="transport lease quiesce")
        if registration.readiness_status(now) is ReadinessStatus.OFFLINE:
            raise XpoolDaemonError("not_ready", "local attention atnagent process is not live")
        with self.lock:
            if self.registrations.atnagents.query(device) is not registration:
                raise XpoolDaemonError("not_ready", "atnagent registration changed during lease quiesce")
            self.transport_broker.quiesce(device, registration.proc)
            leased_instances = self.transport_broker.leased_instances(device)
            candidates = [
                owner
                for instance in leased_instances
                if (owner := self.registrations.instances.query(instance)) is not None
            ]
        live_remaining = [owner for owner in candidates if owner.proc.is_alive()]
        return AtnAgentTransportLeaseQuiesceResponse(
            in_use=[
                InstanceRankRef(
                    pid=owner.proc.pid,
                    abi_version=owner.abi_version,
                    model_id=owner.instance.model_id,
                    rank=owner.instance.rank,
                )
                for owner in live_remaining
            ]
        )

    def validate_atnagent_transport_arenas(
        self,
        device: int,
        bindings: list[AtnAgentTransportArenaBinding],
    ) -> list[TransportArenaPublication]:
        """Validate and normalize transport arenas published by one atnagent."""

        atn_devices = get_global_config().atn.devices
        seen: set[InstanceRankId] = set()
        seen_handles: set[str] = set()
        publications: list[TransportArenaPublication] = []
        for binding in bindings:
            handle = binding.handle
            if handle.handle in seen_handles:
                raise XpoolDaemonError("conflict", "atnagent transport arenas contain duplicate arena handle")
            seen_handles.add(handle.handle)
            if binding.model_id not in get_global_config().instance_by_model_id:
                raise XpoolDaemonError("conflict", "atnagent transport arena handle references unknown instance")
            self.validate_instance_rank(binding.model_id, binding.rank)
            expected_device = atn_devices[binding.rank]
            if expected_device != device:
                raise XpoolDaemonError(
                    "conflict",
                    (
                        f"atnagent transport arena handle rank {binding.rank} belongs to device "
                        f"{expected_device}, "
                        f"not {device}"
                    ),
                )
            instance = InstanceRankId(model_id=binding.model_id, rank=binding.rank)
            if instance in seen:
                raise XpoolDaemonError("conflict", "atnagent transport arenas contain duplicate instance-rank handle")
            registration = self.registrations.instances.query(instance)
            if registration is None:
                raise XpoolDaemonError(
                    "not_ready",
                    "atnagent transport arena handle references an instance rank that is not registered",
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

    def readiness_snapshot(self) -> ReadinessSnapshot:
        """Return the single global daemon readiness snapshot."""

        now = monotonic()
        with self.lock:
            return ControlPlaneProjection.capture(
                config=get_global_config(),
                now=now,
                mps_online=None if self.mps_cache_result is None else self.mps_cache_result.online,
                registrations=self.registrations,
                transport=self.transport_broker,
                fabric=self.fabric_controller,
            ).readiness

    def capture_serving_health_targets(self) -> ServingHealthTargets | None:
        """Return current ordered Instance listeners once System Ready is true."""

        config = get_global_config()
        now = monotonic()
        with self.lock:
            fabric = self.fabric_controller.generation
            startup = self.serving_startup
            if (
                fabric is None
                or startup.generation != fabric.plan.generation
                or startup.confirmed
                or not ControlPlaneProjection.capture(
                    config=config,
                    now=now,
                    mps_online=None if self.mps_cache_result is None else self.mps_cache_result.online,
                    registrations=self.registrations,
                    transport=self.transport_broker,
                    fabric=self.fabric_controller,
                ).readiness.ready
            ):
                return None
            model_ids = tuple(instance.model_id for instance in config.instances)
            if set(startup.listeners) != set(model_ids):
                return None
            return ServingHealthTargets(
                generation=fabric.plan.generation,
                listeners=tuple((model_id, startup.listeners[model_id]) for model_id in model_ids),
            )

    def confirm_serving_health(self, targets: ServingHealthTargets) -> bool:
        """Confirm the first successful probe of the unchanged current targets."""

        config = get_global_config()
        with self.lock:
            fabric = self.fabric_controller.generation
            startup = self.serving_startup
            if (
                fabric is None
                or fabric.phase is not FabricGenerationPhase.EXECUTABLE
                or fabric.plan.generation != targets.generation
                or startup.generation != targets.generation
                or startup.confirmed
            ):
                return False
            model_ids = tuple(instance.model_id for instance in config.instances)
            if set(startup.listeners) != set(model_ids):
                return False
            listeners = tuple((model_id, startup.listeners[model_id]) for model_id in model_ids)
            if listeners != targets.listeners:
                return False
            startup.confirmed = True
            return True

    def acquire_instance_transport_arena(
        self,
        model_id: ModelId,
        *,
        rank: int,
        owner: ProcessRef,
    ) -> TransportArenaHandle:
        """Acquire a daemon-brokered transport arena handle for one registered instance rank."""

        config = get_global_config()
        atn_devices = config.atn.devices
        self.validate_instance_rank(model_id, rank)
        instance_uid = InstanceRankId(model_id=model_id, rank=rank)
        device = atn_devices[rank]
        now = monotonic()
        with self.lock:
            registration = self.registrations.instances.query(instance_uid)
        if registration is None:
            raise XpoolDaemonError("not_ready", "instance rank is not registered")
        registration.validate_process_ref(owner, context="transport arena handle acquirer")
        with self.lock:
            atnagent_registration = self.registrations.atnagents.query(device)
        if atnagent_registration is None:
            raise XpoolDaemonError("not_ready", "local attention atnagent is not registered")
        with self.lock:
            if self.registrations.instances.query(instance_uid) is not registration:
                raise XpoolDaemonError("not_ready", "instance registration changed during arena acquisition")
            if self.registrations.atnagents.query(device) is not atnagent_registration:
                raise XpoolDaemonError("not_ready", "atnagent registration changed during arena acquisition")
            registration.require_online(now, context="instance rank")
            atnagent_registration.require_online(now, context="local attention atnagent")
            fabric = self.fabric_controller.generation
            if fabric is None:
                raise XpoolDaemonError("not_ready", "Fabric generation is not executable")
            if fabric is not None:
                if fabric.phase in {
                    FabricGenerationPhase.QUIESCING,
                    FabricGenerationPhase.DRAINING,
                    FabricGenerationPhase.FINALIZING,
                    FabricGenerationPhase.ABORTING,
                    FabricGenerationPhase.STOPPED,
                }:
                    raise XpoolDaemonError("not_ready", "Fabric generation is not accepting Instance-rank arena leases")
                if fabric.phase is not FabricGenerationPhase.EXECUTABLE:
                    raise XpoolDaemonError("not_ready", "Fabric generation is not executable")
                if fabric.instance_owners.get(instance_uid) != registration.proc:
                    raise XpoolDaemonError(
                        "conflict", "Fabric generation owner does not match Instance-rank registration"
                    )
            publication = self.transport_broker.publication(device, instance_uid, atnagent_registration.proc)
            if publication.transport != registration.transport:
                raise XpoolDaemonError("not_ready", "published transport arena geometry is stale")
            self.transport_broker.acquire(
                instance_uid,
                TransportArenaLease(
                    device=device,
                    handle=publication.handle,
                    publisher=atnagent_registration.proc,
                ),
                last_seen_at=registration.last_seen_at,
                now=now,
                heartbeat_timeout_s=TRANSPORT_ARENA_LEASE_HEARTBEAT_TIMEOUT_S,
            )
        return publication.handle

    def validate_instance_rank(self, model_id: ModelId, rank: int) -> None:
        """Reject an unknown model or rank outside its configured attention world."""

        config = get_global_config()
        if model_id not in config.instance_by_model_id:
            raise XpoolDaemonError("not_found", "unknown instance")
        if rank < 0 or rank >= config.atn_world_size:
            raise XpoolDaemonError(
                "conflict",
                f"instance rank {rank} is outside atn_world_size={config.atn_world_size}",
            )
