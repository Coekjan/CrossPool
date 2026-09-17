"""Authoritative orchestration for the daemon control plane."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import monotonic, sleep

import xpool.native
from xpool import ffn
from xpool.config import FfnSchedulingPolicy, XpoolConfig, get_global_config
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
from xpool.mps import MpsProbeResult, probe_mps_controller
from xpool.native import ABI_VERSION
from xpool.service.daemon.fabric import FabricController, FabricGenerationState, FabricMembership
from xpool.service.daemon.ffn_placement import place_ffn_models
from xpool.service.daemon.kv import KvCapacityPolicy
from xpool.service.daemon.readiness import ControlPlaneProjection
from xpool.service.daemon.registration import (
    HEARTBEAT_WARNING_WATERMARK_S,
    AtnAgentRegistrationState,
    CommonRegistration,
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
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    AtnAgentTransportLeaseQuiesceResponse,
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
    ProcessRef,
    ReadinessSnapshot,
    ReadinessStatus,
    ServingListener,
)
from xpool.transport import TransportArenaHandle
from xpool.utils.procs import ProcUniqId

TRANSPORT_ARENA_LEASE_HEARTBEAT_TIMEOUT_S = 2.0 * HEARTBEAT_WARNING_WATERMARK_S
TRANSPORT_DRAIN_TERM_GRACE_S = HEARTBEAT_WARNING_WATERMARK_S
ATNAGENT_REPLACEMENT_TIMEOUT_S = 60.0
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
    listeners: dict[str, ServingListener] = field(default_factory=dict)
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class ServingHealthTargets:
    """Immutable ordered listener snapshot for one Fabric generation."""

    generation: FabricGenerationId
    listeners: tuple[tuple[str, ServingListener], ...]


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

    def mps_readiness_status(self) -> ReadinessStatus:
        """Project the latest watchdog-owned MPS probe result.

        Returns:
            Online when the configured controller responds, otherwise offline.

        This query performs no controller I/O and does not mutate state.
        """

        with self.lock:
            result = self.mps_cache_result
        return ReadinessStatus.ONLINE if result is not None and result.online else ReadinessStatus.OFFLINE

    def refresh_mps_status(self, now: float) -> None:
        """Refresh MPS readiness outside the domain lock when its cache is stale."""

        with self.lock:
            if self.mps_cache_result is not None and now - self.mps_cache_at < MPS_READINESS_CACHE_S:
                return
            previous = self.mps_cache_result
        result = probe_mps_controller()
        with self.lock:
            self.mps_cache_at = now
            self.mps_cache_result = result
        if previous is None or previous.online != result.online:
            log = logger.info if result.online else logger.warning
            log(
                "mps readiness changed status=%s detail=%s", "online" if result.online else "offline", result.diagnostic
            )

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
                mps_online=self.mps_cache_result is not None and self.mps_cache_result.online,
                registrations=self.registrations,
                transport=self.transport_broker,
                fabric=self.fabric_controller,
            )
            previous_warnings = self.warning_cache
            self.warning_cache_at = now
            self.warning_cache = projection.warnings
            warnings = projection.warnings
        previous_by_key = {(warning.kind, warning.cuda_device) for warning in previous_warnings}
        current_by_key = {(warning.kind, warning.cuda_device) for warning in warnings}
        for kind, cuda_device in sorted(current_by_key - previous_by_key):
            logger.warning("global warning entered kind=%s device=%s", kind, cuda_device)
        for kind, cuda_device in sorted(previous_by_key - current_by_key):
            logger.info("global warning cleared kind=%s device=%s", kind, cuda_device)
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
        """Install a atnagent registration after draining a dead generation.

        Args:
            registration: Candidate atnagent process registration.

        Raises:
            XpoolDaemonError: If the device or ABI is invalid, a conflicting
                generation remains live, or old arena users outlive the bounded
                replacement cleanup.

        Side Effects:
            Concurrently terminates live instance processes leasing arenas from
            a dead prior generation and removes their registrations after death.
        """

        self.retire_terminal_generation()
        if registration.cuda_device not in get_global_config().atnagent_by_cuda_device:
            raise XpoolDaemonError("not_found", "unknown AtnAgent")
        if registration.abi_version != ABI_VERSION:
            raise XpoolDaemonError("conflict", "atnagent ABI version does not match daemon ABI")
        with self.lock:
            existing = self.registrations.atnagents.query(registration.cuda_device)
            fabric = self.fabric_controller.generation
            if fabric is not None:
                pe = next(
                    pe
                    for pe, item in enumerate(fabric.plan.pe_placements)
                    if item.role is FabricRole.ATNAGENT and item.cuda_device == registration.cuda_device
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
                leased_instances = self.transport_broker.leased_instances(registration.cuda_device, existing.proc)
                owners = [
                    owner
                    for instance in leased_instances
                    if (owner := self.registrations.instances.query(instance)) is not None
                ]
            live_owners = [owner for owner in owners if owner.proc.is_alive()]
            if live_owners:
                deadline = monotonic() + ATNAGENT_REPLACEMENT_TIMEOUT_S
                with ThreadPoolExecutor(max_workers=len(live_owners)) as executor:
                    futures = [
                        executor.submit(owner.proc.terminate_tree, term_grace_s=TRANSPORT_DRAIN_TERM_GRACE_S)
                        for owner in live_owners
                    ]
                    for future in futures:
                        future.result()
                while any(owner.proc.is_alive() for owner in live_owners) and monotonic() < deadline:
                    sleep(0.05)
                if any(owner.proc.is_alive() for owner in live_owners):
                    raise XpoolDaemonError("not_ready", "previous atnagent generation still has live arena users")
            with self.lock:
                for owner in live_owners:
                    if self.registrations.instances.query(owner.instance) is owner:
                        self.registrations.instances.remove_snapshot(owner.instance, owner)
                        self.transport_broker.remove_instance(owner.instance)
                        logger.info(
                            "registration removed role=instance instance=%s rank=%s pid=%s",
                            owner.instance.instance_id,
                            owner.instance.rank,
                            owner.proc.pid,
                        )
                if self.registrations.atnagents.query(registration.cuda_device) is not existing:
                    raise XpoolDaemonError("not_ready", "atnagent registration changed during generation cleanup")
        with self.lock:
            changed = self.registrations.atnagents.install_snapshot(
                registration,
                existing,
                existing_alive=existing_alive,
            )
            installed = self.registrations.atnagents.query(registration.cuda_device)
            if installed is None:
                raise RuntimeError("AtnAgent registration disappeared during installation")
            self.transport_broker.install_atnagent(registration.cuda_device, installed.proc)
            if changed:
                self.membership_revision += 1
                logger.info(
                    "registration accepted role=atnagent device=%s pid=%s",
                    registration.cuda_device,
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
        if registration.cuda_device not in get_global_config().ffnagent_by_cuda_device:
            raise XpoolDaemonError("not_found", "unknown FfnAgent")
        if registration.abi_version != ABI_VERSION:
            raise XpoolDaemonError("conflict", "ffnagent ABI version does not match daemon ABI")
        with self.lock:
            existing = self.registrations.ffnagents.query(registration.cuda_device)
            fabric = self.fabric_controller.generation
            if fabric is not None:
                pe = next(
                    pe
                    for pe, item in enumerate(fabric.plan.pe_placements)
                    if item.role is FabricRole.FFNAGENT and item.cuda_device == registration.cuda_device
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
                    registration.cuda_device,
                    registration.proc.pid,
                )
        self.ensure_fabric_plan()

    def register_instance(self, registration: InstanceRankRegistrationState) -> None:
        """Install an instance registration unless a different live one exists."""

        self.retire_terminal_generation()
        instance_id = registration.instance.instance_id
        rank = registration.instance.rank
        if registration.abi_version != ABI_VERSION:
            raise XpoolDaemonError("conflict", "instance ABI version does not match daemon ABI")
        if instance_id not in get_global_config().instance_by_id:
            raise XpoolDaemonError("not_found", "unknown instance")
        self.validate_instance_rank(rank)
        transport = registration.transport
        if transport.atn_tp_size * transport.atn_dp_size > get_global_config().atn_world_size:
            raise XpoolDaemonError(
                "conflict",
                "instance transport TP-by-DP topology exceeds the configured AtnAgent world",
            )
        if transport.atn_dp_rank * transport.atn_tp_size + transport.atn_tp_rank != rank:
            raise XpoolDaemonError("conflict", "instance transport coordinates do not use TP-fastest rank order")
        with self.lock:
            existing = self.registrations.instances.query(registration.instance)
            fabric = self.fabric_controller.generation
            if fabric is not None:
                expected_owner = fabric.instance_owners.get(registration.instance)
                if expected_owner != registration.proc:
                    fabric.record_owner_failure(
                        FabricInstanceRankOwnerFailure(
                            instance_id=instance_id,
                            rank=rank,
                            reason=FabricOwnerFailureReason.REPLACED,
                        )
                    )
                    self.fabric_controller.quiesce(now=monotonic())
                    raise XpoolDaemonError("conflict", "retained Fabric generation forbids Instance-rank replacement")
        existing_alive = False if existing is None else existing.proc.is_alive()
        with self.lock:
            if self.registrations.instances.query(registration.instance) is not existing:
                raise XpoolDaemonError("not_ready", "instance registration changed during validation")
            for peer in self.registrations.instances.values():
                if peer.instance.instance_id != instance_id or peer.instance == registration.instance:
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
                    registration.instance.instance_id,
                    registration.instance.rank,
                    registration.proc.pid,
                    get_global_config().atn.devices[registration.instance.rank],
                )
        self.ensure_fabric_plan()

    def deregister_instance(self, instance_id: str, rank: int, owner: ProcessRef) -> None:
        """Remove an instance-rank registration when the owner process requests it."""

        if instance_id not in get_global_config().instance_by_id:
            raise XpoolDaemonError("not_found", "unknown instance")
        self.validate_instance_rank(rank)
        instance = InstanceRankId(instance_id=instance_id, rank=rank)
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
                    instance_id=instance_id,
                    rank=rank,
                    reason=FabricOwnerFailureReason.EXITED,
                ),
                termination_requested=lease is not None and lease.termination_requested,
                now=monotonic(),
            )
        logger.info(
            "registration removed role=instance instance=%s rank=%s pid=%s",
            instance_id,
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
                if self.membership_revision != membership.revision:
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

    def close(self) -> None:
        """Release daemon-owned Generation resources during lifespan shutdown."""

        with self.lock:
            if self.kv_capacity_policy is not None:
                self.kv_capacity_policy.close()
                self.kv_capacity_policy = None

    def request_fabric_quiesce(self, request: FabricQuiesceRequest) -> None:
        """Authenticate an Agent owner and stop generation admission."""

        with self.lock:
            fabric = self.fabric_controller.generation
            if fabric is None or request.generation != fabric.plan.generation:
                raise XpoolDaemonError("conflict", "fabric quiesce generation is not retained")
            owner_matches = 0
            for pe, placement in enumerate(fabric.plan.pe_placements):
                registration = (
                    self.registrations.atnagents.query(placement.cuda_device)
                    if placement.role is FabricRole.ATNAGENT
                    else self.registrations.ffnagents.query(placement.cuda_device)
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
                self.registrations.atnagents.query(placement.cuda_device)
                if placement.role is FabricRole.ATNAGENT
                else self.registrations.ffnagents.query(placement.cuda_device)
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
        instance_id: str,
        *,
        rank: int,
        publication: InstanceRankInitializedPublication,
    ) -> None:
        """Record one Instance Rank's scheduler construction and listener."""

        self.validate_instance_rank(rank)
        instance = InstanceRankId(instance_id=instance_id, rank=rank)
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
            listener = self.serving_startup.listeners.get(instance_id)
            if listener is not None and listener != publication.serving_listener:
                raise XpoolDaemonError("conflict", "serving listener disagrees across instance ranks")
            self.fabric_controller.record_initialized(
                registration.instance,
                registration.proc,
                generation=publication.generation,
            )
            self.serving_startup.listeners.setdefault(instance_id, publication.serving_listener)

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

    def heartbeat_atnagent(self, cuda_device: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh a atnagent heartbeat and return global daemon warnings."""

        now = monotonic()
        with self.lock:
            registration = self.registrations.atnagents.query(cuda_device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "registration is not registered")
        registration.validate_process_ref(heartbeat, context="heartbeat")
        with self.lock:
            self.registrations.atnagents.commit_heartbeat(cuda_device, registration, now=now)
        return self.heartbeat_response(now)

    def heartbeat_ffnagent(self, cuda_device: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh an FfnAgent heartbeat and return global daemon warnings."""

        now = monotonic()
        with self.lock:
            registration = self.registrations.ffnagents.query(cuda_device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "registration is not registered")
        registration.validate_process_ref(heartbeat, context="heartbeat")
        with self.lock:
            self.registrations.ffnagents.commit_heartbeat(cuda_device, registration, now=now)
        return self.heartbeat_response(now)

    def heartbeat_instance(self, instance_id: str, rank: int, heartbeat: ProcessRef) -> HeartbeatResponse:
        """Refresh an instance-rank heartbeat and return global daemon warnings."""

        now = monotonic()
        instance = InstanceRankId(instance_id=instance_id, rank=rank)
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
            registrations: tuple[CommonRegistration, ...] = tuple(
                [
                    *self.registrations.atnagents.values(),
                    *self.registrations.ffnagents.values(),
                    *self.registrations.instances.values(),
                ]
            )
            fabric_snapshot = self.fabric_controller.generation
            generation_owners = (
                ()
                if fabric_snapshot is None
                else tuple({*fabric_snapshot.agent_owners.values(), *fabric_snapshot.instance_owners.values()})
            )
        liveness = {registration.proc: registration.proc.is_alive() for registration in registrations}
        owner_liveness = {owner: liveness.get(owner, owner.is_alive()) for owner in generation_owners}
        with self.lock:
            for registration in registrations:
                registration.alive = liveness[registration.proc]

        abort_owners: tuple[ProcUniqId, ...] = ()
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
                    self.registrations.atnagents.query(placement.cuda_device)
                    if placement.role is FabricRole.ATNAGENT
                    else self.registrations.ffnagents.query(placement.cuda_device)
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
                        if (
                            fabric.phase is FabricGenerationPhase.QUIESCING
                            and self.transport_broker.termination_requested(instance)
                        ):
                            continue
                        fabric.record_owner_failure(
                            FabricInstanceRankOwnerFailure(
                                instance_id=instance.instance_id,
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
                abort_owners = tuple({*fabric.agent_owners.values(), *fabric.instance_owners.values()})

        # Fail-stop process trees outside the daemon lock; state advances to
        # Stopped only after every generation owner is proven dead.
        live_abort_owners = tuple(owner for owner in abort_owners if owner.is_alive())
        if live_abort_owners:
            with ThreadPoolExecutor(max_workers=len(live_abort_owners)) as executor:
                futures = [executor.submit(owner.kill_tree) for owner in live_abort_owners]
                for future in futures:
                    future.result()
        if abort_owners:
            any_alive = any(owner.is_alive() for owner in abort_owners)
            with self.lock:
                fabric = self.fabric_controller.generation
                if fabric is fabric_snapshot and fabric.phase is FabricGenerationPhase.ABORTING and not any_alive:
                    fabric.transition(FabricGenerationPhase.STOPPED, now=monotonic())
        self.retire_terminal_generation()

    def upsert_atnagent_transport_arenas(
        self,
        cuda_device: int,
        bindings: list[AtnAgentTransportArenaBinding],
        publisher: ProcessRef,
    ) -> None:
        """Upsert transport arenas if the owning atnagent registration is live."""

        now = monotonic()
        if cuda_device not in get_global_config().atnagent_by_cuda_device:
            raise XpoolDaemonError("not_found", "unknown AtnAgent")
        with self.lock:
            registration = self.registrations.atnagents.query(cuda_device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "atnagent must register before upserting transport arenas")
        registration.validate_process_ref(publisher, context="transport arena publisher")
        registration.require_online(now, context="local attention atnagent")
        with self.lock:
            if self.registrations.atnagents.query(cuda_device) is not registration:
                raise XpoolDaemonError("not_ready", "atnagent registration changed during arena publication")
            publications = self.validate_atnagent_transport_arenas(cuda_device, bindings)
            self.transport_broker.publish(cuda_device, registration.proc, publications)

    def quiesce_atnagent_transport_leases(
        self,
        cuda_device: int,
        publisher: ProcessRef,
    ) -> AtnAgentTransportLeaseQuiesceResponse:
        """Close lease admission and terminate every live lease owner."""

        now = monotonic()
        if cuda_device not in get_global_config().atnagent_by_cuda_device:
            raise XpoolDaemonError("not_found", "unknown AtnAgent")
        with self.lock:
            registration = self.registrations.atnagents.query(cuda_device)
        if registration is None:
            raise XpoolDaemonError("not_ready", "atnagent must register before quiescing transport leases")
        registration.validate_process_ref(publisher, context="transport lease quiesce")
        if registration.readiness_status(now) is ReadinessStatus.OFFLINE:
            raise XpoolDaemonError("not_ready", "local attention atnagent process is not live")
        with self.lock:
            if self.registrations.atnagents.query(cuda_device) is not registration:
                raise XpoolDaemonError("not_ready", "atnagent registration changed during lease quiesce")
            self.transport_broker.quiesce(cuda_device, registration.proc)
            leased_instances = self.transport_broker.leased_instances(cuda_device)
            candidates = [
                owner
                for instance in leased_instances
                if (owner := self.registrations.instances.query(instance)) is not None
            ]
        live_candidates = [owner for owner in candidates if owner.proc.is_alive()]
        with self.lock:
            active_instances = [
                owner.instance
                for owner in live_candidates
                if self.registrations.instances.query(owner.instance) is owner
                and owner.instance in self.transport_broker.leased_instances(cuda_device)
            ]
            requested_instances = self.transport_broker.mark_termination_requested(active_instances)
            procs_to_terminate = [owner.proc for owner in live_candidates if owner.instance in requested_instances]
        if procs_to_terminate:
            with ThreadPoolExecutor(max_workers=len(procs_to_terminate)) as executor:
                futures = [
                    executor.submit(proc.terminate_tree, term_grace_s=TRANSPORT_DRAIN_TERM_GRACE_S)
                    for proc in procs_to_terminate
                ]
                for future in futures:
                    future.result()
        with self.lock:
            remaining = [
                owner
                for instance in self.transport_broker.leased_instances(cuda_device)
                if (owner := self.registrations.instances.query(instance)) is not None
            ]
        live_remaining = [owner for owner in remaining if owner.proc.is_alive()]
        return AtnAgentTransportLeaseQuiesceResponse(
            in_use=[
                InstanceRankRef(
                    pid=owner.proc.pid,
                    abi_version=owner.abi_version,
                    instance_id=owner.instance.instance_id,
                    rank=owner.instance.rank,
                )
                for owner in live_remaining
            ]
        )

    def validate_atnagent_transport_arenas(
        self,
        cuda_device: int,
        bindings: list[AtnAgentTransportArenaBinding],
    ) -> list[TransportArenaPublication]:
        """Validate and normalize transport arenas published by one atnagent."""

        atn_cuda_devices = get_global_config().atn.devices
        seen: set[InstanceRankId] = set()
        seen_handles: set[str] = set()
        publications: list[TransportArenaPublication] = []
        for binding in bindings:
            handle = binding.handle
            if handle.handle in seen_handles:
                raise XpoolDaemonError("conflict", "atnagent transport arenas contain duplicate arena handle")
            seen_handles.add(handle.handle)
            if binding.instance_id not in get_global_config().instance_by_id:
                raise XpoolDaemonError("conflict", "atnagent transport arena handle references unknown instance")
            self.validate_instance_rank(binding.rank)
            expected_cuda_device = atn_cuda_devices[binding.rank]
            if expected_cuda_device != cuda_device:
                raise XpoolDaemonError(
                    "conflict",
                    (
                        f"atnagent transport arena handle rank {binding.rank} belongs to CUDA device "
                        f"{expected_cuda_device}, "
                        f"not {cuda_device}"
                    ),
                )
            instance = InstanceRankId(instance_id=binding.instance_id, rank=binding.rank)
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
                mps_online=self.mps_cache_result is not None and self.mps_cache_result.online,
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
                    mps_online=self.mps_cache_result is not None and self.mps_cache_result.online,
                    registrations=self.registrations,
                    transport=self.transport_broker,
                    fabric=self.fabric_controller,
                ).readiness.ready
            ):
                return None
            instance_ids = tuple(instance.id for instance in config.instances)
            if set(startup.listeners) != set(instance_ids):
                return None
            return ServingHealthTargets(
                generation=fabric.plan.generation,
                listeners=tuple((instance_id, startup.listeners[instance_id]) for instance_id in instance_ids),
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
            instance_ids = tuple(instance.id for instance in config.instances)
            if set(startup.listeners) != set(instance_ids):
                return False
            listeners = tuple((instance_id, startup.listeners[instance_id]) for instance_id in instance_ids)
            if listeners != targets.listeners:
                return False
            startup.confirmed = True
            return True

    def acquire_instance_transport_arena(
        self,
        instance_id: str,
        *,
        rank: int,
        owner: ProcessRef,
    ) -> TransportArenaHandle:
        """Acquire a daemon-brokered transport arena handle for one registered instance rank."""

        config = get_global_config()
        instance = config.instance_by_id.get(instance_id)
        if instance is None:
            raise XpoolDaemonError("not_found", "unknown instance")

        atn_cuda_devices = config.atn.devices
        self.validate_instance_rank(rank)
        instance_uid = InstanceRankId(instance_id=instance_id, rank=rank)
        cuda_device = atn_cuda_devices[rank]
        now = monotonic()
        with self.lock:
            registration = self.registrations.instances.query(instance_uid)
        if registration is None:
            raise XpoolDaemonError("not_ready", "instance rank is not registered")
        registration.validate_process_ref(owner, context="transport arena handle acquirer")
        with self.lock:
            atnagent_registration = self.registrations.atnagents.query(cuda_device)
        if atnagent_registration is None:
            raise XpoolDaemonError("not_ready", "local attention atnagent is not registered")
        with self.lock:
            if self.registrations.instances.query(instance_uid) is not registration:
                raise XpoolDaemonError("not_ready", "instance registration changed during arena acquisition")
            if self.registrations.atnagents.query(cuda_device) is not atnagent_registration:
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
            publication = self.transport_broker.publication(cuda_device, instance_uid, atnagent_registration.proc)
            if publication.transport != registration.transport:
                raise XpoolDaemonError("not_ready", "published transport arena geometry is stale")
            self.transport_broker.acquire(
                instance_uid,
                TransportArenaLease(
                    cuda_device=cuda_device,
                    handle=publication.handle,
                    publisher=atnagent_registration.proc,
                ),
                last_seen_at=registration.last_seen_at,
                now=now,
                heartbeat_timeout_s=TRANSPORT_ARENA_LEASE_HEARTBEAT_TIMEOUT_S,
            )
        return publication.handle

    def validate_instance_rank(self, rank: int) -> None:
        """Reject an instance rank outside the configured ATN world."""

        if rank < 0 or rank >= get_global_config().atn_world_size:
            raise XpoolDaemonError(
                "conflict",
                f"instance rank {rank} is outside atn_world_size={get_global_config().atn_world_size}",
            )
