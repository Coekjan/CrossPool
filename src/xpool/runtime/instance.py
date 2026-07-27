"""Instance runtime registration and transport arena handoff."""

from __future__ import annotations

import logging
import os
import time

import xpool.native
from xpool.abi import ABI_VERSION, FfnResultCode
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationPhase, FabricPlan, FfnWorkload
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.client import XpoolClient, XpoolClientError, XpoolDaemonError
from xpool.service.wire import (
    HeartbeatResponse,
    InstanceInitializedPublication,
    InstanceRegistration,
    ProcessRef,
)
from xpool.transport import TransportArenaHandle
from xpool.utils.background import BackgroundThread
from xpool.utils.procs import bail

__all__ = [
    "Instance",
    "InstanceError",
    "InstanceFailureMonitor",
    "InstanceHeartbeat",
]

INSTANCE_HEARTBEAT_INTERVAL_S = 5.0
STALE_ATNAGENT_RECOVERY_GRACE_S = 60.0
TRANSPORT_METADATA_RECOVERY_DEADLINE_S = 60.0
INSTANCE_TRANSPORT_ACQUIRE_INTERVAL_S = 0.5
INSTANCE_HEARTBEAT_STOP_JOIN_TIMEOUT_S = 5.0
INSTANCE_FAILURE_MONITOR_INTERVAL_S = 0.1
INSTANCE_STARTUP_BARRIER_TIMEOUT_S = 600.0
INSTANCE_FAILURE_MONITOR_STOP_JOIN_TIMEOUT_S = 5.0
logger = logging.getLogger(__name__)


class InstanceError(RuntimeError):
    """Raised when daemon-brokered FFN shim transport arenas cannot be attached."""


class InstanceFailureMonitor:
    """Fail-close an instance when its transport arena records a failure.

    Args:
        instance_id: Configured instance id used in fatal diagnostics.
        instance_index: Integer instance index used by the native arena map.
        rank: Rank-local process index within the instance.

    Attributes:
        worker: Periodic background thread that polls native sticky failure state.
    """

    def __init__(self, *, instance_id: str, instance_index: int, rank: int) -> None:
        """Create a stopped instance failure monitor."""

        self.instance_id = instance_id
        self.instance_index = instance_index
        self.rank = rank
        self.worker = BackgroundThread.periodic(
            name=f"xpool-instance-failure-monitor-{instance_id}-{rank}",
            interval_s=INSTANCE_FAILURE_MONITOR_INTERVAL_S,
            target=self.step,
            join_timeout_s=INSTANCE_FAILURE_MONITOR_STOP_JOIN_TIMEOUT_S,
        )

    def start(self) -> None:
        """Start polling native sticky failure state."""

        self.worker.start()

    def stop(self) -> None:
        """Stop polling native sticky failure state."""

        self.worker.close()

    def step(self) -> bool:
        """Poll once and terminate the process when the arena has failed."""

        failure = FfnResultCode(xpool.native.transport.read_generation_failure())
        if failure != FfnResultCode.OK:
            bail(
                logger,
                "xpool transport executor failed for instance %s rank %s: %s",
                self.instance_id,
                self.rank,
                failure.name,
            )
        return True


class InstanceHeartbeat:
    """Background heartbeat owner for one registered instance rank.

    Args:
        instance_id: Configured instance id.
        rank: ATN rank-local SGLang process index.
        local_cuda_device: CUDA device represented by ``rank``.
        registration: Stable registration payload used to repair missing daemon
            registration state.
        heartbeat: Stable heartbeat payload for the current process.

    Attributes:
        client: Dedicated daemon client used only by the heartbeat thread.
        arena_handle: Original attached CUDA IPC arena handle, if attachment
            has completed.
        arena_lease_recovery_pending: Whether daemon restart recovery still
            needs to reacquire a lease for ``arena_handle``.
        recovery_deadline: Deadline for transient daemon metadata failures.
        stale_atnagent_deadline: Deadline for a local stale-agent warning.
        worker: Periodic background thread that invokes :meth:`step`.
    """

    def __init__(
        self,
        *,
        instance_id: str,
        rank: int,
        local_cuda_device: int,
        registration: InstanceRegistration,
        heartbeat: ProcessRef,
    ) -> None:
        """Create a stopped heartbeat worker owner."""

        self.instance_id = instance_id
        self.rank = rank
        self.local_cuda_device = local_cuda_device
        self.registration = registration
        self.heartbeat = heartbeat
        self.client = XpoolClient()
        self.arena_handle: TransportArenaHandle | None = None
        self.arena_lease_recovery_pending = False
        self.recovery_deadline: float | None = None
        self.stale_atnagent_deadline: float | None = None
        self.worker = BackgroundThread.periodic(
            name=f"xpool-instance-heartbeat-{self.instance_id}-{self.rank}",
            interval_s=INSTANCE_HEARTBEAT_INTERVAL_S,
            target=self.step,
            join_timeout_s=INSTANCE_HEARTBEAT_STOP_JOIN_TIMEOUT_S,
        )

    def start(self) -> None:
        """Start this process's background heartbeat worker."""

        if self.worker.is_running:
            return
        self.worker.start()

    def stop(self) -> None:
        """Stop this process's background heartbeat worker for the instance rank."""

        self.worker.close()
        self.client.close()

    def step(self) -> bool:
        """Run one heartbeat tick and keep the periodic worker alive."""

        try:
            response = self.heartbeat_once()
            self.recovery_deadline = None
            self.handle_warnings(response)
        except (XpoolDaemonError, XpoolClientError) as exc:
            self.handle_recoverable_failure(exc)
        except Exception as exc:
            bail(
                logger,
                "xpool instance heartbeat failed with unexpected error for rank %s: %s",
                self.rank,
                exc,
            )
        return True

    def heartbeat_once(self) -> HeartbeatResponse:
        """Refresh this instance registration, repairing missing daemon state once."""

        try:
            response = self.client.heartbeat_instance(
                self.instance_id,
                rank=self.rank,
                heartbeat=self.heartbeat,
            )
        except XpoolDaemonError as exc:
            if exc.kind != "not_ready":
                bail(
                    logger,
                    "xpool instance heartbeat received unrecoverable daemon error for rank %s: %s",
                    self.rank,
                    exc,
                )
            logger.warning(
                "xpool instance registration for %s rank %s is missing; re-registering",
                self.instance_id,
                self.rank,
            )
            self.client.register_instance(self.registration)
            self.arena_lease_recovery_pending = self.arena_handle is not None
            self.recover_arena_lease()
            response = self.client.heartbeat_instance(
                self.instance_id,
                rank=self.rank,
                heartbeat=self.heartbeat,
            )
        self.recover_arena_lease()
        return response

    def recover_arena_lease(self) -> None:
        """Reacquire the original arena lease after daemon registration loss."""

        if not self.arena_lease_recovery_pending or self.arena_handle is None:
            return
        recovered_handle = self.client.acquire_instance_transport_arena(
            self.instance_id,
            rank=self.rank,
            owner=ProcessRef(abi_version=self.heartbeat.abi_version, pid=self.heartbeat.pid),
        )
        if recovered_handle != self.arena_handle:
            bail(
                logger,
                "daemon returned a different transport arena after instance recovery for %s rank %s",
                self.instance_id,
                self.rank,
            )
        self.arena_lease_recovery_pending = False

    def handle_warnings(self, response: HeartbeatResponse) -> None:
        """Apply fail-closed policy for daemon heartbeat warnings."""

        warning_kinds = {warning.kind for warning in response.warnings if warning.cuda_device == self.local_cuda_device}
        if "quiescing_atnagent" in warning_kinds:
            logger.warning(
                "xpool daemon reported quiescing agent on CUDA device %s for instance %s rank %s; "
                "waiting for daemon-scoped quiesce",
                self.local_cuda_device,
                self.instance_id,
                self.rank,
            )
        if "stale_atnagent" not in warning_kinds:
            self.stale_atnagent_deadline = None
            return
        now = time.monotonic()
        if self.stale_atnagent_deadline is None:
            self.stale_atnagent_deadline = now + STALE_ATNAGENT_RECOVERY_GRACE_S
        if now >= self.stale_atnagent_deadline:
            bail(
                logger,
                "xpool daemon reported stale agent on CUDA device %s for instance %s rank %s beyond recovery grace",
                self.local_cuda_device,
                self.instance_id,
                self.rank,
            )
        logger.warning(
            "xpool daemon reported stale agent on CUDA device %s for instance %s rank %s; waiting for recovery",
            self.local_cuda_device,
            self.instance_id,
            self.rank,
        )

    def handle_recoverable_failure(self, exc: XpoolDaemonError | XpoolClientError) -> None:
        """Apply retry and fail-closed policy for heartbeat transport failures."""

        if not exc.is_recoverable:
            bail(
                logger,
                "xpool instance heartbeat received unrecoverable daemon error for rank %s: %s",
                self.rank,
                exc,
            )
        if self.recovery_deadline is None:
            self.recovery_deadline = time.monotonic() + TRANSPORT_METADATA_RECOVERY_DEADLINE_S
        if time.monotonic() >= self.recovery_deadline:
            bail(
                logger,
                "xpool transport arena did not recover for instance rank %s: %s",
                self.rank,
                exc,
            )
        logger.warning("xpool instance heartbeat failed: %s", exc)


class Instance:
    """Runtime lifecycle and resources for one SGLang instance rank."""

    client: XpoolClient
    instance_id: str
    instance_index: int
    rank: int
    local_cuda_device: int
    process_ref: ProcessRef
    registration: InstanceRegistration | None
    heartbeat: ProcessRef
    heartbeat_worker: InstanceHeartbeat | None
    failure_monitor: InstanceFailureMonitor | None
    arena_handle: TransportArenaHandle | None

    def __init__(
        self,
        *,
        instance_id: str,
        rank: int,
    ) -> None:
        """Construct an unstarted runtime for one configured instance rank.

        Args:
            instance_id: Configured model/instance id owned by this SGLang rank.
            rank: Rank-local SGLang process index within the instance.

        Raises:
            InstanceError: If the identity or rank is absent from global config.
        """

        config = get_global_config()
        config_instance = config.instance_by_id.get(instance_id)
        if config_instance is None:
            raise InstanceError(f"unknown instance id for daemon registration: {instance_id}")
        if rank < 0 or rank >= config.atn_world_size:
            raise InstanceError(f"instance rank {rank} is outside configured ATN devices")
        pid = os.getpid()
        self.client = XpoolClient()
        self.instance_id = instance_id
        self.instance_index = config_instance.instance_index
        self.rank = rank
        self.local_cuda_device = config.devices.atn_cuda_devices[rank]
        self.process_ref = ProcessRef(abi_version=ABI_VERSION, pid=pid)
        self.registration = None
        self.heartbeat = self.process_ref
        self.heartbeat_worker = None
        self.failure_monitor = None
        self.arena_handle = None
        self.fabric_plan: FabricPlan | None = None

    @classmethod
    def start(
        cls,
        *,
        instance_id: str,
        rank: int,
        transport: InstanceTransportAttributes,
        workload: FfnWorkload,
    ) -> Instance:
        """Construct and transactionally start one runner-owned runtime.

        Args:
            instance_id: Configured model/instance id owned by this rank.
            rank: Rank-local SGLang process index within the instance.
            transport: Transport geometry declared by this rank.
            workload: Rank-independent FFN execution contract.

        Returns:
            Registered runtime with a running heartbeat worker. Transport
            attachment remains an explicit post-``EXECUTABLE`` operation.

        Raises:
            InstanceError: If identity, registration, or attachment fails.

        Side Effects:
            Registers with the daemon and starts the heartbeat worker.
        """

        runtime = cls(instance_id=instance_id, rank=rank)
        try:
            runtime.start_runtime(transport, workload)
        except Exception:
            runtime.close()
            raise
        return runtime

    def start_runtime(self, transport: InstanceTransportAttributes, workload: FfnWorkload) -> None:
        """Register this instance rank and start its heartbeat transactionally."""

        if self.registration is not None:
            self.expect_transport(transport)
            if self.registration.workload != workload:
                raise InstanceError("instance runtime is already registered with a different workload")
            return
        self.register_runtime(transport, workload)
        try:
            self.start_heartbeat_worker()
        except Exception:
            try:
                self.deregister_runtime()
            except Exception as cleanup_exc:
                logger.warning("failed to clean up xpool instance registration: %s", cleanup_exc)
            raise

    def register_runtime(self, transport: InstanceTransportAttributes, workload: FfnWorkload) -> None:
        """Register this instance rank with the daemon."""

        if self.registration is not None:
            self.expect_transport(transport)
            if self.registration.workload != workload:
                raise InstanceError("instance runtime is already registered with a different workload")
            return
        registration = InstanceRegistration(
            instance_id=self.instance_id,
            rank=self.rank,
            abi_version=ABI_VERSION,
            pid=self.process_ref.pid,
            transport=transport,
            workload=workload,
        )
        self.client.register_instance(registration)
        self.registration = registration

    def wait_for_fabric_executable(self) -> FabricPlan:
        """Wait until every Agent and FFN Executor is device-executable.

        Returns:
            Immutable active fabric plan observed at the executable barrier.

        Raises:
            InstanceError: If the barrier times out or the generation fails or drains.
        """

        deadline = time.monotonic() + INSTANCE_STARTUP_BARRIER_TIMEOUT_S
        while time.monotonic() < deadline:
            readiness = self.client.readiness()
            match readiness.fabric_phase:
                case FabricGenerationPhase.EXECUTABLE:
                    plan = self.client.fabric_plan()
                    if readiness.generation is None or plan.generation != readiness.generation:
                        raise InstanceError("daemon returned inconsistent executable Fabric generation facts")
                    self.fabric_plan = plan
                    return plan
                case None | FabricGenerationPhase.JOINING:
                    time.sleep(INSTANCE_TRANSPORT_ACQUIRE_INTERVAL_S)
                case phase:
                    failures = tuple(
                        failure
                        for failure in (
                            readiness.fabric_invocation_failure,
                            readiness.fabric_owner_failure,
                            readiness.fabric_protocol_failure,
                        )
                        if failure is not None
                    )
                    detail = f": {failures}" if failures else ""
                    raise InstanceError(f"fabric entered {phase.value} during startup{detail}")
        raise InstanceError("timed out waiting for executable fabric generation")

    def publish_initialized(self) -> None:
        """Publish the post-ModelRunner.initialize startup barrier.

        Raises:
            InstanceError: If this runtime did not observe an executable plan.
        """

        plan = self.fabric_plan
        if plan is None:
            raise InstanceError("instance cannot publish initialized before the executable fabric barrier")
        self.client.publish_instance_initialized(
            self.instance_id,
            rank=self.rank,
            publication=InstanceInitializedPublication(
                owner=self.process_ref,
                generation=plan.generation,
                plan_digest=plan.digest(),
            ),
        )

    def wait_for_ready(self) -> None:
        """Wait for every configured SGLang rank to finish initialization.

        Raises:
            InstanceError: If the barrier times out or the generation fails or drains.
        """

        deadline = time.monotonic() + INSTANCE_STARTUP_BARRIER_TIMEOUT_S
        while time.monotonic() < deadline:
            readiness = self.client.readiness()
            if readiness.ready:
                return
            match readiness.fabric_phase:
                case None | FabricGenerationPhase.JOINING | FabricGenerationPhase.EXECUTABLE:
                    time.sleep(INSTANCE_TRANSPORT_ACQUIRE_INTERVAL_S)
                case phase:
                    failures = tuple(
                        failure
                        for failure in (
                            readiness.fabric_invocation_failure,
                            readiness.fabric_owner_failure,
                            readiness.fabric_protocol_failure,
                        )
                        if failure is not None
                    )
                    detail = f": {failures}" if failures else ""
                    raise InstanceError(f"fabric entered {phase.value} during initialization{detail}")
        raise InstanceError("timed out waiting for every instance rank to initialize")

    def deregister_runtime(self) -> None:
        """Detach transport and remove this rank's registration.

        Native detach runs before daemon deregistration. The deregistration is
        the authoritative Instance-departure event that makes the daemon select
        generation-wide cooperative quiesce. If native cleanup fails, daemon
        deregistration is intentionally skipped so agents do not treat a
        still-attached CUDA IPC arena as detached.
        """

        registration = self.registration
        self.stop_failure_monitor()
        self.detach_arena()
        self.stop_heartbeat_worker()
        if registration is None:
            return
        self.client.deregister_instance(
            self.instance_id,
            rank=self.rank,
            owner=self.process_ref,
        )
        self.registration = None

    def attach_arena(self, handle: TransportArenaHandle) -> None:
        """Attach a daemon-brokered transport arena handle to the native shim."""

        if self.arena_handle is not None:
            if self.arena_handle != handle:
                raise InstanceError("instance transport arena is already attached with a different handle")
            return
        xpool.native.transport.attach_arena(self.instance_index, self.rank, handle.handle)
        self.arena_handle = handle
        if self.heartbeat_worker is not None:
            self.heartbeat_worker.arena_handle = handle
            self.heartbeat_worker.arena_lease_recovery_pending = False

    def detach_arena(self) -> None:
        """Detach this rank's native transport arena, if one is attached."""

        self.stop_failure_monitor()
        if self.arena_handle is None:
            return
        xpool.native.transport.detach_arena()
        self.arena_handle = None
        if self.heartbeat_worker is not None:
            self.heartbeat_worker.arena_handle = None
            self.heartbeat_worker.arena_lease_recovery_pending = False

    def attach_arena_from_daemon(self) -> None:
        """Acquire this rank's daemon-published arena and attach it natively."""

        deadline = time.monotonic() + TRANSPORT_METADATA_RECOVERY_DEADLINE_S
        while True:
            try:
                handle = self.client.acquire_instance_transport_arena(
                    self.instance_id,
                    rank=self.rank,
                    owner=self.process_ref,
                )
                self.attach_arena(handle)
                return
            except (XpoolDaemonError, XpoolClientError) as exc:
                if not exc.is_recoverable or time.monotonic() >= deadline:
                    raise
                time.sleep(INSTANCE_TRANSPORT_ACQUIRE_INTERVAL_S)

    def start_heartbeat_worker(self) -> None:
        """Start or replace this process's background heartbeat worker."""

        if self.registration is None:
            raise InstanceError("instance must be registered before starting heartbeat")
        if self.heartbeat_worker is None:
            self.heartbeat_worker = InstanceHeartbeat(
                instance_id=self.instance_id,
                rank=self.rank,
                local_cuda_device=self.local_cuda_device,
                registration=self.registration,
                heartbeat=self.heartbeat,
            )
            self.heartbeat_worker.arena_handle = self.arena_handle
        self.heartbeat_worker.start()

    def stop_heartbeat_worker(self) -> None:
        """Stop this process's background heartbeat worker for the instance rank."""

        if self.heartbeat_worker is None:
            return
        self.heartbeat_worker.stop()
        self.heartbeat_worker = None

    def start_failure_monitor(self) -> None:
        """Start the sticky failure monitor for the attached arena."""

        if self.arena_handle is None:
            raise InstanceError("instance failure monitor requires an attached arena")
        if self.failure_monitor is None:
            self.failure_monitor = InstanceFailureMonitor(
                instance_id=self.instance_id,
                instance_index=self.instance_index,
                rank=self.rank,
            )
        self.failure_monitor.start()

    def stop_failure_monitor(self) -> None:
        """Stop the sticky failure monitor, if one is active."""

        if self.failure_monitor is None:
            return
        self.failure_monitor.stop()
        self.failure_monitor = None

    def close(self) -> None:
        """Release all runtime resources in dependency order.

        Cleanup is idempotent after success. If native detach fails, the
        registration and client remain available so the caller can retry.
        """

        try:
            self.deregister_runtime()
        finally:
            if self.registration is None and self.arena_handle is None:
                self.client.close()

    def expect_transport(self, transport: InstanceTransportAttributes) -> None:
        """Raise when a started runtime receives different transport attributes."""

        if self.registration is None:
            return
        installed = self.registration.transport.model_dump(mode="json")
        requested = transport.model_dump(mode="json")
        if installed != requested:
            raise InstanceError("xpool instance runtime already started with different transport attributes")
