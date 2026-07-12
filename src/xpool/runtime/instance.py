"""Instance runtime registration and transport arena handoff."""

from __future__ import annotations

import logging
import os
import threading
import time

import xpool.ops
from xpool.abi import ABI_VERSION, FfnResultErrorCode, TransportArenaHandle
from xpool.config import get_global_config
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.client import XpoolClient, XpoolClientError, XpoolDaemonError
from xpool.service.wire import (
    HeartbeatResponse,
    InstanceRegistration,
    ProcessHeartbeat,
    ProcessRef,
)
from xpool.utils.background import BackgroundThread
from xpool.utils.procs import bail

__all__ = [
    "Instance",
    "InstanceError",
    "InstanceHeartbeat",
    "InstanceTransportMonitor",
    "del_instance",
    "get_instance",
    "init_instance",
]

INSTANCE_HEARTBEAT_INTERVAL_S = 5.0
STALE_ATNAGENT_RECOVERY_GRACE_S = 60.0
TRANSPORT_METADATA_RECOVERY_DEADLINE_S = 60.0
INSTANCE_TRANSPORT_ACQUIRE_INTERVAL_S = 0.5
INSTANCE_HEARTBEAT_STOP_JOIN_TIMEOUT_S = 5.0
INSTANCE_TRANSPORT_MONITOR_INTERVAL_S = 0.1
INSTANCE_TRANSPORT_MONITOR_STOP_JOIN_TIMEOUT_S = 5.0
logger = logging.getLogger(__name__)


class InstanceError(RuntimeError):
    """Raised when daemon-brokered FFN shim transport arenas cannot be attached."""


class InstanceTransportMonitor:
    """Fail-close an instance when its device transport records an error.

    Args:
        instance_id: Configured instance id used in fatal diagnostics.
        instance_index: Integer instance index used by the native arena map.
        rank: Rank-local process index within the instance.

    Attributes:
        worker: Periodic background thread that polls native sticky error state.
    """

    def __init__(self, *, instance_id: str, instance_index: int, rank: int) -> None:
        """Create a stopped transport error monitor."""

        self.instance_id = instance_id
        self.instance_index = instance_index
        self.rank = rank
        self.worker = BackgroundThread.periodic(
            name=f"xpool-instance-transport-monitor-{instance_id}-{rank}",
            interval_s=INSTANCE_TRANSPORT_MONITOR_INTERVAL_S,
            target=self.step,
            join_timeout_s=INSTANCE_TRANSPORT_MONITOR_STOP_JOIN_TIMEOUT_S,
        )

    def start(self) -> None:
        """Start polling native transport error state."""

        self.worker.start()

    def stop(self) -> None:
        """Stop polling native transport error state."""

        self.worker.close()

    def step(self) -> bool:
        """Poll once and terminate the process when the arena has failed."""

        error = xpool.ops.instance.transport_error_snapshot(self.instance_index, self.rank)
        if error != FfnResultErrorCode.OK:
            bail(
                logger,
                "xpool transport executor failed for instance %s rank %s: %s",
                self.instance_id,
                self.rank,
                error.name,
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
        heartbeat: ProcessHeartbeat,
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
        if "terminating_atnagent" in warning_kinds:
            logger.warning(
                "xpool daemon reported terminating agent on CUDA device %s for instance %s rank %s; "
                "waiting for daemon-scoped termination",
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
    heartbeat: ProcessHeartbeat
    heartbeat_worker: InstanceHeartbeat | None
    transport_monitor: InstanceTransportMonitor | None
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
        self.heartbeat = ProcessHeartbeat(abi_version=ABI_VERSION, pid=pid)
        self.heartbeat_worker = None
        self.transport_monitor = None
        self.arena_handle = None

    def start_runtime(self, transport: InstanceTransportAttributes) -> None:
        """Register, heartbeat, and attach this instance rank transactionally."""

        if self.registration is not None:
            self.expect_transport(transport)
            return
        self.register_runtime(transport)
        try:
            self.start_heartbeat_worker()
            self.attach_arena_from_daemon()
            self.start_transport_monitor()
        except Exception:
            try:
                self.deregister_runtime()
            except Exception as cleanup_exc:
                logger.warning("failed to clean up xpool instance registration: %s", cleanup_exc)
            raise

    def register_runtime(self, transport: InstanceTransportAttributes) -> None:
        """Register this instance rank with the daemon."""

        if self.registration is not None:
            self.expect_transport(transport)
            return
        registration = InstanceRegistration(
            instance_id=self.instance_id,
            rank=self.rank,
            abi_version=ABI_VERSION,
            pid=self.process_ref.pid,
            transport=transport,
        )
        self.client.register_instance(registration)
        self.registration = registration

    def deregister_runtime(self) -> None:
        """Detach native transport and remove this rank's daemon registration.

        Native detach runs first. If native cleanup fails, daemon deregistration
        is intentionally skipped so agents do not treat a still-attached CUDA
        IPC arena as detached.
        """

        registration = self.registration
        self.stop_transport_monitor()
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
        xpool.ops.instance.attach_transport_arena(self.instance_index, self.rank, handle)
        self.arena_handle = handle
        if self.heartbeat_worker is not None:
            self.heartbeat_worker.arena_handle = handle
            self.heartbeat_worker.arena_lease_recovery_pending = False

    def detach_arena(self) -> None:
        """Detach this rank's native transport arena, if one is attached."""

        self.stop_transport_monitor()
        xpool.ops.instance.detach_transport_arena(self.instance_index, self.rank)
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

    def start_transport_monitor(self) -> None:
        """Start the device transport error monitor for the attached arena."""

        if self.arena_handle is None:
            raise InstanceError("instance transport monitor requires an attached arena")
        if self.transport_monitor is None:
            self.transport_monitor = InstanceTransportMonitor(
                instance_id=self.instance_id,
                instance_index=self.instance_index,
                rank=self.rank,
            )
        self.transport_monitor.start()

    def stop_transport_monitor(self) -> None:
        """Stop the device transport error monitor, if one is active."""

        if self.transport_monitor is None:
            return
        self.transport_monitor.stop()
        self.transport_monitor = None

    def close(self) -> None:
        """Stop heartbeat activity and close the daemon client."""

        self.stop_transport_monitor()
        self.stop_heartbeat_worker()
        self.client.close()

    def expect_identity(self, instance_id: str, rank: int) -> None:
        """Raise when a singleton request targets a different instance identity."""

        if self.instance_id != instance_id or self.rank != rank:
            raise InstanceError(
                "xpool instance runtime singleton already owns "
                f"{self.instance_id} rank {self.rank}, got {instance_id} rank {rank}"
            )

    def expect_transport(self, transport: InstanceTransportAttributes) -> None:
        """Raise when a started singleton is requested with different transport attributes."""

        if self.registration is None:
            return
        installed = self.registration.transport.model_dump(mode="json")
        requested = transport.model_dump(mode="json")
        if installed != requested:
            raise InstanceError("xpool instance runtime singleton already started with different transport attributes")


instance_runtime: Instance | None = None
instance_runtime_lock = threading.RLock()


def init_instance(
    *,
    instance_id: str,
    rank: int,
    transport: InstanceTransportAttributes,
) -> Instance:
    """Initialize the process-global instance runtime exactly once.

    Args:
        instance_id: Configured model/instance id owned by this SGLang rank.
        rank: Rank-local SGLang process index within the instance.
        transport: Transport capacity and topology reported by this rank.

    Returns:
        Active process-global instance runtime.

    Raises:
        InstanceError: If an existing runtime has a different identity or
            transport contract.

    Side Effects:
        Registers the rank, starts heartbeat and transport monitoring, and
        publishes the singleton only after startup succeeds.
    """

    global instance_runtime
    with instance_runtime_lock:
        if instance_runtime is not None:
            instance_runtime.expect_identity(instance_id, rank)
            if instance_runtime.registration is None:
                instance_runtime.start_runtime(transport)
            else:
                instance_runtime.expect_transport(transport)
            return instance_runtime

        candidate = Instance(instance_id=instance_id, rank=rank)
        try:
            candidate.start_runtime(transport)
        except Exception:
            candidate.close()
            raise
        instance_runtime = candidate
        return candidate


def get_instance() -> Instance:
    """Return the initialized process-global instance runtime.

    Returns:
        Active instance runtime.

    Raises:
        InstanceError: If this process has not initialized an instance runtime.
    """

    with instance_runtime_lock:
        if instance_runtime is None:
            raise InstanceError("xpool instance runtime is not initialized")
        return instance_runtime


def del_instance() -> None:
    """Deregister and delete the process-global instance runtime.

    Side Effects:
        Stops monitor and heartbeat workers, detaches native transport,
        deregisters from the daemon, and closes the daemon client. The
        singleton remains installed if any cleanup step fails so cleanup can be
        retried safely.
    """

    global instance_runtime
    with instance_runtime_lock:
        if instance_runtime is None:
            return
        instance_runtime.deregister_runtime()
        instance_runtime.close()
        instance_runtime = None
