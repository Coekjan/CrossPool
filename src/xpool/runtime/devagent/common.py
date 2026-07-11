"""Shared devagent runtime utilities."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

from xpool import bootstrap, devkit
from xpool.abi import ABI_VERSION, RuntimeRole
from xpool.config import DeviceRole
from xpool.service.client import XpoolClient, XpoolClientError, XpoolDaemonError
from xpool.service.wire import DevagentRegistration, ProcessHeartbeat, ProcessRef
from xpool.utils.background import BackgroundThread
from xpool.utils.procs import ProcUniqId

DEVAGENT_CONTROL_INTERVAL_S = 0.5
DEVAGENT_HEARTBEAT_INTERVAL_S = 5.0
DEVAGENT_HEARTBEAT_STOP_JOIN_TIMEOUT_S = 5.0
DEVAGENT_SHUTDOWN_POLL_INTERVAL_S = 0.5
logger = logging.getLogger(__name__)


class DevagentError(RuntimeError):
    """Raised when a devagent runtime cannot be launched or kept running."""


class Devagent(ABC):
    """Runtime lifecycle controller for one configured devagent process."""

    cuda_device: int
    role: DeviceRole
    proc_id: ProcUniqId
    registration: DevagentRegistration
    heartbeat_payload: ProcessHeartbeat
    process_ref: ProcessRef
    client: XpoolClient
    registered: bool

    def __init__(self, *, cuda_device: int, role: DeviceRole) -> None:
        """Create native state, registration payloads, and the daemon client."""

        bootstrap.init(cuda_device, RuntimeRole.DEVAGENT)
        devkit.install()
        self.cuda_device = cuda_device
        self.role = role
        self.proc_id = ProcUniqId.current()
        self.registration = DevagentRegistration(
            cuda_device=cuda_device,
            abi_version=ABI_VERSION,
            pid=self.proc_id.pid,
        )
        self.heartbeat_payload = ProcessHeartbeat(abi_version=ABI_VERSION, pid=self.proc_id.pid)
        self.process_ref = ProcessRef(abi_version=ABI_VERSION, pid=self.proc_id.pid)
        self.client = XpoolClient()
        self.registered = False

    @abstractmethod
    def run(self) -> None:
        """Run the role-specific devagent lifecycle until interrupted."""

        raise NotImplementedError

    def register(self) -> None:
        """Register this devagent and update process-local registration state."""

        try:
            self.client.register_devagent(self.registration)
        except (XpoolDaemonError, XpoolClientError) as exc:
            self.registered = False
            if not exc.is_recoverable:
                raise DevagentError(f"devagent registration received unrecoverable daemon error: {exc}") from exc
            logger.warning("devagent registration failed: %s", exc)
            return
        self.registered = True


class DevagentHeartbeat:
    """Background heartbeat worker for one devagent process.

    Args:
        cuda_device: CUDA device owned by the devagent.
        heartbeat: Stable process heartbeat payload.
        interval_s: Heartbeat period in seconds. Defaults to the devagent
            heartbeat interval.

    Side Effects:
        Owns one daemon client for the worker lifetime. A missing daemon
        registration is reported to the owner instead of re-registering here.
    """

    cuda_device: int
    heartbeat: ProcessHeartbeat
    interval_s: float
    client: XpoolClient
    worker: BackgroundThread

    def __init__(
        self,
        *,
        cuda_device: int,
        heartbeat: ProcessHeartbeat,
        interval_s: float = DEVAGENT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        """Create a stopped devagent heartbeat worker."""

        self.cuda_device = cuda_device
        self.heartbeat = heartbeat
        self.interval_s = interval_s
        self.client = XpoolClient()
        self.worker = BackgroundThread.periodic(
            name=f"xpool-devagent-heartbeat-{self.cuda_device}",
            interval_s=interval_s,
            target=self.heartbeat_once,
            join_timeout_s=DEVAGENT_HEARTBEAT_STOP_JOIN_TIMEOUT_S,
        )
        self.lock = threading.Lock()
        self.registration_missing = False
        self.closed = False

    @property
    def thread(self) -> threading.Thread | None:
        """Return the current heartbeat thread, if one has been started."""

        return self.worker.thread

    def start(self) -> None:
        """Start the heartbeat thread when it is not already running."""

        with self.lock:
            if self.closed:
                raise DevagentError("cannot restart a closed devagent heartbeat worker")
            if self.worker.is_running:
                return
            self.registration_missing = False
        self.worker.start()

    def stop(self) -> None:
        """Stop the heartbeat thread and wait briefly for cleanup."""

        self.worker.stop()

    def close(self) -> None:
        """Stop the heartbeat thread and close the owned daemon client."""

        with self.lock:
            if self.closed:
                return
        self.worker.close()
        self.client.close()
        with self.lock:
            self.closed = True

    def consume_registration_missing(self) -> bool:
        """Return and clear the missing-registration signal."""

        with self.lock:
            registration_missing = self.registration_missing
            self.registration_missing = False
        return registration_missing

    def raise_if_failed(self) -> None:
        """Raise the fatal heartbeat failure, if the worker recorded one."""

        self.worker.raise_if_failed()

    def heartbeat_once(self) -> bool:
        """Send one heartbeat and report whether periodic execution should continue."""

        try:
            self.client.heartbeat_devagent(self.cuda_device, self.heartbeat)
        except XpoolDaemonError as exc:
            if exc.is_recoverable:
                logger.warning(
                    "xpool devagent registration for CUDA device %s is missing; re-registering",
                    self.cuda_device,
                )
                with self.lock:
                    self.registration_missing = True
                return False
            raise DevagentError(f"devagent heartbeat received unrecoverable daemon error: {exc}") from exc
        except XpoolClientError as exc:
            if not exc.is_recoverable:
                raise DevagentError(f"devagent heartbeat received unrecoverable client error: {exc}") from exc
            logger.warning("xpool devagent heartbeat failed: %s", exc)
        except Exception as exc:
            raise DevagentError(f"devagent heartbeat failed with unexpected error: {exc}") from exc
        return True
