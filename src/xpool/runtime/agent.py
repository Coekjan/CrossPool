"""Shared runtime lifecycle support for xpool agents."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from xpool import bootstrap, devkit
from xpool.abi import ABI_VERSION, RuntimeRole
from xpool.service.client import XpoolClient, XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import HeartbeatResponse, ProcessHeartbeat, ProcessRef
from xpool.utils.background import BackgroundThread
from xpool.utils.procs import ProcUniqId

AGENT_CONTROL_INTERVAL_S = 0.5
AGENT_HEARTBEAT_INTERVAL_S = 5.0
AGENT_HEARTBEAT_STOP_JOIN_TIMEOUT_S = 5.0
AGENT_SHUTDOWN_POLL_INTERVAL_S = 0.5
logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Raised when an xpool agent cannot be launched or kept running."""


class Agent(ABC):
    """Common process-local resources for one configured xpool agent."""

    cuda_device: int
    proc_id: ProcUniqId
    heartbeat_payload: ProcessHeartbeat
    process_ref: ProcessRef
    client: XpoolClient
    registered: bool

    def __init__(self, *, cuda_device: int, runtime_role: RuntimeRole) -> None:
        """Initialize native state and daemon client for one agent role."""

        bootstrap.init(cuda_device, runtime_role)
        devkit.install()
        self.cuda_device = cuda_device
        self.proc_id = ProcUniqId.current()
        self.heartbeat_payload = ProcessHeartbeat(abi_version=ABI_VERSION, pid=self.proc_id.pid)
        self.process_ref = ProcessRef(abi_version=ABI_VERSION, pid=self.proc_id.pid)
        self.client = XpoolClient()
        self.registered = False

    @abstractmethod
    def register(self) -> None:
        """Register the concrete agent type with the daemon."""

        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """Run the concrete agent lifecycle until interrupted."""

        raise NotImplementedError


class AgentHeartbeat:
    """Background heartbeat worker for one concrete agent endpoint."""

    def __init__(
        self,
        *,
        cuda_device: int,
        heartbeat: ProcessHeartbeat,
        sender: Callable[[int, ProcessHeartbeat], HeartbeatResponse],
        interval_s: float = AGENT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        """Create a stopped agent heartbeat worker."""

        self.cuda_device = cuda_device
        self.heartbeat = heartbeat
        self.sender = sender
        self.interval_s = interval_s
        self.worker = BackgroundThread.periodic(
            name=f"xpool-agent-heartbeat-{cuda_device}",
            interval_s=interval_s,
            target=self.heartbeat_once,
            join_timeout_s=AGENT_HEARTBEAT_STOP_JOIN_TIMEOUT_S,
        )
        self.lock = threading.Lock()
        self.registration_missing = False
        self.closed = False

    @property
    def thread(self) -> threading.Thread | None:
        """Return the current heartbeat thread, if started."""

        return self.worker.thread

    def start(self) -> None:
        """Start periodic heartbeats."""

        with self.lock:
            if self.closed:
                raise AgentError("cannot restart a closed agent heartbeat worker")
            if self.worker.is_running:
                return
            self.registration_missing = False
        self.worker.start()

    def stop(self) -> None:
        """Stop periodic heartbeats."""

        self.worker.stop()

    def close(self) -> None:
        """Stop and permanently close the worker."""

        with self.lock:
            if self.closed:
                return
        self.worker.close()
        with self.lock:
            self.closed = True

    def consume_registration_missing(self) -> bool:
        """Return and clear the missing-registration signal."""

        with self.lock:
            missing = self.registration_missing
            self.registration_missing = False
        return missing

    def raise_if_failed(self) -> None:
        """Raise a fatal worker failure, if recorded."""

        self.worker.raise_if_failed()

    def heartbeat_once(self) -> bool:
        """Send one heartbeat and indicate whether periodic work should continue."""

        try:
            self.sender(self.cuda_device, self.heartbeat)
        except XpoolDaemonError as exc:
            if exc.is_recoverable:
                logger.warning("agent registration for CUDA device %s is missing", self.cuda_device)
                with self.lock:
                    self.registration_missing = True
                return False
            raise AgentError(f"agent heartbeat received unrecoverable daemon error: {exc}") from exc
        except XpoolClientError as exc:
            if not exc.is_recoverable:
                raise AgentError(f"agent heartbeat received unrecoverable client error: {exc}") from exc
            logger.warning("xpool agent heartbeat failed: %s", exc)
        except Exception as exc:
            raise AgentError(f"agent heartbeat failed with unexpected error: {exc}") from exc
        return True
