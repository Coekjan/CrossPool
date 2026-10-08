"""Owned CUDA MPS lifecycle, client termination and availability observations."""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import shutil
import stat
import subprocess
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import BinaryIO

import psutil

from xpool.utils.procs import ProcUniqId
from xpool.utils.sighandler import defer_signal_exceptions

__all__ = [
    "MPS_CLEANUP_TIMEOUT_S",
    "MPS_STARTUP_TIMEOUT_S",
    "MPS_TERMINATION_TIMEOUT_S",
    "MpsEndpoint",
    "MpsProbeResult",
    "MpsScope",
]

MPS_CONTROL_COMMAND = "nvidia-cuda-mps-control"
MPS_PROBE_LOCK_DIRECTORY = Path("/tmp") / f"xpool-mps-probe-locks-{os.getuid()}"
MPS_STARTUP_TIMEOUT_S = 30.0
MPS_CLEANUP_TIMEOUT_S = 300.0
MPS_TERMINATION_TIMEOUT_S = 30.0
MPS_SCOPE_DIRECTORY = Path("/tmp") / f"xpool-mps-{os.getuid()}"


@dataclass(frozen=True, slots=True)
class MpsProbeResult:
    """Result of one bounded CUDA MPS controller probe.

    Attributes:
        online: Successful availability, unsuccessful availability, or None
            when command serialization is unavailable.
        active_thread_percentage: Integral controller percentage when online.
        diagnostic: Stable human-readable reason for logs when offline.
    """

    online: bool | None
    active_thread_percentage: int | None
    diagnostic: str


@dataclass(frozen=True, slots=True)
class MpsEndpoint:
    """Address and inspect MPS for an ordered physical attention-device view.

    Construction validates full canonical UUIDs and creates no resources.
    Address computation sorts identities; execution visibility retains rank order.

    Attributes:
        device_uuids: Nonempty, unique physical UUIDs in attention rank order.
    """

    device_uuids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject empty, duplicate or noncanonical physical identities."""

        if not self.device_uuids or len(self.device_uuids) != len(set(self.device_uuids)):
            raise ValueError("MPS addressing requires nonempty unique device UUIDs")
        if any(not value.startswith("GPU-") or value != f"GPU-{uuid.UUID(value[4:])}" for value in self.device_uuids):
            raise ValueError("MPS addressing requires full canonical physical UUIDs")

    @property
    def directory(self) -> Path:
        """Return the per-user address derived from the unordered device set."""

        key = hashlib.sha256("\n".join(sorted(self.device_uuids)).encode("ascii")).hexdigest()[:32]
        return MPS_SCOPE_DIRECTORY / key

    @property
    def pipe_directory(self) -> Path:
        """Return the management and client pipe address."""

        return self.directory / "pipe"

    @property
    def log_directory(self) -> Path:
        """Return the controller and server diagnostic directory."""

        return self.directory / "log"

    def environment(self) -> dict[str, str]:
        """Return pipe/log overrides without modifying process visibility."""

        return {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_directory),
            "CUDA_MPS_LOG_DIRECTORY": str(self.log_directory),
        }

    def require_environment(self, environment: Mapping[str, str]) -> None:
        """Require matching pipe/log fields in an environment.

        Unrelated fields are ignored. A mismatch raises ``ValueError``.
        """

        for name, value in self.environment().items():
            if environment.get(name) != value:
                raise ValueError(f"MPS environment requires {name}={value!r}")

    def probe(self, *, deadline: float | None = None) -> MpsProbeResult:
        """Return whether the configured CUDA MPS control daemon is reachable.

        The probe queries controller state that exists before an MPS server is
        created, avoiding a readiness dependency on the first CUDA client.

        Args:
            deadline: Applicable absolute monotonic owner deadline, when present.

        Returns:
            Structured controller availability, active-thread percentage, and
            diagnostic detail. A busy serialization lock returns unavailable.

        Raises:
            subprocess.TimeoutExpired: The actual management command times out.
            TimeoutError: The enclosing owner deadline has expired.

        Side Effects:
            Queries this endpoint without initializing a context or acquiring ownership.
        """

        try:
            output = self.run_control("get_default_active_thread_percentage", deadline=deadline)
        except BlockingIOError:
            return MpsProbeResult(None, None, "MPS controller probe serialization is busy")
        except TimeoutError:
            raise
        except (OSError, RuntimeError) as exc:
            return MpsProbeResult(False, None, f"failed to execute serialized MPS controller probe: {exc}")
        try:
            active_thread_percentage = float(output)
        except ValueError:
            return MpsProbeResult(
                False,
                None,
                f"MPS controller returned an invalid active-thread percentage: {output!r}",
            )
        if (
            not math.isfinite(active_thread_percentage)
            or not active_thread_percentage.is_integer()
            or not 1 <= active_thread_percentage <= 100
        ):
            return MpsProbeResult(
                False,
                None,
                f"MPS controller returned an out-of-range or non-integral active-thread percentage: {output}",
            )
        percentage = int(active_thread_percentage)
        return MpsProbeResult(True, percentage, f"MPS controller is online with {percentage}% active threads")

    def run_control(
        self,
        command: str,
        *,
        deadline: float | None = None,
    ) -> str:
        """Execute one serialized management command at the selected endpoint.

        Args:
            command: One nonempty management command without a line terminator.
            deadline: Applicable absolute monotonic owner deadline, when present.

        Returns:
            Stripped controller output. Its command-specific fields remain untrusted.

        Raises:
            ValueError: The command is empty or contains a line terminator.
            BlockingIOError: Another command owns the serialization lock and no
                enclosing deadline permits bounded waiting.
            OSError: The binary or lock storage is unavailable.
            RuntimeError: The command reports failure.
            subprocess.TimeoutExpired: An executed command exceeds its bound.
            TimeoutError: The enclosing deadline expires before execution.

        Side Effects:
            Takes the endpoint lock, waiting only when the caller supplied an owner
            deadline. Serialization and execution share the operation bound capped
            by the owner's remaining budget.
        """

        if not command.strip() or "\n" in command or "\r" in command:
            raise ValueError("MPS control requires one nonempty command")
        endpoint = self.pipe_directory.resolve(strict=False)
        executable = shutil.which(MPS_CONTROL_COMMAND)
        if executable is None:
            raise FileNotFoundError(f"{MPS_CONTROL_COMMAND} is not installed or not on PATH")
        MPS_PROBE_LOCK_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
        operation_deadline = monotonic() + MPS_STARTUP_TIMEOUT_S
        if deadline is not None:
            operation_deadline = min(operation_deadline, deadline)
        digest = hashlib.sha256(os.fsencode(endpoint)).hexdigest()
        with (MPS_PROBE_LOCK_DIRECTORY / f"{digest}.lock").open("a+", encoding="utf-8") as lock_file:
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if deadline is None:
                        raise
                    remaining = operation_deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError("MPS control deadline expired during serialization") from None
                    sleep(min(0.01, remaining))
            timeout = operation_deadline - monotonic()
            if timeout <= 0:
                raise TimeoutError("MPS control owner deadline expired before command execution")
            environment = dict(os.environ)
            environment["CUDA_MPS_PIPE_DIRECTORY"] = str(endpoint)
            completed = subprocess.run(
                [executable],
                input=f"{command}\n",
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"MPS control command {command!r} failed: {detail}")
        return completed.stdout.strip()

    def require_client(self) -> None:
        """Require this initialized PID to be an actual client of this endpoint.

        Inspection shares one thirty-second budget across serialized queries.
        An absent client or invalid reply raises ``RuntimeError``; management
        and deadline errors propagate. This check creates no device context
        and acquires no controller ownership.
        """

        deadline = monotonic() + MPS_STARTUP_TIMEOUT_S
        servers = self.parse_process_ids(self.run_control("get_server_list", deadline=deadline), allow_empty=True)
        for server in servers:
            clients = self.parse_process_ids(
                self.run_control(f"get_client_list {server}", deadline=deadline), allow_empty=True
            )
            if os.getpid() in clients:
                return
        raise RuntimeError(f"process {os.getpid()} is not connected to attention MPS at {self.pipe_directory}")

    @staticmethod
    def parse_process_ids(output: str, *, allow_empty: bool) -> tuple[int, ...]:
        """Parse unique positive process IDs from one management reply.

        Empty replies are accepted only for commands whose protocol permits them,
        including lazy server startup and an empty client list. Invalid external
        replies raise ``RuntimeError`` rather than becoming proof of an empty domain.

        Args:
            output: Stripped command output from ``run_control``.
            allow_empty: Whether this command permits an empty result.

        Returns:
            Unique positive process IDs in reply order.

        Raises:
            RuntimeError: The reply is invalid or unexpectedly empty.
        """

        if not output:
            if allow_empty:
                return ()
            raise RuntimeError("MPS controller returned no process IDs")
        values = output.splitlines()
        if any(not value.isascii() or not value.isdigit() or int(value) <= 0 for value in values):
            raise RuntimeError(f"MPS controller returned invalid process IDs: {output!r}")
        process_ids = tuple(int(value) for value in values)
        if len(process_ids) != len(set(process_ids)):
            raise RuntimeError(f"MPS controller returned duplicate process IDs: {process_ids}")
        return process_ids


class MpsScope:
    """Own one foreground controller and its attention-side MPS endpoint.

    The enclosing deployment or topology owner retains this object before
    calling ``start`` and proves client retirement before ``stop``. This scope
    does not register clients or finalize Fabric.

    Attributes:
        endpoint: Non-owning attention visibility, address and client-query value.
        controller: Retained subprocess even after a failed startup or stop.
        controller_identity: Exact live identity captured after launch.
        servers: Exact observed server identities, retained after their exit.
        closed: Whether resource cleanup has been verified successfully.
        cleanup_deadline: Owning retirement's absolute monotonic deadline, never renewed.
    """

    def __init__(self, endpoint: MpsEndpoint) -> None:
        """Retain an endpoint without creating files or starting its controller."""

        self.endpoint = endpoint
        self.directory_identity: tuple[int, int] | None = None
        self.controller: subprocess.Popen[bytes] | None = None
        self.controller_identity: ProcUniqId | None = None
        self.controller_log: BinaryIO | None = None
        self.servers: dict[int, ProcUniqId] = {}
        self.startup_deadline: float | None = None
        self.cleanup_deadline: float | None = None
        self.stop_lock = threading.Lock()
        self.closed = False

    def start(self) -> None:
        """Claim the scope endpoint and start its controller within thirty seconds.

        Raises:
            RuntimeError: Startup was already attempted or the created controller
                exits before availability.
            OSError: Resource creation or process observation fails.
            InterruptedError: The enclosing owner has started retirement.
            TimeoutError: The startup deadline expires.
            subprocess.TimeoutExpired: A bounded management command
                times out. The enclosing owner decides whether recovery applies.

        Side Effects:
            Creates private pipe/log directories and launches the foreground
            controller in a new session. Partial startup retains every actual
            handle for the enclosing owner's ordered rollback.
        """

        if self.closed or self.startup_deadline is not None:
            raise RuntimeError("MPS scope startup was already attempted")
        self.startup_deadline = monotonic() + MPS_STARTUP_TIMEOUT_S
        executable = shutil.which(MPS_CONTROL_COMMAND)
        if executable is None:
            raise FileNotFoundError(f"{MPS_CONTROL_COMMAND} is not installed or not on PATH")

        if monotonic() >= self.startup_deadline:
            raise TimeoutError("MPS startup deadline expired before controller creation")
        if self.cleanup_deadline is not None:
            raise InterruptedError("MPS scope is retiring")

        MPS_SCOPE_DIRECTORY.mkdir(mode=0o700, exist_ok=True)
        facts = MPS_SCOPE_DIRECTORY.lstat()
        if not stat.S_ISDIR(facts.st_mode) or facts.st_uid != os.getuid() or facts.st_mode & 0o022:
            raise PermissionError(f"MPS directory has incompatible ownership: {MPS_SCOPE_DIRECTORY}")
        with defer_signal_exceptions():
            self.endpoint.directory.mkdir(mode=0o700, exist_ok=False)
            facts = self.endpoint.directory.lstat()
            self.directory_identity = facts.st_dev, facts.st_ino
        self.endpoint.pipe_directory.mkdir(mode=0o700)
        self.endpoint.log_directory.mkdir(mode=0o700)
        with defer_signal_exceptions():
            self.controller_log = (self.endpoint.log_directory / "controller.log").open("ab")
        environment = dict(os.environ)
        environment.update(self.endpoint.environment())
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(self.endpoint.device_uuids)
        if monotonic() >= self.startup_deadline:
            raise TimeoutError("MPS startup deadline expired before controller creation")
        if self.cleanup_deadline is not None:
            raise InterruptedError("MPS scope is retiring")
        with defer_signal_exceptions():
            self.controller = subprocess.Popen(
                [executable, "-f"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self.controller_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.controller_identity = ProcUniqId(self.controller.pid)
        if os.getpgid(self.controller.pid) != self.controller.pid:
            raise RuntimeError("owned MPS controller did not establish its isolated process group")
        while monotonic() < self.startup_deadline:
            if self.cleanup_deadline is not None:
                raise InterruptedError("MPS scope is retiring")
            if self.controller.poll() is not None:
                raise RuntimeError(
                    f"owned MPS controller exited during startup; diagnostics: {self.endpoint.log_directory}"
                )
            result = self.endpoint.probe(deadline=self.startup_deadline)
            if result.online is True:
                return
            sleep(min(0.05, max(0.0, self.startup_deadline - monotonic())))
        raise TimeoutError(f"MPS startup deadline expired; diagnostics: {self.endpoint.log_directory}")

    def probe(self) -> MpsProbeResult:
        """Observe this retained controller within the shared startup query budget.

        Exact process-observation errors and actual command timeouts propagate
        to the enclosing owner. A confirmed exited controller is offline; a
        successful reply does not establish client membership.
        """

        if self.controller is None or self.controller.poll() is not None:
            return MpsProbeResult(False, None, "owned MPS controller is not running")
        if self.controller_identity is None or not self.controller_identity.is_alive():
            return MpsProbeResult(False, None, "owned MPS controller identity is absent")
        deadline = monotonic() + MPS_STARTUP_TIMEOUT_S
        if self.cleanup_deadline is not None:
            deadline = min(deadline, self.cleanup_deadline)
        return self.endpoint.probe(deadline=deadline)

    def observe_servers(self, *, deadline: float | None = None) -> tuple[ProcUniqId, ...]:
        """Retain exact server identities after validating their controller link.

        Returns:
            Current server identities, empty before lazy server creation.

        Raises:
            RuntimeError: The controller is absent or a server's ownership/group
                cannot be established from the retained live controller.
            OSError: Process or management observation fails.
            subprocess.TimeoutExpired: The actual management command times out.

        Side Effects:
            Queries the owned endpoint and retains observed server identities for
            retirement and fault evidence, without registering client ownership.
        """

        if self.controller_identity is None or not self.controller_identity.is_alive():
            raise RuntimeError("MPS server observation requires the retained live controller")
        output = self.endpoint.run_control("get_server_list", deadline=deadline)
        identities = []
        for process_id in self.endpoint.parse_process_ids(output, allow_empty=True):
            identity = ProcUniqId(process_id)
            process = psutil.Process(process_id)
            if (
                process.create_time() != identity.create_time
                or process.ppid() != self.controller_identity.pid
                or os.getpgid(process_id) != self.controller_identity.pid
            ):
                raise RuntimeError(f"MPS server {process_id} is outside the owned controller domain")
            previous = self.servers.get(process_id)
            if previous is not None and previous != identity:
                raise RuntimeError(f"MPS server process identity was reused: {process_id}")
            self.servers[process_id] = identity
            identities.append(identity)
        return tuple(identities)

    def stop(self, *, deadline: float) -> None:
        """Stop after the enclosing owner prevents new startup and retires clients.

        Args:
            deadline: Absolute monotonic deadline of the enclosing cleanup.
                Repeated calls keep the first deadline, including failed stop.

        Raises:
            RuntimeError: Clients, unknown descendants or owned resources remain.
            OSError: Resource observation, management or housekeeping fails.
            TimeoutError: The original cleanup envelope expires.
            subprocess.TimeoutExpired: A bounded management/wait operation expires.

        Side Effects:
            Serializes close attempts, requests owned controller exit, verifies
            process-domain retirement, closes its log and removes only its own
            directory. Failure retains handles and diagnostics.
            The caller orders client-before-controller retirement. An empty
            client snapshot is an additional check, not a future-connection fence.
        """

        with self.stop_lock:
            if self.closed:
                return
            if self.cleanup_deadline is None:
                self.cleanup_deadline = deadline
            deadline = self.cleanup_deadline
            if self.controller is not None:
                if self.controller.poll() is None:
                    identities = self.observe_servers(deadline=deadline)
                    for identity in identities:
                        clients = self.endpoint.run_control(f"get_client_list {identity.pid}", deadline=deadline)
                        if self.endpoint.parse_process_ids(clients, allow_empty=True):
                            raise RuntimeError(f"MPS server {identity.pid} still has clients; retaining ownership")
                    if self.controller_identity is None:
                        raise RuntimeError("MPS controller identity is unavailable; retaining ownership")
                    descendants = self.controller_identity.child_process_ids()
                    if set(descendants) != set(identities):
                        raise RuntimeError("MPS controller has unobserved descendants; retaining ownership")
                    self.endpoint.run_control("quit", deadline=deadline)
                    self.controller.wait(timeout=max(0.0, deadline - monotonic()))
                # A dead leader alone is insufficient. The isolated group must
                # be gone and every retained server identity confirmed absent.
                while True:
                    try:
                        os.killpg(self.controller.pid, 0)
                    except ProcessLookupError:
                        if any(identity.is_alive() for identity in self.servers.values()):
                            raise RuntimeError("MPS server outlived the owned process domain; retaining ownership")
                        break
                    if monotonic() >= deadline:
                        raise TimeoutError(
                            f"MPS process domain remains alive; diagnostics: {self.endpoint.log_directory}"
                        )
                    sleep(min(0.05, max(0.0, deadline - monotonic())))
            if self.controller_log is not None:
                self.controller_log.close()
            if self.directory_identity is not None:
                facts = self.endpoint.directory.lstat()
                if (facts.st_dev, facts.st_ino) != self.directory_identity or not stat.S_ISDIR(facts.st_mode):
                    raise RuntimeError(
                        f"owned MPS directory was replaced; retaining diagnostics: {self.endpoint.directory}"
                    )
                shutil.rmtree(self.endpoint.directory)
                self.directory_identity = None
            self.closed = True

    def terminate_client(self, target: ProcUniqId, *, deadline: float) -> None:
        """Terminate this exact client's contexts without killing its process.

        The caller selects its own target and stops further resource creation.
        This operation confirms actual membership at the retained controller;
        only NVIDIA's successful result authorizes a subsequent host signal.
        It neither finalizes Fabric nor releases the controller.

        Args:
            target: Exact live identity in the controller's user/PID namespace.
            deadline: Absolute monotonic cleanup deadline, capped by the scope's
                retained deadline and the thirty-second termination budget.

        Raises:
            ValueError: The deadline is nonfinite.
            RuntimeError: Ownership, client membership or CUDA termination is
                unconfirmed. Partial termination does not authorize killing.
            OSError: Process, directory or management observation fails.
            psutil.Error: Exact process identity, ancestry or user observation fails.
            TimeoutError: The operation or serialization budget expires.
            subprocess.TimeoutExpired: A management command exceeds its bound.

        Side Effects:
            Terminates contexts on every owned server associated with this
            client. The process remains alive with CUDA's sticky termination
            error; its owner remains responsible for exact-identity exit/reaping.
        """

        if not math.isfinite(deadline):
            raise ValueError("MPS client termination requires a finite deadline")
        deadline = min(deadline, monotonic() + MPS_TERMINATION_TIMEOUT_S)
        if self.cleanup_deadline is not None:
            deadline = min(deadline, self.cleanup_deadline)
        remaining = deadline - monotonic()
        if remaining <= 0 or not self.stop_lock.acquire(timeout=remaining):
            raise TimeoutError("MPS client termination deadline expired before serialization")
        try:
            if self.closed or self.controller_identity is None or not self.controller_identity.is_alive():
                raise RuntimeError("MPS client termination requires the retained live controller")
            if self.directory_identity is None:
                raise RuntimeError("MPS client termination requires the owned directory")
            facts = self.endpoint.directory.lstat()
            if (facts.st_dev, facts.st_ino) != self.directory_identity or not stat.S_ISDIR(facts.st_mode):
                raise RuntimeError("owned MPS directory was replaced; client termination is unconfirmed")
            if not target.is_alive():
                raise RuntimeError("MPS client termination requires the exact live target")
            process = psutil.Process(target.pid)
            namespace = Path(f"/proc/{target.pid}/ns/pid").stat()
            own_namespace = Path("/proc/self/ns/pid").stat()
            if process.uids().real != os.getuid() or (namespace.st_dev, namespace.st_ino) != (
                own_namespace.st_dev,
                own_namespace.st_ino,
            ):
                raise RuntimeError("MPS client termination requires the controller's user and PID namespace")
            matched = False
            for server in self.observe_servers(deadline=deadline):
                clients = self.endpoint.parse_process_ids(
                    self.endpoint.run_control(f"get_client_list {server.pid}", deadline=deadline),
                    allow_empty=True,
                )
                if target.pid not in clients:
                    continue
                if not target.is_alive() or not server.is_alive() or not self.controller_identity.is_alive():
                    raise RuntimeError("MPS termination process identity changed during observation")
                result = self.endpoint.run_control(
                    f"terminate_client {server.pid} {target.pid}",
                    deadline=deadline,
                )
                if result != "0":
                    raise RuntimeError(f"MPS client {target.pid} termination was not CUDA_SUCCESS: {result!r}")
                matched = True
            if not matched:
                raise RuntimeError(f"process {target.pid} is not a client of the owned MPS controller")
        finally:
            self.stop_lock.release()
