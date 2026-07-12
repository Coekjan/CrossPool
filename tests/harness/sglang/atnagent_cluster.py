from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

import httpx

from xpool.config import XpoolConfig
from xpool.service.client import XpoolClient, XpoolClientError, XpoolDaemonError
from xpool.service.wire import ReadinessScope, ReadinessStatus

STARTUP_TIMEOUT_S = 30.0
ATNAGENT_SHUTDOWN_TIMEOUT_S = 75.0
DAEMON_SHUTDOWN_TIMEOUT_S = 15.0
POLL_INTERVAL_S = 0.1
DAEMON_START_ATTEMPTS = 3
LOG_TAIL_CHARS = 12000
CLI_BOOTSTRAP = "from xpool.cli import main; raise SystemExit(main())"


@dataclass(slots=True)
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    log_path: Path
    log_file: TextIO

    def stop(self, *, timeout_s: float) -> str | None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
                return f"{self.name} did not stop within {timeout_s:.0f}s and was killed"
        if self.process.returncode != 0 and not (self.name == "daemon" and self.process.returncode == -signal.SIGTERM):
            return f"{self.name} exited with code {self.process.returncode}"
        return None

    def tail(self) -> str:
        self.log_file.flush()
        try:
            return self.log_path.read_text(encoding="utf-8")[-LOG_TAIL_CHARS:]
        except OSError as exc:
            return f"<failed to read {self.log_path}: {exc}>"


class AtnAgentLoopbackCluster:
    def __init__(
        self,
        *,
        config: XpoolConfig,
        config_path: Path,
        source_config: XpoolConfig,
        model_id: str,
        model_path: Path,
        graph_observer_outdir: Path,
        workdir: Path,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.source_config = source_config
        self.model_id = model_id
        self.model_path = model_path
        self.graph_observer_outdir = graph_observer_outdir
        self.workdir = workdir
        self.processes: list[ManagedProcess] = []
        self.client: XpoolClient | None = None

    @classmethod
    def create(
        cls,
        *,
        source_config: XpoolConfig,
        model_id: str,
        model_path: Path,
        graph_observer_outdir: Path,
        workdir: Path,
    ) -> AtnAgentLoopbackCluster:
        workdir.mkdir(parents=True, exist_ok=True)
        config_path = workdir / "xpool.atn-atnagent-loopback.toml"
        daemon_port = cls.allocate_daemon_port()
        cls.write_config(
            config_path,
            source_config=source_config,
            model_id=model_id,
            model_path=model_path,
            daemon_port=daemon_port,
        )
        debug_env = {
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(graph_observer_outdir),
            "XPOOL_DEBUG_LOOPBACK_ENABLE": "1",
            "XPOOL_DEBUG_LOOPBACK_SITE": "atnagent",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(graph_observer_outdir),
        }
        config = XpoolConfig.from_file(config_path, env=debug_env)
        return cls(
            config=config,
            config_path=config_path,
            source_config=source_config,
            model_id=model_id,
            model_path=model_path,
            graph_observer_outdir=graph_observer_outdir,
            workdir=workdir,
        )

    @staticmethod
    def allocate_daemon_port() -> int:
        """Return an ephemeral loopback port candidate."""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    @staticmethod
    def write_config(
        config_path: Path,
        *,
        source_config: XpoolConfig,
        model_id: str,
        model_path: Path,
        daemon_port: int,
    ) -> None:
        """Write one retryable ATN-atnagent-loopback config generation."""

        config_path.write_text(
            "\n".join(
                (
                    "[daemon]",
                    'host = "127.0.0.1"',
                    f"port = {daemon_port}",
                    "",
                    "[scheduler]",
                    f"atn_concurrency = {source_config.scheduler.atn_concurrency}",
                    f"ffn_concurrency = {source_config.scheduler.ffn_concurrency}",
                    "",
                    "[devices]",
                    f"atn_cuda_devices = {json.dumps(source_config.devices.atn_cuda_devices)}",
                    f"ffn_cuda_devices = {json.dumps(source_config.devices.ffn_cuda_devices)}",
                    "",
                    "[[models]]",
                    f"id = {json.dumps(model_id)}",
                    f"path = {json.dumps(str(model_path.resolve()))}",
                    "",
                )
            ),
            encoding="utf-8",
        )

    def debug_env(self) -> dict[str, str]:
        """Return debug environment overrides shared by cluster processes."""

        return {
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(self.graph_observer_outdir),
            "XPOOL_DEBUG_LOOPBACK_ENABLE": "1",
            "XPOOL_DEBUG_LOOPBACK_SITE": "atnagent",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(self.graph_observer_outdir),
        }

    def select_daemon_port(self) -> None:
        """Allocate a new port and rebuild the effective harness config."""

        self.write_config(
            self.config_path,
            source_config=self.source_config,
            model_id=self.model_id,
            model_path=self.model_path,
            daemon_port=self.allocate_daemon_port(),
        )
        self.config = XpoolConfig.from_file(self.config_path, env=self.debug_env())

    def __enter__(self) -> AtnAgentLoopbackCluster:
        env = dict(os.environ)
        env.pop("XPOOL_DEBUG_LOOPBACK_ENABLE", None)
        env.pop("XPOOL_DEBUG_LOOPBACK_SITE", None)
        env["XPOOL_CONFIG"] = str(self.config_path)
        env.update(self.debug_env())
        try:
            startup_failures: list[str] = []
            for attempt in range(DAEMON_START_ATTEMPTS):
                if attempt > 0:
                    self.select_daemon_port()
                self.spawn("daemon", ["daemon", "serve"], env=env)
                try:
                    self.wait_for_daemon()
                except Exception as exc:
                    startup_failures.append(f"attempt {attempt + 1}: {exc}\n{self.diagnostics()}")
                    self.discard_failed_daemon()
                    if attempt + 1 == DAEMON_START_ATTEMPTS:
                        raise RuntimeError("daemon startup retries exhausted\n" + "\n".join(startup_failures)) from exc
                else:
                    break
            self.client = XpoolClient(
                http_client=httpx.Client(
                    base_url=f"http://{self.config.daemon.host}:{self.config.daemon.port}",
                    timeout=5.0,
                )
            )
            for cuda_device in self.config.devices.atn_cuda_devices:
                self.spawn(
                    f"atnagent-{cuda_device}",
                    ["atnagent", "--cuda-device", str(cuda_device)],
                    env=env,
                )
            self.wait_for_atnagents()
        except Exception as exc:
            diagnostics = self.diagnostics()
            self.close_processes(force=True)
            self.close_logs()
            raise RuntimeError(f"failed to start ATN-atnagent-loopback cluster: {exc}\n{diagnostics}") from exc
        return self

    def discard_failed_daemon(self) -> None:
        """Stop and forget the most recent failed daemon attempt."""

        managed = self.processes.pop()
        if managed.name != "daemon":
            raise RuntimeError(f"expected failed daemon process, got {managed.name}")
        if managed.process.poll() is None:
            managed.process.kill()
            managed.process.wait(timeout=10)
        managed.log_file.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        failures = self.close_processes(force=False)
        diagnostics = self.diagnostics() if failures else ""
        self.close_logs()
        if failures:
            message = "; ".join(failures)
            if exc is not None:
                raise RuntimeError(f"{exc}\nATN-atnagent-loopback cleanup failed: {message}\n{diagnostics}") from exc
            raise RuntimeError(f"ATN-atnagent-loopback cleanup failed: {message}\n{diagnostics}")
        return False

    def diagnostics(self) -> str:
        sections = [f"--- {managed.name}: {managed.log_path} ---\n{managed.tail()}" for managed in self.processes]
        return "\n".join(sections)

    def spawn(self, name: str, arguments: list[str], *, env: dict[str, str]) -> None:
        log_path = self.workdir / f"{name}.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-c", CLI_BOOTSTRAP, *arguments],
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        self.processes.append(ManagedProcess(name=name, process=process, log_path=log_path, log_file=log_file))

    def wait_for_daemon(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            self.raise_for_exited_process()
            try:
                response = httpx.get(
                    f"http://{self.config.daemon.host}:{self.config.daemon.port}/health",
                    timeout=POLL_INTERVAL_S,
                )
                if response.is_success:
                    return
            except httpx.HTTPError:
                time.sleep(POLL_INTERVAL_S)
        raise RuntimeError("timed out waiting for daemon health")

    def wait_for_atnagents(self) -> None:
        if self.client is None:
            raise RuntimeError("ATN-atnagent-loopback daemon client is not initialized")
        expected_devices = set(self.config.devices.atn_cuda_devices)
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            self.raise_for_exited_process()
            try:
                readiness = self.client.readiness((ReadinessScope.ATN,))
            except (XpoolClientError, XpoolDaemonError):
                time.sleep(POLL_INTERVAL_S)
                continue
            online_devices = {
                entry.cuda_device for entry in readiness.atnagents if entry.status is ReadinessStatus.ONLINE
            }
            if online_devices == expected_devices:
                return
            time.sleep(POLL_INTERVAL_S)
        raise RuntimeError("timed out waiting for ATN atnagent registration")

    def raise_for_exited_process(self) -> None:
        for managed in self.processes:
            return_code = managed.process.poll()
            if return_code is not None:
                raise RuntimeError(f"{managed.name} exited during startup with code {return_code}")

    def close_processes(self, *, force: bool) -> list[str]:
        failures: list[str] = []
        if self.client is not None:
            self.client.close()
            self.client = None
        atnagents = [managed for managed in self.processes if managed.name.startswith("atnagent-")]
        daemons = [managed for managed in self.processes if managed.name == "daemon"]
        for managed in (*reversed(atnagents), *reversed(daemons)):
            if force and managed.process.poll() is None:
                managed.process.kill()
                managed.process.wait(timeout=10)
            else:
                timeout_s = (
                    ATNAGENT_SHUTDOWN_TIMEOUT_S if managed.name.startswith("atnagent-") else DAEMON_SHUTDOWN_TIMEOUT_S
                )
                if failure := managed.stop(timeout_s=timeout_s):
                    failures.append(failure)
        return failures

    def close_logs(self) -> None:
        for managed in self.processes:
            managed.log_file.close()
