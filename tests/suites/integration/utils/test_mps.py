"""CPU subprocess evidence for owned MPS lifecycle operations."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import xpool.service.daemon.app
import xpool.service.daemon.control
from xpool.service.daemon.control import ControlPlane
from xpool.utils import mps
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import install_test_config, reset_global_config, synthetic_config


@pytest.fixture
def scoped_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[mps.MpsScope, None, None]:
    # This CPU stand-in exercises real subprocess/session ownership, not CUDA.
    executable = tmp_path / "control"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import os, signal, subprocess, sys, time
from pathlib import Path
pipe = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"])
if "-f" in sys.argv:
    server = None
    def terminate(signum, frame):
        raise SystemExit()
    signal.signal(signal.SIGTERM, terminate)
    try:
        while not (pipe / "quit").exists():
            if server is None and (pipe / "create-server").exists():
                server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                (pipe / "server").write_text(str(server.pid))
            time.sleep(0.01)
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=5)
else:
    command = sys.stdin.read().strip()
    if command == "get_default_active_thread_percentage":
        if (pipe / "query-delay").exists():
            time.sleep(float((pipe / "query-delay").read_text()))
        print("100.0")
    elif command == "get_server_list":
        if (pipe / "server").exists():
            print((pipe / "server").read_text())
    elif command.startswith("get_client_list "):
        if (pipe / "clients").exists():
            print((pipe / "clients").read_text())
    elif command.startswith("terminate_client "):
        (pipe / "termination-command").write_text(command)
        print((pipe / "termination-reply").read_text())
    elif command == "quit":
        (pipe / "quit").touch()
    else:
        raise SystemExit(1)
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    monkeypatch.setattr(mps, "MPS_SCOPE_DIRECTORY", tmp_path / "mps")
    monkeypatch.setattr(mps, "MPS_PROBE_LOCK_DIRECTORY", tmp_path / "probe-locks")
    monkeypatch.setattr(mps.shutil, "which", lambda command: str(executable))
    uuids = ("GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002")
    scope = mps.MpsScope(mps.MpsEndpoint(uuids[:1]))
    yield scope
    # Only CPU stand-ins are eligible for this fixture's emergency teardown.
    if scope.controller is not None and scope.controller.poll() is None:
        scope.controller.terminate()
        scope.controller.wait(timeout=5)
    if scope.controller_log is not None:
        scope.controller_log.close()


def test_management_queries_allow_startup_latency_and_respect_owner_deadline(
    scoped_controller: mps.MpsScope,
) -> None:
    scope = scoped_controller
    scope.start()
    delay = scope.endpoint.pipe_directory / "query-delay"
    delay.write_text("1.2", encoding="utf-8")
    assert scope.probe().online is True
    with pytest.raises(subprocess.TimeoutExpired):
        scope.endpoint.run_control(
            "get_default_active_thread_percentage",
            deadline=time.monotonic() + 0.2,
        )
    delay.unlink()
    scope.stop(deadline=time.monotonic() + mps.MPS_CLEANUP_TIMEOUT_S)
    assert scope.closed


def test_scope_stops_only_after_clients_retire(
    scoped_controller: mps.MpsScope,
) -> None:
    scope = scoped_controller
    scope.start()
    assert scope.controller is not None
    assert os.getpgid(scope.controller.pid) == scope.controller.pid
    assert scope.observe_servers() == ()
    assert scope.probe().online is True
    (scope.endpoint.pipe_directory / "create-server").touch()
    deadline = time.monotonic() + 5
    while not scope.observe_servers(deadline=deadline):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    clients = scope.endpoint.pipe_directory / "clients"
    clients.write_text(str(os.getpid()), encoding="utf-8")
    cleanup_deadline = time.monotonic() + mps.MPS_CLEANUP_TIMEOUT_S
    with pytest.raises(RuntimeError, match="still has clients"):
        scope.stop(deadline=cleanup_deadline)
    assert scope.controller.poll() is None
    assert not scope.closed
    clients.unlink()
    scope.stop(deadline=cleanup_deadline + 100)
    assert scope.cleanup_deadline == cleanup_deadline
    assert scope.closed
    assert scope.controller.returncode == 0
    assert not scope.endpoint.directory.exists()
    scope.stop(deadline=cleanup_deadline + 200)


@pytest.mark.parametrize("reply", ["0", "700", "", 'Unknown command "terminate_client"'])
def test_scope_requires_device_success_without_killing_client(
    reply: str,
    scoped_controller: mps.MpsScope,
) -> None:
    scope = scoped_controller
    scope.start()
    (scope.endpoint.pipe_directory / "create-server").touch()
    deadline = time.monotonic() + 5
    while not (servers := scope.observe_servers(deadline=deadline)):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    target = ProcUniqId.current()
    clients = scope.endpoint.pipe_directory / "clients"
    clients.write_text(str(target.pid), encoding="utf-8")
    (scope.endpoint.pipe_directory / "termination-reply").write_text(reply, encoding="utf-8")

    if reply == "0":
        scope.terminate_client(target, deadline=deadline)
    else:
        with pytest.raises(RuntimeError, match="not CUDA_SUCCESS"):
            scope.terminate_client(target, deadline=deadline)
    assert (scope.endpoint.pipe_directory / "termination-command").read_text() == (
        f"terminate_client {servers[0].pid} {target.pid}"
    )
    assert target.is_alive()
    assert scope.controller is not None and scope.controller.poll() is None
    assert not scope.closed
    clients.unlink()
    scope.stop(deadline=deadline)


def test_scope_does_not_terminate_an_unconnected_process(
    scoped_controller: mps.MpsScope,
) -> None:
    scope = scoped_controller
    scope.start()
    deadline = time.monotonic() + 5
    with pytest.raises(RuntimeError, match="not a client"):
        scope.terminate_client(ProcUniqId.current(), deadline=deadline)
    assert not (scope.endpoint.pipe_directory / "termination-command").exists()
    assert not scope.closed
    scope.stop(deadline=deadline)


def test_scope_preserves_an_existing_endpoint(scoped_controller: mps.MpsScope) -> None:
    scope = scoped_controller
    mps.MPS_SCOPE_DIRECTORY.mkdir(mode=0o700)
    scope.endpoint.directory.parent.mkdir(mode=0o700)
    scope.endpoint.directory.mkdir(mode=0o700)
    diagnostic = scope.endpoint.directory / "owner.log"
    diagnostic.write_text("retained owner", encoding="utf-8")

    with pytest.raises(FileExistsError):
        scope.start()
    assert scope.controller is None
    scope.stop(deadline=time.monotonic() + mps.MPS_CLEANUP_TIMEOUT_S)

    assert diagnostic.read_text(encoding="utf-8") == "retained owner"


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("injected startup observation failure"), subprocess.TimeoutExpired("MPS startup probe", 1.0)],
)
def test_partial_scope_startup_retains_controller_for_owner_rollback(
    failure: RuntimeError | subprocess.TimeoutExpired,
    scoped_controller: mps.MpsScope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = scoped_controller

    def unavailable_probe(self: mps.MpsEndpoint, **kwargs: object) -> mps.MpsProbeResult:
        raise failure

    monkeypatch.setattr(mps.MpsEndpoint, "probe", unavailable_probe)
    with pytest.raises(type(failure)) as caught:
        scope.start()
    assert caught.value is failure
    assert scope.controller is not None
    assert scope.controller.poll() is None
    assert scope.endpoint.directory.exists()
    assert not scope.closed
    scope.stop(deadline=time.monotonic() + mps.MPS_CLEANUP_TIMEOUT_S)
    assert scope.closed
    assert scope.controller.returncode == 0


def test_expired_scope_startup_creates_no_resources(
    scoped_controller: mps.MpsScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = scoped_controller
    monkeypatch.setattr(mps, "MPS_STARTUP_TIMEOUT_S", 0.0)
    with pytest.raises(TimeoutError, match="before controller creation"):
        scope.start()
    assert scope.controller is None
    assert not scope.endpoint.directory.exists()
    assert not scope.closed
    scope.stop(deadline=time.monotonic() + mps.MPS_CLEANUP_TIMEOUT_S)
    assert scope.closed


@pytest.mark.parametrize("failed", [False, True], ids=["normal", "partial-startup"])
@pytest.mark.usefixtures(reset_global_config.__name__)
def test_daemon_lifespan_owns_controller_and_rolls_back_partial_startup(
    failed: bool,
    scoped_controller: mps.MpsScope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uuids = (*scoped_controller.endpoint.device_uuids, "GPU-00000000-0000-0000-0000-000000000002")
    install_test_config(synthetic_config())
    monkeypatch.setattr(xpool.service.daemon.app.bootstrap, "init", lambda selected, role: None)
    monkeypatch.setattr(xpool.service.daemon.control, "normalize_environment", lambda: None)
    monkeypatch.setattr(xpool.service.daemon.control, "visible_uuids", lambda: uuids)
    original_start = mps.MpsScope.start
    failure = RuntimeError("injected controller startup failure")

    def start(scope: mps.MpsScope) -> None:
        original_start(scope)
        if failed:
            raise failure

    monkeypatch.setattr(mps.MpsScope, "start", start)
    app = xpool.service.daemon.app.create_daemon()
    control: ControlPlane = app.state.control_plane

    async def run() -> None:
        async with app.router.lifespan_context(app):
            assert control.device_uuids == uuids
            assert control.mps_scope is not None
            assert control.mps_scope.endpoint.device_uuids == uuids[:1]
            assert control.mps_scope.controller is not None
            assert control.mps_scope.controller.poll() is None

    try:
        if failed:
            with pytest.raises(RuntimeError) as caught:
                asyncio.run(run())
            assert caught.value is failure
            assert app.state.daemon_failure.exception is failure
        else:
            asyncio.run(run())
            assert not app.state.daemon_failure.failed
        assert control.closed
        assert control.mps_scope is not None
        assert control.mps_scope.closed
        assert control.mps_scope.controller is not None
        assert control.mps_scope.controller.returncode == 0
        assert not control.mps_scope.endpoint.directory.exists()
    finally:
        # This test owns only a CPU stand-in, including failures in its assertions.
        scope = control.mps_scope
        if scope is not None and scope.controller is not None and scope.controller.poll() is None:
            scope.controller.terminate()
            scope.controller.wait(timeout=5)
        control.close()


def test_retirement_during_scope_startup_stops_controller_creation(
    scoped_controller: mps.MpsScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = scoped_controller

    def resolve(command: str) -> str:
        scope.cleanup_deadline = time.monotonic() + mps.MPS_CLEANUP_TIMEOUT_S
        return "/unused/control"

    monkeypatch.setattr(mps.shutil, "which", resolve)
    with pytest.raises(InterruptedError, match="retiring"):
        scope.start()
    assert scope.controller is None
    assert not scope.endpoint.directory.exists()
    assert scope.cleanup_deadline is not None
    scope.stop(deadline=scope.cleanup_deadline)
    assert scope.closed


def test_probe_serializes_concurrent_queries_for_one_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def run_probe(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal active, max_active
        del args, kwargs
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with state_lock:
            active -= 1
        return SimpleNamespace(returncode=0, stdout="100.0\n", stderr="")

    endpoint = mps.MpsEndpoint(("GPU-00000000-0000-0000-0000-000000000001",))
    monkeypatch.setattr(mps, "MPS_PROBE_LOCK_DIRECTORY", tmp_path / "locks")
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps.subprocess, "run", run_probe)

    with ThreadPoolExecutor(max_workers=7) as executor:
        results = tuple(executor.map(lambda index: endpoint.probe(), range(14)))

    assert any(result.online is True for result in results)
    assert all(result.online is True or result.online is None for result in results)
    assert max_active == 1


def test_busy_probe_is_unavailable_without_executing_a_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = mps.MpsEndpoint(("GPU-00000000-0000-0000-0000-000000000001",))
    monkeypatch.setattr(mps, "MPS_PROBE_LOCK_DIRECTORY", tmp_path / "locks")
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")

    started = threading.Event()
    release = threading.Event()

    def delayed_command(*args: object, **kwargs: object) -> SimpleNamespace:
        started.set()
        assert release.wait(5)
        return SimpleNamespace(returncode=0, stdout="100.0\n", stderr="")

    monkeypatch.setattr(mps.subprocess, "run", delayed_command)
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(endpoint.probe)
        try:
            assert started.wait(5)
            result = endpoint.probe()
        finally:
            release.set()
        assert running.result().online is True
    assert result.online is None
    assert result.active_thread_percentage is None
