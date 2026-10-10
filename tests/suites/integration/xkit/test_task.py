from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from xkit.supervisor import SupervisedTaskScope, TaskCompletionKind, TaskScopeState, TaskSupervisionFailure
from xpool.utils.procs import ProcUniqId


@pytest.mark.parametrize("mode", ["passed", "failed", "cancelled", "owner-expired"])
def test_protected_invocation_keeps_owner_until_cleanup(tmp_path: Path, mode: str) -> None:
    # CPU stand-ins expose ordering and lifetime without creating device state.
    program = '''
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from xkit.task import TaskCancelled, TaskRoot

directory = Path(sys.argv[1])
mode = sys.argv[2]
root = TaskRoot.from_environment()
assert root is not None
verdict = 0
try:
    first = root.register_scope()
    root.activate()
    first.complete()
    (directory / "between-scopes").touch()
    while not (directory / "continue").exists():
        time.sleep(0.01)

    second = root.register_scope()
    root.activate()
    child_program = """
import signal
import sys
import time
from pathlib import Path
directory = Path(sys.argv[1])
def unordered(signum, frame):
    (directory / "unordered-signal").touch()
    raise SystemExit(99)
signal.signal(signal.SIGTERM, unordered)
signal.signal(signal.SIGINT, unordered)
print("ready", flush=True)
assert input() == "retire"
(directory / "client-retiring").touch()
while not (directory / "allow-cleanup").exists():
    time.sleep(0.01)
(directory / "client-cleaned").touch()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_program, str(directory)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert child.stdout.readline().strip() == "ready"
    (directory / "ready").write_text(str(child.pid))
    try:
        while not (directory / "finish").exists():
            time.sleep(0.01)
        verdict = 1 if mode in ("failed", "owner-expired") else 0
    except TaskCancelled:
        verdict = 130
    finally:
        child.stdin.write("retire\\n")
        child.stdin.flush()
        if mode == "owner-expired":
            deadline = time.monotonic() - 1
            root.request_retirement(deadline)
            root.request_retirement(deadline + 30)
            try:
                root.register_scope()
            except TaskCancelled:
                pass
            else:
                raise AssertionError("retirement must prohibit new resources")
        (directory / "deadline").write_text(str(root.cleanup_deadline))
        assert child.wait() == 0
        child.stdin.close()
        child.stdout.close()
        second.complete()
finally:
    root.finish()
raise SystemExit(verdict)
'''
    scope = SupervisedTaskScope.start(
        mode,
        [sys.executable, "-c", program, str(tmp_path), mode],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "task.log",
        timeout_seconds=30,
    )
    cancellation: threading.Thread | None = None
    errors: list[BaseException] = []

    def cancel_scope() -> None:
        try:
            SupervisedTaskScope.terminate_all((scope,))
        except BaseException as error:
            errors.append(error)

    try:
        for marker in ("between-scopes", "ready"):
            observation_deadline = time.monotonic() + 10
            while time.monotonic() < observation_deadline:
                assert scope.poll() is None, (tmp_path / "task.log").read_text(encoding="utf-8")
                if (tmp_path / marker).exists():
                    break
                time.sleep(0.01)
            else:
                pytest.fail((tmp_path / "task.log").read_text(encoding="utf-8"))
            assert scope.is_protected
            if marker == "between-scopes":
                (tmp_path / "continue").touch()
        child_identity = ProcUniqId(int((tmp_path / "ready").read_text(encoding="utf-8")))
        assert scope.root is not None

        if mode in ("passed", "failed"):
            (tmp_path / "allow-cleanup").touch()
            (tmp_path / "finish").touch()
            completion = scope.wait()
            assert completion.kind is TaskCompletionKind.EXITED
            assert completion.returncode == (1 if mode == "failed" else 0)
        else:
            if mode == "owner-expired":
                (tmp_path / "finish").touch()
                observation_deadline = time.monotonic() + 10
                while not (tmp_path / "deadline").exists():
                    assert time.monotonic() < observation_deadline
                    time.sleep(0.01)
                deadline = float((tmp_path / "deadline").read_text(encoding="utf-8"))
            else:
                deadline = time.monotonic() + 0.3
                scope.request_retirement(deadline)
                scope.request_retirement(deadline + 30)
                assert scope.cleanup_deadline == deadline
                observation_deadline = time.monotonic() + 10
                while time.monotonic() < observation_deadline:
                    if (tmp_path / "deadline").exists() and time.monotonic() > deadline:
                        break
                    time.sleep(0.01)
            with pytest.raises(TaskSupervisionFailure, match="cleanup expired"):
                scope.poll()
            assert scope.cleanup_deadline == deadline
            # This thread is now the sole root-channel reader. The main thread
            # observes markers, never concurrent scope.poll().
            cancellation = threading.Thread(target=cancel_scope, daemon=True)
            cancellation.start()
            observation_deadline = time.monotonic() + 10
            while time.monotonic() < observation_deadline:
                if (
                    (tmp_path / "client-retiring").exists()
                    and (tmp_path / "deadline").exists()
                    and time.monotonic() > deadline
                ):
                    break
                time.sleep(0.01)
            assert (tmp_path / "client-retiring").exists()
            assert float((tmp_path / "deadline").read_text(encoding="utf-8")) == deadline
            assert cancellation.is_alive()
            assert scope.is_protected
            assert scope.root.is_alive()
            assert child_identity.is_alive()
            scope.root.send_signal(signal.SIGTERM)
            scope.root.send_signal(signal.SIGINT)
            (tmp_path / "allow-cleanup").touch()
            cancellation.join(10)
            assert not cancellation.is_alive()
            assert not errors
            assert scope.state is TaskScopeState.DRAINED
        assert not scope.is_protected
        assert (tmp_path / "client-cleaned").exists()
        assert not (tmp_path / "unordered-signal").exists()
        assert not scope.root.is_alive()
        assert not child_identity.is_alive()
        scope.close()
    finally:
        (tmp_path / "continue").touch()
        (tmp_path / "finish").touch()
        (tmp_path / "allow-cleanup").touch()
        if cancellation is not None:
            cancellation.join(10)
        elif scope.state is not TaskScopeState.CLOSED:
            SupervisedTaskScope.terminate_all((scope,))
        if scope.state in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED):
            scope.close()


def test_cancellation_services_other_roots_while_one_owner_is_waiting(tmp_path: Path) -> None:
    program = """
import sys
import time
from pathlib import Path
from xkit.task import TaskCancelled, TaskRoot
directory = Path(sys.argv[1])
name = sys.argv[2]
root = TaskRoot.from_environment()
assert root is not None
scope = root.register_scope()
root.activate()
(directory / (name + "-ready")).touch()
try:
    while True:
        time.sleep(0.01)
except TaskCancelled:
    if name == "first":
        while not (directory / "second-closed").exists():
            time.sleep(0.01)
    scope.complete()
finally:
    root.finish()
(directory / (name + "-closed")).touch()
"""
    scopes: list[SupervisedTaskScope] = []
    cancellation: threading.Thread | None = None
    errors: list[BaseException] = []

    def cancel_scopes() -> None:
        try:
            SupervisedTaskScope.terminate_all(scopes)
        except BaseException as error:
            errors.append(error)

    try:
        for name in ("first", "second"):
            scopes.append(
                SupervisedTaskScope.start(
                    name,
                    [sys.executable, "-c", program, str(tmp_path), name],
                    cwd=tmp_path,
                    env=dict(os.environ),
                    log_path=tmp_path / f"{name}.log",
                    timeout_seconds=30,
                )
            )
        observation_deadline = time.monotonic() + 10
        while time.monotonic() < observation_deadline:
            for scope in scopes:
                assert scope.poll() is None
            if all((tmp_path / f"{scope.name}-ready").exists() for scope in scopes):
                break
            time.sleep(0.01)
        assert all(scope.is_protected for scope in scopes)
        cancellation = threading.Thread(target=cancel_scopes, daemon=True)
        cancellation.start()
        cancellation.join(10)
        assert not cancellation.is_alive()
        assert not errors
        assert all(scope.state is TaskScopeState.DRAINED for scope in scopes)
        assert all((tmp_path / f"{scope.name}-closed").exists() for scope in scopes)
    finally:
        # Release only the CPU stand-in's ordering barrier on assertion failure.
        (tmp_path / "second-closed").touch()
        if cancellation is None:
            SupervisedTaskScope.terminate_all(scopes)
        else:
            cancellation.join(10)
        for scope in scopes:
            if scope.state in (TaskScopeState.DRAINED, TaskScopeState.COMPLETED):
                scope.close()


@pytest.mark.parametrize("nested", [False, True])
def test_startup_cancellation_retains_created_child_through_rollback(tmp_path: Path, nested: bool) -> None:
    # Interrupt after real process creation, before the factory publishes it.
    # The CPU daemon exits with the same controlled status as an ordered owner.
    program = '''
import os
import signal
import sys
import time
from pathlib import Path
from types import MappingProxyType
import pytest
from xkit.network import TcpEndpointReservation, TcpPortSpace
from xkit.process import OwnedProcessGroup
from xkit.serving.cluster import XpoolCluster, XpoolClusterLaunch
from xkit.task import TaskCancelled, TaskRoot, TaskRootUpdate, TaskResourceEvent
from xpool.config import XpoolConfig

directory = Path(sys.argv[1])
nested = sys.argv[2] == "True"
root = TaskRoot.from_environment()
assert root is not None
config = XpoolConfig.from_mapping({
    "vendor": {"model_base_uri": str(directory)},
    "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
    "atn": {"devices": [0]}, "ffn": {"devices": [1]},
    "models": [{"id": "test/model"}],
}, env={})
launch = XpoolClusterLaunch(config, directory / "config.toml", MappingProxyType({}), directory)
endpoint = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
original = OwnedProcessGroup.spawn_logged.__func__
child_program = """
import signal
import sys
import time
from pathlib import Path
directory = Path(sys.argv[1])
def retire(signum, frame):
    (directory / "child-retiring").touch()
    while not (directory / "allow-cleanup").exists():
        time.sleep(0.01)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, retire)
(directory / "child-ready").touch()
while True:
    time.sleep(0.01)
"""

def create_child(cls, name, command, **kwargs):
    child = original(cls, name, [sys.executable, "-c", child_program, str(directory)], **kwargs)
    (directory / "child-pid").write_text(str(child.process.pid))
    deadline = time.monotonic() + 10
    while not (directory / "child-ready").exists():
        assert child.process.poll() is None
        assert time.monotonic() < deadline
        time.sleep(0.01)
    os.kill(os.getpid(), signal.SIGTERM)
    assert root.sealed
    (directory / "deadline").write_text(str(root.cleanup_deadline))
    return child

verdict = 0
outer_scope = None
events = []
send = root.connection.send
def send_control(message):
    if isinstance(message, TaskRootUpdate):
        events.append(message.event)
    send(message)
root.connection.send = send_control
try:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(OwnedProcessGroup, "spawn_logged", classmethod(create_child))
        try:
            if nested:
                outer_scope = root.register_scope()
                root.activate()
            cluster = XpoolCluster(launch)
            cluster.start(endpoint, enclosing_scope=outer_scope)
        except TaskCancelled:
            assert cluster.closed
            if outer_scope is not None:
                outer_scope.complete()
            verdict = 143
finally:
    endpoint.close()
    root.finish()
assert events == [TaskResourceEvent.ACTIVE, TaskResourceEvent.CLEANED]
raise SystemExit(verdict)
'''
    scope = SupervisedTaskScope.start(
        "startup-cancellation",
        [sys.executable, "-c", program, str(tmp_path), str(nested)],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "task.log",
        timeout_seconds=30,
    )
    try:
        observation_deadline = time.monotonic() + 15
        while not (tmp_path / "child-retiring").exists():
            assert scope.poll() is None, (tmp_path / "task.log").read_text(encoding="utf-8")
            assert time.monotonic() < observation_deadline, (tmp_path / "task.log").read_text(encoding="utf-8")
            time.sleep(0.01)
        child = ProcUniqId(int((tmp_path / "child-pid").read_text(encoding="utf-8")))
        assert scope.is_protected
        assert scope.root is not None and scope.root.is_alive()
        assert child.is_alive()
        deadline = float((tmp_path / "deadline").read_text(encoding="utf-8"))
        scope.root.send_signal(signal.SIGTERM)
        scope.root.send_signal(signal.SIGINT)
        assert float((tmp_path / "deadline").read_text(encoding="utf-8")) == deadline
        (tmp_path / "allow-cleanup").touch()
        completion = scope.wait()
        assert completion.kind is TaskCompletionKind.EXITED
        assert completion.returncode == 143
        assert not scope.is_protected
        assert not child.is_alive()
        scope.close()
    finally:
        (tmp_path / "allow-cleanup").touch()
        if scope.state is not TaskScopeState.CLOSED:
            SupervisedTaskScope.terminate_all((scope,))
        if scope.state in (TaskScopeState.DRAINED, TaskScopeState.COMPLETED):
            scope.close()
