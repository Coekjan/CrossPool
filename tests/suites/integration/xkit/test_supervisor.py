from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from multiprocessing.connection import Connection
from pathlib import Path

import psutil
import pytest

import xkit.process
import xkit.supervisor
from xkit.child import PythonChildProcess
from xkit.process import (
    OwnedProcessGroup,
)
from xkit.supervisor import (
    SupervisedTaskScope,
    TaskCompletionKind,
    TaskScopeState,
    prepare_task_supervision,
    unprotected_subreaper_roots,
)
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.process_probe import command

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_task_supervision_preserves_existing_helpers_and_drains_task_children() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import subprocess
import sys
import warnings
from multiprocessing import resource_tracker

from xkit.supervisor import (
    drain_unprotected_subreaper_descendants,
    prepare_task_supervision,
    unprotected_subreaper_roots,
)
from xpool.utils.procs import ProcUniqId

resource_tracker.ensure_running()
task = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
try:
    prepare_task_supervision()
    assert unprotected_subreaper_roots() == (ProcUniqId(task.pid),)
    drain_unprotected_subreaper_descendants()
    assert not unprotected_subreaper_roots()
    warnings.simplefilter("error")
    resource_tracker.ensure_running()
    assert not unprotected_subreaper_roots()
finally:
    task.kill()
    task.wait()
""",
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("timeout_seconds", [None, 10.0])
def test_supervised_scope_drains_with_optional_total_deadline(tmp_path: Path, timeout_seconds: float | None) -> None:
    completion = SupervisedTaskScope.run(
        "optional-deadline",
        command("spawn-detached-child", seconds=0.05),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=tmp_path / "optional-deadline.log",
        timeout_seconds=timeout_seconds,
    )

    assert completion.kind is TaskCompletionKind.EXITED
    assert completion.returncode == 0


def run_nested_supervised_scope(connection: Connection, workdir: str) -> None:
    interrupted = False

    def request_stop(signal_number: int, frame: object) -> None:
        nonlocal interrupted
        del signal_number, frame
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    path = Path(workdir)
    scope = SupervisedTaskScope.start(
        "nested-signal",
        command("sleep", seconds=60),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=path / "nested-signal-task.log",
        timeout_seconds=60,
    )
    connection.send("ready")
    while not interrupted:
        time.sleep(0.01)
    SupervisedTaskScope.terminate_all((scope,))
    if scope.state is not TaskScopeState.DRAINED:
        raise RuntimeError(f"nested task was not drained after SIGINT: {scope.state}")
    scope.close()
    connection.send("drained")


def run_abandoned_supervised_scope(connection: Connection, workdir: str) -> None:
    task_log_path = Path(workdir) / "abandoned-task.log"
    scope = SupervisedTaskScope.start(
        "abandoned-runner",
        command("spawn-child", seconds=60),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=task_log_path,
        timeout_seconds=60,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        contents = task_log_path.read_text(encoding="utf-8").strip()
        if contents:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("nested task child did not report readiness")
    supervisor_pid = scope.supervisor.pid
    if supervisor_pid is None:
        raise RuntimeError("nested task supervisor has no process id")
    supervisor_children = psutil.Process(supervisor_pid).children(recursive=False)
    if len(supervisor_children) != 1:
        raise RuntimeError(f"nested task supervisor owns {len(supervisor_children)} roots")
    connection.send((supervisor_pid, supervisor_children[0].pid, int(contents)))
    while True:
        time.sleep(60)


def test_nested_runner_sigint_does_not_reach_isolated_task_session(tmp_path: Path) -> None:
    prepare_task_supervision()
    runner = PythonChildProcess.start(
        "nested-runner",
        run_nested_supervised_scope,
        str(tmp_path),
        log_path=tmp_path / "nested-runner.log",
        import_paths=(REPO_ROOT,),
    )
    runner_pid = runner.process.pid
    assert runner_pid is not None
    try:
        assert runner.receive(str, timeout_seconds=5) == "ready"
        os.killpg(runner_pid, signal.SIGINT)
        assert runner.receive(str, timeout_seconds=10) == "drained"
        runner.wait(timeout_seconds=5)
    finally:
        if runner.process.is_alive():
            PythonChildProcess.terminate_all((runner,))
        runner.close()


def test_supervisor_drains_task_scope_after_runner_is_killed(tmp_path: Path) -> None:
    prepare_task_supervision()
    runner = PythonChildProcess.start(
        "abandoned-runner",
        run_abandoned_supervised_scope,
        str(tmp_path),
        log_path=tmp_path / "abandoned-runner.log",
        import_paths=(REPO_ROOT,),
    )
    try:
        supervisor_pid, root_pid, child_pid = runner.receive(tuple, timeout_seconds=5)
        task_process_ids = tuple(ProcUniqId(pid) for pid in (supervisor_pid, root_pid, child_pid))

        runner.process.kill()
        runner.process.join(5)
        assert not runner.process.is_alive()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not any(process_id.is_alive() for process_id in task_process_ids) and not unprotected_subreaper_roots():
                break
            time.sleep(0.01)
        else:
            pytest.fail(
                "task scope remained live after abrupt runner death: "
                f"{tuple(process_id.pid for process_id in task_process_ids if process_id.is_alive())}"
            )
    finally:
        if runner.process.is_alive():
            runner.process.kill()
            runner.process.join(5)
        runner.close()


def test_supervised_scope_waits_for_detached_descendant_before_completion(tmp_path: Path) -> None:
    log_path = tmp_path / "detached-child.log"
    scope = SupervisedTaskScope.start(
        "detached-child",
        command("spawn-detached-child", seconds=2.0),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=log_path,
        timeout_seconds=10,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            contents = log_path.read_text(encoding="utf-8").strip()
            if contents:
                break
            time.sleep(0.01)
        else:
            pytest.fail("detached child did not report its process ID")
        child = ProcUniqId(int(contents))
        assert child.is_alive()
        assert scope.poll() is None
        completion = scope.wait()
        assert completion.kind is TaskCompletionKind.EXITED
        assert completion.returncode == 0
        assert not child.is_alive()
    finally:
        SupervisedTaskScope.terminate_all((scope,))
        scope.close()


def test_supervised_scope_adopts_descendant_after_intermediate_exits(tmp_path: Path) -> None:
    log_path = tmp_path / "exiting-intermediate.log"
    scope = SupervisedTaskScope.start(
        "exiting-intermediate",
        command("spawn-exiting-intermediate", seconds=2.0),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=log_path,
        timeout_seconds=10,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            contents = log_path.read_text(encoding="utf-8").strip()
            if contents:
                break
            time.sleep(0.01)
        else:
            pytest.fail("adopted child did not report its process ID")
        child = ProcUniqId(int(contents))
        assert child.is_alive()
        assert scope.poll() is None
        completion = scope.wait()
        assert completion.kind is TaskCompletionKind.EXITED
        assert completion.returncode == 0
        assert not child.is_alive()
    finally:
        SupervisedTaskScope.terminate_all((scope,))
        scope.close()


def test_owned_process_group_leaves_adopted_zombie_reaping_to_scope_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_task_supervision()
    monkeypatch.setattr(xkit.process, "PROCESS_TERMINATE_TIMEOUT_SECONDS", 0.2)
    owner = OwnedProcessGroup(
        name="adopted-zombie",
        process=subprocess.Popen(
            command("spawn-ignoring-child", seconds=60),
            cwd=REPO_ROOT,
            env=dict(os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        ),
    )
    assert owner.process.stdout is not None
    child_pid = int(owner.process.stdout.readline())

    try:
        owner.terminate()

        assert owner.process.returncode == -signal.SIGTERM
        assert psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        with suppress(ChildProcessError, ProcessLookupError):
            os.waitpid(child_pid, 0)
        owner.close()


def test_supervised_scope_owns_timeout_and_cleanup(tmp_path: Path) -> None:
    scope = SupervisedTaskScope.start(
        "timeout",
        command("sleep", seconds=60),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=tmp_path / "timeout.log",
        timeout_seconds=0.1,
    )

    completion = scope.wait()
    scope.close()

    assert completion.kind is TaskCompletionKind.TIMED_OUT


def test_supervisor_reports_local_failure_after_proving_scope_empty(tmp_path: Path) -> None:
    scope = SupervisedTaskScope.start(
        "local-infrastructure-failure",
        command("sleep", seconds=60),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=tmp_path / "local-infrastructure-failure.log",
        timeout_seconds=60,
    )

    scope.connection.send("invalid supervisor command")
    completion = scope.wait()
    scope.close()

    assert completion.kind is TaskCompletionKind.INFRASTRUCTURE_FAILED
    assert completion.returncode is None
    assert completion.diagnostics is not None
    assert "invalid command" in completion.diagnostics


def test_supervised_scope_cancels_multiple_tasks_before_closing(tmp_path: Path) -> None:
    scopes = tuple(
        SupervisedTaskScope.start(
            f"cancel-{index}",
            command("sleep", seconds=60),
            cwd=REPO_ROOT,
            env=dict(os.environ),
            log_path=tmp_path / f"cancel-{index}.log",
            timeout_seconds=60,
        )
        for index in range(2)
    )

    SupervisedTaskScope.terminate_all(scopes)
    for scope in scopes:
        assert scope.state is TaskScopeState.DRAINED
        scope.close()
        assert scope.state is TaskScopeState.CLOSED


def test_supervisor_and_root_use_distinct_session_and_process_groups(tmp_path: Path) -> None:
    log_path = tmp_path / "topology.log"
    scope = SupervisedTaskScope.start(
        "topology",
        command("report-topology", seconds=0.25),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=log_path,
        timeout_seconds=10,
    )
    supervisor_pid = scope.supervisor.pid
    assert supervisor_pid is not None
    supervisor_sid = os.getsid(supervisor_pid)
    supervisor_pgid = os.getpgid(supervisor_pid)

    completion = scope.wait()
    topology = tuple(int(value) for value in log_path.read_text(encoding="utf-8").strip().split())
    scope.close()

    root_pid, root_parent, root_sid, root_pgid = topology
    assert root_parent == supervisor_pid
    assert supervisor_sid == supervisor_pid
    assert supervisor_pgid == supervisor_pid
    assert root_sid == supervisor_pid
    assert root_pgid == root_pid
    assert completion.kind is TaskCompletionKind.EXITED


def test_supervised_scope_escalates_from_term_to_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("xkit.supervisor.PROCESS_TERMINATE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr("xkit.supervisor.PROCESS_KILL_TIMEOUT_SECONDS", 5.0)
    term_path = tmp_path / "term-observed"
    scope = SupervisedTaskScope.start(
        "term-to-kill",
        command("record-term-ignore", seconds=60, output=term_path),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=tmp_path / "term-to-kill.log",
        timeout_seconds=60,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if term_path.exists() and term_path.read_text(encoding="utf-8") == "ready\n":
            break
        time.sleep(0.01)
    else:
        pytest.fail("TERM-recording task did not become ready")

    SupervisedTaskScope.terminate_all((scope,))
    assert term_path.read_text(encoding="utf-8") == "ready\nSIGTERM\n"
    assert scope.state is TaskScopeState.DRAINED
    scope.close()


def test_runner_subreaper_drains_descendants_after_supervisor_failure(tmp_path: Path) -> None:
    prepare_task_supervision()
    scope = SupervisedTaskScope.start(
        "failed-supervisor",
        command("sleep", seconds=60),
        cwd=REPO_ROOT,
        env=dict(os.environ),
        log_path=tmp_path / "failed-supervisor.log",
        timeout_seconds=60,
    )
    scope.supervisor.kill()
    scope.supervisor.join(5)

    SupervisedTaskScope.terminate_all((scope,))

    assert unprotected_subreaper_roots() == ()
    assert scope.state is TaskScopeState.DRAINED
    scope.close()
