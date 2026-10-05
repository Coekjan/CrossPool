from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from xkit.supervisor import SupervisedTaskScope, TaskCompletionKind, TaskScopeState


@pytest.mark.parametrize("timer_method", ["signal", "thread"])
def test_grouped_item_expiry_preserves_fixture_cleanup(tmp_path: Path, timer_method: str) -> None:
    # Real pytest timers have different destructive paths. CPU owner scopes
    # exercise their handoff without launching MPS or initializing a device.
    (tmp_path / "pytest.ini").write_text(f"[pytest]\ntimeout = 15\ntimeout_method = {timer_method}\n", encoding="utf-8")
    source = tmp_path / "test_items.py"
    source.write_text(
        """
import time
from pathlib import Path
import pytest
from xkit.task import get_task_root

directory = Path(__file__).parent

@pytest.fixture(autouse=True)
def owned_scope(request):
    root = get_task_root()
    assert root is not None
    scope = root.register_scope()
    root.activate()
    try:
        yield
    finally:
        if request.node.name == "test_slow":
            (directory / "retiring").touch()
            while not (directory / "allow-cleanup").exists():
                time.sleep(0.01)
            (directory / "cleaned").touch()
        scope.complete()

def test_first():
    (directory / "first-passed").touch()

@pytest.mark.timeout(0.5)
def test_slow():
    time.sleep(20)

def test_later():
    (directory / "later-ran").touch()
""",
        encoding="utf-8",
    )
    scope = SupervisedTaskScope.start(
        "pytest-item-timeout",
        [sys.executable, "-m", "xtest.harness.runner.worker", "-q", "-c", str(tmp_path / "pytest.ini"), str(source)],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "task.log",
        timeout_seconds=30,
    )
    try:
        observation_deadline = time.monotonic() + 15
        while time.monotonic() < observation_deadline:
            assert scope.poll() is None, (tmp_path / "task.log").read_text(encoding="utf-8")
            if (tmp_path / "retiring").exists():
                break
            time.sleep(0.01)
        assert (tmp_path / "first-passed").is_file()
        assert (tmp_path / "retiring").is_file(), (tmp_path / "task.log").read_text(encoding="utf-8")
        assert scope.is_protected
        assert scope.root is not None
        os.kill(scope.root.pid, signal.SIGTERM)
        os.kill(scope.root.pid, signal.SIGINT)
        # Leave teardown beyond the item limit. Neither the native timer nor a
        # repeated cancellation may interrupt the resource owner's release.
        observation_deadline = time.monotonic() + 0.8
        while time.monotonic() < observation_deadline:
            assert scope.poll() is None
            assert scope.root.is_alive()
            time.sleep(0.01)
        (tmp_path / "allow-cleanup").touch()
        completion = scope.wait()
        assert completion.kind is TaskCompletionKind.TIMED_OUT
        assert completion.diagnostics == "task item exceeded its execution timeout"
        assert (tmp_path / "cleaned").is_file()
        assert not (tmp_path / "later-ran").exists()
    finally:
        (tmp_path / "allow-cleanup").touch()
        if scope.state not in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED, TaskScopeState.CLOSED):
            SupervisedTaskScope.terminate_all((scope,))
        if scope.state in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED):
            scope.close()
