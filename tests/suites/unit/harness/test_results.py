from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

import tests.harness.runner.results


def test_retention_is_disabled_when_environment_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPOOL_TEST_KEEP_RUNS", raising=False)

    assert tests.harness.runner.results.TestResultRetention.from_environment() is None


@pytest.mark.parametrize("value", ["", "0", "-1", "invalid"])
def test_retention_rejects_nonpositive_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("XPOOL_TEST_KEEP_RUNS", value)

    with pytest.raises(ValueError, match="positive integer"):
        tests.harness.runner.results.TestResultRetention.from_environment()


def test_cleanup_retains_newest_completed_and_interrupted_runs(tmp_path: Path) -> None:
    store = tests.harness.runner.results.TestResultStore(tmp_path, tests.harness.runner.results.TestResultRetention(2))
    run_ids = (
        "20260727-100000-1-1",
        "20260727-100001-1-2",
        "20260727-100002-1-3",
    )
    for timestamp, run_id in enumerate(run_ids, start=1):
        run = store.start(run_id)
        if run_id != run_ids[-1]:
            run.complete()
        else:
            fcntl.flock(run.lock_file.fileno(), fcntl.LOCK_UN)
            run.lock_file.close()
            run.closed = True
        os.utime(run.directory / ".run.lock", ns=(timestamp, timestamp))

    store.cleanup()

    assert not (tmp_path / run_ids[0]).exists()
    assert (tmp_path / run_ids[1]).is_dir()
    assert (tmp_path / run_ids[2]).is_dir()


def test_cleanup_ignores_active_unknown_and_symlink_entries(tmp_path: Path) -> None:
    store = tests.harness.runner.results.TestResultStore(tmp_path, tests.harness.runner.results.TestResultRetention(1))
    old_run = store.start("20260727-100000-1-1")
    old_run.complete()
    active_run = store.start("20260727-100001-1-2")
    os.utime(old_run.directory / ".run.lock", ns=(1, 1))
    os.utime(active_run.directory / ".run.lock", ns=(2, 2))
    unknown = tmp_path / "manual-results"
    unknown.mkdir()
    symlink = tmp_path / "20260727-100002-1-3"
    symlink.symlink_to(unknown, target_is_directory=True)

    store.cleanup()

    assert not old_run.directory.exists()
    assert active_run.directory.is_dir()
    assert unknown.is_dir()
    assert symlink.is_symlink()
    fcntl.flock(active_run.lock_file.fileno(), fcntl.LOCK_UN)
    active_run.lock_file.close()
    active_run.closed = True
