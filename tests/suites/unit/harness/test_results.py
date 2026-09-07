from __future__ import annotations

import fcntl
import os
import threading
from pathlib import Path

import pytest

import tests.harness.runner.results


def test_cleanup_retains_newest_inactive_runs(tmp_path: Path) -> None:
    store = tests.harness.runner.results.TestResultStore(tmp_path)
    run_ids = (
        "20260727-100000-1-1",
        "20260727-100001-1-2",
        "20260727-100002-1-3",
    )
    for timestamp, run_id in enumerate(run_ids, start=1):
        run = store.start(run_id)
        run.complete()
        os.utime(run.directory, ns=(timestamp, timestamp))

    cleanup = store.cleanup(keep_runs=2)

    assert cleanup.removable == (tmp_path / run_ids[0],)
    assert cleanup.retained == (tmp_path / run_ids[2], tmp_path / run_ids[1])
    assert cleanup.active == ()
    assert not (tmp_path / run_ids[0]).exists()
    assert (tmp_path / run_ids[1]).is_dir()
    assert (tmp_path / run_ids[2]).is_dir()


def test_cleanup_removes_legacy_and_symlink_entries_without_touching_active_runs(tmp_path: Path) -> None:
    store = tests.harness.runner.results.TestResultStore(tmp_path)
    completed = store.start("20260727-100000-1-1")
    completed.complete()
    active = store.start("20260727-100001-1-2")
    legacy = tmp_path / "20260724T192347.527844Z-1527645"
    legacy.mkdir()
    external = tmp_path.parent / "external-results"
    external.mkdir()
    symlink = tmp_path / "old-results-link"
    symlink.symlink_to(external, target_is_directory=True)

    cleanup = store.cleanup(keep_runs=0)

    assert set(cleanup.removable) == {completed.directory, legacy, symlink}
    assert cleanup.retained == ()
    assert cleanup.active == (active.directory,)
    assert not completed.directory.exists()
    assert not legacy.exists()
    assert not symlink.exists()
    assert external.is_dir()
    assert active.directory.is_dir()
    active.complete()


def test_cleanup_dry_run_and_missing_root_do_not_mutate_results(tmp_path: Path) -> None:
    missing = tests.harness.runner.results.TestResultStore(tmp_path / "missing")
    assert missing.cleanup(keep_runs=0, dry_run=True) == tests.harness.runner.results.TestResultCleanup((), (), ())

    store = tests.harness.runner.results.TestResultStore(tmp_path / "results")
    run = store.start("20260727-100000-1-1")
    run.complete()

    cleanup = store.cleanup(keep_runs=0, dry_run=True)

    assert cleanup.removable == (run.directory,)
    assert run.directory.is_dir()


@pytest.mark.parametrize("linked_component", ["root", "parent"])
def test_store_resolves_symbolic_link_root_once(tmp_path: Path, linked_component: str) -> None:
    target_parent = tmp_path / "target"
    target_parent.mkdir()
    if linked_component == "root":
        target_root = target_parent / "results"
        target_root.mkdir()
        root = tmp_path / "results"
        root.symlink_to(target_root, target_is_directory=True)
    else:
        linked_parent = tmp_path / "cache"
        linked_parent.symlink_to(target_parent, target_is_directory=True)
        root = linked_parent / "results"
        target_root = target_parent / "results"

    store = tests.harness.runner.results.TestResultStore(root)
    run = store.start("20260727-100000-1-1")
    run.complete()

    cleanup = store.cleanup(keep_runs=0)

    assert cleanup.removable == (target_root / run.directory.name,)
    assert not run.directory.exists()
    assert target_root.is_dir()
    if linked_component == "root":
        assert root.is_symlink()
    else:
        assert root.parent.is_symlink()


def test_start_holds_cleanup_lock_until_run_lock_is_acquired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tests.harness.runner.results.TestResultStore(tmp_path)
    run_id = "20260727-100000-1-1"
    run_lock_pending = threading.Event()
    cleaner_pending = threading.Event()
    allow_run_lock = threading.Event()
    real_flock = fcntl.flock

    def controlled_flock(file_descriptor: int, operation: int) -> None:
        path = Path(os.readlink(f"/proc/self/fd/{file_descriptor}"))
        thread_name = threading.current_thread().name
        if thread_name == "starter" and path.name == tests.harness.runner.results.RUN_LOCK_NAME:
            run_lock_pending.set()
            assert allow_run_lock.wait(timeout=5)
        elif thread_name == "cleaner" and path.name == tests.harness.runner.results.CLEANUP_LOCK_NAME:
            cleaner_pending.set()
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", controlled_flock)
    started: list[tests.harness.runner.results.TestRun] = []
    cleaned: list[tests.harness.runner.results.TestResultCleanup] = []
    starter = threading.Thread(target=lambda: started.append(store.start(run_id)), name="starter")
    cleaner = threading.Thread(target=lambda: cleaned.append(store.cleanup(keep_runs=0)), name="cleaner")

    starter.start()
    assert run_lock_pending.wait(timeout=5)
    cleaner.start()
    assert cleaner_pending.wait(timeout=5)
    allow_run_lock.set()
    starter.join(timeout=5)
    cleaner.join(timeout=5)

    assert not starter.is_alive()
    assert not cleaner.is_alive()
    assert cleaned[0].active == (started[0].directory,)
    assert started[0].directory.is_dir()
    started[0].complete()
