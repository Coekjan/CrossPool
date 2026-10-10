from __future__ import annotations

import fcntl
import json
import os
import threading
from pathlib import Path

import pytest
from pydantic import JsonValue

import xkit.results


def test_json_checkpoint_preserves_previous_value_on_serialization_failure(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    xkit.results.write_json(path, {"ready": True})
    with pytest.raises(ValueError):
        xkit.results.write_json(path, {"elapsed": float("nan")})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ready": True}
    assert tuple(tmp_path.iterdir()) == (path,)


def test_resume_keeps_existing_evidence_locked_through_reopening(tmp_path: Path) -> None:
    store = xkit.results.RunStore(tmp_path)
    run = store.start("20261009-100000-1-1")
    xkit.results.write_json(run.directory / "run.json", {"finished": True})
    run.complete()
    with store.resume(run.directory.name) as entry:
        assert (entry.directory / ".completed").is_file()
        assert (entry.directory / "run.json").read_bytes() == (run.directory / "run.json").read_bytes()
        with pytest.raises(BlockingIOError):
            with store.read(run.directory.name):
                pytest.fail("resuming run became readable")
        entry.reopen()
        assert not (entry.directory / ".completed").exists()
        assert store.cleanup(keep_runs=0).active == (entry.directory,)
        entry.complete()
    assert (run.directory / ".completed").is_file()


def test_jsonl_round_trip_preserves_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    values: tuple[JsonValue, ...] = ({"text": "你好", "count": 2}, None)
    xkit.results.write_jsonl(path, (value for value in values))

    assert tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) == values
    with pytest.raises(FileExistsError):
        xkit.results.write_jsonl(path, ())
    with pytest.raises(ValueError):
        xkit.results.write_jsonl(tmp_path / "invalid.jsonl", ({"elapsed": float("nan")},))
    assert tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) == values


@pytest.mark.parametrize("exclusive", [False, True])
def test_report_read_protects_inactive_run_from_cleanup(tmp_path: Path, exclusive: bool) -> None:
    store = xkit.results.RunStore(tmp_path)
    run = store.start("20260927-100000-1-1")
    with pytest.raises(BlockingIOError):
        with store.read(run.directory.name):
            pytest.fail("active run became readable")
    run.complete()

    with store.read(run.directory.name, exclusive=exclusive) as directory:
        assert directory == run.directory
        assert store.cleanup(keep_runs=0).active == (run.directory,)
        if exclusive:
            with pytest.raises(BlockingIOError):
                with store.read(run.directory.name, exclusive=True):
                    pytest.fail("concurrent writer acquired an exclusive lock")
    assert store.cleanup(keep_runs=0).removable == (run.directory,)


def test_cleanup_retains_newest_inactive_runs(tmp_path: Path) -> None:
    store = xkit.results.RunStore(tmp_path)
    run_ids = (
        "20260727-100000-1-1",
        "20260727-100001-1-2",
        "20260727-100002-1-3",
    )
    for timestamp, run_id in enumerate(run_ids, start=1):
        run = store.start(run_id)
        run.complete()
        os.utime(run.directory, ns=(timestamp, timestamp))
        os.utime(run.directory / ".run.lock", ns=(timestamp, timestamp))

    # Derived reports can change directory age, but not execution start age.
    (tmp_path / run_ids[0] / "report").mkdir()

    cleanup = store.cleanup(keep_runs=2)

    assert cleanup.removable == (tmp_path / run_ids[0],)
    assert cleanup.retained == (tmp_path / run_ids[2], tmp_path / run_ids[1])
    assert cleanup.active == ()
    assert not (tmp_path / run_ids[0]).exists()
    assert (tmp_path / run_ids[1]).is_dir()
    assert (tmp_path / run_ids[2]).is_dir()


def test_cleanup_removes_unrecognized_and_symlink_entries_without_touching_active_runs(tmp_path: Path) -> None:
    store = xkit.results.RunStore(tmp_path)
    completed = store.start("20260727-100000-1-1")
    completed.complete()
    active = store.start("20260727-100001-1-2")
    unrecognized = tmp_path / "20260724T192347.527844Z-1527645"
    unrecognized.mkdir()
    external = tmp_path.parent / "external-results"
    external.mkdir()
    symlink = tmp_path / "old-results-link"
    symlink.symlink_to(external, target_is_directory=True)

    cleanup = store.cleanup(keep_runs=0)

    assert set(cleanup.removable) == {completed.directory, unrecognized, symlink}
    assert cleanup.retained == ()
    assert cleanup.active == (active.directory,)
    assert not completed.directory.exists()
    assert not unrecognized.exists()
    assert not symlink.exists()
    assert external.is_dir()
    assert active.directory.is_dir()
    active.complete()


def test_cleanup_dry_run_and_missing_root_do_not_mutate_results(tmp_path: Path) -> None:
    missing = xkit.results.RunStore(tmp_path / "missing")
    assert missing.cleanup(keep_runs=0, dry_run=True) == xkit.results.RunCleanup((), (), ())

    store = xkit.results.RunStore(tmp_path / "results")
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

    store = xkit.results.RunStore(root)
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
    store = xkit.results.RunStore(tmp_path)
    run_id = "20260727-100000-1-1"
    run_lock_pending = threading.Event()
    cleaner_pending = threading.Event()
    allow_run_lock = threading.Event()
    real_flock = fcntl.flock

    def controlled_flock(file_descriptor: int, operation: int) -> None:
        path = Path(os.readlink(f"/proc/self/fd/{file_descriptor}"))
        thread_name = threading.current_thread().name
        if thread_name == "starter" and path.name == xkit.results.RUN_LOCK_NAME:
            run_lock_pending.set()
            assert allow_run_lock.wait(timeout=5)
        elif thread_name == "cleaner" and path.name == xkit.results.CLEANUP_LOCK_NAME:
            cleaner_pending.set()
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", controlled_flock)
    started: list[xkit.results.RunEntry] = []
    cleaned: list[xkit.results.RunCleanup] = []
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


def test_inactive_inventory_is_read_only_and_excludes_active_and_unrecognized_entries(tmp_path: Path) -> None:
    missing = xkit.results.RunStore(tmp_path / "missing")
    assert missing.inactive_runs() == ()
    assert not missing.root.exists()
    store = xkit.results.RunStore(tmp_path / "runs")
    finished = store.start("20260727-100000-1-1")
    finished.complete()
    active = store.start("20260727-100001-1-2")
    (store.root / "notes").mkdir()
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in store.root.rglob("*") if path.is_file()}
    try:
        assert store.inactive_runs() == (finished.directory,)
        assert before == {
            path: (path.stat().st_mtime_ns, path.read_bytes()) for path in store.root.rglob("*") if path.is_file()
        }
    finally:
        active.complete()
