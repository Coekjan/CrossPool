from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from xpool.utils.background import BackgroundThread


def test_background_thread_start_stop_sets_event_and_clears_thread() -> None:
    started = threading.Event()
    stopped = threading.Event()

    def target(stop_event: threading.Event) -> None:
        started.set()
        stop_event.wait(5.0)
        stopped.set()

    worker = BackgroundThread(
        name="xpool-test-background",
        target=target,
        join_timeout_s=1.0,
    )

    worker.start()
    assert started.wait(1.0)
    assert worker.is_running
    worker.stop()

    assert stopped.wait(1.0)
    assert worker.thread is None
    assert not worker.is_running


def test_background_thread_start_is_idempotent_while_running() -> None:
    started = threading.Event()
    calls = 0

    def target(stop_event: threading.Event) -> None:
        nonlocal calls
        calls += 1
        started.set()
        stop_event.wait(5.0)

    worker = BackgroundThread(
        name="xpool-test-background",
        target=target,
        join_timeout_s=1.0,
    )

    worker.start()
    assert started.wait(1.0)
    worker.start()
    worker.stop()

    assert calls == 1


def test_background_thread_start_replaces_dead_thread() -> None:
    calls = 0

    def target(stop_event: threading.Event) -> None:
        nonlocal calls
        calls += 1

    worker = BackgroundThread(
        name="xpool-test-background",
        target=target,
        join_timeout_s=1.0,
    )

    worker.start()
    assert wait_until(lambda: worker.thread is not None and not worker.thread.is_alive())
    worker.start()
    assert wait_until(lambda: calls == 2)
    worker.stop()

    assert calls == 2


def test_background_thread_close_prevents_restart() -> None:
    worker = BackgroundThread(
        name="xpool-test-background",
        target=lambda stop_event: None,
        join_timeout_s=1.0,
    )

    worker.close()

    with pytest.raises(RuntimeError, match="closed background thread"):
        worker.start()


def test_background_thread_timeout_preserves_live_worker_for_retry() -> None:
    started = threading.Event()
    release = threading.Event()

    def target(stop_event: threading.Event) -> None:
        started.set()
        release.wait(5.0)

    worker = BackgroundThread(
        name="xpool-test-background",
        target=target,
        join_timeout_s=0.01,
    )
    worker.start()
    assert started.wait(1.0)

    with pytest.raises(RuntimeError, match="did not stop"):
        worker.close()
    assert worker.thread is not None
    assert worker.is_running

    release.set()
    worker.close()
    assert worker.thread is None
    with pytest.raises(RuntimeError, match="closed background thread"):
        worker.start()


def test_background_thread_surfaces_target_failure() -> None:
    failure = RuntimeError("worker crashed")
    worker = BackgroundThread(
        name="xpool-test-background",
        target=lambda stop_event: (item for item in ()).throw(failure),
        join_timeout_s=1.0,
    )

    worker.start()
    assert wait_until(lambda: worker.thread is not None and not worker.thread.is_alive())

    with pytest.raises(RuntimeError, match="worker crashed") as exc_info:
        worker.raise_if_failed()
    assert exc_info.value is failure


def test_background_thread_failure_prevents_restart() -> None:
    failure = RuntimeError("worker crashed")
    worker = BackgroundThread(
        name="xpool-test-background",
        target=lambda stop_event: (item for item in ()).throw(failure),
        join_timeout_s=1.0,
    )

    worker.start()
    assert wait_until(lambda: worker.thread is not None and not worker.thread.is_alive())

    with pytest.raises(RuntimeError, match="worker crashed") as exc_info:
        worker.start()
    assert exc_info.value is failure


def test_background_thread_periodic_repeats_until_stopped() -> None:
    calls = 0

    def target() -> None:
        nonlocal calls
        calls += 1

    worker = BackgroundThread.periodic(
        name="xpool-test-background",
        interval_s=0.01,
        target=target,
        join_timeout_s=1.0,
    )

    worker.start()
    assert wait_until(lambda: calls >= 2)
    worker.stop()
    worker.raise_if_failed()


def test_background_thread_periodic_false_return_stops_without_failure() -> None:
    calls = 0

    def target() -> bool:
        nonlocal calls
        calls += 1
        return False

    worker = BackgroundThread.periodic(
        name="xpool-test-background",
        interval_s=0.01,
        target=target,
        join_timeout_s=1.0,
    )

    worker.start()
    assert wait_until(lambda: worker.thread is not None and not worker.thread.is_alive())
    worker.raise_if_failed()

    assert calls == 1


def test_background_thread_does_not_store_base_exception() -> None:
    worker = BackgroundThread(
        name="xpool-test-background",
        target=lambda stop_event: (item for item in ()).throw(SystemExit(7)),
        join_timeout_s=1.0,
    )

    with pytest.raises(SystemExit) as exc_info:
        worker.run_target(threading.Event())
    assert exc_info.value.code == 7
    worker.raise_if_failed()


def wait_until(predicate: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False
