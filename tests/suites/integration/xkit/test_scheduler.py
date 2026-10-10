from __future__ import annotations

import os
from pathlib import Path

from xkit.scheduler import ActiveTask, TaskScheduler
from xkit.supervisor import TaskCompletion
from xtest.harness.support.process_probe import command

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_scheduler_cancellation_retires_all_active_task_domains(tmp_path: Path) -> None:
    completed: list[int] = []

    def complete(running: ActiveTask[int], completion: TaskCompletion) -> None:
        completed.append(running.task)

    scheduler = TaskScheduler[int](device_pool=None, on_complete=complete)

    def start(index: int) -> None:
        scheduler.start(
            index,
            name=f"cancel-{index}",
            device_count=0,
            command=command("sleep", seconds=60),
            cwd=REPO_ROOT,
            env=dict(os.environ),
            log_path=tmp_path / f"task-{index}.log",
            timeout_seconds=60,
        )

    for index in range(2):
        scheduler.submit(index, device_count=0)
    scheduler.admit(start)
    scheduler.request_cancel()
    scheduler.run(start)

    assert set(completed) == {0, 1}
    assert scheduler.resources_releasable
