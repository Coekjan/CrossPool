from __future__ import annotations

import os
import time
from pathlib import Path

import xtest.harness.runner.plan
import xtest.harness.runner.suite
from xkit import ResourceRequirements
from xkit.supervisor import (
    SupervisedTaskScope,
)
from xtest.harness.runner.task import ExecutionTask
from xtest.harness.support.process_probe import command

REPO_ROOT = Path(__file__).resolve().parents[6]


def test_suite_runner_cancellation_reaps_each_supervisor_through_its_scope(tmp_path: Path) -> None:
    requirements = ResourceRequirements(0, False, ())
    cases = tuple(
        xtest.harness.runner.plan.CollectedTestCase(
            path="tests/suites/integration/xtest/harness/runner/test_suite.py",
            nodeid=f"tests/suites/integration/xtest/harness/runner/test_suite.py::synthetic_cancel_{index}",
            stage=xtest.harness.runner.plan.TestStage.INTEGRATION,
            requirements=requirements,
            estimated_duration_seconds=1.0,
            timeout_seconds=60.0,
            artifact_group=None,
        )
        for index in range(2)
    )
    runner = xtest.harness.runner.suite.SuiteRunner(
        xtest.harness.runner.plan.TestPlan(cases),
        repository_root=REPO_ROOT,
        run_directory=tmp_path,
        strict_requirements=False,
        catalogue_path=REPO_ROOT / "tests/tests.toml",
    )
    for index, case in enumerate(cases):
        task = ExecutionTask(
            key=f"cancel-{index}",
            stage=xtest.harness.runner.plan.TestStage.INTEGRATION,
            cases=(case,),
            requirements=requirements,
            estimated_duration_seconds=1.0,
            timeout_seconds=60.0,
        )
        directory = tmp_path / task.key
        directory.mkdir()
        scope = SupervisedTaskScope.start(
            task.key,
            command("sleep", seconds=60),
            cwd=REPO_ROOT,
            env=dict(os.environ),
            log_path=directory / "task.log",
            timeout_seconds=60,
        )
        runner.active[task.key] = xtest.harness.runner.suite.RunningTask(
            task, directory, scope, None, device_assignments="none", started_at=time.monotonic()
        )

    runner.cancel_active_tasks()

    assert runner.active == {}
    assert runner.resources_releasable
