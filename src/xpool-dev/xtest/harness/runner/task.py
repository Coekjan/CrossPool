"""Deterministic compilation of collected pytest cases into execution tasks."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from xkit import ResourceRequirements
from xtest.harness.runner.plan import CollectedTestCase, TestPlan, TestStage

TASK_KEY_CHARACTER_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
TASK_KEY_PREFIX_LENGTH = 72


@dataclass(frozen=True, slots=True)
class ExecutionTask:
    """One isolated pytest process with homogeneous scheduling resources."""

    key: str
    stage: TestStage
    cases: tuple[CollectedTestCase, ...]
    requirements: ResourceRequirements
    estimated_duration_seconds: float
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.key or TASK_KEY_CHARACTER_PATTERN.search(self.key):
            raise ValueError(f"execution task key is not filesystem-safe: {self.key!r}")
        if not self.cases:
            raise ValueError("execution task must contain at least one collected case")
        if any(case.stage is not self.stage for case in self.cases):
            raise ValueError("execution task cases must belong to its stage")
        if self.estimated_duration_seconds <= 0:
            raise ValueError("execution task estimate must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("execution task timeout must be positive")


def compile_execution_tasks(plan: TestPlan) -> tuple[ExecutionTask, ...]:
    """Compile a validated Test Plan into deterministic pytest process units."""

    tasks: list[ExecutionTask] = []
    unit_cases = tuple(case for case in plan.cases if case.stage is TestStage.UNIT)
    if unit_cases:
        tasks.append(build_task("unit", unit_cases))

    integration_cases = tuple(case for case in plan.cases if case.stage is TestStage.INTEGRATION)
    integration_cpu_cases = tuple(case for case in integration_cases if case.requirements.device_count == 0)
    if integration_cpu_cases:
        tasks.append(build_task("integration-cpu", integration_cpu_cases))

    integration_device_groups: dict[tuple[str, ResourceRequirements], list[CollectedTestCase]] = defaultdict(list)
    for case in integration_cases:
        if case.requirements.device_count:
            integration_device_groups[(case.path, case.requirements)].append(case)
    for group in sorted(integration_device_groups, key=lambda group: group[0]):
        key = task_key("integration-device", group[0], len(tasks) + 1)
        tasks.append(build_task(key, tuple(integration_device_groups[group])))

    for case in plan.cases:
        if case.stage in {TestStage.E2E, TestStage.MODELS}:
            key = task_key(case.stage.value, case.nodeid, len(tasks) + 1)
            tasks.append(build_task(key, (case,)))
    return tuple(tasks)


def build_task(key: str, cases: tuple[CollectedTestCase, ...]) -> ExecutionTask:
    """Build one task using the accepted member aggregation rules."""

    model_ids = tuple(dict.fromkeys(model_id for case in cases for model_id in case.requirements.model_ids))
    return ExecutionTask(
        key=key,
        stage=cases[0].stage,
        cases=cases,
        requirements=ResourceRequirements(
            device_count=max(case.requirements.device_count for case in cases),
            requires_config=any(case.requirements.requires_config for case in cases),
            model_ids=model_ids,
        ),
        estimated_duration_seconds=sum(case.estimated_duration_seconds or case.timeout_seconds for case in cases),
        timeout_seconds=sum(case.timeout_seconds for case in cases),
    )


def task_key(category: str, label: str, ordinal: int) -> str:
    """Derive a readable filesystem key unique within this run."""

    readable = TASK_KEY_CHARACTER_PATTERN.sub("-", label).strip("-._")[:TASK_KEY_PREFIX_LENGTH]
    return f"{category}-{readable or 'task'}-{ordinal:04d}"
