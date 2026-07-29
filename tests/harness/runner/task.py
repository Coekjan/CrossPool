"""Deterministic compilation of collected pytest cases into execution tasks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from tests.harness.runner.plan import CollectedTestCase, TestPlan, TestRequirements, TestStage

TASK_KEY_CHARACTER_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
TASK_KEY_PREFIX_LENGTH = 72


@dataclass(frozen=True, slots=True)
class ExecutionTask:
    """One isolated pytest process with homogeneous scheduling resources."""

    key: str
    stage: TestStage
    cases: tuple[CollectedTestCase, ...]
    requirements: TestRequirements
    estimated_duration_seconds: float
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.key or TASK_KEY_CHARACTER_PATTERN.search(self.key):
            raise ValueError(f"execution task key is not filesystem-safe: {self.key!r}")
        if not self.cases:
            raise ValueError("execution task must contain at least one collected case")
        if any(case.stage is not self.stage for case in self.cases):
            raise ValueError("execution task cases must belong to its stage")
        if self.requirements != merge_requirements(self.cases):
            raise ValueError("execution task requirements must equal its merged case requirements")
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
    integration_cpu_cases = tuple(case for case in integration_cases if case.requirements.cuda_count == 0)
    if integration_cpu_cases:
        tasks.append(build_task("integration-cpu", integration_cpu_cases))

    integration_gpu_groups: dict[tuple[str, TestRequirements], list[CollectedTestCase]] = defaultdict(list)
    for case in integration_cases:
        if case.requirements.cuda_count:
            integration_gpu_groups[(case.path, case.requirements)].append(case)
    for (path, requirements), cases in sorted(
        integration_gpu_groups.items(), key=lambda item: (item[0][0], requirements_identity(item[0][1]))
    ):
        identity = f"{path}\0{requirements_identity(requirements)}"
        tasks.append(build_task(task_key("integration-gpu", path, identity), tuple(cases)))

    for case in plan.cases:
        if case.stage is TestStage.E2E:
            tasks.append(build_task(task_key("e2e", case.nodeid, case.nodeid), (case,)))

    keys = tuple(task.key for task in tasks)
    if len(keys) != len(set(keys)):
        raise ValueError("compiled execution task keys must be unique")
    return tuple(tasks)


def build_task(key: str, cases: tuple[CollectedTestCase, ...]) -> ExecutionTask:
    """Build one task using the accepted member aggregation rules."""

    return ExecutionTask(
        key=key,
        stage=cases[0].stage,
        cases=cases,
        requirements=merge_requirements(cases),
        estimated_duration_seconds=sum(case.estimated_duration_seconds or case.timeout_seconds for case in cases),
        timeout_seconds=sum(case.timeout_seconds for case in cases),
    )


def merge_requirements(cases: Iterable[CollectedTestCase]) -> TestRequirements:
    """Merge item requirements for one pytest process without losing model order."""

    materialized = tuple(cases)
    if not materialized:
        raise ValueError("cannot merge requirements for an empty case set")
    model_ids = tuple(dict.fromkeys(model_id for case in materialized for model_id in case.requirements.model_ids))
    return TestRequirements(
        cuda_count=max(case.requirements.cuda_count for case in materialized),
        requires_mps=any(case.requirements.requires_mps for case in materialized),
        requires_config=any(case.requirements.requires_config for case in materialized),
        model_ids=model_ids,
    )


def task_key(category: str, label: str, identity: str) -> str:
    """Derive one readable collision-resistant filesystem key."""

    readable = TASK_KEY_CHARACTER_PATTERN.sub("-", label).strip("-._")[:TASK_KEY_PREFIX_LENGTH]
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return f"{category}-{readable or 'task'}-{digest}"


def requirements_identity(requirements: TestRequirements) -> str:
    """Return a stable grouping identity for one exact resource signature."""

    return json.dumps(requirements.raw(), sort_keys=True, separators=(",", ":"))
