from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests.harness.runner.artifact
import tests.harness.runner.plan


def test_plan_round_trips_strict_unversioned_json(tmp_path: Path) -> None:
    plan = tests.harness.runner.plan.TestPlan((e2e_case("eager"), e2e_case("full")))
    path = tmp_path / "plan.json"

    plan.write(path)

    assert tests.harness.runner.plan.TestPlan.read(path) == plan
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"cases"}
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_plan_rejects_unknown_case_fields(tmp_path: Path) -> None:
    case = unit_case().raw()
    case["unknown"] = True
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        tests.harness.runner.plan.TestPlan.read(path)


def test_collected_case_rejects_stage_and_resource_drift() -> None:
    with pytest.raises(ValueError, match="stage disagrees"):
        tests.harness.runner.plan.CollectedTestCase(
            path="tests/suites/unit/test_example.py",
            nodeid="tests/suites/unit/test_example.py::test_example",
            stage=tests.harness.runner.plan.TestStage.INTEGRATION,
            requirements=tests.harness.runner.plan.TestRequirements(0, False, False, ()),
            estimated_duration_seconds=None,
            timeout_seconds=10,
            artifact_group=None,
        )
    with pytest.raises(ValueError, match="unit tests cannot require CUDA"):
        tests.harness.runner.plan.CollectedTestCase(
            path="tests/suites/unit/test_example.py",
            nodeid="tests/suites/unit/test_example.py::test_example",
            stage=tests.harness.runner.plan.TestStage.UNIT,
            requirements=tests.harness.runner.plan.TestRequirements(1, False, False, ()),
            estimated_duration_seconds=None,
            timeout_seconds=10,
            artifact_group=None,
        )


def test_plan_rejects_duplicate_nodes_and_incomplete_parity_groups() -> None:
    case = unit_case()
    with pytest.raises(ValueError, match="nodeids must be unique"):
        tests.harness.runner.plan.TestPlan((case, case))
    with pytest.raises(ValueError, match="every expected case"):
        tests.harness.runner.plan.TestPlan((e2e_case("eager"),))


def test_plan_rejects_inconsistent_parity_group_cardinality() -> None:
    eager = e2e_case("eager")
    full = tests.harness.runner.plan.CollectedTestCase(
        path=eager.path,
        nodeid=f"{eager.path}::test_e2e_example[full]",
        stage=eager.stage,
        requirements=eager.requirements,
        estimated_duration_seconds=eager.estimated_duration_seconds,
        timeout_seconds=eager.timeout_seconds,
        artifact_group=tests.harness.runner.artifact.ArtifactGroupRef("token_parity", "example", 3),
    )

    with pytest.raises(ValueError, match="inconsistent expected_case_count"):
        tests.harness.runner.plan.TestPlan((eager, full))


def unit_case() -> tests.harness.runner.plan.CollectedTestCase:
    return tests.harness.runner.plan.CollectedTestCase(
        path="tests/suites/unit/test_example.py",
        nodeid="tests/suites/unit/test_example.py::test_example",
        stage=tests.harness.runner.plan.TestStage.UNIT,
        requirements=tests.harness.runner.plan.TestRequirements(0, False, False, ()),
        estimated_duration_seconds=None,
        timeout_seconds=10,
        artifact_group=None,
    )


def e2e_case(mode: str) -> tests.harness.runner.plan.CollectedTestCase:
    path = "tests/suites/e2e/test_e2e_example.py"
    return tests.harness.runner.plan.CollectedTestCase(
        path=path,
        nodeid=f"{path}::test_e2e_example[{mode}]",
        stage=tests.harness.runner.plan.TestStage.E2E,
        requirements=tests.harness.runner.plan.TestRequirements(2, True, True, ("organization/model",)),
        estimated_duration_seconds=60,
        timeout_seconds=120,
        artifact_group=tests.harness.runner.artifact.ArtifactGroupRef("token_parity", "example", 2),
    )
