from __future__ import annotations

import json
from pathlib import Path

import pytest

import xtest.harness.runner.artifact
import xtest.harness.runner.plan
from xkit import ResourceRequirements
from xtest.harness.support.config import TEST_MODEL_ID


def test_plan_round_trips_strict_unversioned_json(tmp_path: Path) -> None:
    plan = xtest.harness.runner.plan.TestPlan((e2e_case("eager"), e2e_case("full")))
    path = tmp_path / "plan.json"

    plan.write(path)

    assert xtest.harness.runner.plan.TestPlan.read(path) == plan
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"cases"}
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_plan_rejects_unknown_case_fields(tmp_path: Path) -> None:
    case = unit_case().raw()
    case["unknown"] = True
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        xtest.harness.runner.plan.TestPlan.read(path)


def test_collected_case_rejects_stage_and_resource_drift() -> None:
    with pytest.raises(ValueError, match="stage disagrees"):
        xtest.harness.runner.plan.CollectedTestCase(
            path="tests/suites/unit/test_example.py",
            nodeid="tests/suites/unit/test_example.py::test_example",
            stage=xtest.harness.runner.plan.TestStage.INTEGRATION,
            requirements=ResourceRequirements(0, False, ()),
            estimated_duration_seconds=None,
            timeout_seconds=10,
            artifact_group=None,
        )
    with pytest.raises(ValueError, match="unit tests cannot require devices"):
        xtest.harness.runner.plan.CollectedTestCase(
            path="tests/suites/unit/test_example.py",
            nodeid="tests/suites/unit/test_example.py::test_example",
            stage=xtest.harness.runner.plan.TestStage.UNIT,
            requirements=ResourceRequirements(1, False, ()),
            estimated_duration_seconds=None,
            timeout_seconds=10,
            artifact_group=None,
        )


def test_plan_rejects_duplicate_nodes_and_incomplete_serving_graph_groups() -> None:
    case = unit_case()
    with pytest.raises(ValueError, match="nodeids must be unique"):
        xtest.harness.runner.plan.TestPlan((case, case))
    with pytest.raises(ValueError, match="every expected case"):
        xtest.harness.runner.plan.TestPlan((e2e_case("eager"),))


def test_plan_rejects_inconsistent_serving_graph_group_cardinality() -> None:
    eager = e2e_case("eager")
    full = xtest.harness.runner.plan.CollectedTestCase(
        path=eager.path,
        nodeid=f"{eager.path}::test_e2e_example[full]",
        stage=eager.stage,
        requirements=eager.requirements,
        estimated_duration_seconds=eager.estimated_duration_seconds,
        timeout_seconds=eager.timeout_seconds,
        artifact_group=xtest.harness.runner.artifact.ArtifactGroupRef("serving_graph", "example", 3),
    )

    with pytest.raises(ValueError, match="inconsistent expected_case_count"):
        xtest.harness.runner.plan.TestPlan((eager, full))


def unit_case() -> xtest.harness.runner.plan.CollectedTestCase:
    return xtest.harness.runner.plan.CollectedTestCase(
        path="tests/suites/unit/test_example.py",
        nodeid="tests/suites/unit/test_example.py::test_example",
        stage=xtest.harness.runner.plan.TestStage.UNIT,
        requirements=ResourceRequirements(0, False, ()),
        estimated_duration_seconds=None,
        timeout_seconds=10,
        artifact_group=None,
    )


def e2e_case(mode: str) -> xtest.harness.runner.plan.CollectedTestCase:
    path = "tests/suites/e2e/test_e2e_example.py"
    return xtest.harness.runner.plan.CollectedTestCase(
        path=path,
        nodeid=f"{path}::test_e2e_example[{mode}]",
        stage=xtest.harness.runner.plan.TestStage.E2E,
        requirements=ResourceRequirements(2, True, (TEST_MODEL_ID,)),
        estimated_duration_seconds=60,
        timeout_seconds=120,
        artifact_group=xtest.harness.runner.artifact.ArtifactGroupRef("serving_graph", "example", 2),
    )
