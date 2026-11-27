from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import xtest.harness.runner.plan
from xkit import ResourceRequirements
from xtest.harness.support.config import TEST_MODEL_ID


def test_plan_round_trips_strict_unversioned_json(tmp_path: Path) -> None:
    plan = xtest.harness.runner.plan.TestPlan((e2e_case("eager"), e2e_case("full")), ())
    path = tmp_path / "plan.json"

    plan.write(path)

    assert xtest.harness.runner.plan.TestPlan.read(path) == plan
    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document) == {"cases", "selected_suites"}
    assert document["cases"][0]["requirements"]["model_ids"] == [str(TEST_MODEL_ID)]
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_plan_rejects_unknown_case_fields(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    xtest.harness.runner.plan.TestPlan((unit_case(),), ()).write(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cases"][0]["unknown"] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
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
            inspection=None,
        )
    with pytest.raises(ValueError, match="unit tests cannot require devices"):
        xtest.harness.runner.plan.CollectedTestCase(
            path="tests/suites/unit/test_example.py",
            nodeid="tests/suites/unit/test_example.py::test_example",
            stage=xtest.harness.runner.plan.TestStage.UNIT,
            requirements=ResourceRequirements(1, False, ()),
            estimated_duration_seconds=None,
            timeout_seconds=10,
            inspection=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("estimated_duration_seconds", float("nan")), ("timeout_seconds", float("inf"))),
)
def test_collection_rejects_nonfinite_times(field: str, value: float, tmp_path: Path) -> None:
    case = unit_case()
    with pytest.raises(ValueError, match="finite"):
        replace(case, **{field: value})
    path = tmp_path / "plan.json"
    xtest.harness.runner.plan.TestPlan((case,), ()).write(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cases"][0][field] = value
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        xtest.harness.runner.plan.TestPlan.read(path)


def test_plan_rejects_duplicate_nodes() -> None:
    case = unit_case()
    with pytest.raises(ValueError, match="nodeids must be unique"):
        xtest.harness.runner.plan.TestPlan((case, case), ())


def unit_case() -> xtest.harness.runner.plan.CollectedTestCase:
    return xtest.harness.runner.plan.CollectedTestCase(
        path="tests/suites/unit/test_example.py",
        nodeid="tests/suites/unit/test_example.py::test_example",
        stage=xtest.harness.runner.plan.TestStage.UNIT,
        requirements=ResourceRequirements(0, False, ()),
        estimated_duration_seconds=None,
        timeout_seconds=10,
        inspection=None,
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
        inspection=None,
    )
