import shutil
import tomllib
from pathlib import Path

import pytest
import tomli_w
from tests import TEST_CATALOG_PATH

import xtest.harness.runner.plan
from xpool.model import ModelId
from xtest.harness.support.config import TEST_MODEL_ID, write_minimal_config

pytest_plugins = ["pytester"]


@pytest.fixture
def bound_catalogue(pytester: pytest.Pytester) -> Path:
    """Rebind two existing portable scenes to one isolated source module."""

    pytester.makeini("[pytest]\ntimeout = 15\n")
    root = TEST_CATALOG_PATH.resolve().parent.parent
    shutil.copytree(root / "configs/deployments", pytester.path / "configs/deployments")
    with TEST_CATALOG_PATH.open("rb") as source:
        declarations = tomllib.load(source)
    cases = {name: declarations["serving_cases"][name] for name in ("serving-001", "serving-002")}
    for case in cases.values():
        case["module"] = "integration.test_program"
    catalogue = pytester.path / "tests/tests.toml"
    catalogue.parent.mkdir()
    catalogue.write_text(tomli_w.dumps({"serving_cases": cases}), encoding="utf-8")
    (pytester.path / "tests/suites/integration").mkdir(parents=True)
    return catalogue


def test_bound_cases_preserve_native_fixtures_explicit_rows_and_resource_deselection(
    pytester: pytest.Pytester, bound_catalogue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    program = pytester.path / "tests/suites/integration/test_program.py"
    program.write_text(
        """
import pytest
import xtest
from xkit import ResourceRequirements

def resources(case):
    return ResourceRequirements(case.required_device_count, True, case.models)

@xtest.parameterize("case")
@xtest.requirements(resources)
def test_bound(case, tmp_path):
    assert tmp_path.is_dir()

@xtest.parameterize("size", (16, 32))
def test_explicit(size, tmp_path):
    assert size in (16, 32) and tmp_path.is_dir()

@pytest.mark.parametrize("device_count", (1, 2))
@xtest.requirements(lambda device_count: ResourceRequirements(device_count, False, ()))
def test_native(device_count, tmp_path):
    assert device_count in (1, 2) and tmp_path.is_dir()

def test_plain(tmp_path):
    assert tmp_path.is_dir()
""",
        encoding="utf-8",
    )
    output = pytester.path / "plan.json"
    collected = pytester.runpytest(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={bound_catalogue}",
        f"--xpool-test-plan={output}",
        "--collect-only",
        "-q",
        str(program),
    )
    assert collected.ret == 0
    plan = xtest.harness.runner.plan.TestPlan.read(output)
    assert len(plan.cases) == 7
    assert tuple(case.requirements.device_count for case in plan.cases) == (2, 3, 0, 0, 1, 2, 0)
    assert plan.cases[0].nodeid.endswith("test_bound[serving-001]")
    assert plan.cases[1].nodeid.endswith("test_bound[serving-002]")
    assert plan.cases[0].requirements.model_ids == (ModelId("Qwen/Qwen3-0.6B"),)
    executed = pytester.runpytest(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={bound_catalogue}",
        "-m",
        "not requires_device",
        "-q",
        str(program),
    )
    executed.assert_outcomes(passed=3, deselected=4)


@pytest.mark.parametrize(
    ("source", "entry_count"),
    [
        ("", 0),
        ("def test_plain(): pass\n", 0),
        (
            "import xtest\n@xtest.parameterize('case')\ndef test_one(case): pass\n"
            "@xtest.parameterize('case')\ndef test_two(case): pass\n",
            2,
        ),
    ],
)
def test_referenced_module_requires_one_collectable_bound_entry(
    pytester: pytest.Pytester, bound_catalogue: Path, source: str, entry_count: int
) -> None:
    program = pytester.path / "tests/suites/integration/test_program.py"
    program.write_text(source, encoding="utf-8")
    result = pytester.runpytest(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={bound_catalogue}",
        "--collect-only",
        "-q",
        str(program),
    )
    assert result.ret != 0
    assert f"expected one catalogue-bound entry, found {entry_count}" in result.stdout.str()


def test_expanded_graph_inputs_keep_distinct_groups_and_stable_task_recollection(
    pytester: pytest.Pytester, bound_catalogue: Path
) -> None:
    program = pytester.path / "tests/suites/integration/test_graph_program.py"
    program.write_text(
        f"""
from functools import partial
from pathlib import Path

import xtest
from xkit.serving.sglang.graph import SglangGraphMode
from xtest.harness.sglang.catalog import TestCatalog
from xtest.harness.sglang.serving.qualification import graph_rows

base = TestCatalog.load(Path({str(bound_catalogue)!r})).serving_cases[0]
cases = tuple(
    base.model_copy(update={{
        "id": None, "description": str(index), "graph_modes": tuple(SglangGraphMode),
        "estimated_duration_seconds": 3.0, "timeout_seconds": 12.0,
    }})
    for index in range(2)
)

@xtest.parameterize(("case", "graph_mode"), cases, rows=partial(graph_rows, compare_modes=True))
def test_graph(case, graph_mode, request, tmp_path):
    marker = request.node.get_closest_marker("serving_graph_group")
    function_nodeid = request.node.nodeid.split("[", 1)[0]
    assert marker.kwargs == {{"name": f"{{function_nodeid}}[case-{{case.description}}]", "expected_case_count": 3}}
    assert tmp_path.is_dir()
""",
        encoding="utf-8",
    )
    output = pytester.path / "plan.json"
    arguments = (
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={bound_catalogue}",
        f"--rootdir={pytester.path}",
        "-q",
    )
    collected = pytester.runpytest_subprocess(*arguments, "--collect-only", f"--xpool-test-plan={output}", str(program))
    assert collected.ret == 0
    plan = xtest.harness.runner.plan.TestPlan.read(output)
    assert len(plan.cases) == 6
    assert len({case.nodeid for case in plan.cases}) == 6
    for index in range(2):
        rows = plan.cases[index * 3 : (index + 1) * 3]
        assert all(case.artifact_group is not None for case in rows)
        assert {case.artifact_group.name for case in rows if case.artifact_group is not None} == {
            f"tests/suites/integration/test_graph_program.py::test_graph[case-{index}]"
        }
        assert all(case.estimated_duration_seconds == 3 for case in rows)
        assert all(case.timeout_seconds == 12 for case in rows)
    recollected = pytester.runpytest_subprocess(
        *arguments, "--collect-only", f"--xpool-test-plan={output}", str(program)
    )
    assert recollected.ret == 0
    assert xtest.harness.runner.plan.TestPlan.read(output) == plan


def test_config_requirements_apply_default_and_strict_outcomes(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    pytester.makepyfile(
        """
        import xtest

        @xtest.requirements(requires_config=True)
        def test_config():
            pass
        """
    )

    default_result = pytester.runpytest(
        "-p", "xtest.harness.runner.pytest_plugin", f"--xpool-test-catalog={TEST_CATALOG_PATH}", "-q"
    )
    default_result.assert_outcomes(skipped=1)
    strict_result = pytester.runpytest(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={TEST_CATALOG_PATH}",
        "--strict-requirements",
        "-q",
    )
    strict_result.assert_outcomes(errors=1)

    invalid_config = pytester.path / "invalid.toml"
    invalid_config.write_text("not = [valid", encoding="utf-8")
    monkeypatch.setenv("XPOOL_CONFIG", str(invalid_config))
    invalid_result = pytester.runpytest(
        "-p", "xtest.harness.runner.pytest_plugin", f"--xpool-test-catalog={TEST_CATALOG_PATH}", "-q"
    )
    invalid_result.assert_outcomes(errors=1)


def test_deselected_requirement_is_not_resolved(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    pytester.makepyfile(
        """
        import xtest

        @xtest.requirements(requires_config=True)
        def test_config():
            pass

        def test_plain():
            pass
        """
    )

    result = pytester.runpytest(
        "-p", "xtest.harness.runner.pytest_plugin", f"--xpool-test-catalog={TEST_CATALOG_PATH}", "-k", "plain", "-q"
    )

    result.assert_outcomes(passed=1, deselected=1)


def test_declared_resources_and_config_are_shared_through_plan_setup_and_fixtures(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_model_id = ModelId("test/other-model")
    model_paths = tuple(pytester.path / model_id.relative_path for model_id in (TEST_MODEL_ID, other_model_id))
    for path in model_paths:
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
    config_path = write_minimal_config(pytester.path / "xpool.toml", model_path=model_paths[0])
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    payload["models"].append({"id": str(other_model_id), "path": str(model_paths[1])})
    config_path.write_text(tomli_w.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    pytester.makeini("[pytest]\ntimeout = 15\n")
    pytester.makeconftest(
        """
from unittest.mock import patch

import pytest

import xtest.harness.runner.requirements
from xpool.config import XpoolConfig

observations_key = pytest.StashKey()

@pytest.hookimpl(wrapper=True)
def pytest_runtestloop(session):
    with (
        patch.object(XpoolConfig, "from_file", wraps=XpoolConfig.from_file) as reads,
        patch.object(
            xtest.harness.runner.requirements,
            "resolve_model_weights",
            wraps=xtest.harness.runner.requirements.resolve_model_weights,
        ) as weights,
    ):
        session.config.stash[observations_key] = (reads, weights)
        return (yield)

@pytest.fixture
def observations(request):
    return request.config.stash[observations_key]
"""
    )
    directory = pytester.path / "tests/suites/integration"
    directory.mkdir(parents=True)
    program = directory / "test_configuration.py"
    program.write_text(
        f"""
import os

import pytest
import xtest
from xkit import ResourceRequirements
from xpool.model import ModelId
from xtest.harness.support.config import TEST_MODEL_ID, e2e_base_config

model_ids = (TEST_MODEL_ID, ModelId({str(other_model_id)!r}))
evaluated_rows = []

def resources(row):
    evaluated_rows.append(row)
    return ResourceRequirements(0, True, model_ids)

@pytest.mark.parametrize("row", (0, 1))
@xtest.requirements(resources)
def test_configuration(row, e2e_base_config, observations):
    reads, weights = observations
    assert evaluated_rows == [0, 1]
    assert reads.call_count == row + 1
    assert reads.call_args.kwargs["env"] is os.environ
    checks = weights.call_args_list[row * 2 : (row + 1) * 2]
    assert tuple(call.args[1] for call in checks) == model_ids
    assert all(call.args[0] is e2e_base_config.config for call in checks)
""",
        encoding="utf-8",
    )
    output = pytester.path / "plan.json"
    result = pytester.runpytest_subprocess(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={TEST_CATALOG_PATH}",
        f"--xpool-test-plan={output}",
        "--strict-requirements",
        "-q",
        str(program),
    )
    result.assert_outcomes(passed=2)
    plan = xtest.harness.runner.plan.TestPlan.read(output)
    assert len(plan.cases) == 2
    assert all(case.requirements.model_ids == (TEST_MODEL_ID, other_model_id) for case in plan.cases)


def test_collection_requires_complete_serving_graph_group(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    test_directory = pytester.path / "tests" / "suites" / "e2e"
    test_directory.mkdir(parents=True)
    (test_directory / "test_e2e_example.py").write_text(
        """
import pytest

@pytest.mark.serving_graph_group(name="example", expected_case_count=2)
@pytest.mark.parametrize("mode", ["eager", "full"])
def test_e2e_example(mode):
    pass
""",
        encoding="utf-8",
    )

    result = pytester.runpytest(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={TEST_CATALOG_PATH}",
        f"--rootdir={pytester.path}",
        "--collect-only",
        "-q",
        "-o",
        "timeout=10",
        "-k",
        "eager",
        f"--xpool-test-plan={pytester.path / 'plan.json'}",
        str(test_directory),
    )

    assert result.ret != 0
    assert "artifact groups must contain every expected case" in result.stderr.str()


def test_collection_rejects_xfail_serving_graph_case(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(reason="not supported")
        @pytest.mark.serving_graph_group(name="example", expected_case_count=2)
        def test_e2e_example():
            pass
        """
    )

    result = pytester.runpytest(
        "-p", "xtest.harness.runner.pytest_plugin", f"--xpool-test-catalog={TEST_CATALOG_PATH}", "--collect-only", "-q"
    )

    result.stderr.fnmatch_lines(["*serving graph cases cannot use xfail*"])


def test_collection_worker_writes_final_typed_item_metadata(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    test_directory = pytester.path / "tests" / "suites" / "integration"
    test_directory.mkdir(parents=True)
    (test_directory / "test_example.py").write_text(
        f"""
import pytest
import xtest
from xpool.model import ModelId

@xtest.requirements(device_count=2, requires_config=True, model_ids=(ModelId({str(TEST_MODEL_ID)!r}),))
@pytest.mark.estimated_duration(seconds=3)
@pytest.mark.timeout(12)
def test_example():
    pass
""",
        encoding="utf-8",
    )
    output = pytester.path / "plan.json"

    result = pytester.runpytest(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={TEST_CATALOG_PATH}",
        f"--rootdir={pytester.path}",
        "--collect-only",
        "-q",
        f"--xpool-test-plan={output}",
        str(test_directory),
    )

    result.assert_outcomes()
    plan = xtest.harness.runner.plan.TestPlan.read(output)
    assert len(plan.cases) == 1
    case = plan.cases[0]
    assert case.path == "tests/suites/integration/test_example.py"
    assert case.stage is xtest.harness.runner.plan.TestStage.INTEGRATION
    assert case.requirements.device_count == 2
    assert case.requirements.requires_config
    assert case.requirements.model_ids == (TEST_MODEL_ID,)
    assert case.estimated_duration_seconds == 3
    assert case.timeout_seconds == 12


def test_collection_worker_rejects_unbounded_item(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    test_directory = pytester.path / "tests" / "suites" / "unit"
    test_directory.mkdir(parents=True)
    (test_directory / "test_example.py").write_text("def test_example(): pass\n", encoding="utf-8")

    result = pytester.runpytest(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={TEST_CATALOG_PATH}",
        f"--rootdir={pytester.path}",
        "--collect-only",
        "-q",
        f"--xpool-test-plan={pytester.path / 'plan.json'}",
        str(test_directory),
    )

    result.stderr.fnmatch_lines(["*requires a timeout marker or configured pytest timeout*"])


def test_collection_worker_uses_configured_pytest_timeout(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    test_directory = pytester.path / "tests" / "suites" / "unit"
    test_directory.mkdir(parents=True)
    (test_directory / "test_example.py").write_text("def test_example(): pass\n", encoding="utf-8")
    output = pytester.path / "plan.json"

    result = pytester.runpytest(
        "-p",
        "xtest.harness.runner.pytest_plugin",
        f"--xpool-test-catalog={TEST_CATALOG_PATH}",
        f"--rootdir={pytester.path}",
        "--collect-only",
        "-q",
        "-o",
        "timeout=45",
        f"--xpool-test-plan={output}",
        str(test_directory),
    )

    result.assert_outcomes()
    (case,) = xtest.harness.runner.plan.TestPlan.read(output).cases
    assert case.timeout_seconds == 45
