"""Pytest hooks implementing CrossPool test resource requirements."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast

import pytest
from _pytest.mark.structures import Mark, MarkDecorator, ParameterSet

from xkit import ResourceRequirements
from xkit.declaration import Parameterization
from xkit.source import resolve_source_path
from xpool.model import ModelId
from xtest.harness.runner.artifact import ArtifactGroupRef
from xtest.harness.runner.bootstrap import TestBootstrapError, ensure_test_native
from xtest.harness.runner.plan import (
    CollectedTestCase,
    TestPlan,
    TestStage,
)
from xtest.harness.runner.requirements import (
    CudaRequirement,
    RequirementGuard,
    RequirementResolver,
)
from xtest.harness.sglang.catalog import E2eFfnTopologyCase, E2eServingCase, TestCatalog

requirement_resolver_key = pytest.StashKey[RequirementResolver]()
catalogue_key = pytest.StashKey[tuple[Path, TestCatalog]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CrossPool test requirement command-line options."""

    parser.addoption(
        "--strict-requirements",
        action="store_true",
        default=False,
        help="fail instead of skip when a selected test resource is unavailable",
    )
    parser.addoption(
        "--xpool-test-catalog",
        default=None,
        help="internal portable test catalogue path, defaulting to tests/tests.toml",
    )
    parser.addoption(
        "--xpool-test-plan",
        default=None,
        help="internal collection-worker Test Plan output path",
    )
    parser.addoption(
        "--xpool-task-artifact-dir",
        default=None,
        help="internal tests task artifact directory",
    )


@pytest.fixture
def task_artifact_dir(request: pytest.FixtureRequest) -> Path | None:
    """Return the runner-owned artifact directory, absent during ordinary pytest."""

    value = request.config.getoption("--xpool-task-artifact-dir")
    return Path(value) if value is not None else None


def pytest_configure(config: pytest.Config) -> None:
    """Create one resource resolver for the pytest session."""

    config.addinivalue_line("markers", "requires_config: requires XPOOL_CONFIG")
    config.addinivalue_line("markers", "requires_cuda(min_devices=1): requires CUDA resources")
    config.addinivalue_line("markers", "requires_mps: requires a responsive CUDA MPS controller")
    config.addinivalue_line("markers", "requires_model_weights(model_id): requires configured model weights")
    config.addinivalue_line("markers", "estimated_duration(seconds): estimated test runtime used by tests")
    config.addinivalue_line(
        "markers",
        "serving_graph_group(name, expected_case_count): complete cross-task serving graph group",
    )
    config.stash[requirement_resolver_key] = RequirementResolver()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Preflight native ops and load one complete portable catalogue per process."""

    try:
        ensure_test_native()
    except TestBootstrapError as exc:
        pytest.exit(str(exc), returncode=2)
    configured_path = session.config.getoption("--xpool-test-catalog")
    catalogue_path = (
        (Path(configured_path) if configured_path is not None else session.config.rootpath / "tests/tests.toml")
        .expanduser()
        .resolve()
    )
    try:
        session.config.stash[catalogue_key] = (catalogue_path, TestCatalog.load(catalogue_path))
    except (OSError, ValueError) as error:
        pytest.exit(f"xpool test catalogue failure: {error}", returncode=2)


def catalogue_cases(config: pytest.Config, path: Path) -> tuple[E2eServingCase | E2eFfnTopologyCase, ...]:
    """Select assignments by resolved source path, independently of pytest module names."""

    catalogue_path, catalogue = config.stash[catalogue_key]
    source_path = path.resolve()
    return tuple(
        case
        for case in (*catalogue.serving_cases, *catalogue.topology_cases)
        if case.module is not None and resolve_source_path(catalogue_path, case.module) == source_path
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Expand only explicitly declared parameters through native pytest machinery."""

    # Attributes are the shared decorators' metadata boundary; functions stay unwrapped.
    declaration = cast(Parameterization | None, getattr(metafunc.function, "xpool_parameters", None))
    if declaration is None:
        return
    if declaration.values is None:
        cases = catalogue_cases(metafunc.config, metafunc.definition.path)
        if not cases:
            raise pytest.UsageError(f"{metafunc.definition.nodeid}: bound entry has no catalogue-assigned cases")
        if declaration.rows is None:
            values = tuple(
                pytest.param(
                    case,
                    id=case.id,
                    marks=(
                        pytest.mark.timeout(case.timeout_seconds),
                        pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
                    ),
                )
                for case in cases
            )
        inputs: tuple[object, ...] = cases
    else:
        inputs = declaration.values
        values = declaration.values
    if declaration.rows is not None:
        expanded: list[object] = []
        for input_index, value in enumerate(inputs):
            group_name = f"{metafunc.definition.nodeid}[case-{input_index}]"
            for row in declaration.rows(value):
                if isinstance(row, ParameterSet):
                    marks: list[Mark | MarkDecorator] = []
                    for marker in row.marks:
                        mark = marker.mark if isinstance(marker, MarkDecorator) else marker
                        if mark.name == "serving_graph_group" and "name" not in mark.kwargs:
                            marks.append(pytest.mark.serving_graph_group(*mark.args, name=group_name, **mark.kwargs))
                        else:
                            marks.append(marker)
                    row = pytest.param(*row.values, id=row.id, marks=marks)
                expanded.append(row)
        values = tuple(expanded)
    names = declaration.argument_names[0] if len(declaration.argument_names) == 1 else declaration.argument_names
    metafunc.parametrize(names, values)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """Validate the collected entry even when a referenced module produces no items."""

    report = yield
    if (
        not isinstance(collector, pytest.Module)
        or not report.passed
        or not catalogue_cases(collector.config, collector.path)
    ):
        return report
    entries = set()
    for item in report.result:
        if not isinstance(item, pytest.Function) or item.obj.__module__ != collector.obj.__name__:
            continue
        declaration = cast(Parameterization | None, getattr(item.obj, "xpool_parameters", None))
        if declaration is not None and declaration.values is None:
            entries.add(item.obj)
    if len(entries) == 1:
        return report
    return pytest.CollectReport(
        collector.nodeid,
        "failed",
        longrepr=f"{collector.path}: expected one catalogue-bound entry, found {len(entries)}",
        result=[],
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Adapt declared resources before marker deselection and validate merged marks."""

    for item in items:
        if isinstance(item, pytest.Function):
            declaration = cast(
                ResourceRequirements | Callable[..., ResourceRequirements] | None,
                getattr(item.obj, "xpool_requirements", None),
            )
            if declaration is not None:
                if isinstance(declaration, ResourceRequirements):
                    resources = declaration
                else:
                    try:
                        resources = declaration(**item.callspec.params) if hasattr(item, "callspec") else declaration()
                    except (TypeError, ValueError) as error:
                        raise pytest.UsageError(f"{item.nodeid}: invalid declared resources: {error}") from error
                if resources.cuda_count:
                    item.add_marker(pytest.mark.requires_cuda(min_devices=resources.cuda_count))
                if resources.requires_config:
                    item.add_marker(pytest.mark.requires_config)
                if resources.requires_mps:
                    item.add_marker(pytest.mark.requires_mps)
                for model_id in resources.model_ids:
                    item.add_marker(pytest.mark.requires_model_weights(model_id))
        config_markers = list(item.iter_markers("requires_config"))
        weight_markers = list(item.iter_markers("requires_model_weights"))
        if weight_markers and not config_markers:
            raise pytest.UsageError(f"{item.nodeid}: requires_model_weights must be paired with requires_config")
        for marker in config_markers:
            if marker.args or marker.kwargs:
                raise pytest.UsageError(f"{item.nodeid}: requires_config accepts no arguments")
        for marker in item.iter_markers("requires_mps"):
            if marker.args or marker.kwargs:
                raise pytest.UsageError(f"{item.nodeid}: requires_mps accepts no arguments")
        cuda_requirement(item)
        model_ids(item)
        estimated_duration(item)
        serving_graph_group_ref = serving_graph_group(item)
        if serving_graph_group_ref is not None and list(item.iter_markers("xfail")):
            raise pytest.UsageError(f"{item.nodeid}: serving graph cases cannot use xfail")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Resolve resource requirements for one selected test before fixtures."""

    resolver = item.config.stash[requirement_resolver_key]
    guard = RequirementGuard(
        strict=item.config.getoption("--strict-requirements"),
        skip=lambda reason: pytest.skip(reason),
        fail=lambda reason: pytest.fail(reason, pytrace=False),
    )
    operations: list[Callable[[], object]] = []
    if list(item.iter_markers("requires_cuda")):
        requirement = cuda_requirement(item)
        operations.append(lambda: resolver.require_cuda(requirement))
    if list(item.iter_markers("requires_config")):
        operations.append(resolver.require_config)
    if list(item.iter_markers("requires_mps")):
        operations.append(resolver.require_mps)
    operations.extend(
        lambda model_id=model_id: resolver.require_model_weights(model_id) for model_id in model_ids(item)
    )
    for operation in operations:
        guard.run(operation)


def cuda_requirement(item: pytest.Item) -> CudaRequirement:
    """Merge all CUDA markers attached to an item."""

    min_devices = 1
    for marker in item.iter_markers("requires_cuda"):
        if marker.args:
            raise pytest.UsageError(f"{item.nodeid}: requires_cuda accepts keyword arguments only")
        unknown = set(marker.kwargs) - {"min_devices"}
        if unknown:
            raise pytest.UsageError(f"{item.nodeid}: unknown requires_cuda arguments: {sorted(unknown)}")
        marker_min_devices = marker.kwargs.get("min_devices", 1)
        if isinstance(marker_min_devices, bool) or not isinstance(marker_min_devices, int) or marker_min_devices < 1:
            raise pytest.UsageError(f"{item.nodeid}: requires_cuda min_devices must be a positive integer")
        min_devices = max(min_devices, marker_min_devices)
    return CudaRequirement(min_devices=min_devices)


def model_ids(item: pytest.Item) -> tuple[ModelId, ...]:
    """Validate and return model IDs required by an item."""

    result: list[ModelId] = []
    for marker in item.iter_markers("requires_model_weights"):
        if len(marker.args) != 1 or marker.kwargs:
            raise pytest.UsageError(f"{item.nodeid}: requires_model_weights expects one namespace/name model ID")
        try:
            result.append(ModelId.model_validate(marker.args[0]))
        except ValueError as error:
            raise pytest.UsageError(
                f"{item.nodeid}: requires_model_weights expects one namespace/name model ID"
            ) from error
    return tuple(dict.fromkeys(result))


def estimated_duration(item: pytest.Item) -> float | None:
    """Return the closest validated scheduling estimate."""

    marker = item.get_closest_marker("estimated_duration")
    if marker is None:
        return None
    if marker.args or set(marker.kwargs) != {"seconds"}:
        raise pytest.UsageError(f"{item.nodeid}: estimated_duration expects seconds=<positive number>")
    return positive_number(item, "estimated_duration", marker.kwargs["seconds"])


def serving_graph_group(item: pytest.Item) -> ArtifactGroupRef | None:
    """Return the closest validated complete serving-graph group reference."""

    marker = item.get_closest_marker("serving_graph_group")
    if marker is None:
        return None
    if marker.args or set(marker.kwargs) != {"name", "expected_case_count"}:
        raise pytest.UsageError(
            f"{item.nodeid}: serving_graph_group expects name=<non-empty string>, expected_case_count=<integer>"
        )
    name = marker.kwargs["name"]
    expected_case_count = marker.kwargs["expected_case_count"]
    if not isinstance(name, str) or not name:
        raise pytest.UsageError(f"{item.nodeid}: serving_graph_group expects name=<non-empty string>")
    if not isinstance(expected_case_count, int) or isinstance(expected_case_count, bool) or expected_case_count < 2:
        raise pytest.UsageError(f"{item.nodeid}: serving_graph_group expected_case_count must be at least two")
    return ArtifactGroupRef(kind="serving_graph", name=name, expected_case_count=expected_case_count)


def timeout_seconds(item: pytest.Item) -> float:
    """Return the closest pytest-timeout deadline or reject an unbounded worker item."""

    marker = item.get_closest_marker("timeout")
    if marker is not None:
        if len(marker.args) == 1 and not marker.kwargs:
            return positive_number(item, "timeout", marker.args[0])
        if not marker.args and "seconds" in marker.kwargs:
            return positive_number(item, "timeout", marker.kwargs["seconds"])
        raise pytest.UsageError(f"{item.nodeid}: timeout must provide one positive seconds value")
    configured = item.config.getoption("timeout", default=None)
    if configured is None:
        configured = item.config.getini("timeout")
    if configured is None or configured == 0 or configured == "0" or configured == "":
        raise pytest.UsageError(
            f"{item.nodeid}: collection worker requires a timeout marker or configured pytest timeout"
        )
    try:
        configured_seconds = float(configured)
    except (TypeError, ValueError) as error:
        raise pytest.UsageError(f"{item.nodeid}: configured pytest timeout must be a positive number") from error
    if configured_seconds <= 0:
        raise pytest.UsageError(f"{item.nodeid}: configured pytest timeout must be a positive number")
    return configured_seconds


def positive_number(item: pytest.Item, marker_name: str, value: object) -> float:
    """Validate one positive non-boolean marker number."""

    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise pytest.UsageError(f"{item.nodeid}: {marker_name} seconds must be a positive number")
    return float(value)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Atomically publish final concrete items for an isolated collection worker."""

    output = session.config.getoption("--xpool-test-plan")
    if output is None:
        return
    root = Path(session.config.rootpath).resolve()
    cases: list[CollectedTestCase] = []
    for item in session.items:
        try:
            path = item.path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise pytest.UsageError(f"{item.nodeid}: collected path is outside repository root") from error
        cuda = cuda_requirement(item)
        cases.append(
            CollectedTestCase(
                path=path,
                nodeid=item.nodeid,
                stage=TestStage.from_path(path),
                requirements=ResourceRequirements(
                    cuda_count=cuda.min_devices if list(item.iter_markers("requires_cuda")) else 0,
                    requires_mps=bool(list(item.iter_markers("requires_mps"))),
                    requires_config=bool(list(item.iter_markers("requires_config"))),
                    model_ids=model_ids(item),
                ),
                estimated_duration_seconds=estimated_duration(item),
                timeout_seconds=timeout_seconds(item),
                artifact_group=serving_graph_group(item),
            )
        )
    try:
        TestPlan(tuple(cases)).write(Path(output))
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
