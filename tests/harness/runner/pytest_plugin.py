"""Pytest hooks implementing xpool test resource requirements."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.harness.runner.artifact import ArtifactGroupRef
from tests.harness.runner.bootstrap import TestBootstrapError, ensure_test_native
from tests.harness.runner.plan import (
    CollectedTestCase,
    TestPlan,
    TestRequirements,
    TestStage,
)
from tests.harness.runner.requirements import (
    CudaRequirement,
    RequirementGuard,
    RequirementResolver,
)

requirement_resolver_key = pytest.StashKey[RequirementResolver]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register xpool test requirement command-line options."""

    parser.addoption(
        "--strict-requirements",
        action="store_true",
        default=False,
        help="fail instead of skip when a selected test resource is unavailable",
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
    """Fail cleanly before collection when mandatory native ops cannot load."""

    try:
        ensure_test_native()
    except TestBootstrapError as exc:
        pytest.exit(str(exc), returncode=2)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Validate requirement marker syntax and dependencies during collection."""

    for item in items:
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


def model_ids(item: pytest.Item) -> tuple[str, ...]:
    """Validate and return model IDs required by an item."""

    result: list[str] = []
    for marker in item.iter_markers("requires_model_weights"):
        if len(marker.args) != 1 or marker.kwargs or not isinstance(marker.args[0], str) or not marker.args[0]:
            raise pytest.UsageError(f"{item.nodeid}: requires_model_weights expects one non-empty model ID")
        result.append(marker.args[0])
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
                requirements=TestRequirements(
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
