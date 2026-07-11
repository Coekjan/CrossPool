"""Pytest hooks implementing xpool test resource requirements."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.harness.bootstrap import TestBootstrapError, ensure_test_native_ops
from tests.harness.requirements import (
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


def pytest_configure(config: pytest.Config) -> None:
    """Create one resource resolver for the pytest session."""

    config.addinivalue_line("markers", "requires_config: requires XPOOL_CONFIG")
    config.addinivalue_line("markers", "requires_cuda(min_devices=1, bf16=False): requires CUDA resources")
    config.addinivalue_line("markers", "requires_model_weights(model_id): requires configured model weights")
    config.stash[requirement_resolver_key] = RequirementResolver()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail cleanly before collection when mandatory native ops cannot load."""

    try:
        ensure_test_native_ops()
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
        cuda_requirement(item)
        model_ids(item)


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
    operations.extend(
        lambda model_id=model_id: resolver.require_model_weights(model_id) for model_id in model_ids(item)
    )
    for operation in operations:
        guard.run(operation)


def cuda_requirement(item: pytest.Item) -> CudaRequirement:
    """Merge all CUDA markers attached to an item."""

    min_devices = 1
    bf16 = False
    for marker in item.iter_markers("requires_cuda"):
        if marker.args:
            raise pytest.UsageError(f"{item.nodeid}: requires_cuda accepts keyword arguments only")
        unknown = set(marker.kwargs) - {"min_devices", "bf16"}
        if unknown:
            raise pytest.UsageError(f"{item.nodeid}: unknown requires_cuda arguments: {sorted(unknown)}")
        marker_min_devices = marker.kwargs.get("min_devices", 1)
        marker_bf16 = marker.kwargs.get("bf16", False)
        if isinstance(marker_min_devices, bool) or not isinstance(marker_min_devices, int) or marker_min_devices < 1:
            raise pytest.UsageError(f"{item.nodeid}: requires_cuda min_devices must be a positive integer")
        if not isinstance(marker_bf16, bool):
            raise pytest.UsageError(f"{item.nodeid}: requires_cuda bf16 must be a boolean")
        min_devices = max(min_devices, marker_min_devices)
        bf16 = bf16 or marker_bf16
    return CudaRequirement(min_devices=min_devices, bf16=bf16)


def model_ids(item: pytest.Item) -> tuple[str, ...]:
    """Validate and return model IDs required by an item."""

    result: list[str] = []
    for marker in item.iter_markers("requires_model_weights"):
        if len(marker.args) != 1 or marker.kwargs or not isinstance(marker.args[0], str) or not marker.args[0]:
            raise pytest.UsageError(f"{item.nodeid}: requires_model_weights expects one non-empty model ID")
        result.append(marker.args[0])
    return tuple(result)
