"""Explicit base-config fixture for parameterized SGLang E2E cases."""

from __future__ import annotations

import pytest

from tests.harness.pytest_plugin import requirement_resolver_key
from tests.harness.requirements import RequirementGuard
from xpool.config import XpoolConfig


@pytest.fixture
def e2e_base_config(request: pytest.FixtureRequest) -> XpoolConfig:
    """Return the externally supplied config after normal requirement handling."""

    resolver = request.config.stash[requirement_resolver_key]
    guard = RequirementGuard(
        strict=request.config.getoption("--strict-requirements"),
        skip=lambda reason: pytest.skip(reason),
        fail=lambda reason: pytest.fail(reason, pytrace=False),
    )
    return guard.run(resolver.require_config).config
