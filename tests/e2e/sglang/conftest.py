"""Shared resource fixture for SGLang end-to-end tests."""

import pytest
import torch

from tests.harness.pytest_plugin import requirement_resolver_key
from tests.harness.requirements import RequirementGuard
from tests.harness.sglang.environment import SglangTestEnvironment, create_sglang_test_environment
from tests.harness.sglang.offline_probe import MODEL_ID


@pytest.fixture(scope="module")
def sglang_environment(request: pytest.FixtureRequest) -> SglangTestEnvironment:
    """Create one SGLang environment from plugin-validated resources."""

    resolver = request.config.stash[requirement_resolver_key]
    guard = RequirementGuard(
        strict=request.config.getoption("--strict-requirements"),
        skip=lambda reason: pytest.skip(reason),
        fail=lambda reason: pytest.fail(reason, pytrace=False),
    )
    resolved_config = guard.run(resolver.require_config)
    resolved_weights = guard.run(lambda: resolver.require_model_weights(MODEL_ID))
    return guard.run(
        lambda: create_sglang_test_environment(
            resolved_config=resolved_config,
            resolved_weights=resolved_weights,
            device_count=torch.cuda.device_count(),
        )
    )
