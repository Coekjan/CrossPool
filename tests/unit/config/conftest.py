"""Configuration test isolation fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolate_config_state(reset_global_config: None) -> None:
    """Reset process-global config for every configuration test."""
