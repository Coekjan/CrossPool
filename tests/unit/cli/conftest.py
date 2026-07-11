"""CLI test isolation fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolate_cli_config(reset_global_config: None) -> None:
    """Reset process-global config for every CLI test."""
