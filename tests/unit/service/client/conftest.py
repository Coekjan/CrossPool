"""Daemon client fixture registration."""

import pytest

from tests.harness.service.client import config
from xpool.config import init_global_config


@pytest.fixture(autouse=True)
def initialize_client_config(reset_global_config: None) -> None:
    """Install the default process-global config for client unit tests."""

    init_global_config(config=config())
