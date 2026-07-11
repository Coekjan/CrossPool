from __future__ import annotations

from collections.abc import Iterator

import pytest

import xpool.bootstrap as bootstrap_module
import xpool.config as config_module

pytest_plugins = ["tests.harness.pytest_plugin"]


@pytest.fixture(autouse=True)
def reset_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset the process-global xpool config around one test."""

    monkeypatch.setattr(config_module, "global_config", None)
    monkeypatch.setattr(bootstrap_module, "runtime_role", None)
    monkeypatch.setattr(bootstrap_module, "runtime_cuda_device", None)
    yield
    monkeypatch.setattr(config_module, "global_config", None)
    monkeypatch.setattr(bootstrap_module, "runtime_role", None)
    monkeypatch.setattr(bootstrap_module, "runtime_cuda_device", None)
