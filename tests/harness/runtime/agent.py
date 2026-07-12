"""Reusable Agent runtime fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import xpool.runtime.agent as agent_module


@pytest.fixture
def reset_agent_runtime(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    """Replace process-global Agent bootstrap dependencies for one test."""

    class HealthyClient:
        def close(self) -> None:
            """Close the fake client."""

    monkeypatch.setattr(agent_module.bootstrap, "init", lambda cuda_device, role: None)
    monkeypatch.setattr(agent_module.devkit, "install", lambda: None)
    monkeypatch.setattr(agent_module, "XpoolClient", HealthyClient)
    yield
