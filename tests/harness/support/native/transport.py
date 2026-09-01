"""Instance-side Transport setup shared by native integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

import xpool.native
from xpool.native import RuntimeRole


@pytest.fixture
def instance_transport_runtime(reset_global_config: None) -> Iterator[None]:
    """Initialize and reset one Instance-role Transport binding boundary."""

    xpool.native.initialize(RuntimeRole.INSTANCE, torch.cuda.current_device(), None)
    xpool.native.transport.detach_arena()
    yield
    xpool.native.transport.detach_arena()
