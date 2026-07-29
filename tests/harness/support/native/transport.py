"""Instance-side Transport setup shared by native integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

import xpool.native
from tests.harness.native.debug import native_debug_options
from xpool.config import LoopbackSite
from xpool.runtime import RuntimeRole


@pytest.fixture
def instance_transport_runtime(reset_global_config: None) -> Iterator[None]:
    """Initialize and reset one Instance-role Transport binding boundary."""

    xpool.native.initialize(RuntimeRole.INSTANCE, torch.cuda.current_device(), None)
    xpool.native.transport.detach_arena()
    yield
    xpool.native.transport.detach_arena()


def initialize_instance_transport(*, atnagent_loopback: bool) -> None:
    """Initialize one isolated Instance-role native Transport runtime."""

    debug_options = native_debug_options(loopback_site=LoopbackSite.ATNAGENT) if atnagent_loopback else None
    xpool.native.initialize(
        RuntimeRole.INSTANCE,
        cuda_device=torch.cuda.current_device(),
        debug_options=debug_options,
    )
