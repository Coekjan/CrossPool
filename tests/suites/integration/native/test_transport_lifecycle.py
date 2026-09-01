"""Native Transport lifecycle contracts that do not require a test responder."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.debug import native_debug_options
from xpool.native import RuntimeRole

pytestmark = [pytest.mark.requires_cuda(), pytest.mark.timeout(180)]


def isolated_transport_activation_requires_joined_fabric() -> None:
    xpool.native.initialize(
        RuntimeRole.ATNAGENT,
        cuda_device=torch.cuda.current_device(),
        debug_options=native_debug_options(),
    )
    arena = xpool.native.transport.create_arena(0, 0, 8, 4, torch.bfloat16, 0, 1, 0, 1)
    try:
        with pytest.raises(RuntimeError, match="unavailable outside a joined generation"):
            xpool.native.transport.activate()
    finally:
        xpool.native.transport.destroy_arenas([arena])


def test_transport_activation_requires_joined_fabric(tmp_path: Path) -> None:
    run_native_case(isolated_transport_activation_requires_joined_fabric, workdir=tmp_path / "case")
