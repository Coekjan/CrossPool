"""Typed native debug-option binding contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.debug import native_debug_options
from xpool.native import RuntimeRole


def isolated_default_debug_options_are_frozen() -> None:
    cuda_device = torch.cuda.current_device()
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, None)
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, None)
    with pytest.raises(RuntimeError, match="debug options differ"):
        xpool.native.initialize(
            RuntimeRole.INSTANCE,
            cuda_device,
            native_debug_options(transport_observer=True),
        )


def test_debug_options_are_read_only() -> None:
    options = native_debug_options()
    with pytest.raises(AttributeError):
        setattr(options.transport_observer, "enable", True)


@pytest.mark.requires_cuda()
def test_debug_defaults_are_frozen_by_first_initialization(tmp_path: Path) -> None:
    run_native_case(isolated_default_debug_options_are_frozen, workdir=tmp_path / "case")
