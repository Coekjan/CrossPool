"""Typed native debug-option binding contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
import xtest
from xpool.config import DebugConfig, TransportObserverDebugConfig
from xpool.native import RuntimeRole
from xtest.harness.native.case import run_native_case


def isolated_default_debug_options_are_frozen() -> None:
    cuda_device = torch.cuda.current_device()
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, None)
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, None)
    with pytest.raises(RuntimeError, match="debug options differ"):
        xpool.native.initialize(
            RuntimeRole.INSTANCE,
            cuda_device,
            DebugConfig(
                transport_observer=TransportObserverDebugConfig(enable=True, outdir=Path.cwd())
            ).native_options(),
        )


def test_debug_options_are_read_only() -> None:
    options = DebugConfig().native_options()
    with pytest.raises(AttributeError):
        setattr(options.transport_observer, "enable", True)


@xtest.requirements(cuda_count=1)
def test_debug_defaults_are_frozen_by_first_initialization(tmp_path: Path) -> None:
    run_native_case(isolated_default_debug_options_are_frozen, workdir=tmp_path / "case")
