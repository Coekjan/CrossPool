"""Native debug JSON binding contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.debug import native_debug_options
from xpool.config import LoopbackSite
from xpool.runtime import RuntimeRole


def isolated_first_explicit_debug_options() -> None:
    cuda_device = torch.cuda.current_device()
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, None)
    options = native_debug_options(loopback_site=LoopbackSite.INSTANCE)
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, options)
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, options)
    with pytest.raises(RuntimeError, match="debug options differ"):
        xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, native_debug_options())


def isolated_unknown_debug_field() -> None:
    options = json.loads(native_debug_options())
    options["unknown"] = {"enable": False}
    with pytest.raises(RuntimeError, match="exactly three sections"):
        xpool.native.initialize(RuntimeRole.INSTANCE, torch.cuda.current_device(), json.dumps(options))


def isolated_debug_parse_failure() -> None:
    cuda_device = torch.cuda.current_device()
    with pytest.raises(RuntimeError, match="native debug JSON is invalid"):
        xpool.native.initialize(RuntimeRole.ATNAGENT, cuda_device, "{")
    with pytest.raises(RuntimeError, match="current process was initialized as atnagent"):
        xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, None)
    xpool.native.initialize(RuntimeRole.ATNAGENT, cuda_device, native_debug_options())


@pytest.mark.requires_cuda()
def test_debug_allows_first_explicit_configuration_after_none(tmp_path: Path) -> None:
    run_native_case(isolated_first_explicit_debug_options, workdir=tmp_path / "case")


@pytest.mark.requires_cuda()
def test_debug_binding_propagates_representative_schema_failure(tmp_path: Path) -> None:
    run_native_case(isolated_unknown_debug_field, workdir=tmp_path / "case")


@pytest.mark.requires_cuda()
def test_debug_parse_failure_preserves_runtime_role(tmp_path: Path) -> None:
    run_native_case(isolated_debug_parse_failure, workdir=tmp_path / "case")
