"""Native process-runtime initialization contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.debug import native_debug_options
from xpool.runtime import RuntimeRole


def isolated_runtime_identity() -> None:
    cuda_device = torch.cuda.current_device()
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, None)
    xpool.native.initialize(RuntimeRole.INSTANCE, cuda_device, None)
    with pytest.raises(RuntimeError, match="current process was initialized as instance"):
        xpool.native.initialize(RuntimeRole.ATNAGENT, cuda_device, None)


def isolated_invalid_runtime_values() -> None:
    cuda_device = torch.cuda.current_device()
    with pytest.raises(RuntimeError, match="invalid runtime role"):
        xpool.native.initialize(99, cuda_device, None)


def isolated_negative_cuda_device() -> None:
    with pytest.raises(RuntimeError, match="non-negative CUDA device"):
        xpool.native.initialize(RuntimeRole.INSTANCE, -1, None)


def isolated_cuda_configuration_failure() -> None:
    invalid_device = torch.cuda.device_count() + 100
    with pytest.raises(RuntimeError):
        xpool.native.initialize(RuntimeRole.INSTANCE, invalid_device, native_debug_options())
    with pytest.raises(RuntimeError, match="current process was initialized as instance"):
        xpool.native.initialize(RuntimeRole.ATNAGENT, torch.cuda.current_device(), None)


def isolated_daemon_runtime() -> None:
    with pytest.raises(RuntimeError, match="daemon init requires a null CUDA device"):
        xpool.native.initialize(RuntimeRole.DAEMON, 0, None)
    with pytest.raises(RuntimeError, match="daemon initialize requires null debug options"):
        xpool.native.initialize(RuntimeRole.DAEMON, None, native_debug_options())
    xpool.native.initialize(RuntimeRole.DAEMON)
    uid = xpool.native.fabric.create_uid()
    assert isinstance(uid, str)
    assert uid
    with pytest.raises(RuntimeError, match="requires runtime role atnagent"):
        xpool.native.transport.create_arena(0, 0, 1, 2, 2, 0, 1, 0, 1)
    with pytest.raises(RuntimeError, match="requires one of the accepted runtime roles"):
        xpool.native.fabric.join(
            xpool.native.FabricJoinMetadata(uid, 0, 1, 1, 1, xpool.native.FfnSchedulerPolicy.fifo(), [])
        )


@pytest.mark.requires_cuda()
def test_runtime_initialization_is_idempotent_and_role_immutable(tmp_path: Path) -> None:
    run_native_case(isolated_runtime_identity, workdir=tmp_path / "case")


@pytest.mark.requires_cuda()
def test_runtime_rejects_invalid_role(tmp_path: Path) -> None:
    run_native_case(isolated_invalid_runtime_values, workdir=tmp_path / "case")


@pytest.mark.requires_cuda()
def test_runtime_rejects_negative_cuda_device(tmp_path: Path) -> None:
    run_native_case(isolated_negative_cuda_device, workdir=tmp_path / "case")


@pytest.mark.requires_cuda()
def test_runtime_locks_role_after_cuda_configuration_failure(tmp_path: Path) -> None:
    run_native_case(isolated_cuda_configuration_failure, workdir=tmp_path / "case")


def test_daemon_runtime_owns_uid_creation_without_cuda(tmp_path: Path) -> None:
    run_native_case(isolated_daemon_runtime, workdir=tmp_path / "case")
