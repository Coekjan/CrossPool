"""CUDA MPS control results and bounded client inspection."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cuda.bindings.runtime import cudaError_t  # ty: ignore[unresolved-import]

from xpool.utils import mps


def test_terminal_device_error_requires_retained_numeric_evidence() -> None:
    failure = torch.AcceleratorError("device execution failed")
    setattr(failure, "error_code", int(cudaError_t.cudaErrorLaunchFailure))
    wrapped = RuntimeError("runtime failed")
    wrapped.__cause__ = failure

    assert mps.is_terminal_device_error(failure)
    assert mps.is_terminal_device_error(wrapped)
    wrapped.__cause__ = None
    wrapped.__context__ = failure
    assert mps.is_terminal_device_error(wrapped)
    setattr(failure, "error_code", int(cudaError_t.cudaErrorInvalidValue))
    assert not mps.is_terminal_device_error(wrapped)
    assert not mps.is_terminal_device_error(torch.AcceleratorError("unspecified launch failure"))
    assert not mps.is_terminal_device_error(RuntimeError("CUDA error 719"))


@pytest.fixture
def endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> mps.MpsEndpoint:
    monkeypatch.setattr(mps, "MPS_SCOPE_DIRECTORY", tmp_path / "mps")
    monkeypatch.setattr(mps, "MPS_PROBE_LOCK_DIRECTORY", tmp_path / "locks")
    monkeypatch.setattr(mps.fcntl, "flock", lambda file_descriptor, operation: None)
    return mps.MpsEndpoint(("GPU-00000000-0000-0000-0000-000000000001",))


def test_endpoint_uses_a_private_per_user_root() -> None:
    endpoint = mps.MpsEndpoint(("GPU-00000000-0000-0000-0000-000000000001",))
    root = Path("/tmp") / f"xpool-mps-{os.getuid()}"

    assert endpoint.directory.parent == root
    assert endpoint.environment() == {
        "CUDA_MPS_PIPE_DIRECTORY": str(root / endpoint.directory.name / "pipe"),
        "CUDA_MPS_LOG_DIRECTORY": str(root / endpoint.directory.name / "log"),
    }


@pytest.fixture
def scope(endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch) -> mps.MpsScope:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")

    def launch(*args: object, **kwargs: object) -> None:
        raise RuntimeError("controller launch boundary")

    monkeypatch.setattr(mps.subprocess, "Popen", launch)
    return mps.MpsScope(endpoint)


@pytest.mark.parametrize("existing_root", [False, True])
def test_scope_creates_private_directories(scope: mps.MpsScope, existing_root: bool) -> None:
    if existing_root:
        mps.MPS_SCOPE_DIRECTORY.mkdir(mode=0o700)

    try:
        with pytest.raises(RuntimeError, match="controller launch boundary"):
            scope.start()
    finally:
        if scope.controller_log is not None:
            scope.controller_log.close()

    assert scope.endpoint.directory.parent == mps.MPS_SCOPE_DIRECTORY
    for directory in (
        mps.MPS_SCOPE_DIRECTORY,
        scope.endpoint.directory,
        scope.endpoint.pipe_directory,
        scope.endpoint.log_directory,
    ):
        facts = directory.lstat()
        assert stat.S_IMODE(facts.st_mode) == 0o700
        assert facts.st_uid == os.getuid()


@pytest.mark.parametrize("invalid", ["writable", "foreign-owner", "symlink", "file"])
def test_scope_rejects_incompatible_root_without_claiming_endpoint(
    scope: mps.MpsScope, invalid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = mps.MPS_SCOPE_DIRECTORY
    if invalid == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        root.symlink_to(target, target_is_directory=True)
    elif invalid == "file":
        root.write_text("retained", encoding="utf-8")
    else:
        root.mkdir(mode=0o700)
        if invalid == "writable":
            root.chmod(0o722)
        else:
            original_lstat = Path.lstat

            def lstat(path: Path) -> os.stat_result:
                facts = original_lstat(path)
                if path == root:
                    values = list(facts)
                    values[4] = os.getuid() + 1
                    return os.stat_result(values)
                return facts

            monkeypatch.setattr(Path, "lstat", lstat)
    original = root.lstat()

    error = FileExistsError if invalid == "file" else PermissionError
    with pytest.raises(error) as raised:
        scope.start()

    assert str(root) in str(raised.value)
    assert root.lstat() == original
    assert not scope.endpoint.directory.exists()
    assert scope.directory_identity is None
    assert scope.controller is None


def test_endpoint_preserves_rank_order_and_derives_one_address(endpoint: mps.MpsEndpoint) -> None:
    second = "GPU-00000000-0000-0000-0000-000000000002"
    forward = mps.MpsEndpoint((*endpoint.device_uuids, second))
    reverse = mps.MpsEndpoint(tuple(reversed(forward.device_uuids)))

    assert forward.directory == reverse.directory
    assert forward.device_uuids == (*endpoint.device_uuids, second)
    assert reverse.device_uuids == tuple(reversed(forward.device_uuids))
    assert forward.environment() == {
        "CUDA_MPS_PIPE_DIRECTORY": str(forward.pipe_directory),
        "CUDA_MPS_LOG_DIRECTORY": str(forward.log_directory),
    }
    environment = {**forward.environment(), "unrelated": "preserved"}
    forward.require_environment(environment)
    reverse.require_environment(environment)
    environment["CUDA_MPS_PIPE_DIRECTORY"] = "/another/pipe"
    with pytest.raises(ValueError, match="CUDA_MPS_PIPE_DIRECTORY"):
        forward.require_environment(environment)
    assert not forward.directory.exists()


@pytest.mark.parametrize(
    "identities",
    [(), ("0",), ("MIG-invalid",), ("GPU-00000000",), ("GPU-00000000-0000-0000-0000-000000000001",) * 2],
)
def test_endpoint_requires_unique_full_physical_uuids(identities: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        mps.MpsEndpoint(identities)


def test_probe_reports_missing_control_binary(endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: None)

    result = endpoint.probe()

    assert result.online is False
    assert "not installed" in result.diagnostic


@pytest.mark.parametrize(
    ("stdout", "expected_detail"),
    [
        ("not-a-number\n", "invalid"),
        ("nan\n", "out-of-range"),
        ("0\n", "out-of-range"),
        ("101\n", "out-of-range"),
        ("50.5\n", "non-integral"),
    ],
)
def test_probe_rejects_invalid_controller_output(
    endpoint: mps.MpsEndpoint,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    expected_detail: str,
) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(
        mps.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )

    result = endpoint.probe()

    assert result.online is False
    assert result.active_thread_percentage is None
    assert expected_detail in result.diagnostic


def test_probe_reports_unreachable_controller(endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(
        mps.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="controller unavailable"),
    )

    result = endpoint.probe()

    assert result.online is False
    assert "controller unavailable" in result.diagnostic


def test_probe_preserves_actual_command_timeout(endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    error = subprocess.TimeoutExpired("nvidia-cuda-mps-control", mps.MPS_STARTUP_TIMEOUT_S)

    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(mps.subprocess, "run", raise_timeout)

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        endpoint.probe()
    assert raised.value is error


def test_probe_respects_the_owner_deadline(endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps, "monotonic", lambda: 10.0)

    def command(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["timeout"] == 0.25
        return SimpleNamespace(returncode=0, stdout="100.0\n", stderr="")

    monkeypatch.setattr(mps.subprocess, "run", command)
    assert endpoint.probe(deadline=10.25).online is True
    with pytest.raises(TimeoutError, match="owner deadline"):
        endpoint.probe(deadline=10.0)


def test_probe_accepts_reachable_controller(endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(
        mps.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="100.0\n", stderr=""),
    )

    result = endpoint.probe()

    assert result.online is True
    assert result.active_thread_percentage == 100
    assert "100%" in result.diagnostic


def test_control_operation_timeout_is_capped_by_owner_deadline(
    endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mps.shutil, "which", lambda command: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps, "monotonic", lambda: 1000.0)
    timeouts: list[float] = []

    def run(command: list[str], *, input: str, timeout: float, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert input == "terminate_client 11 12\n"
        timeouts.append(timeout)
        return subprocess.CompletedProcess(command, 0, "0\n", "")

    monkeypatch.setattr(mps.subprocess, "run", run)
    assert endpoint.run_control("terminate_client 11 12", deadline=1002.0) == "0"
    assert timeouts == [2.0]


@pytest.mark.parametrize("reply", ["0", "not-a-pid", "11\n11"])
def test_invalid_process_reply_is_not_an_empty_domain(reply: str) -> None:
    with pytest.raises(RuntimeError):
        mps.MpsEndpoint.parse_process_ids(reply, allow_empty=True)


def test_process_reply_preserves_ids_and_explicit_empty_semantics() -> None:
    assert mps.MpsEndpoint.parse_process_ids("11\n12", allow_empty=False) == (11, 12)
    assert mps.MpsEndpoint.parse_process_ids("", allow_empty=True) == ()
    with pytest.raises(RuntimeError):
        mps.MpsEndpoint.parse_process_ids("", allow_empty=False)


@pytest.mark.parametrize("connected", [True, False])
def test_client_inspection_requires_this_actual_process(
    connected: bool, endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[str] = []

    def control(self: mps.MpsEndpoint, command: str, *, deadline: float) -> str:
        assert self == endpoint
        commands.append(command)
        if command == "get_server_list":
            return "41"
        assert command == "get_client_list 41"
        return str(os.getpid() if connected else os.getpid() + 1)

    monkeypatch.setattr(mps.MpsEndpoint, "run_control", control)
    if connected:
        endpoint.require_client()
    else:
        with pytest.raises(RuntimeError, match="not connected to attention MPS"):
            endpoint.require_client()
    assert commands == ["get_server_list", "get_client_list 41"]


def test_client_inspection_shares_one_query_deadline(
    endpoint: mps.MpsEndpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadlines: list[float] = []

    def control(self: mps.MpsEndpoint, command: str, *, deadline: float) -> str:
        deadlines.append(deadline)
        return "41" if command == "get_server_list" else str(os.getpid())

    monkeypatch.setattr(mps.MpsEndpoint, "run_control", control)
    endpoint.require_client()
    assert len(set(deadlines)) == 1
