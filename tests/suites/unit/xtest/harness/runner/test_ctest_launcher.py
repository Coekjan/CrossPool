from __future__ import annotations

import os
from pathlib import Path

import pytest

import xtest.harness.runner.ctest_launcher
from xtest.envs import CTEST_CANONICAL, CTEST_INFRASTRUCTURE_SENTINEL
from xtest.harness.runner.ctest_launcher import configure_device_visibility


def test_launch_prepends_native_dependency_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = (Path("/torch/lib"), Path("/nvshmem/lib"))
    monkeypatch.setattr(xtest.harness.runner.ctest_launcher, "native_library_paths", lambda: paths)
    monkeypatch.setenv(CTEST_CANONICAL.name, "1")
    monkeypatch.setenv("CTEST_RESOURCE_GROUP_COUNT", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    captured: tuple[str, list[str], dict[str, str]] | None = None

    def capture(executable: str, arguments: list[str], environment: dict[str, str]) -> None:
        nonlocal captured
        captured = executable, arguments, environment
        raise SystemExit

    monkeypatch.setattr(os, "execvpe", capture)
    with pytest.raises(SystemExit):
        xtest.harness.runner.ctest_launcher.launch(["native-test", "--gtest_filter=Suite.Case"])

    assert captured is not None
    executable, arguments, environment = captured
    assert executable == "native-test"
    assert arguments == ["native-test", "--gtest_filter=Suite.Case"]
    assert environment["LD_LIBRARY_PATH"] == "/torch/lib:/nvshmem/lib:/existing"
    assert "DEVICE ASSIGNMENT devices=none" in capsys.readouterr().out


def test_launch_requires_an_executable() -> None:
    with pytest.raises(RuntimeError, match="requires an executable"):
        xtest.harness.runner.ctest_launcher.launch([])


def test_ctest_launcher_hides_cuda_from_canonical_cpu_test() -> None:
    environment = {"CUDA_VISIBLE_DEVICES": "GPU-old"}
    CTEST_CANONICAL.set("1", env=environment)

    configure_device_visibility(environment)

    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_ctest_launcher_projects_one_device() -> None:
    identifier = "device_aaaaaaaa_bbbb_cccc_dddd_eeeeeeeeeeee"
    environment = {
        "CTEST_RESOURCE_GROUP_COUNT": "1",
        "CTEST_RESOURCE_GROUP_0": "devices",
        "CTEST_RESOURCE_GROUP_0_DEVICES": f"id:{identifier},slots:1",
    }
    CTEST_CANONICAL.set("1", env=environment)

    configure_device_visibility(environment)

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.mark.parametrize("canonical", [None, "0", "true", " 1 "])
def test_noncanonical_launcher_preserves_device_visibility(canonical: str | None) -> None:
    environment = {"CUDA_VISIBLE_DEVICES": "GPU-visible"}
    if canonical is not None:
        CTEST_CANONICAL.set(canonical, env=environment)

    configure_device_visibility(environment)

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-visible"


@pytest.mark.parametrize("sentinel", [None, "failure.txt"])
def test_launcher_reports_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], sentinel: str | None
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CTEST_INFRASTRUCTURE_SENTINEL.name, raising=False)
    if sentinel is not None:
        monkeypatch.setenv(CTEST_INFRASTRUCTURE_SENTINEL.name, sentinel)
    monkeypatch.setenv(CTEST_CANONICAL.name, "1")
    monkeypatch.setenv("CTEST_RESOURCE_GROUP_COUNT", "2")

    with pytest.raises(SystemExit) as error:
        xtest.harness.runner.ctest_launcher.launch(["native-test"])

    assert error.value.code == 125
    diagnostic = capsys.readouterr().err
    assert "exactly one CTest devices resource group" in diagnostic
    if sentinel is not None:
        assert (tmp_path / sentinel).read_text(encoding="utf-8") == diagnostic


def test_launcher_preserves_sentinel_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(CTEST_INFRASTRUCTURE_SENTINEL.name, str(tmp_path / "missing" / "failure.txt"))

    with pytest.raises(FileNotFoundError):
        xtest.harness.runner.ctest_launcher.report_infrastructure_failure("resource unavailable")

    assert capsys.readouterr().err == "resource unavailable\n"
