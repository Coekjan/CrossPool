from __future__ import annotations

import os
from pathlib import Path

import pytest

import xtest.harness.runner.ctest_launcher
from xtest.harness.runner.ctest_launcher import configure_device_visibility


def test_launch_prepends_native_dependency_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = (Path("/torch/lib"), Path("/nvshmem/lib"))
    monkeypatch.setattr(xtest.harness.runner.ctest_launcher, "native_library_paths", lambda: paths)
    monkeypatch.setenv("XPOOL_CTEST_CANONICAL", "1")
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
    environment = {"XPOOL_CTEST_CANONICAL": "1", "CUDA_VISIBLE_DEVICES": "GPU-old"}

    configure_device_visibility(environment)

    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_ctest_launcher_projects_one_device() -> None:
    identifier = "device_aaaaaaaa_bbbb_cccc_dddd_eeeeeeeeeeee"
    environment = {
        "XPOOL_CTEST_CANONICAL": "1",
        "CTEST_RESOURCE_GROUP_COUNT": "1",
        "CTEST_RESOURCE_GROUP_0": "devices",
        "CTEST_RESOURCE_GROUP_0_DEVICES": f"id:{identifier},slots:1",
    }

    configure_device_visibility(environment)

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
