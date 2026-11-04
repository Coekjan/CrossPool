from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from xpool.memory import (
    FfnMemoryCalibration,
    FfnMemoryCalibrationCoefficients,
    MemoryCalibrationEnvironment,
    XpoolMemoryCalibrationProfile,
)
from xpool.runtime import memory_calibration
from xtest.harness.support.config import synthetic_memory_context


@pytest.fixture
def observed_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    inputs: dict[str, str] = {}
    monkeypatch.setattr(
        memory_calibration,
        "os",
        SimpleNamespace(environ=inputs, getuid=lambda: 0, fsencode=os.fsencode, fsdecode=os.fsdecode),
    )
    monkeypatch.setattr(
        memory_calibration, "pwd", SimpleNamespace(getpwuid=lambda uid: SimpleNamespace(pw_dir="/user"))
    )
    monkeypatch.setattr(memory_calibration, "Path", lambda path: tmp_path / Path(path).name)
    monkeypatch.setattr(memory_calibration, "cuda_versions", lambda: (13030, 13030))
    monkeypatch.setattr(
        memory_calibration,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(
                    preferred_blas_library=lambda: "cublas",
                    cublas_workspace_size=lambda: 1024,
                    cublaslt_workspace_size=lambda: 1024,
                )
            ),
            cuda=SimpleNamespace(
                get_device_properties=lambda device: SimpleNamespace(
                    name="observed-device", major=8, minor=0, total_memory=1234
                ),
                nccl=SimpleNamespace(version=lambda: (2, 29, 7)),
                memory=SimpleNamespace(get_allocator_backend=lambda: "native"),
            ),
        ),
    )
    return inputs


def test_observation_records_inputs_without_interpreting_or_mutating_them(
    observed_inputs: dict[str, str], tmp_path: Path
) -> None:
    observed_inputs.update(
        NVSHMEM_SYMMETRIC_SIZE="2G",
        NVSHMEM_G_BUF_SIZE="4194304",
        NCCL_ENV_PLUGIN="custom",
        CUBLAS_WORKSPACE_CONFIG=":16:8",
        CUDA_MODULE_LOADING="EAGER",
    )
    contents = b"NVSHMEM_DISABLE_NCCL=1\n"
    (tmp_path / "nvshmem.conf").write_bytes(contents)
    before = observed_inputs.copy()
    observed = memory_calibration.observe_environment(1)
    inputs = dict(observed.configuration_inputs)
    assert observed.name == "observed-device"
    assert observed.total_memory_bytes == 1234
    assert observed.nccl_version == (2, 29, 7)
    assert inputs["environment.NVSHMEM_SYMMETRIC_SIZE"] == "2G"
    assert inputs["environment.NVSHMEM_G_BUF_SIZE"] == "4194304"
    assert inputs["environment.NCCL_ENV_PLUGIN"] == "custom"
    assert inputs["environment.CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert inputs["environment.CUDA_MODULE_LOADING"] == "EAGER"
    assert inputs["file.nvshmem.system"] == hashlib.sha256(contents).hexdigest()
    assert inputs["file.nccl.system"] is None
    assert observed_inputs == before
    assert MemoryCalibrationEnvironment.model_validate(observed.model_dump(mode="json")) == observed


def test_unreadable_configuration_is_distinct_from_absent_configuration(
    monkeypatch: pytest.MonkeyPatch, observed_inputs: dict[str, str], tmp_path: Path
) -> None:
    absent = memory_calibration.observe_environment(1)
    (tmp_path / "nvshmem.conf").write_bytes(b"NVSHMEM_DISABLE_NCCL=1\n")

    def fail_read() -> bytes:
        raise PermissionError("configuration read denied")

    path_factory = memory_calibration.Path

    def input_path(path: str) -> Path:
        if Path(path).name == "nvshmem.conf":
            return cast(Path, SimpleNamespace(read_bytes=fail_read))
        return path_factory(path)

    monkeypatch.setattr(memory_calibration, "Path", input_path)
    assert dict(absent.configuration_inputs)["file.nvshmem.system"] is None
    with pytest.raises(RuntimeError, match=r"device 1: file.nvshmem.system.*configuration read denied"):
        memory_calibration.observe_environment(1)


@pytest.mark.parametrize("variable", ("NVSHMEM_CONF_FILE", "NCCL_CONF_FILE"))
def test_profile_reuse_follows_selected_configuration_contents(
    observed_inputs: dict[str, str], tmp_path: Path, variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original.conf"
    relocated = tmp_path / "relocated.conf"
    original.write_bytes(b"BUFFER_SIZE=4194304\n")
    relocated.write_bytes(original.read_bytes())
    observed_inputs[variable] = str(original)
    environment = memory_calibration.observe_environment(1)
    context = synthetic_memory_context()
    assert context.participants is not None
    context = context.model_copy(update={"participants": (environment, *context.participants[1:])})
    profile = XpoolMemoryCalibrationProfile(
        memory_context=context,
        ffn=FfnMemoryCalibration(
            executor_lane_count=1,
            minimum_held_out_headroom_bytes=0,
            coefficients=FfnMemoryCalibrationCoefficients(
                base_bytes=0,
                bytes_per_tensor_storage_mib=0,
                bytes_per_tensor_storage_allocation=0,
                bytes_per_dense_graph_capture=0,
                bytes_per_moe_graph_capture=0,
                bytes_per_executor_lane=0,
                bytes_per_compute_branch=0,
                dense_implementation_bytes=0,
                moe_implementation_bytes=0,
                joined_ffnagent_bytes=0,
            ),
        ),
    )
    assert context.participants is not None
    observed_inputs[variable] = str(relocated)
    relocated_environment = memory_calibration.observe_environment(1)
    relocated_context = context.model_copy(update={"participants": (relocated_environment, *context.participants[1:])})
    profile.require_compatible(context=relocated_context, executor_lane_count=1)
    assert observed_inputs[variable] == str(relocated)

    before = observed_inputs.copy()
    for query, changed_value in (
        ("preferred_blas_library", "cublaslt"),
        ("cublas_workspace_size", 2048),
        ("cublaslt_workspace_size", 2048),
    ):
        with monkeypatch.context() as settings:
            settings.setattr(memory_calibration.torch.backends.cuda, query, lambda: changed_value)
            changed_environment = memory_calibration.observe_environment(1)
            with pytest.raises(RuntimeError, match="incompatible with the current Fabric memory context"):
                profile.require_compatible(
                    context=context.model_copy(
                        update={"participants": (changed_environment, *context.participants[1:])}
                    ),
                    executor_lane_count=1,
                )
        assert observed_inputs == before

    relocated.write_bytes(b"BUFFER_SIZE=8388608\n")
    changed = memory_calibration.observe_environment(1)
    with pytest.raises(RuntimeError, match="incompatible with the current Fabric memory context"):
        profile.require_compatible(
            context=context.model_copy(update={"participants": (changed, *context.participants[1:])}),
            executor_lane_count=1,
        )


def test_missing_uid_fails_unless_nccl_configuration_is_selected(
    monkeypatch: pytest.MonkeyPatch, observed_inputs: dict[str, str]
) -> None:
    def missing_user(uid: int) -> None:
        raise KeyError(uid)

    monkeypatch.setattr(memory_calibration.pwd, "getpwuid", missing_user)
    with pytest.raises(RuntimeError, match=r"file.nccl.user.*passwd lookup"):
        memory_calibration.observe_environment(1)
    observed_inputs["NCCL_CONF_FILE"] = "/selected.conf"
    selected = memory_calibration.observe_environment(1)
    assert dict(selected.configuration_inputs)["file.nccl.selected"] is None


def test_diagnostic_settings_do_not_change_matching_inputs(observed_inputs: dict[str, str]) -> None:
    observed_inputs.update(NCCL_DEBUG_FILE="/first/log", NCCL_TOPO_DUMP_FILE="/first/dump")
    before = memory_calibration.observe_environment(1)
    observed_inputs.update(
        CUBLAS_LOGINFO_DBG="1",
        CUBLAS_LOGDEST_DBG="/other/blas.log",
        NCCL_DEBUG_FILE="/other/log",
        NCCL_TOPO_DUMP_FILE="/other/dump",
    )
    assert memory_calibration.observe_environment(1) == before


def test_topology_identity_follows_contents_across_paths(observed_inputs: dict[str, str], tmp_path: Path) -> None:
    original = tmp_path / "original.xml"
    relocated = tmp_path / "relocated.xml"
    original.write_bytes(b"<topology/>\n")
    relocated.write_bytes(original.read_bytes())
    observed_inputs["NCCL_TOPO_FILE"] = str(original)
    before = memory_calibration.observe_environment(1)
    observed_inputs["NCCL_TOPO_FILE"] = str(relocated)
    assert memory_calibration.observe_environment(1) == before
    relocated.write_bytes(b"<changed/>\n")
    assert memory_calibration.observe_environment(1) != before
    inputs = dict(before.configuration_inputs)
    assert "environment.NCCL_TOPO_FILE" not in inputs
    assert inputs["file.nccl.topology"] == hashlib.sha256(original.read_bytes()).hexdigest()


@pytest.mark.parametrize("selection", ("environment", "selected", "user", "system", "default"))
def test_topology_selection_respects_nccl_precedence(
    observed_inputs: dict[str, str],
    tmp_path: Path,
    selection: str,
) -> None:
    for name in ("environment", "selected", "user", "system", "virtualTopology"):
        (tmp_path / f"{name}.xml").write_text(name)
    (tmp_path / "selected.conf").write_text("NCCL_TOPO_FILE=selected.xml\n")
    (tmp_path / ".nccl.conf").write_text(
        "# NCCL_TOPO_FILE=ignored.xml\n NCCL_TOPO_FILE=ignored.xml\n"
        "NCCL_TOPO_FILE=user.xml\nNCCL_TOPO_FILE=ignored.xml\n"
    )
    (tmp_path / "nccl.conf").write_text("NCCL_TOPO_FILE=system.xml\n")
    if selection in ("environment", "selected"):
        observed_inputs["NCCL_CONF_FILE"] = str(tmp_path / "selected.conf")
    if selection == "environment":
        observed_inputs["NCCL_TOPO_FILE"] = "environment.xml"
    if selection in ("system", "default"):
        (tmp_path / ".nccl.conf").write_bytes(b"")
    if selection == "default":
        (tmp_path / "nccl.conf").write_bytes(b"")
    expected = "virtualTopology" if selection == "default" else selection
    observed = memory_calibration.observe_environment(1)
    assert dict(observed.configuration_inputs)["file.nccl.topology"] == hashlib.sha256(expected.encode()).hexdigest()


@pytest.mark.parametrize("name", ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING"))
def test_allocator_configuration_is_recorded_for_profile_reuse(observed_inputs: dict[str, str], name: str) -> None:
    absent = memory_calibration.observe_environment(0)
    observed_inputs[name] = ""
    empty = memory_calibration.observe_environment(0)
    observed_inputs[name] = "backend:native" if name.endswith("ALLOC_CONF") else "0"
    configured = memory_calibration.observe_environment(0)
    key = f"environment.{name}"
    assert dict(absent.configuration_inputs)[key] is None
    assert dict(empty.configuration_inputs)[key] == ""
    assert dict(configured.configuration_inputs)[key] == observed_inputs[name]
    assert absent != empty != configured
