from __future__ import annotations

import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

import xpool.memory
from xpool.config import XpoolConfig
from xpool.memory import (
    ExecutionResourceDeclaration,
    FfnMemoryCalibration,
    FfnMemoryCalibrationCoefficients,
    MemoryCalibrationDevice,
    MemoryCalibrationEnvironment,
    XpoolMemoryCalibrationProfile,
)
from xpool.native import ABI_VERSION
from xtest.harness.support.config import (
    TEST_MODEL_ID,
    install_test_config,
    lane_resource_declaration,
    reset_global_config,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def profile() -> XpoolMemoryCalibrationProfile:
    """Build one strict Profile matching the installed software."""

    return XpoolMemoryCalibrationProfile(
        environment=MemoryCalibrationEnvironment(
            native_abi_version=ABI_VERSION,
            ffnagent_devices=(
                MemoryCalibrationDevice(
                    name="NVIDIA A100-SXM4-40GB",
                    compute_capability=(8, 0),
                    total_memory_bytes=40 * 1024**3,
                    uuid="GPU-test",
                ),
            ),
            cuda_driver_version=13030,
            cuda_runtime_version=13030,
            torch_version=importlib.metadata.version("torch"),
            triton_version=importlib.metadata.version("triton"),
            sglang_version=importlib.metadata.version("sglang"),
            sglang_kernel_version=importlib.metadata.version("sglang-kernel"),
            nvshmem_version=importlib.metadata.version("nvidia-nvshmem-cu13"),
        ),
        ffn=FfnMemoryCalibration(
            atnagent_count=1,
            executor_lane_count=1,
            minimum_held_out_headroom_bytes=1,
            coefficients=FfnMemoryCalibrationCoefficients(
                base_bytes=1,
                bytes_per_tensor_storage_mib=2,
                bytes_per_tensor_storage_allocation=3,
                bytes_per_dense_graph_capture=4,
                bytes_per_moe_graph_capture=5,
                bytes_per_executor_lane=6,
                bytes_per_compute_branch=7,
                dense_implementation_bytes=8,
                moe_implementation_bytes=9,
                joined_ffnagent_bytes=10,
            ),
        ),
    )


def install_config(path: Path | None) -> None:
    """Install one single-agent config with an optional Profile path."""

    ffn: dict[str, object] = {"devices": [1]}
    payload: dict[str, object] = {
        "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
        "atn": {"devices": [0]},
        "ffn": ffn,
        "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
    }
    if path is not None:
        ffn["device_memory_calibration"] = str(path)
    install_test_config(XpoolConfig.from_mapping(payload))


def test_profile_json_rejects_string_for_numeric_field() -> None:
    payload = profile().model_dump(mode="json")
    payload["ffn"]["coefficients"]["base_bytes"] = "1"

    with pytest.raises(ValidationError) as error:
        XpoolMemoryCalibrationProfile.model_validate_json(json.dumps(payload))
    assert any(item["loc"][-1] == "base_bytes" for item in error.value.errors())


def test_absent_profile_selects_analytic_admission_without_host_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_config(None)

    assert xpool.memory.load_memory_calibration_profile() is None


def test_load_accepts_matching_host_and_config_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "memory.json"
    path.write_text(profile().model_dump_json(), encoding="utf-8")
    install_config(path)
    monkeypatch.setattr(xpool.memory, "cuda_versions", lambda: (13030, 13030))

    assert xpool.memory.load_memory_calibration_profile() == profile()


def test_configured_incompatible_profile_fails_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "memory.json"
    incompatible = profile().model_copy(update={"ffn": profile().ffn.model_copy(update={"executor_lane_count": 2})})
    path.write_text(incompatible.model_dump_json(), encoding="utf-8")
    install_config(path)
    monkeypatch.setattr(xpool.memory, "cuda_versions", lambda: (13030, 13030))

    with pytest.raises(RuntimeError, match="Executor Lane count"):
        xpool.memory.load_memory_calibration_profile()


def test_writer_serializes_profile_to_the_configured_path(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text("old", encoding="utf-8")
    install_config(path)
    expected = profile()
    xpool.memory.write_memory_calibration_profile(expected)

    assert XpoolMemoryCalibrationProfile.model_validate_json(path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    ("capacity", "alignment", "aliases"),
    (
        (0, 256, ("scratch",)),
        (1024, 3, ("scratch",)),
        (1024, 256, ("other",)),
        (1024, 256, ("scratch", "scratch")),
    ),
)
def test_resource_declaration_rejects_unusable_backing(capacity: int, alignment: int, aliases: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        lane_resource_declaration(key="scratch", max_bytes=capacity, alignment_bytes=alignment, aliases=aliases)


@pytest.mark.parametrize(
    "change",
    (
        {"request_bytes": ()},
        {"request_bytes": (0,)},
        {"alignment_bytes": 512},
        {"exact_logical_bytes": 1025},
        {"request_bytes": (1,), "exact_logical_bytes": 1024},
    ),
)
def test_requirement_rejects_incoherent_contract(change: dict[str, object]) -> None:
    requirement = lane_resource_declaration(
        key="scratch", max_bytes=1024, alignment_bytes=256, aliases=("scratch",)
    ).requirement
    with pytest.raises(ValueError):
        replace(requirement, **change)


@pytest.mark.parametrize("uses", ([], ["dense", "dense"], ["unknown"]))
def test_resource_requires_distinct_known_computation_sites(uses: list[str]) -> None:
    declaration = lane_resource_declaration(key="scratch", max_bytes=1024, alignment_bytes=256, aliases=("scratch",))
    values = declaration.model_dump(mode="json")
    values["uses"] = uses
    with pytest.raises(ValidationError):
        ExecutionResourceDeclaration.model_validate(values)
