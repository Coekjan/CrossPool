from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import xpool.memory
from tests.harness.support.config import install_test_config, reset_global_config
from xpool.config import XpoolConfig
from xpool.memory import (
    FfnMemoryCalibration,
    FfnMemoryCalibrationCoefficients,
    MemoryCalibrationEnvironment,
    MemoryCalibrationGpu,
    XpoolMemoryCalibrationProfile,
)
from xpool.mps import MpsProbeResult
from xpool.native import ABI_VERSION

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def profile() -> XpoolMemoryCalibrationProfile:
    """Build one strict Profile matching the installed software."""

    return XpoolMemoryCalibrationProfile(
        environment=MemoryCalibrationEnvironment(
            native_abi_version=ABI_VERSION,
            ffnagent_gpus=(
                MemoryCalibrationGpu(
                    name="NVIDIA A100-SXM4-40GB",
                    compute_capability=(8, 0),
                    total_memory_bytes=40 * 1024**3,
                    uuid="GPU-test",
                ),
            ),
            cuda_driver_version=13030,
            cuda_runtime_version=13030,
            mps_active_thread_percentage=100,
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

    payload: dict[str, object] = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    if path is not None:
        payload["memory"] = {"calibration_path": str(path)}
    install_test_config(XpoolConfig.from_mapping(payload))


def test_profile_json_is_strict_and_forbids_extra_fields() -> None:
    payload = profile().model_dump(mode="json")
    payload["ffn"]["coefficients"]["base_bytes"] = "1"

    with pytest.raises(ValidationError):
        XpoolMemoryCalibrationProfile.model_validate_json(json.dumps(payload))


def test_absent_profile_selects_analytic_admission_without_host_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_config(None)
    monkeypatch.setattr(
        xpool.memory,
        "probe_mps_controller",
        lambda: pytest.fail("absent Profile must not probe Host compatibility"),
    )

    assert xpool.memory.load_memory_calibration_profile() is None


def test_load_requires_exact_host_and_config_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "memory.json"
    path.write_text(profile().model_dump_json(), encoding="utf-8")
    install_config(path)
    monkeypatch.setattr(xpool.memory, "cuda_versions", lambda: (13030, 13030))
    monkeypatch.setattr(
        xpool.memory,
        "probe_mps_controller",
        lambda: MpsProbeResult(True, 100, "online"),
    )

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
    monkeypatch.setattr(
        xpool.memory,
        "probe_mps_controller",
        lambda: MpsProbeResult(True, 100, "online"),
    )

    with pytest.raises(RuntimeError, match="Executor Lane count"):
        xpool.memory.load_memory_calibration_profile()


def test_writer_atomically_replaces_one_complete_profile(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text("old", encoding="utf-8")
    install_config(path)

    xpool.memory.write_memory_calibration_profile(profile())

    assert path.read_text(encoding="utf-8") == profile().model_dump_json(indent=2) + "\n"
    assert list(tmp_path.iterdir()) == [path]
