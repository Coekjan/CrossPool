"""Pure behavior tests for offline FFN memory profiling."""

from __future__ import annotations

import multiprocessing
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.harness.support.config import install_test_config, reset_global_config, synthetic_config
from xpool import ffn
from xpool.config import MemoryConfig
from xpool.fabric import FabricPlan, FabricRole
from xpool.memory import FfnMemoryCalibrationCoefficients, MemoryCalibrationGpu
from xpool.mps import MpsProbeResult
from xpool.native import ABI_VERSION
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent.device_memory import DeviceMemoryFeatures, DeviceMemoryPoint
from xpool.runtime.ffnagent.memory_profile import corpus, fitting, runner

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def software_environment() -> fitting.MemoryProfileSoftwareEnvironment:
    """Return one self-consistent synthetic software environment."""

    return fitting.MemoryProfileSoftwareEnvironment(
        native_abi_version=ABI_VERSION,
        cuda_driver_version=1,
        cuda_runtime_version=1,
        mps_active_thread_percentage=100,
        torch_version="1",
        triton_version="1",
        sglang_version="1",
        sglang_kernel_version="1",
        nvshmem_version="1",
    )


def gpu(index: int) -> MemoryCalibrationGpu:
    """Return one ordered synthetic GPU record."""

    return MemoryCalibrationGpu(
        name="gpu",
        compute_capability=(8, 0),
        total_memory_bytes=100,
        uuid=f"gpu-{index}",
    )


def feature_row(index: int) -> DeviceMemoryFeatures:
    """Exercise one nonconstant coefficient feature per row."""

    return DeviceMemoryFeatures(
        tensor_storage_bytes=(1 << 20) if index == 1 else 0,
        tensor_storage_allocation_count=int(index == 2),
        dense_graph_capture_count=int(index == 3),
        moe_graph_capture_count=int(index == 4),
        executor_lane_count=int(index == 5),
        compute_branch_count=int(index == 6),
        dense_implementation_present=index == 7,
        moe_implementation_present=index == 8,
        joined_ffnagent=index == 9,
    )


def test_fit_worlds_covers_all_features_and_held_out_evidence() -> None:
    observations = tuple(
        fitting.MemoryObservation(
            point=DeviceMemoryPoint(
                point=str(index),
                exact_resource_ledger_bytes=0,
                allocator_allowance_bytes=2,
                features=feature_row(index),
            ),
            observed_device_bytes=2 * (index + 1),
        )
        for index in range(10)
    )
    fit = fitting.MemoryProfileWorld(
        gpus=(gpu(0),),
        environment=software_environment(),
        observations=observations,
    )
    held_out = fitting.MemoryProfileWorld(
        gpus=(gpu(0),),
        environment=software_environment(),
        observations=(observations[0],),
    )

    coefficients, headroom = fitting.fit_worlds((fit,), (held_out,))

    assert all(value >= 0 for value in coefficients.values())
    assert all(value % 2 == 0 for value in coefficients.values())
    assert coefficients.base_bytes == 2
    assert headroom == 2


def test_profile_ffn_memory_runs_fixed_complete_fleet_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = synthetic_config(ffn_cuda_devices=(1, 2)).model_copy(
        update={"memory": MemoryConfig(calibration_path=tmp_path / "profile.json")}
    )
    install_test_config(config)
    world = fitting.MemoryProfileWorld(
        gpus=(gpu(0), gpu(1)),
        environment=software_environment(),
        observations=(),
    )
    coordinates = []
    monkeypatch.setattr(runner, "refuse_live_daemon", lambda config: None)
    monkeypatch.setattr(
        runner,
        "probe_mps_controller",
        lambda: MpsProbeResult(True, 100, "online"),
    )
    monkeypatch.setattr(
        runner,
        "run_world",
        lambda coordinate, source: coordinates.append(coordinate) or world,
    )
    coefficients = FfnMemoryCalibrationCoefficients(*([1] * 10))
    monkeypatch.setattr(runner, "fit_worlds", lambda fit, held_out: (coefficients, 7))

    profile = runner.profile_ffn_memory()

    assert Counter(coordinates) == Counter({coordinate: 3 for coordinate in (*corpus.FIT_COORDINATES, "H0")})
    assert profile.environment.ffnagent_gpus == (gpu(0), gpu(1))
    assert profile.ffn.minimum_held_out_headroom_bytes == 7
    assert profile.ffn.coefficients == coefficients


def test_coordinate_matrix_omits_unreachable_dense_tp2() -> None:
    assert corpus.coordinate_members("C2", atnagent_count=1, ffnagent_count=1)[0][1] == 1
    with pytest.raises(ValueError, match="unreachable"):
        corpus.coordinate_members("D-T", atnagent_count=1, ffnagent_count=1)


def test_calibration_corpus_uses_one_copy_of_each_signature_shape() -> None:
    members = (
        corpus.calibration_corpus_spec("gated-dense"),
        corpus.calibration_corpus_spec("softmax-shared-moe64"),
        corpus.calibration_corpus_spec("corrected-shared-moe64"),
        corpus.calibration_corpus_spec("softmax-moe128"),
    )

    assert tuple(tuple(layer.kind for layer in member.layers) for member in members) == (
        (LayerKind.DENSE,),
        (LayerKind.DENSE, LayerKind.MOE),
        (LayerKind.DENSE, LayerKind.MOE),
        (LayerKind.MOE,),
    )
    dense = members[0].layers[0]
    softmax_dense = members[1].layers[0]
    softmax_moe = members[1].layers[1]
    corrected_dense = members[2].layers[0]
    corrected_moe = members[2].layers[1]
    moe128 = members[3].layers[0]
    assert isinstance(dense, ffn.DenseFfnSpec)
    assert isinstance(softmax_dense, ffn.DenseFfnSpec)
    assert isinstance(softmax_moe, ffn.MoeFfnSpec)
    assert isinstance(corrected_dense, ffn.DenseFfnSpec)
    assert isinstance(corrected_moe, ffn.MoeFfnSpec)
    assert isinstance(moe128, ffn.MoeFfnSpec)
    assert (members[0].hidden_size, dense.intermediate_size) == (5120, 17408)
    assert (
        members[1].hidden_size,
        softmax_dense.intermediate_size,
        softmax_moe.expert_intermediate_size,
        softmax_moe.routed_expert_count,
        softmax_moe.shared_expert_count,
        softmax_moe.routed_topk,
        softmax_moe.checkpoint.router_correction_bias_key is not None,
        softmax_moe.renormalize,
        softmax_moe.routed_scaling_factor,
    ) == (2048, 10944, 1408, 64, 2, 6, False, False, 1.0)
    assert (
        members[2].hidden_size,
        corrected_dense.intermediate_size,
        corrected_moe.expert_intermediate_size,
        corrected_moe.routed_expert_count,
        corrected_moe.shared_expert_count,
        corrected_moe.routed_topk,
        corrected_moe.checkpoint.router_correction_bias_key is not None,
        corrected_moe.renormalize,
        corrected_moe.routed_scaling_factor,
    ) == (2048, 10240, 1536, 64, 1, 4, True, True, 1.8)
    assert (
        members[3].hidden_size,
        moe128.expert_intermediate_size,
        moe128.routed_expert_count,
        moe128.shared_expert_count,
        moe128.routed_topk,
        moe128.checkpoint.router_correction_bias_key is not None,
        moe128.renormalize,
        moe128.routed_scaling_factor,
    ) == (2048, 768, 128, 0, 8, False, True, 1.0)


def test_participant_rejects_allocator_before_cuda_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_allocator() -> None:
        raise RuntimeError("unsupported allocator")

    parent, child = multiprocessing.Pipe()
    plan = cast(
        FabricPlan,
        SimpleNamespace(pe_placements=(SimpleNamespace(role=FabricRole.FFNAGENT, cuda_device=1),)),
    )
    monkeypatch.setattr(runner, "init_global_config", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(runner.bootstrap, "init", lambda cuda_device, role: None)
    monkeypatch.setattr(runner, "ensure_supported_cuda_allocator", reject_allocator)
    monkeypatch.setattr(
        runner.torch,
        "empty",
        lambda *args, **kwargs: pytest.fail("CUDA warmup must not run with an unsupported allocator"),
    )

    with pytest.raises(RuntimeError, match="unsupported allocator"):
        runner.participant_child(
            child,
            config_path=Path("config.toml"),
            fabric_plan=plan,
            pe=0,
        )

    assert parent.recv() == ("failure", "RuntimeError: unsupported allocator")
    parent.close()
