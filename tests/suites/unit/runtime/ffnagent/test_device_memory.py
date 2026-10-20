"""Pure golden tests for exact FFN device-memory accounting."""

from __future__ import annotations

import pytest
import torch

from xpool.config import XpoolConfig
from xpool.fabric import (
    FabricGenerationId,
    FabricInstancePlan,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    FabricUid,
    FifoSchedulerPolicy,
    InstanceFfnProfile,
    InstanceRankTopology,
)
from xpool.ffn import FfnModelSpec
from xpool.memory import MIB, SIGNED_INT64_MAX, FfnMemoryCalibrationCoefficients
from xpool.runtime.ffnagent import device_memory, execution
from xpool.runtime.ffnagent.memory_profile import corpus
from xpool.service.daemon.ffn_placement import place_ffn_models
from xtest.harness.support.config import TEST_MODEL_ID, install_test_config, reset_global_config, synthetic_config
from xtest.harness.support.native.sizing import install_native_allocation_sizing
from xtest.harness.support.service.daemon import ffn_model_spec, ffn_profile

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.fixture(autouse=True)
def native_allocation_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    install_native_allocation_sizing(monkeypatch)


def estimator_and_plan() -> tuple[device_memory.DeviceMemoryEstimator, FabricPlan]:
    """Build one small two-rank Dense world with four Capacity buckets."""

    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1, 2]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    install_test_config(config)
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id=str(TEST_MODEL_ID)))
    profile = InstanceFfnProfile.model_validate(ffn_profile())
    instance_plan = FabricInstancePlan(
        model_id=TEST_MODEL_ID,
        ffn_profile=profile,
        instance_rank_topology=InstanceRankTopology(atn_tp_size=1, atn_dp_size=1, atnagent_indices=(0,)),
    )
    model_plans = place_ffn_models(
        model_specs=(spec,),
        instance_plans=(instance_plan,),
        ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
    )
    plan = FabricPlan(
        generation=FabricGenerationId(high=1, low=1),
        uid=FabricUid("a" * 256),
        pe_placements=(
            FabricPePlacement(role=FabricRole.ATNAGENT, device=0),
            FabricPePlacement(role=FabricRole.FFNAGENT, device=1),
            FabricPePlacement(role=FabricRole.FFNAGENT, device=2),
        ),
        executor_lane_count=1,
        scheduler=FifoSchedulerPolicy(),
        model_plans=model_plans,
        instance_plans=(instance_plan,),
    )
    return device_memory.DeviceMemoryEstimator(model_specs=(spec,), instance_profiles=(profile,)), plan


def test_fp32_router_keeps_fp32_weights_and_input_workspace() -> None:
    experiments, _ = corpus.calibration_experiments(synthetic_config(ffn_devices=(1, 2)))
    model = experiments[2].models[0]
    spec = model.spec
    profile = model.instance_plan().ffn_profile
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1, 2]},
                "models": [{"id": spec.model_id, "path": "/models/glm"}],
            }
        )
    )
    estimator = device_memory.DeviceMemoryEstimator(model_specs=(spec,), instance_profiles=(profile,))

    assert estimator.packed_weight_storage_bytes(0, 1, 0)[-2:] == (4 * 64 * spec.hidden_size, 4 * 64)
    signatures = execution.required_execution_signatures(
        model_spec=spec, profile=profile, layer_ordinal=1, tp_rank=0, tp_size=2
    )
    for signature in signatures:
        assert isinstance(signature, execution.MoeFfnExecutionSignature)
        assert signature.router is not None
        assert signature.router.router_weight_dtype is torch.float32
        assert signature.router.router_workspace_bytes == 4 * signature.payload_row_capacity * (spec.hidden_size + 64)


def test_exact_dense_allocation_ledger_and_feature_rows() -> None:
    estimator, plan = estimator_and_plan()

    assert estimator.packed_weight_storage_bytes(0, 0, 0) == (64, 32)
    signature = execution.required_execution_signatures(
        model_spec=estimator.model_specs[0],
        profile=estimator.instance_profiles[0],
        layer_ordinal=0,
        tp_size=2,
        tp_rank=0,
    )[0]
    assert execution.control_capture_probe_storage_bytes(signature) == (64, 32)
    assert execution.graph_capture_capacity_storage_bytes(signature) == (8, 8, 24)
    assert estimator.fabric_arena_bytes() == 3328
    assert (
        estimator.execution_runtime_bytes(
            local_layer_count=1,
            local_layer_capacity_count=4,
            local_signature_count=4,
            coordinator=True,
        )
        == 440
    )

    points = estimator.allocation_ledger(fabric_plan=plan, ffnagent_index=0)
    assert tuple(point.point for point in points) == (
        "weight_materialization",
        "fabric_join",
        "graph_capture",
        "execution_installation",
        "retained",
    )
    assert tuple(point.exact_resource_ledger_bytes for point in points) == (96, 3424, 4120, 4752, 4056)
    weight_allowance = 2 * MIB - 96
    capture_allowance = 10 * MIB - 696
    assert tuple(point.allocator_allowance_bytes for point in points) == (
        weight_allowance,
        weight_allowance,
        weight_allowance + capture_allowance,
        weight_allowance + capture_allowance,
        weight_allowance,
    )
    assert points[2].features.tensor_storage_bytes == 792
    assert points[2].features.tensor_storage_allocation_count == 16
    assert points[2].features.dense_graph_capture_count == 4
    assert points[3].features.executor_lane_count == 1
    assert points[3].features.compute_branch_count == 4
    assert points[4].features.tensor_storage_bytes == 96
    assert points[4].features.dense_graph_capture_count == 0

    assert estimator.estimate(fabric_plan=plan, ffnagent_index=0).retained_bytes == 2 * MIB + 3960
    assert estimator.estimate(fabric_plan=plan, ffnagent_index=0).peak_bytes == 12 * MIB + 3960

    estimator.coefficients = FfnMemoryCalibrationCoefficients(
        base_bytes=10,
        bytes_per_tensor_storage_mib=0,
        bytes_per_tensor_storage_allocation=0,
        bytes_per_dense_graph_capture=0,
        bytes_per_moe_graph_capture=0,
        bytes_per_executor_lane=0,
        bytes_per_compute_branch=0,
        dense_implementation_bytes=0,
        moe_implementation_bytes=0,
        joined_ffnagent_bytes=0,
    )
    assert estimator.estimate(fabric_plan=plan, ffnagent_index=0).retained_bytes == 2 * MIB + 3970
    assert estimator.estimate(fabric_plan=plan, ffnagent_index=0).peak_bytes == 12 * MIB + 3970

    estimator.coefficients = FfnMemoryCalibrationCoefficients(
        base_bytes=8000,
        bytes_per_tensor_storage_mib=0,
        bytes_per_tensor_storage_allocation=0,
        bytes_per_dense_graph_capture=0,
        bytes_per_moe_graph_capture=0,
        bytes_per_executor_lane=0,
        bytes_per_compute_branch=0,
        dense_implementation_bytes=0,
        moe_implementation_bytes=0,
        joined_ffnagent_bytes=0,
    )
    assert estimator.estimate(fabric_plan=plan, ffnagent_index=0).retained_bytes == 2 * MIB + 11960
    assert estimator.estimate(fabric_plan=plan, ffnagent_index=0).peak_bytes == 12 * MIB + 11960


def test_fabric_observer_allocation_begins_at_fabric_join(monkeypatch: pytest.MonkeyPatch) -> None:
    estimator, plan = estimator_and_plan()
    baseline = estimator.allocation_ledger(fabric_plan=plan, ffnagent_index=0)
    install_native_allocation_sizing(monkeypatch, fabric_observer_bytes=128)
    observed = estimator.allocation_ledger(fabric_plan=plan, ffnagent_index=0)

    assert tuple(
        after.exact_resource_ledger_bytes - before.exact_resource_ledger_bytes
        for before, after in zip(baseline, observed, strict=True)
    ) == (0, 128, 128, 128, 128)


@pytest.mark.parametrize(
    ("requests", "reserved"),
    (
        ((), 0),
        ((1, 512), 2 * MIB),
        ((MIB, MIB), 2 * MIB),
        ((MIB, MIB, 512), 6 * MIB),
        ((MIB + 1,), 20 * MIB),
        ((8_716_288, 4_358_144), 20 * MIB),
        ((10 * MIB - 512, 10 * MIB - 512), 20 * MIB),
        ((10 * MIB,), 10 * MIB),
        ((10 * MIB + 1,), 12 * MIB),
        ((MIB, MIB + 1), 22 * MIB),
        ((2 * MIB, 10 * MIB, 9 * MIB), 50 * MIB),
    ),
)
def test_allocator_reservation_bounds_segments_without_assuming_cached_reuse(
    requests: tuple[int, ...], reserved: int
) -> None:
    assert device_memory.allocator_reservation_bytes(requests) == reserved


@pytest.mark.parametrize("requests", ((0,), (-1,), (True,), (SIGNED_INT64_MAX,)))
def test_allocator_reservation_rejects_invalid_or_overflowing_byte_counts(requests: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        device_memory.allocator_reservation_bytes(requests)


def test_moe_capture_budget_includes_transient_initialization_reservations() -> None:
    _, experiment = corpus.calibration_experiments(synthetic_config(ffn_devices=(1, 2)))
    model = experiment.models[0]
    spec = model.spec
    profile = model.instance_plan().ffn_profile
    signature = execution.required_execution_signatures(
        model_spec=spec, profile=profile, layer_ordinal=0, tp_rank=0, tp_size=2
    )[0]
    retained = execution.graph_capture_capacity_storage_bytes(signature)
    temporary = execution.graph_capture_initialization_storage_bytes(signature)

    assert temporary == (32, 32)
    assert device_memory.graph_capture_allocator_allowance_bytes(signature) + sum(retained) == sum(
        device_memory.allocator_reservation_bytes((value,)) for value in retained + temporary
    )


@pytest.mark.parametrize(
    "name",
    (
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
    ),
)
def test_allocator_compatibility_rejects_control_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    for candidate in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
    ):
        monkeypatch.delenv(candidate, raising=False)
    monkeypatch.setenv(name, "")

    with pytest.raises(RuntimeError, match=name):
        device_memory.ensure_supported_cuda_allocator()


def test_allocator_compatibility_requires_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(device_memory.torch.cuda.memory, "get_allocator_backend", lambda: "cudaMallocAsync")

    with pytest.raises(RuntimeError, match="cudaMallocAsync"):
        device_memory.ensure_supported_cuda_allocator()


def test_calibrated_overhead_uses_one_mib_ceiling_and_rejects_overflow() -> None:
    coefficients = FfnMemoryCalibrationCoefficients(
        base_bytes=1,
        bytes_per_tensor_storage_mib=2,
        bytes_per_tensor_storage_allocation=3,
        bytes_per_dense_graph_capture=5,
        bytes_per_moe_graph_capture=7,
        bytes_per_executor_lane=11,
        bytes_per_compute_branch=13,
        dense_implementation_bytes=17,
        moe_implementation_bytes=19,
        joined_ffnagent_bytes=23,
    )

    assert (
        coefficients.evaluate(
            tensor_storage_bytes=MIB + 1,
            tensor_storage_allocation_count=2,
            dense_graph_capture_count=1,
            moe_graph_capture_count=0,
            executor_lane_count=1,
            compute_branch_count=4,
            dense_implementation_present=True,
            moe_implementation_present=False,
            joined_ffnagent=True,
        )
        == 119
    )
    with pytest.raises(ValueError, match="signed 64-bit"):
        coefficients.evaluate(
            tensor_storage_bytes=0,
            tensor_storage_allocation_count=0,
            dense_graph_capture_count=0,
            moe_graph_capture_count=0,
            executor_lane_count=0,
            compute_branch_count=SIGNED_INT64_MAX,
            dense_implementation_present=False,
            moe_implementation_present=False,
            joined_ffnagent=False,
        )
