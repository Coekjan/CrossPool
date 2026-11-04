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
    FfnModelPlan,
    FifoSchedulerPolicy,
    InstanceFfnProfile,
    InstanceRankTopology,
    place_ffn_layer,
)
from xpool.ffn import FfnModelSpec
from xpool.memory import (
    MIB,
    DeviceMemoryFeatures,
    ExecutionResourceDeclaration,
    ExecutionResourceRequirement,
    FfnMemoryCalibrationCoefficients,
)
from xpool.runtime.ffnagent import device_memory, execution, preparation
from xpool.runtime.ffnagent.memory_profile import corpus
from xpool.utils.integer import SINT64_MAX
from xtest.harness.support.config import (
    TEST_EXECUTION_CONTRACT,
    TEST_MODEL_ID,
    install_test_config,
    reset_global_config,
    synthetic_config,
)
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
    model_plans = (
        FfnModelPlan(
            model_spec_digest=spec.digest(),
            layers=(place_ffn_layer(spec.layers[0], (0, 1), local_intermediate_size=4),),
        ),
    )
    plan = FabricPlan(
        generation=FabricGenerationId(high=1, low=1),
        uid=FabricUid("a" * 256),
        ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
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
    return device_memory.DeviceMemoryEstimator(
        coefficients=None,
        model_specs=(spec,),
        instance_profiles=(profile,),
        ffn_execution_contracts=plan.ffn_execution_contracts,
    ), plan


def test_fp32_router_keeps_fp32_weights_and_input_workspace() -> None:
    experiments, _ = corpus.calibration_experiments(synthetic_config(ffn_devices=(1, 2)))
    model = experiments[2].models[0]
    spec = model.spec
    profile = model.instance_plan().ffn_profile
    layout = preparation.prepare_ffn_layer(
        spec,
        layer_ordinal=1,
        payload_dtype=profile.payload_dtype,
        tp_size=2,
        tp_rank=0,
    )
    assert layout.weight_allocation_bytes()[-2:] == (4 * 64 * spec.hidden_size, 4 * 64)
    signatures = layout.execution_signatures(profile=profile)
    for signature in signatures:
        assert isinstance(signature, execution.MoeFfnExecutionSignature)
        assert signature.router is not None
        assert signature.router.projection.values.dtype is torch.float32
        assert signature.router.router_workspace_bytes == 4 * signature.payload_row_capacity * (spec.hidden_size + 64)


@pytest.mark.parametrize("calibration_base", (None, 1 << 30))
def test_unmodeled_overhead_allows_analytic_or_calibrated_estimation(calibration_base: int | None) -> None:
    estimator, plan = estimator_and_plan()
    if calibration_base is not None:
        estimator.coefficients = FfnMemoryCalibrationCoefficients(
            base_bytes=calibration_base,
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
    points = estimator.allocation_ledger_for_model_plans(model_plans=plan.model_plans, ffnagent_index=0)
    estimate = estimator.estimate_points(points)
    assert estimator.estimate_points(tuple(reversed(points))) == estimate
    correction = calibration_base or 0
    assert estimate.peak_bytes == max(point.analytic_estimate_bytes for point in points) + correction
    assert estimate.retained_bytes == points[-1].analytic_estimate_bytes + correction
    assert "cuda.graph_driver" in points[3].unmodeled_costs


def test_estimator_requires_one_execution_contract_per_configured_ffnagent() -> None:
    estimator, _ = estimator_and_plan()
    with pytest.raises(ValueError, match="execution contracts do not match configured FfnAgent count"):
        device_memory.DeviceMemoryEstimator(
            model_specs=estimator.model_specs,
            instance_profiles=estimator.instance_profiles,
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,),
            coefficients=None,
        )


def test_fabric_observer_allocation_begins_at_fabric_join(monkeypatch: pytest.MonkeyPatch) -> None:
    estimator, plan = estimator_and_plan()
    baseline = estimator.allocation_ledger_for_model_plans(model_plans=plan.model_plans, ffnagent_index=0)
    install_native_allocation_sizing(monkeypatch, fabric_observer_bytes=128)
    observed = estimator.allocation_ledger_for_model_plans(model_plans=plan.model_plans, ffnagent_index=0)

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


@pytest.mark.parametrize("requests", ((0,), (-1,), (True,), (SINT64_MAX,)))
def test_allocator_reservation_rejects_invalid_or_overflowing_byte_counts(requests: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        device_memory.allocator_reservation_bytes(requests)


@pytest.mark.parametrize("storage_bytes", [-1, SINT64_MAX + 1])
def test_features_reject_invalid_bytes_at_construction(storage_bytes: int) -> None:
    with pytest.raises(ValueError, match="sint64"):
        DeviceMemoryFeatures(
            tensor_storage_bytes=storage_bytes,
            tensor_storage_allocation_count=0,
            dense_graph_capture_count=0,
            moe_graph_capture_count=0,
            executor_lane_count=0,
            compute_branch_count=0,
            dense_implementation_present=False,
            moe_implementation_present=False,
            joined_ffnagent=False,
        )


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
            DeviceMemoryFeatures(
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
        )
        == 119
    )
    with pytest.raises(ValueError, match="sint64"):
        coefficients.evaluate(
            DeviceMemoryFeatures(
                tensor_storage_bytes=0,
                tensor_storage_allocation_count=0,
                dense_graph_capture_count=0,
                moe_graph_capture_count=0,
                executor_lane_count=0,
                compute_branch_count=SINT64_MAX,
                dense_implementation_present=False,
                moe_implementation_present=False,
                joined_ffnagent=False,
            )
        )


def test_prebaseline_shared_storage_is_recorded_without_incremental_charge() -> None:
    estimator, plan = estimator_and_plan()
    baseline = estimator.allocation_ledger_for_model_plans(model_plans=plan.model_plans, ffnagent_index=0)
    declaration = ExecutionResourceDeclaration(
        requirement=ExecutionResourceRequirement(
            key="table",
            request_bytes=(4096,),
            logical_extent_bound_bytes=1024,
            exact_logical_bytes=1024,
            alignment_bytes=256,
        ),
        aliases=("table",),
        uses=("dense",),
    )
    contract = TEST_EXECUTION_CONTRACT.model_copy(update={"shared_resources": (declaration,)})
    contracts = (contract, TEST_EXECUTION_CONTRACT)
    plan = plan.model_copy(update={"ffn_execution_contracts": contracts})
    estimator = device_memory.DeviceMemoryEstimator(
        model_specs=estimator.model_specs,
        instance_profiles=estimator.instance_profiles,
        ffn_execution_contracts=contracts,
        coefficients=None,
    )
    points = estimator.allocation_ledger_for_model_plans(model_plans=plan.model_plans, ffnagent_index=0)
    assert points == baseline
