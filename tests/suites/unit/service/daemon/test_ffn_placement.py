from __future__ import annotations

from dataclasses import replace

import pytest

import xpool.service.daemon.ffn_placement
from xpool.config import XpoolConfig
from xpool.fabric import FabricInstancePlan, InstanceFfnLayerProfile, InstanceFfnProfile, InstanceRankTopology
from xpool.ffn import FfnModelSpec
from xpool.memory import (
    ExecutionResourceDeclaration,
    FfnExecutionContract,
    FfnMemoryCalibrationCoefficients,
)
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import device_memory
from xpool.service.daemon.ffn_placement import place_ffn_models
from xpool.service.errors import XpoolDaemonError
from xtest.harness.native.ffn.qualification import execution_model_specs
from xtest.harness.support.config import (
    TEST_EXECUTION_CONTRACT,
    TEST_MODEL_ID,
    install_test_config,
    lane_resource_declaration,
    reset_global_config,
)
from xtest.harness.support.native.sizing import install_native_allocation_sizing
from xtest.harness.support.service.daemon import ffn_model_spec, ffn_profile

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.fixture(autouse=True)
def native_allocation_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    install_native_allocation_sizing(monkeypatch)


def config(*, tp_size: int | None = None) -> XpoolConfig:
    """Build one two-FfnAgent configuration."""

    model: dict[str, object] = {"id": str(TEST_MODEL_ID), "path": "/models/m"}
    if tp_size is not None:
        model["ffn_tp_size"] = tp_size
    return XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1, 2]},
            "models": [model],
        }
    )


def instance_plan(model_id: ModelId = TEST_MODEL_ID) -> FabricInstancePlan:
    """Build the co-indexed Instance declaration."""

    return FabricInstancePlan(
        model_id=model_id,
        ffn_profile=InstanceFfnProfile.model_validate(ffn_profile()),
        instance_rank_topology=InstanceRankTopology(atn_tp_size=1, atn_dp_size=1, atnagent_indices=(0,)),
    )


def test_placement_publishes_calibrated_budgets_for_idle_ffnagents() -> None:
    tp_size = 1
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id=str(TEST_MODEL_ID)))
    install_test_config(config(tp_size=tp_size))
    coefficients = FfnMemoryCalibrationCoefficients(
        base_bytes=10,
        bytes_per_tensor_storage_mib=0,
        bytes_per_tensor_storage_allocation=0,
        bytes_per_dense_graph_capture=0,
        bytes_per_moe_graph_capture=0,
        bytes_per_executor_lane=0,
        bytes_per_compute_branch=0,
        dense_implementation_bytes=1024,
        moe_implementation_bytes=4096,
        joined_ffnagent_bytes=0,
    )

    plans = place_ffn_models(
        coefficients=coefficients,
        model_specs=(spec,),
        instance_plans=(instance_plan(),),
        ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
        ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
    )

    assert plans.model_plans[0].tp_size == tp_size
    assert plans.model_plans[0].layers[0].ffnagent_indices == tuple(range(tp_size))
    assert plans.model_plans[0].layers[0].local_intermediate_size == 8 // tp_size
    estimator = device_memory.DeviceMemoryEstimator(
        coefficients=coefficients,
        model_specs=(spec,),
        instance_profiles=(instance_plan().ffn_profile,),
        ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
    )
    assert plans.memory_estimates == tuple(
        estimator.estimate_model_plans(model_plans=plans.model_plans, ffnagent_index=index) for index in range(2)
    )


def test_placement_uses_each_devices_source_capacity() -> None:
    install_test_config(config(tp_size=2))
    spec = FfnModelSpec.model_validate(ffn_model_spec())
    environment = FfnExecutionContract(
        lane_resources=(
            lane_resource_declaration(
                key="projection.scratch", max_bytes=32 << 20, alignment_bytes=256, aliases=("projection.scratch",)
            ),
        ),
        shared_resources=(),
    )
    declaration = instance_plan()
    plans = place_ffn_models(
        coefficients=None,
        model_specs=(spec,),
        instance_plans=(declaration,),
        ffnagent_free_memory_bytes=(2 << 30, 2 << 30),
        ffn_execution_contracts=(environment, TEST_EXECUTION_CONTRACT),
    )
    assert plans.model_plans[0].layers[0].ffnagent_indices == (0, 1)
    estimator = device_memory.DeviceMemoryEstimator(
        coefficients=None,
        model_specs=(spec,),
        instance_profiles=(declaration.ffn_profile,),
        ffn_execution_contracts=(environment, TEST_EXECUTION_CONTRACT),
    )
    first = estimator.estimate_model_plans(model_plans=plans.model_plans, ffnagent_index=0)
    second = estimator.estimate_model_plans(model_plans=plans.model_plans, ffnagent_index=1)
    assert plans.memory_estimates == (first, second)
    assert first.retained_bytes > second.retained_bytes
    assert first.peak_bytes > second.peak_bytes
    assert (
        place_ffn_models(
            coefficients=None,
            model_specs=(spec,),
            instance_plans=(declaration,),
            ffnagent_free_memory_bytes=(first.peak_bytes, second.peak_bytes),
            ffn_execution_contracts=(environment, TEST_EXECUTION_CONTRACT),
        )
        == plans
    )


@pytest.mark.parametrize("lanes", (1, 2))
@pytest.mark.parametrize("kind", (LayerKind.DENSE, LayerKind.MOE))
def test_placement_admission_matches_numeric_budget_with_reused_and_shared_resources(
    lanes: int, kind: LayerKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = config(tp_size=2)
    install_test_config(
        base.model_copy(
            update={
                "scheduler": base.scheduler.model_copy(update={"ffn_concurrency": lanes}),
                "ffn": base.ffn.model_copy(update={"device_memory_extra_margin_bytes": 123}),
                "debug": base.debug.model_copy(
                    update={"ffn_routing_observer": base.debug.ffn_routing_observer.model_copy(update={"enable": True})}
                ),
            }
        )
    )
    monkeypatch.setattr(
        device_memory.xpool.native.devkit.ffn_routing_observer, "allocation_bytes", lambda capacity: 64 * capacity
    )
    (spec,) = execution_model_specs(
        instance_count=1, execution_kind=kind, layer_count=2, hidden_size=4, intermediate_size=8
    )
    spec = spec.model_copy(update={"model_id": TEST_MODEL_ID})
    instance = instance_plan()
    profile = instance.ffn_profile
    instance = instance.model_copy(
        update={
            "ffn_profile": profile.model_copy(
                update={
                    "layers": tuple(
                        InstanceFfnLayerProfile(layer_id=layer.layer_id, kind=layer.kind) for layer in spec.layers
                    )
                }
            )
        }
    )
    scratch = lane_resource_declaration(key="scratch", max_bytes=4096, alignment_bytes=256, aliases=("scratch",))
    table = ExecutionResourceDeclaration(
        requirement=replace(scratch.requirement, key="table", exact_logical_bytes=4096),
        aliases=("table",),
        uses=("dense", "moe_router"),
    )
    contract = FfnExecutionContract(lane_resources=(scratch,), shared_resources=(table,))
    contracts = (contract, contract)
    coefficients = FfnMemoryCalibrationCoefficients(
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
    )
    plans = place_ffn_models(
        model_specs=(spec,),
        instance_plans=(instance,),
        ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
        ffn_execution_contracts=contracts,
        coefficients=coefficients,
    )
    estimator = device_memory.DeviceMemoryEstimator(
        model_specs=(spec,),
        instance_profiles=(instance.ffn_profile,),
        ffn_execution_contracts=contracts,
        coefficients=coefficients,
    )
    estimates = tuple(
        estimator.estimate_model_plans(model_plans=plans.model_plans, ffnagent_index=index) for index in range(2)
    )
    assert plans.memory_estimates == estimates
    required = tuple(estimate.peak_bytes + 123 for estimate in estimates)
    assert (
        place_ffn_models(
            model_specs=(spec,),
            instance_plans=(instance,),
            ffnagent_free_memory_bytes=required,
            ffn_execution_contracts=contracts,
            coefficients=coefficients,
        )
        == plans
    )
    with pytest.raises(XpoolDaemonError):
        place_ffn_models(
            model_specs=(spec,),
            instance_plans=(instance,),
            ffnagent_free_memory_bytes=(required[0] - 1, required[1]),
            ffn_execution_contracts=contracts,
            coefficients=coefficients,
        )


def test_equal_optimum_placement_uses_lowest_index_groups() -> None:
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1, 2, 3, 4]},
                "models": [
                    {"id": "test/wide", "path": "/models/wide", "ffn_tp_size": 4},
                    {"id": "test/narrow", "path": "/models/narrow", "ffn_tp_size": 2},
                ],
            }
        )
    )
    plans = place_ffn_models(
        coefficients=None,
        model_specs=tuple(
            FfnModelSpec.model_validate(ffn_model_spec(model_id=model_id)) for model_id in ("test/wide", "test/narrow")
        ),
        instance_plans=(instance_plan(ModelId("test/wide")), instance_plan(ModelId("test/narrow"))),
        ffnagent_free_memory_bytes=(1 << 40,) * 4,
        ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 4,
    )

    assert tuple(frozenset(plan.layers[0].ffnagent_indices) for plan in plans.model_plans) == (
        frozenset((0, 1, 2, 3)),
        frozenset((0, 1)),
    )


def test_structural_placement_rejects_tp_wider_than_fleet() -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id=str(TEST_MODEL_ID)))
    install_test_config(config(tp_size=3))

    with pytest.raises(XpoolDaemonError, match="exceeds the FfnAgent Fleet"):
        place_ffn_models(
            coefficients=None,
            model_specs=(spec,),
            instance_plans=(instance_plan(),),
            ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
        )


def test_placement_rejects_insufficient_memory_without_fallback() -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id=str(TEST_MODEL_ID)))
    install_test_config(config())

    with pytest.raises(XpoolDaemonError, match="INFEASIBLE"):
        place_ffn_models(
            coefficients=None,
            model_specs=(spec,),
            instance_plans=(instance_plan(),),
            ffnagent_free_memory_bytes=(1, 1),
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
        )


def test_placement_accounts_for_fabric_observer_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id=str(TEST_MODEL_ID)))
    install_test_config(config())
    install_native_allocation_sizing(monkeypatch, fabric_observer_bytes=1 << 30)

    with pytest.raises(XpoolDaemonError, match="INFEASIBLE"):
        place_ffn_models(
            coefficients=None,
            model_specs=(spec,),
            instance_plans=(instance_plan(),),
            ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
        )


def test_placement_uses_one_deadline_before_solver_work(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id=str(TEST_MODEL_ID)))
    install_test_config(config())
    times = iter((0.0, 61.0))
    monkeypatch.setattr(xpool.service.daemon.ffn_placement, "monotonic", lambda: next(times))

    with pytest.raises(XpoolDaemonError, match="deadline expired during resource calculation"):
        place_ffn_models(
            coefficients=None,
            model_specs=(spec,),
            instance_plans=(instance_plan(),),
            ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
        )
