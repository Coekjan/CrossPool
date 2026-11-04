"""Exact FFN device-memory resources shared by placement and qualification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import xpool.native
from xpool import ffn
from xpool.config import get_global_config
from xpool.fabric import FabricPlan, FfnModelPlan, InstanceFfnProfile
from xpool.memory import (
    MIB,
    DeviceMemoryEstimate,
    DeviceMemoryFeatures,
    FfnExecutionContract,
    FfnMemoryCalibrationCoefficients,
)
from xpool.runtime.ffnagent import execution, preparation
from xpool.utils import align_up
from xpool.utils.integer import ensure_sint64_nonnegative

logger = logging.getLogger(__name__)

type StartupPoint = Literal[
    "weight_materialization",
    "fabric_join",
    "graph_capture",
    "execution_installation",
    "retained",
]


@dataclass(frozen=True, slots=True)
class DeviceMemoryCharge:
    """One disjoint physical-cost estimate, with any unmodeled overhead named."""

    domain: str
    exact_logical_bytes: int
    estimated_physical_bytes: int
    unmodeled_costs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ensure_sint64_nonnegative(self.exact_logical_bytes, self.estimated_physical_bytes)
        if not self.domain or any(not cost for cost in self.unmodeled_costs):
            raise ValueError("memory charge needs a domain and nonempty unmodeled-cost names")


@dataclass(frozen=True, slots=True)
class DeviceMemoryPoint:
    """Disjoint physical charges and resource-derived features at one stage."""

    point: StartupPoint
    charges: tuple[DeviceMemoryCharge, ...]
    features: DeviceMemoryFeatures

    def __post_init__(self) -> None:
        if len({charge.domain for charge in self.charges}) != len(self.charges):
            raise ValueError("device memory point charges a physical domain more than once")

    @property
    def exact_resource_ledger_bytes(self) -> int:
        return sum(charge.exact_logical_bytes for charge in self.charges)

    @property
    def unmodeled_costs(self) -> tuple[str, ...]:
        return tuple(sorted({source for charge in self.charges for source in charge.unmodeled_costs}))

    @property
    def analytic_estimate_bytes(self) -> int:
        """Sum known contributions; unmodeled overhead remains explicit evidence."""

        result = sum(charge.estimated_physical_bytes for charge in self.charges)
        ensure_sint64_nonnegative(result)
        return result


def allocator_reservation_bytes(storage_bytes: tuple[int, ...]) -> int:
    """Bound new reserved bytes for one consecutive same-stream allocation group.

    Requests use the pinned default native allocator. A group in one pool
    fitting a single small or medium segment needs at most one new segment;
    mixed or larger groups charge each request its entire fresh segment.
    Existing cached blocks can reduce this charge, but are never assumed.
    """

    ensure_sint64_nonnegative(*storage_bytes)
    if not storage_bytes:
        return 0
    if any(value == 0 for value in storage_bytes):
        raise ValueError("allocator reservation requires positive Tensor storage byte counts")
    rounded = tuple(align_up(max(value, 512), 512) for value in storage_bytes)
    if max(rounded) <= MIB and sum(rounded) <= 2 * MIB:
        result = 2 * MIB
    elif min(rounded) > MIB and max(rounded) < 10 * MIB and sum(rounded) <= 20 * MIB:
        result = 20 * MIB
    else:
        result = sum(
            2 * MIB if value <= MIB else 20 * MIB if value < 10 * MIB else align_up(value, 2 * MIB) for value in rounded
        )
    ensure_sint64_nonnegative(result)
    return result


def graph_capture_allocator_allowance_bytes(signature: execution.ExecutionSignature) -> int:
    """Bound reservation beyond retained placeholders during signature construction.

    Dense placeholders form one consecutive group. MoE initialization interleaves
    temporary route-ID tensors, so every known request is charged independently.
    Transient reservations include their logical bytes, absent from the ledger.
    """

    storage_bytes = preparation.graph_capture_capacity_storage_bytes(signature)
    if isinstance(signature, execution.DenseFfnExecutionSignature):
        reserved = allocator_reservation_bytes(storage_bytes)
    else:
        requests = storage_bytes + execution.graph_capture_initialization_storage_bytes(signature)
        reserved = sum(allocator_reservation_bytes((value,)) for value in requests)
    result = reserved - sum(storage_bytes)
    ensure_sint64_nonnegative(result)
    return result


class DeviceMemoryEstimator:
    """Exact allocation and optional calibrated opaque-overhead authority."""

    def __init__(
        self,
        *,
        model_specs: tuple[ffn.FfnModelSpec, ...],
        instance_profiles: tuple[InstanceFfnProfile, ...],
        ffn_execution_contracts: tuple[FfnExecutionContract, ...],
        coefficients: FfnMemoryCalibrationCoefficients | None,
    ) -> None:
        """Retain co-indexed semantics and the generation's selected correction."""

        if not model_specs or len(model_specs) != len(instance_profiles):
            raise ValueError("device-memory estimation requires nonempty co-indexed Models and Profiles")
        config = get_global_config()
        if len(ffn_execution_contracts) != len(config.ffn.devices):
            raise ValueError("execution contracts do not match configured FfnAgent count")
        if len(model_specs) != len(config.models):
            raise ValueError("device-memory Models do not match configured Model count")
        for index, (model, spec, profile) in enumerate(zip(config.models, model_specs, instance_profiles, strict=True)):
            if model.id != spec.model_id:
                raise ValueError(f"device-memory Model {index} does not follow configured order")
            profile.require_compatible_model(spec)
        self.model_specs = model_specs
        self.instance_profiles = instance_profiles
        self.ffn_execution_contracts = ffn_execution_contracts
        self.coefficients = coefficients

    def tp_size(self, model_plan_index: int) -> int:
        """Resolve one configured Model's fixed FFN TP width."""

        model = get_global_config().models[model_plan_index]
        return model.ffn_tp_size or len(get_global_config().ffn.devices)

    def execution_runtime_bytes(
        self,
        *,
        local_layer_count: int,
        local_layer_capacity_count: int,
        local_signature_count: int,
        coordinator: bool,
    ) -> int:
        """Return exact native readiness, lookup, binding, and Lane storage."""

        ensure_sint64_nonnegative(local_layer_count, local_layer_capacity_count, local_signature_count)
        lanes = get_global_config().scheduler.ffn_concurrency
        return xpool.native.fabric.ffnagent_control_allocation_bytes(
            coordinator,
            len(self.model_specs),
        ) + xpool.native.ffnagent.execution_state_allocation_bytes(
            local_layer_count,
            local_layer_capacity_count,
            local_signature_count,
            lanes,
        )

    def routing_record_buffer_bytes(self, element_capacity: int) -> int:
        """Return exact optional Routing Observer allocation bytes."""

        ensure_sint64_nonnegative(element_capacity)
        return xpool.native.devkit.ffn_routing_observer.allocation_bytes(element_capacity)

    def fabric_observer_bytes(self) -> int:
        """Return exact optional Fabric Observer allocation bytes."""

        return xpool.native.devkit.fabric_observer.allocation_bytes(
            len(self.model_specs),
            get_global_config().scheduler.ffn_concurrency,
        )

    def fabric_arena_bytes(self) -> int:
        """Return exact native Fabric Arena bytes from semantic geometry."""

        atnagent_count = len(get_global_config().atn.devices)
        ffnagent_count = len(get_global_config().ffn.devices)
        instance_count = len(self.model_specs)
        executor_lane_count = get_global_config().scheduler.ffn_concurrency
        layer_count = sum(len(spec.layers) for spec in self.model_specs)
        atnagent_pe_count = atnagent_count * instance_count
        ffnagent_pe_count = sum(len(spec.layers) * self.tp_size(index) for index, spec in enumerate(self.model_specs))
        maximum_lane_payload_bytes = max(
            max(profile.decode_payload_row_capacity, profile.prefill_payload_row_capacity)
            * profile.hidden_size
            * profile.payload_dtype.itemsize
            for profile in self.instance_profiles
        )
        maximum_routing_metadata_elements = 0
        for spec, profile in zip(self.model_specs, self.instance_profiles, strict=True):
            maximum_rows = max(profile.decode_payload_row_capacity, profile.prefill_payload_row_capacity)
            for layer in spec.layers:
                if isinstance(layer, ffn.MoeFfnSpec):
                    maximum_routing_metadata_elements = max(
                        maximum_routing_metadata_elements,
                        maximum_rows * (layer.routed_topk + layer.shared_expert_count),
                    )
        return xpool.native.fabric.arena_allocation_bytes(
            atnagent_count,
            ffnagent_count,
            instance_count,
            executor_lane_count,
            layer_count,
            atnagent_pe_count,
            ffnagent_pe_count,
            maximum_lane_payload_bytes,
            maximum_routing_metadata_elements,
        )

    def allocation_ledger(self, *, fabric_plan: FabricPlan, ffnagent_index: int) -> tuple[DeviceMemoryPoint, ...]:
        """Compose the five exact startup ledgers and feature rows."""

        if fabric_plan.executor_lane_count != get_global_config().scheduler.ffn_concurrency:
            raise ValueError("Fabric Plan executor Lane count disagrees with the process configuration")
        return self.allocation_ledger_for_model_plans(
            model_plans=fabric_plan.model_plans,
            ffnagent_index=ffnagent_index,
        )

    def allocation_ledger_for_model_plans(
        self,
        *,
        model_plans: tuple[FfnModelPlan, ...],
        ffnagent_index: int,
    ) -> tuple[DeviceMemoryPoint, ...]:
        """Compose ledgers from Model Plans before their Fabric Plan is installed."""

        ffnagent_count = len(get_global_config().ffn.devices)
        if not 0 <= ffnagent_index < ffnagent_count:
            raise ValueError("device-memory FfnAgent index is outside the configured Fleet")
        if len(model_plans) != len(self.model_specs):
            raise ValueError("device-memory Model Plans are not co-indexed with Model Specs")
        signatures: set[execution.ExecutionSignature] = set()
        packed_weight_storage_bytes: list[int] = []
        packed_weight_allowance = 0
        local_layer_count = 0
        local_layer_capacity_count = 0
        for spec, profile, model_plan in zip(self.model_specs, self.instance_profiles, model_plans, strict=True):
            for layer_ordinal, layer_plan in enumerate(model_plan.layers):
                if ffnagent_index not in layer_plan.ffnagent_indices:
                    continue
                tp_rank = layer_plan.ffnagent_indices.index(ffnagent_index)
                layout = preparation.prepare_ffn_layer(
                    spec,
                    layer_ordinal=layer_ordinal,
                    payload_dtype=profile.payload_dtype,
                    tp_size=len(layer_plan.ffnagent_indices),
                    tp_rank=tp_rank,
                )
                local_signatures = layout.execution_signatures(profile=profile)
                signatures.update(local_signatures)
                local_layer_count += 1
                local_layer_capacity_count += len(local_signatures)
                storage_bytes = layout.weight_allocation_bytes()
                packed_weight_storage_bytes.extend(storage_bytes)
                packed_weight_allowance += allocator_reservation_bytes(storage_bytes) - sum(storage_bytes)

        capture_weight_pair_keys = {execution.capture_weight_pair_key(signature) for signature in signatures}
        graph_capture_storage_bytes = tuple(
            value
            for signature in capture_weight_pair_keys
            for value in execution.control_capture_probe_storage_bytes(signature)
        ) + tuple(
            value for signature in signatures for value in preparation.graph_capture_capacity_storage_bytes(signature)
        )
        packed_weight_storage_bytes_tuple = tuple(packed_weight_storage_bytes)
        packed_weight_bytes = sum(packed_weight_storage_bytes_tuple)
        packed_weight_allocations = len(packed_weight_storage_bytes_tuple)
        graph_capture_bytes = sum(graph_capture_storage_bytes)
        graph_capture_allocations = len(graph_capture_storage_bytes)
        graph_capture_allowance = sum(graph_capture_allocator_allowance_bytes(signature) for signature in signatures)
        for signature in capture_weight_pair_keys:
            storage_bytes = execution.control_capture_probe_storage_bytes(signature)
            graph_capture_allowance += allocator_reservation_bytes(storage_bytes) - sum(storage_bytes)
        dense_graph_capture_count = sum(
            isinstance(signature, execution.DenseFfnExecutionSignature) for signature in signatures
        )
        moe_graph_capture_count = len(signatures) - dense_graph_capture_count
        dense_present = dense_graph_capture_count != 0
        moe_present = moe_graph_capture_count != 0
        executor_lane_count = get_global_config().scheduler.ffn_concurrency
        compute_branch_count = executor_lane_count * len(signatures)
        lane_workspace_bytes = executor_lane_count * max(
            (preparation.compute_workspace_bytes(signature) for signature in signatures), default=0
        )
        execution_runtime_bytes = self.execution_runtime_bytes(
            local_layer_count=local_layer_count,
            local_layer_capacity_count=local_layer_capacity_count,
            local_signature_count=len(signatures),
            coordinator=ffnagent_index == 0,
        )
        routing_elements = max(
            (
                signature.payload_row_capacity * signature.effective_topk
                for signature in signatures
                if isinstance(signature, execution.MoeFfnExecutionSignature) and signature.router is not None
            ),
            default=0,
        )
        routing_record_buffer_bytes = self.routing_record_buffer_bytes(routing_elements)
        fabric_join_bytes = self.fabric_arena_bytes() + self.fabric_observer_bytes()

        def features(
            *,
            graph_capture: bool,
            lanes: bool,
            joined: bool,
        ) -> DeviceMemoryFeatures:
            return DeviceMemoryFeatures(
                tensor_storage_bytes=packed_weight_bytes + (graph_capture_bytes if graph_capture else 0),
                tensor_storage_allocation_count=packed_weight_allocations
                + (graph_capture_allocations if graph_capture else 0),
                dense_graph_capture_count=dense_graph_capture_count if graph_capture else 0,
                moe_graph_capture_count=moe_graph_capture_count if graph_capture else 0,
                executor_lane_count=executor_lane_count if lanes else 0,
                compute_branch_count=compute_branch_count if lanes else 0,
                dense_implementation_present=dense_present,
                moe_implementation_present=moe_present,
                joined_ffnagent=joined,
            )

        retained_exact = (
            packed_weight_bytes
            + fabric_join_bytes
            + lane_workspace_bytes
            + execution_runtime_bytes
            + routing_record_buffer_bytes
        )
        return (
            DeviceMemoryPoint(
                point="weight_materialization",
                charges=(
                    DeviceMemoryCharge(
                        domain="execution",
                        exact_logical_bytes=packed_weight_bytes,
                        estimated_physical_bytes=(packed_weight_bytes) + (packed_weight_allowance),
                        unmodeled_costs=(),
                    ),
                ),
                features=features(graph_capture=False, lanes=False, joined=False),
            ),
            DeviceMemoryPoint(
                point="fabric_join",
                charges=(
                    DeviceMemoryCharge(
                        domain="execution",
                        exact_logical_bytes=packed_weight_bytes + fabric_join_bytes,
                        estimated_physical_bytes=(packed_weight_bytes + fabric_join_bytes) + (packed_weight_allowance),
                        unmodeled_costs=(),
                    ),
                ),
                features=features(graph_capture=False, lanes=False, joined=True),
            ),
            DeviceMemoryPoint(
                point="graph_capture",
                charges=(
                    DeviceMemoryCharge(
                        domain="execution",
                        exact_logical_bytes=packed_weight_bytes + fabric_join_bytes + graph_capture_bytes,
                        estimated_physical_bytes=(packed_weight_bytes + fabric_join_bytes + graph_capture_bytes)
                        + (packed_weight_allowance + graph_capture_allowance),
                        unmodeled_costs=("cuda.graph_driver",),
                    ),
                ),
                features=features(graph_capture=True, lanes=False, joined=True),
            ),
            DeviceMemoryPoint(
                point="execution_installation",
                charges=(
                    DeviceMemoryCharge(
                        domain="execution",
                        exact_logical_bytes=retained_exact + graph_capture_bytes,
                        estimated_physical_bytes=(retained_exact + graph_capture_bytes)
                        + (packed_weight_allowance + graph_capture_allowance),
                        unmodeled_costs=("cuda.graph_driver",),
                    ),
                ),
                features=features(graph_capture=True, lanes=True, joined=True),
            ),
            DeviceMemoryPoint(
                point="retained",
                charges=(
                    DeviceMemoryCharge(
                        domain="execution",
                        exact_logical_bytes=retained_exact,
                        estimated_physical_bytes=(retained_exact) + (packed_weight_allowance),
                        unmodeled_costs=("cuda.graph_driver",),
                    ),
                ),
                features=features(graph_capture=False, lanes=True, joined=True),
            ),
        )

    def estimate(self, *, fabric_plan: FabricPlan, ffnagent_index: int) -> DeviceMemoryEstimate:
        """Estimate retained and startup-peak bytes for one concrete FfnAgent."""

        points = self.allocation_ledger(fabric_plan=fabric_plan, ffnagent_index=ffnagent_index)
        return self.estimate_points(points)

    def estimate_model_plans(
        self,
        *,
        model_plans: tuple[FfnModelPlan, ...],
        ffnagent_index: int,
    ) -> DeviceMemoryEstimate:
        """Estimate a solved Placement before its Fabric Plan is installed."""

        points = self.allocation_ledger_for_model_plans(
            model_plans=model_plans,
            ffnagent_index=ffnagent_index,
        )
        return self.estimate_points(points)

    def estimate_points(self, points: tuple[DeviceMemoryPoint, ...]) -> DeviceMemoryEstimate:
        """Reduce one five-point ledger to retained and peak bytes."""

        estimates: dict[StartupPoint, int] = {}
        for point in points:
            calibrated_overhead = 0 if self.coefficients is None else self.coefficients.evaluate(point.features)
            estimates[point.point] = point.analytic_estimate_bytes + calibrated_overhead
            logger.debug(
                "memory stage estimated point=%s analytic_bytes=%s correction_bytes=%s unmodeled_costs=%s",
                point.point,
                point.analytic_estimate_bytes,
                calibrated_overhead,
                point.unmodeled_costs,
            )
        ensure_sint64_nonnegative(*estimates.values())
        logger.info(
            "memory estimated peak_bytes=%s retained_bytes=%s calibrated=%s unmodeled_costs=%s",
            max(estimates.values()),
            estimates["retained"],
            self.coefficients is not None,
            tuple(sorted({cost for point in points for cost in point.unmodeled_costs})),
        )
        return DeviceMemoryEstimate(retained_bytes=estimates["retained"], peak_bytes=max(estimates.values()))
