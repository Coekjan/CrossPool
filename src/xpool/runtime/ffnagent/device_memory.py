"""Exact FFN device-memory resources shared by placement and qualification."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import xpool.native
from xpool import ffn
from xpool.config import get_global_config
from xpool.fabric import FabricPlan, FfnModelPlan, InstanceFfnProfile
from xpool.memory import (
    MIB,
    DeviceMemoryEstimate,
    FfnMemoryCalibrationCoefficients,
    ensure_nonnegative_int64,
    load_memory_calibration_profile,
)
from xpool.runtime.ffnagent import architecture, execution
from xpool.utils import align_up


@dataclass(frozen=True, slots=True)
class DeviceMemoryFeatures:
    """Opaque-overhead inputs at one permanent observation point."""

    tensor_storage_bytes: int
    tensor_storage_allocation_count: int
    dense_graph_capture_count: int
    moe_graph_capture_count: int
    executor_lane_count: int
    compute_branch_count: int
    dense_implementation_present: bool
    moe_implementation_present: bool
    joined_ffnagent: bool


@dataclass(frozen=True, slots=True)
class DeviceMemoryPoint:
    """Resource ledger, allocator allowance, and features at one observation."""

    point: str
    exact_resource_ledger_bytes: int
    allocator_allowance_bytes: int
    features: DeviceMemoryFeatures


def ensure_supported_cuda_allocator() -> None:
    """Require the dependency-pinned default native PyTorch CUDA allocator."""

    for name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
    ):
        if name in os.environ:
            raise RuntimeError(f"unsupported PyTorch CUDA allocator environment variable {name}")
    backend = torch.cuda.memory.get_allocator_backend()
    if backend != "native":
        raise RuntimeError(f"unsupported PyTorch CUDA allocator backend {backend!r}; expected 'native'")


def allocator_block_allowance_bytes(storage_bytes: tuple[int, ...]) -> int:
    """Bound active-block excess for known Tensor storage requests."""

    if any(value <= 0 for value in storage_bytes):
        raise ValueError("allocator allowance requires positive Tensor storage byte counts")
    result = 0
    for value in storage_bytes:
        rounded = align_up(max(value, 512), 512)
        result += rounded - value + (MIB if rounded > MIB else 0)
    ensure_nonnegative_int64(result)
    return result


class DeviceMemoryEstimator:
    """Exact allocation and optional calibrated opaque-overhead authority."""

    def __init__(
        self,
        *,
        model_specs: tuple[ffn.FfnModelSpec, ...],
        instance_profiles: tuple[InstanceFfnProfile, ...],
    ) -> None:
        """Retain co-indexed semantics and one process-global configuration."""

        if not model_specs or len(model_specs) != len(instance_profiles):
            raise ValueError("device-memory estimation requires nonempty co-indexed Models and Profiles")
        config = get_global_config()
        if len(model_specs) != len(config.models):
            raise ValueError("device-memory Models do not match configured Model count")
        for index, (model, spec, profile) in enumerate(zip(config.models, model_specs, instance_profiles, strict=True)):
            if model.id != spec.model_id:
                raise ValueError(f"device-memory Model {index} does not follow configured order")
            if (
                profile.payload_dtype not in (torch.bfloat16, torch.float16)
                or spec.hidden_size != profile.hidden_size
                or tuple((layer.layer_id, layer.kind) for layer in spec.layers)
                != tuple((layer.layer_id, layer.kind) for layer in profile.layers)
            ):
                raise ValueError(f"device-memory Model {spec.model_id!r} disagrees with its Profile")
        self.model_specs = model_specs
        self.instance_profiles = instance_profiles
        self.config = config
        profile = load_memory_calibration_profile()
        self.coefficients: FfnMemoryCalibrationCoefficients | None = (
            None if profile is None else profile.ffn.coefficients
        )

    def tp_size(self, model_plan_index: int) -> int:
        """Resolve one configured Model's fixed FFN TP width."""

        model = self.config.models[model_plan_index]
        return model.ffn_tp_size or len(self.config.ffn.devices)

    def packed_weight_storage_bytes(
        self,
        model_plan_index: int,
        layer_ordinal: int,
        tp_rank: int,
    ) -> tuple[int, ...]:
        """Return logical bytes of every retained packed Tensor storage."""

        spec = self.model_specs[model_plan_index]
        payload_bytes = self.instance_profiles[model_plan_index].payload_dtype.itemsize
        layer = spec.layers[layer_ordinal]
        tp_size = self.tp_size(model_plan_index)
        if not 0 <= tp_rank < tp_size:
            raise ValueError("packed-weight TP rank is outside the configured width")
        if isinstance(layer, ffn.DenseFfnSpec):
            if layer.intermediate_size % tp_size:
                raise ValueError("Dense intermediate size is not divisible by FFN TP")
            local_width = layer.intermediate_size // tp_size
            return (
                2 * payload_bytes * spec.hidden_size * local_width,
                payload_bytes * spec.hidden_size * local_width,
            )
        if layer.expert_intermediate_size % tp_size:
            raise ValueError("MoE Expert intermediate size is not divisible by FFN TP")
        local_width = layer.expert_intermediate_size // tp_size
        expert_count = layer.routed_expert_count + layer.shared_expert_count
        result = [
            2 * payload_bytes * expert_count * spec.hidden_size * local_width,
            payload_bytes * expert_count * spec.hidden_size * local_width,
        ]
        if tp_rank == 0:
            model_adapter = architecture.adapter_for(spec)
            if not issubclass(model_adapter, architecture.MoeFfnModelAdapter):
                raise ValueError("MoE weight sizing requires a MoE FFN Model Adapter")
            router_dtype = model_adapter.router_weight_dtype(
                payload_dtype=self.instance_profiles[model_plan_index].payload_dtype
            )
            result.append(router_dtype.itemsize * layer.routed_expert_count * spec.hidden_size)
            if layer.checkpoint.router_correction_bias_key is not None:
                result.append(4 * layer.routed_expert_count)
        return tuple(result)

    def execution_runtime_bytes(
        self,
        *,
        local_layer_count: int,
        local_layer_capacity_count: int,
        local_signature_count: int,
        coordinator: bool,
    ) -> int:
        """Return exact native readiness, lookup, binding, and Lane storage."""

        ensure_nonnegative_int64(local_layer_count, local_layer_capacity_count, local_signature_count)
        lanes = self.config.scheduler.ffn_concurrency
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

        ensure_nonnegative_int64(element_capacity)
        return xpool.native.devkit.ffn_routing_observer.allocation_bytes(element_capacity)

    def fabric_observer_bytes(self) -> int:
        """Return exact optional Fabric Observer allocation bytes."""

        return xpool.native.devkit.fabric_observer.allocation_bytes(
            len(self.model_specs),
            self.config.scheduler.ffn_concurrency,
        )

    def fabric_arena_bytes(self) -> int:
        """Return exact native Fabric Arena bytes from semantic geometry."""

        atnagent_count = len(self.config.atn.devices)
        ffnagent_count = len(self.config.ffn.devices)
        instance_count = len(self.model_specs)
        executor_lane_count = self.config.scheduler.ffn_concurrency
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

        if fabric_plan.executor_lane_count != self.config.scheduler.ffn_concurrency:
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

        ffnagent_count = len(self.config.ffn.devices)
        if not 0 <= ffnagent_index < ffnagent_count:
            raise ValueError("device-memory FfnAgent index is outside the configured Fleet")
        if len(model_plans) != len(self.model_specs):
            raise ValueError("device-memory Model Plans are not co-indexed with Model Specs")
        signatures: set[execution.ExecutionSignature] = set()
        packed_weight_storage_bytes: list[int] = []
        local_layer_count = 0
        local_layer_capacity_count = 0
        for model_index, (spec, profile, model_plan) in enumerate(
            zip(self.model_specs, self.instance_profiles, model_plans, strict=True)
        ):
            for layer_ordinal, layer_plan in enumerate(model_plan.layers):
                if ffnagent_index not in layer_plan.ffnagent_indices:
                    continue
                tp_rank = layer_plan.ffnagent_indices.index(ffnagent_index)
                local_signatures = execution.required_execution_signatures(
                    model_spec=spec,
                    profile=profile,
                    layer_ordinal=layer_ordinal,
                    tp_size=len(layer_plan.ffnagent_indices),
                    tp_rank=tp_rank,
                )
                signatures.update(local_signatures)
                local_layer_count += 1
                local_layer_capacity_count += len(local_signatures)
                packed_weight_storage_bytes.extend(
                    self.packed_weight_storage_bytes(model_index, layer_ordinal, tp_rank)
                )

        capture_weight_pair_keys = {execution.capture_weight_pair_key(signature) for signature in signatures}
        graph_capture_storage_bytes = tuple(
            value
            for signature in capture_weight_pair_keys
            for value in execution.control_capture_probe_storage_bytes(signature)
        ) + tuple(
            value for signature in signatures for value in execution.graph_capture_capacity_storage_bytes(signature)
        )
        packed_weight_storage_bytes_tuple = tuple(packed_weight_storage_bytes)
        packed_weight_bytes = sum(packed_weight_storage_bytes_tuple)
        packed_weight_allocations = len(packed_weight_storage_bytes_tuple)
        packed_weight_allowance = allocator_block_allowance_bytes(packed_weight_storage_bytes_tuple)
        graph_capture_bytes = sum(graph_capture_storage_bytes)
        graph_capture_allocations = len(graph_capture_storage_bytes)
        graph_capture_allowance = allocator_block_allowance_bytes(graph_capture_storage_bytes)
        dense_graph_capture_count = sum(
            isinstance(signature, execution.DenseFfnExecutionSignature) for signature in signatures
        )
        moe_graph_capture_count = len(signatures) - dense_graph_capture_count
        dense_present = dense_graph_capture_count != 0
        moe_present = moe_graph_capture_count != 0
        executor_lane_count = self.config.scheduler.ffn_concurrency
        compute_branch_count = executor_lane_count * len(signatures)
        lane_workspace_bytes = executor_lane_count * max(
            (execution.compute_workspace_bytes(signature) for signature in signatures), default=0
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
                exact_resource_ledger_bytes=packed_weight_bytes,
                allocator_allowance_bytes=packed_weight_allowance,
                features=features(graph_capture=False, lanes=False, joined=False),
            ),
            DeviceMemoryPoint(
                point="fabric_join",
                exact_resource_ledger_bytes=packed_weight_bytes + fabric_join_bytes,
                allocator_allowance_bytes=packed_weight_allowance,
                features=features(graph_capture=False, lanes=False, joined=True),
            ),
            DeviceMemoryPoint(
                point="graph_capture",
                exact_resource_ledger_bytes=packed_weight_bytes + fabric_join_bytes + graph_capture_bytes,
                allocator_allowance_bytes=packed_weight_allowance + graph_capture_allowance,
                features=features(graph_capture=True, lanes=False, joined=True),
            ),
            DeviceMemoryPoint(
                point="execution_installation",
                exact_resource_ledger_bytes=retained_exact + graph_capture_bytes,
                allocator_allowance_bytes=packed_weight_allowance + graph_capture_allowance,
                features=features(graph_capture=True, lanes=True, joined=True),
            ),
            DeviceMemoryPoint(
                point="retained",
                exact_resource_ledger_bytes=retained_exact,
                allocator_allowance_bytes=packed_weight_allowance,
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

        estimates = []
        for point in points:
            calibrated_overhead = (
                0
                if self.coefficients is None
                else self.coefficients.evaluate(
                    tensor_storage_bytes=point.features.tensor_storage_bytes,
                    tensor_storage_allocation_count=point.features.tensor_storage_allocation_count,
                    dense_graph_capture_count=point.features.dense_graph_capture_count,
                    moe_graph_capture_count=point.features.moe_graph_capture_count,
                    executor_lane_count=point.features.executor_lane_count,
                    compute_branch_count=point.features.compute_branch_count,
                    dense_implementation_present=point.features.dense_implementation_present,
                    moe_implementation_present=point.features.moe_implementation_present,
                    joined_ffnagent=point.features.joined_ffnagent,
                )
            )
            estimates.append(point.exact_resource_ledger_bytes + point.allocator_allowance_bytes + calibrated_overhead)
        ensure_nonnegative_int64(*estimates)
        return DeviceMemoryEstimate(retained_bytes=estimates[-1], peak_bytes=max(estimates))
