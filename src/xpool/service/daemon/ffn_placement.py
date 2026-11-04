"""Generation-static memory-admitted FFN placement."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from ortools.graph.python import max_flow
from ortools.sat.python import cp_model

from xpool import ffn
from xpool.config import get_global_config
from xpool.fabric import (
    FabricInstancePlan,
    FfnModelPlan,
    place_ffn_layer,
)
from xpool.memory import (
    MIB,
    DeviceMemoryEstimate,
    DeviceMemoryFeatures,
    FfnExecutionContract,
    FfnMemoryCalibrationCoefficients,
)
from xpool.runtime.ffnagent import execution, preparation
from xpool.runtime.ffnagent.device_memory import (
    TORCH_GRAPH_CAPTURE_STATE_REQUEST_BYTES,
    DeviceMemoryEstimator,
    allocator_reservation_bytes,
    capture_pool_reservation_bytes,
    graph_capture_allocator_allowance_bytes,
)
from xpool.service.errors import XpoolDaemonError
from xpool.utils import align_up
from xpool.utils.integer import ensure_sint64_nonnegative


@dataclass(frozen=True, slots=True)
class FfnPlacementResult:
    """Model-ordered placement and independently audited FfnAgent-ordered budgets."""

    model_plans: tuple[FfnModelPlan, ...]
    memory_estimates: tuple[DeviceMemoryEstimate, ...]


def remaining_placement_seconds(deadline: float, phase: str) -> float:
    """Return the positive time remaining in the one Placement deadline."""

    remaining = deadline - monotonic()
    if remaining <= 0:
        raise XpoolDaemonError("not_ready", f"FFN Placement deadline expired during {phase}")
    return remaining


def perfect_matching_exists(
    counts: list[list[int]],
    prefix: tuple[int, ...],
    *,
    deadline: float,
) -> bool:
    """Return whether one exact matching extends a fixed lexicographic prefix."""

    remaining_placement_seconds(deadline, "matching projection")
    size = len(counts)
    if len(set(prefix)) != len(prefix):
        return False
    if any(counts[rank][agent] == 0 for rank, agent in enumerate(prefix)):
        return False
    remaining_agents = [agent for agent in range(size) if agent not in prefix]
    if not remaining_agents:
        return True

    source = 0
    rank_begin = 1
    agent_begin = rank_begin + size
    sink = agent_begin + size
    flow = max_flow.SimpleMaxFlow()
    for rank in range(len(prefix), size):
        flow.add_arc_with_capacity(source, rank_begin + rank, 1)
        for agent in remaining_agents:
            if counts[rank][agent] != 0:
                flow.add_arc_with_capacity(rank_begin + rank, agent_begin + agent, 1)
    for agent in remaining_agents:
        flow.add_arc_with_capacity(agent_begin + agent, sink, 1)
    status = flow.solve(source, sink)
    remaining_placement_seconds(deadline, "matching projection")
    if status != flow.OPTIMAL:
        raise RuntimeError(f"FFN Placement matching flow returned status {status}")
    return flow.optimal_flow() == len(remaining_agents)


def project_assignment_counts(
    counts: list[list[int]],
    *,
    deadline: float,
) -> tuple[tuple[int, ...], ...]:
    """Deterministically decompose one rank-by-FfnAgent count matrix."""

    tp_size = len(counts)
    ffnagent_count = len(counts[0])
    degree = sum(counts[0])
    if tp_size > ffnagent_count or any(sum(row) != degree for row in counts):
        raise RuntimeError("FFN Placement aggregate counts are not rank regular")
    deficits = [
        degree - sum(counts[rank][ffnagent_index] for rank in range(tp_size))
        for ffnagent_index in range(ffnagent_count)
    ]
    if any(deficit < 0 for deficit in deficits):
        raise RuntimeError("FFN Placement aggregate counts cannot be regularized")

    # Add dummy TP ranks to turn the rectangular count matrix into a regular
    # bipartite multigraph; every perfect matching then names one valid TP group.
    residual = [row.copy() for row in counts]
    for _ in range(ffnagent_count - tp_size):
        row = [0] * ffnagent_count
        remaining = degree
        for ffnagent_index in range(ffnagent_count):
            assigned = min(remaining, deficits[ffnagent_index])
            row[ffnagent_index] = assigned
            remaining -= assigned
            deficits[ffnagent_index] -= assigned
        if remaining != 0:
            raise RuntimeError("FFN Placement dummy-edge regularization failed")
        residual.append(row)
    if any(deficits) or any(sum(row) != degree for row in residual):
        raise RuntimeError("FFN Placement regularized graph has invalid degree")

    # Repeatedly peel the lexicographically first extendable perfect matching.
    # Exact matching feasibility prevents a locally small choice from trapping
    # the remaining multigraph.
    groups = []
    for _ in range(degree):
        prefix: tuple[int, ...] = ()
        for rank in range(ffnagent_count):
            for ffnagent_index in range(ffnagent_count):
                candidate = (*prefix, ffnagent_index)
                if perfect_matching_exists(residual, candidate, deadline=deadline):
                    prefix = candidate
                    break
            else:
                raise RuntimeError(f"FFN Placement has no matching extension at rank {rank}")
        groups.append(prefix[:tp_size])
        for rank, ffnagent_index in enumerate(prefix):
            residual[rank][ffnagent_index] -= 1
    if any(value != 0 for row in residual for value in row):
        raise RuntimeError("FFN Placement matching projection left residual edges")
    return tuple(groups)


def place_ffn_models(
    *,
    model_specs: tuple[ffn.FfnModelSpec, ...],
    instance_plans: tuple[FabricInstancePlan, ...],
    ffnagent_free_memory_bytes: tuple[int, ...],
    ffn_execution_contracts: tuple[FfnExecutionContract, ...],
    coefficients: FfnMemoryCalibrationCoefficients | None,
) -> FfnPlacementResult:
    """Validate one Placement request and solve it before the shared deadline."""

    try:
        estimator = DeviceMemoryEstimator(
            ffn_execution_contracts=ffn_execution_contracts,
            model_specs=model_specs,
            instance_profiles=tuple(instance.ffn_profile for instance in instance_plans),
            coefficients=coefficients,
        )
    except ValueError as error:
        raise XpoolDaemonError("conflict", str(error)) from error
    config = get_global_config()
    ffnagent_count = len(config.ffn.devices)
    if len(ffnagent_free_memory_bytes) != ffnagent_count:
        raise XpoolDaemonError(
            "conflict",
            "FfnAgent free-memory observations do not match configured count",
        )
    try:
        ensure_sint64_nonnegative(*ffnagent_free_memory_bytes)
        ensure_sint64_nonnegative(config.ffn.device_memory_extra_margin_bytes)
    except ValueError as error:
        raise XpoolDaemonError("conflict", str(error)) from error
    if any(value == 0 for value in ffnagent_free_memory_bytes):
        raise XpoolDaemonError("conflict", "FfnAgent free-memory observations must be positive")

    return solve_placement(
        ffnagent_free_memory_bytes=ffnagent_free_memory_bytes,
        estimator=estimator,
        deadline=monotonic() + config.ffn.placement.timeout_seconds,
    )


def solve_placement(
    *,
    ffnagent_free_memory_bytes: tuple[int, ...],
    estimator: DeviceMemoryEstimator,
    deadline: float,
) -> FfnPlacementResult:
    """Solve a validated request with ordered objectives and deterministic projection.

    Parallel CP-SAT solves fix each objective value in priority order. A final
    single-worker fixed search chooses one reproducible representative before
    aggregate assignment counts are decomposed into concrete TP groups.
    """

    model_specs = estimator.model_specs
    ffnagent_execution_contracts = estimator.ffn_execution_contracts
    config = get_global_config()
    ffnagent_count = len(config.ffn.devices)

    # Layers with identical per-rank resources and Graph
    # signatures are interchangeable to the solver and share one class.
    class_model_indices: list[int] = []
    class_layer_indices: list[tuple[int, ...]] = []
    class_rank_rows: list[tuple[tuple[int, int, int, tuple[execution.ExecutionSignature, ...]], ...]] = []
    for model_index, (model_config, spec, profile) in enumerate(
        zip(config.models, model_specs, estimator.instance_profiles, strict=True)
    ):
        tp_size = model_config.ffn_tp_size or ffnagent_count
        if tp_size > ffnagent_count:
            raise XpoolDaemonError("conflict", f"FFN TP size for {model_config.id!r} exceeds the FfnAgent Fleet")
        grouped: dict[
            tuple[tuple[int, int, int, tuple[execution.ExecutionSignature, ...]], ...],
            list[int],
        ] = {}
        for layer_ordinal in range(len(spec.layers)):
            try:
                rows = []
                for tp_rank in range(tp_size):
                    layout = preparation.prepare_ffn_layer(
                        spec,
                        layer_ordinal=layer_ordinal,
                        payload_dtype=profile.payload_dtype,
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                    )
                    storage_bytes = layout.weight_allocation_bytes()
                    rows.append(
                        (
                            sum(storage_bytes),
                            len(storage_bytes),
                            allocator_reservation_bytes(storage_bytes) - sum(storage_bytes),
                            layout.execution_signatures(profile=profile),
                        )
                    )
                rank_rows = tuple(rows)
            except ValueError as error:
                raise XpoolDaemonError("conflict", str(error)) from error
            grouped.setdefault(rank_rows, []).append(layer_ordinal)
        for rank_rows, layer_indices in grouped.items():
            class_model_indices.append(model_index)
            class_layer_indices.append(tuple(layer_indices))
            class_rank_rows.append(rank_rows)

    # Materialize each distinct Graph and captured-weight
    # resource once so placement variables can express device-local reuse.
    all_signatures = tuple(
        dict.fromkeys(
            signature for rank_rows in class_rank_rows for _, _, _, signatures in rank_rows for signature in signatures
        )
    )
    signature_index = {signature: index for index, signature in enumerate(all_signatures)}
    resource_plans = tuple(
        preparation.ExecutionResourcePlan.from_signatures(all_signatures, contract=contract)
        for contract in ffnagent_execution_contracts
    )
    capture_weight_pair_keys = tuple(dict.fromkeys(execution.capture_weight_pair_key(item) for item in all_signatures))
    signature_storage_bytes = {}
    signature_allocator_allowance_bytes = {}
    dense_signature_count = 0
    router_signatures: list[execution.MoeFfnExecutionSignature] = []
    for signature in all_signatures:
        signature_storage_bytes[signature] = (
            signature.capture_placeholder_layout().allocation_bytes() + TORCH_GRAPH_CAPTURE_STATE_REQUEST_BYTES
        )
        signature_allocator_allowance_bytes[signature] = graph_capture_allocator_allowance_bytes(signature)
        if isinstance(signature, execution.DenseFfnExecutionSignature):
            dense_signature_count += 1
        elif signature.router is not None:
            router_signatures.append(signature)
    moe_signature_count = len(all_signatures) - dense_signature_count
    routing_capacities = sorted({0, *(item.payload_row_capacity * item.effective_topk for item in router_signatures)})
    routing_bytes_table = [
        (capacity, estimator.routing_record_buffer_bytes(capacity)) for capacity in routing_capacities
    ]
    capture_weight_pair_storage_bytes = {
        key: execution.control_capture_probe_storage_bytes(key) for key in capture_weight_pair_keys
    }
    capture_weight_pair_allocator_allowance_bytes = {
        weight_pair_key: allocator_reservation_bytes(storage_bytes) - sum(storage_bytes)
        for weight_pair_key, storage_bytes in capture_weight_pair_storage_bytes.items()
    }
    remaining_placement_seconds(deadline, "resource calculation")

    maximum_weight_bytes = 0
    maximum_weight_allocations = 0
    maximum_weight_allowance = 0
    maximum_layer_capacity_count = 0
    for class_index, rank_rows in enumerate(class_rank_rows):
        layer_count = len(class_layer_indices[class_index])
        maximum_rank_capacity_count = 0
        for weight_bytes, allocation_count, allowance, signatures in rank_rows:
            maximum_weight_bytes += layer_count * weight_bytes
            maximum_weight_allocations += layer_count * allocation_count
            maximum_weight_allowance += layer_count * allowance
            maximum_rank_capacity_count = max(maximum_rank_capacity_count, len(signatures))
        maximum_layer_capacity_count += layer_count * maximum_rank_capacity_count
    maximum_graph_capture_bytes = config.scheduler.ffn_concurrency * sum(
        map(sum, signature_storage_bytes.values())
    ) + sum(map(sum, capture_weight_pair_storage_bytes.values()))
    maximum_graph_capture_allocations = config.scheduler.ffn_concurrency * (
        sum(map(len, signature_storage_bytes.values()))
    ) + sum(map(len, capture_weight_pair_storage_bytes.values()))
    maximum_graph_capture_allowance = config.scheduler.ffn_concurrency * sum(
        signature_allocator_allowance_bytes.values()
    ) + sum(capture_weight_pair_allocator_allowance_bytes.values())
    maximum_pool_bytes = config.scheduler.ffn_concurrency * max(
        (capture_pool_reservation_bytes(plan) for plan in resource_plans), default=0
    )
    maximum_resource_bytes = 0
    maximum_resource_allocations = 0
    for resource_plan in resource_plans:
        resource_bytes = 0
        resource_allocations = 0
        for group in resource_plan.lane_resources:
            resource_bytes += config.scheduler.ffn_concurrency * group.requirement.logical_extent_bound_bytes
            resource_allocations += config.scheduler.ffn_concurrency * len(group.requirement.request_bytes)
        maximum_resource_bytes = max(maximum_resource_bytes, resource_bytes)
        maximum_resource_allocations = max(maximum_resource_allocations, resource_allocations)
    maximum_runtime_bytes = estimator.execution_runtime_bytes(
        local_layer_count=sum(len(item) for item in class_layer_indices),
        local_layer_capacity_count=maximum_layer_capacity_count,
        local_signature_count=len(all_signatures),
        coordinator=True,
    )
    maximum_routing_elements = routing_capacities[-1]
    maximum_routing_bytes = estimator.routing_record_buffer_bytes(maximum_routing_elements)
    maximum_exact_bytes = (
        maximum_weight_bytes
        + estimator.fabric_arena_bytes()
        + estimator.fabric_observer_bytes()
        + maximum_graph_capture_bytes
        + maximum_pool_bytes
        + maximum_runtime_bytes
        + maximum_routing_bytes
    )
    maximum_calibrated_overhead = 0
    if estimator.coefficients is not None:
        maximum_calibrated_overhead = estimator.coefficients.evaluate(
            DeviceMemoryFeatures(
                tensor_storage_bytes=maximum_weight_bytes + maximum_graph_capture_bytes + maximum_resource_bytes,
                tensor_storage_allocation_count=maximum_weight_allocations
                + maximum_graph_capture_allocations
                + maximum_resource_allocations,
                dense_graph_capture_count=config.scheduler.ffn_concurrency * dense_signature_count,
                moe_graph_capture_count=config.scheduler.ffn_concurrency * moe_signature_count,
                executor_lane_count=config.scheduler.ffn_concurrency,
                compute_branch_count=config.scheduler.ffn_concurrency * len(all_signatures),
                dense_implementation_present=dense_signature_count != 0,
                moe_implementation_present=moe_signature_count != 0,
                joined_ffnagent=True,
            )
        )

    try:
        maximum_stage_overhead = (
            maximum_weight_allowance + maximum_graph_capture_allowance + maximum_calibrated_overhead
        )
        ensure_sint64_nonnegative(
            maximum_exact_bytes,
            maximum_stage_overhead,
            maximum_exact_bytes + maximum_stage_overhead + config.ffn.device_memory_extra_margin_bytes,
        )
    except ValueError as error:
        raise XpoolDaemonError("conflict", str(error)) from error

    # Assignment counts describe equivalent placements;
    # presence variables charge each device-local shared resource once.
    model = cp_model.CpModel()
    assignment_count: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for class_index, rank_rows in enumerate(class_rank_rows):
        layer_count = len(class_layer_indices[class_index])
        for tp_rank in range(len(rank_rows)):
            rank_variables = []
            for ffnagent_index in range(ffnagent_count):
                variable = model.new_int_var(
                    0,
                    layer_count,
                    f"assignment_c{class_index}_a{ffnagent_index}_r{tp_rank}",
                )
                assignment_count[class_index, ffnagent_index, tp_rank] = variable
                rank_variables.append(variable)
            model.add(sum(rank_variables) == layer_count)
        for ffnagent_index in range(ffnagent_count):
            model.add(
                sum(assignment_count[class_index, ffnagent_index, tp_rank] for tp_rank in range(len(rank_rows)))
                <= layer_count
            )

    signature_present: dict[tuple[int, int], cp_model.IntVar] = {}
    for ffnagent_index in range(ffnagent_count):
        for current_signature_index, signature in enumerate(all_signatures):
            users = [
                assignment_count[class_index, ffnagent_index, tp_rank]
                for class_index, rank_rows in enumerate(class_rank_rows)
                for tp_rank, row in enumerate(rank_rows)
                if signature in row[3]
            ]
            present = model.new_bool_var(f"signature_a{ffnagent_index}_s{current_signature_index}")
            signature_present[ffnagent_index, current_signature_index] = present
            if users:
                model.add(sum(users) >= 1).only_enforce_if(present)
                model.add(sum(users) == 0).only_enforce_if(~present)
            else:
                model.add(present == 0)

    capture_weight_pair_present: dict[tuple[int, int], cp_model.IntVar] = {}
    for ffnagent_index in range(ffnagent_count):
        for weight_pair_index, weight_pair_key in enumerate(capture_weight_pair_keys):
            users = [
                signature_present[ffnagent_index, signature_index[signature]]
                for signature in all_signatures
                if execution.capture_weight_pair_key(signature) == weight_pair_key
            ]
            present = model.new_bool_var(f"capture_weight_pair_a{ffnagent_index}_b{weight_pair_index}")
            capture_weight_pair_present[ffnagent_index, weight_pair_index] = present
            model.add_max_equality(present, users)

    primary_member: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for model_index, model_config in enumerate(config.models):
        tp_size = model_config.ffn_tp_size or ffnagent_count
        for tp_rank in range(tp_size):
            rank_variables = []
            for ffnagent_index in range(ffnagent_count):
                variable = model.new_bool_var(f"primary_m{model_index}_a{ffnagent_index}_r{tp_rank}")
                primary_member[model_index, ffnagent_index, tp_rank] = variable
                rank_variables.append(variable)
            model.add_exactly_one(rank_variables)
        for ffnagent_index in range(ffnagent_count):
            model.add(sum(primary_member[model_index, ffnagent_index, tp_rank] for tp_rank in range(tp_size)) <= 1)

    primary_layer_count: list[cp_model.IntVar] = []
    primary_assignment_count: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for class_index, rank_rows in enumerate(class_rank_rows):
        layer_count = len(class_layer_indices[class_index])
        model_index = class_model_indices[class_index]
        selected_count = model.new_int_var(0, layer_count, f"primary_count_c{class_index}")
        primary_layer_count.append(selected_count)
        for ffnagent_index in range(ffnagent_count):
            residuals = []
            for tp_rank in range(len(rank_rows)):
                selected = model.new_int_var(
                    0,
                    layer_count,
                    f"primary_assignment_c{class_index}_a{ffnagent_index}_r{tp_rank}",
                )
                model.add_multiplication_equality(
                    selected,
                    [selected_count, primary_member[model_index, ffnagent_index, tp_rank]],
                )
                model.add(assignment_count[class_index, ffnagent_index, tp_rank] >= selected)
                primary_assignment_count[class_index, ffnagent_index, tp_rank] = selected
                residuals.append(assignment_count[class_index, ffnagent_index, tp_rank] - selected)
            model.add(sum(residuals) <= layer_count - selected_count)
    for model_index in range(len(model_specs)):
        model.add(
            sum(
                primary_layer_count[class_index]
                for class_index, owner in enumerate(class_model_indices)
                if owner == model_index
            )
            >= 1
        )

    retained_bytes: list[cp_model.IntVar] = []
    peak_bytes: list[cp_model.IntVar] = []
    headroom: list[cp_model.IntVar] = []
    fabric_join_bytes = estimator.fabric_arena_bytes() + estimator.fabric_observer_bytes()
    lanes = config.scheduler.ffn_concurrency
    margin = config.ffn.device_memory_extra_margin_bytes

    # Every FfnAgent must fit the peak of the modeled startup
    # stages, not merely the retained steady-state allocation.
    for ffnagent_index, free_bytes in enumerate(ffnagent_free_memory_bytes):
        admission_limit = free_bytes - margin
        if admission_limit <= 0:
            raise XpoolDaemonError(
                "conflict",
                f"FfnAgent {ffnagent_index} has no memory after the configured margin",
            )
        packed_weight_bytes: cp_model.LinearExpr | int = 0
        packed_weight_allocations: cp_model.LinearExpr | int = 0
        packed_weight_allowance: cp_model.LinearExpr | int = 0
        local_layer_count: cp_model.LinearExpr | int = 0
        local_layer_capacity_count: cp_model.LinearExpr | int = 0
        for class_index, rank_rows in enumerate(class_rank_rows):
            for tp_rank, (weight_bytes, allocation_count, allowance, signatures) in enumerate(rank_rows):
                assigned = assignment_count[class_index, ffnagent_index, tp_rank]
                packed_weight_bytes += weight_bytes * assigned
                packed_weight_allocations += allocation_count * assigned
                packed_weight_allowance += allowance * assigned
                local_layer_count += assigned
                local_layer_capacity_count += len(signatures) * assigned

        graph_capture_bytes: cp_model.LinearExpr | int = 0
        graph_capture_allocations: cp_model.LinearExpr | int = 0
        graph_capture_allowance: cp_model.LinearExpr | int = 0
        local_signature_count: cp_model.LinearExpr | int = 0
        dense_signatures: list[cp_model.IntVar] = []
        moe_signatures: list[cp_model.IntVar] = []
        for current_signature_index, signature in enumerate(all_signatures):
            present = signature_present[ffnagent_index, current_signature_index]
            local_signature_count += present
            graph_capture_bytes += lanes * sum(signature_storage_bytes[signature]) * present
            graph_capture_allocations += lanes * len(signature_storage_bytes[signature]) * present
            graph_capture_allowance += lanes * signature_allocator_allowance_bytes[signature] * present
            if isinstance(signature, execution.DenseFfnExecutionSignature):
                dense_signatures.append(present)
            else:
                moe_signatures.append(present)
        for weight_pair_index, key in enumerate(capture_weight_pair_keys):
            present = capture_weight_pair_present[ffnagent_index, weight_pair_index]
            graph_capture_bytes += sum(capture_weight_pair_storage_bytes[key]) * present
            graph_capture_allocations += len(capture_weight_pair_storage_bytes[key]) * present
            graph_capture_allowance += capture_weight_pair_allocator_allowance_bytes[key] * present

        private_pool_bytes: cp_model.LinearExpr | int = 0
        resource_storage_bytes: cp_model.LinearExpr | int = 0
        resource_allocations: cp_model.LinearExpr | int = 0
        for resource_index, group in enumerate(resource_plans[ffnagent_index].lane_resources):
            if len(group.signature_indices) == 1:
                present = signature_present[ffnagent_index, group.signature_indices[0]]
            else:
                present = model.new_bool_var(f"resource_present_a{ffnagent_index}_r{resource_index}")
                model.add_max_equality(
                    present,
                    [signature_present[ffnagent_index, index] for index in group.signature_indices],
                )
            private_pool_bytes += lanes * allocator_reservation_bytes(group.requirement.request_bytes) * present
            resource_storage_bytes += lanes * group.requirement.logical_extent_bound_bytes * present
            resource_allocations += lanes * len(group.requirement.request_bytes) * present
        dense_present = model.new_bool_var(f"dense_present_a{ffnagent_index}")
        moe_present = model.new_bool_var(f"moe_present_a{ffnagent_index}")
        if dense_signatures:
            model.add_max_equality(dense_present, dense_signatures)
        else:
            model.add(dense_present == 0)
        if moe_signatures:
            model.add_max_equality(moe_present, moe_signatures)
        else:
            model.add(moe_present == 0)
        dense_graph_capture_count = lanes * sum(dense_signatures)
        moe_graph_capture_count = lanes * sum(moe_signatures)
        compute_branch_count = lanes * local_signature_count
        runtime_base = estimator.execution_runtime_bytes(
            local_layer_count=0,
            local_layer_capacity_count=0,
            local_signature_count=0,
            coordinator=ffnagent_index == 0,
        )
        runtime_per_layer = (
            estimator.execution_runtime_bytes(
                local_layer_count=1,
                local_layer_capacity_count=0,
                local_signature_count=0,
                coordinator=ffnagent_index == 0,
            )
            - runtime_base
        )
        runtime_per_capacity = (
            estimator.execution_runtime_bytes(
                local_layer_count=0,
                local_layer_capacity_count=1,
                local_signature_count=0,
                coordinator=ffnagent_index == 0,
            )
            - runtime_base
        )
        runtime_per_signature = (
            estimator.execution_runtime_bytes(
                local_layer_count=0,
                local_layer_capacity_count=0,
                local_signature_count=1,
                coordinator=ffnagent_index == 0,
            )
            - runtime_base
        )
        execution_runtime_bytes = (
            runtime_base
            + runtime_per_layer * local_layer_count
            + runtime_per_capacity * local_layer_capacity_count
            + runtime_per_signature * local_signature_count
        )
        if config.debug.ffn_routing_observer.enable and router_signatures:
            routing_elements_upper = maximum_routing_elements
            routing_elements = model.new_int_var(0, routing_elements_upper, f"routing_elements_a{ffnagent_index}")
            model.add_max_equality(
                routing_elements,
                [
                    0,
                    *(
                        signature.payload_row_capacity
                        * signature.effective_topk
                        * signature_present[ffnagent_index, signature_index[signature]]
                        for signature in router_signatures
                    ),
                ],
            )
            routing_bytes_upper = estimator.routing_record_buffer_bytes(routing_capacities[-1])
            routing_record_buffer_bytes = model.new_int_var(
                0,
                routing_bytes_upper,
                f"routing_bytes_a{ffnagent_index}",
            )
            model.add_allowed_assignments(
                [routing_elements, routing_record_buffer_bytes],
                routing_bytes_table,
            )
        else:
            routing_record_buffer_bytes = 0

        packed_mib = model.new_int_var(
            0,
            align_up(maximum_weight_bytes, MIB) // MIB,
            f"packed_mib_a{ffnagent_index}",
        )
        model.add(packed_weight_bytes <= MIB * packed_mib)
        model.add(MIB * packed_mib <= packed_weight_bytes + MIB - 1)
        retained_tensor_bytes = packed_weight_bytes + resource_storage_bytes
        retained_mib = model.new_int_var(
            0,
            align_up(maximum_weight_bytes + maximum_resource_bytes, MIB) // MIB,
            f"retained_mib_a{ffnagent_index}",
        )
        model.add(retained_tensor_bytes <= MIB * retained_mib)
        model.add(MIB * retained_mib <= retained_tensor_bytes + MIB - 1)
        graph_capture_tensor_bytes = packed_weight_bytes + resource_storage_bytes + graph_capture_bytes
        graph_capture_mib = model.new_int_var(
            0,
            align_up(
                maximum_weight_bytes + maximum_graph_capture_bytes + maximum_resource_bytes,
                MIB,
            )
            // MIB,
            f"graph_capture_mib_a{ffnagent_index}",
        )
        model.add(graph_capture_tensor_bytes <= MIB * graph_capture_mib)
        model.add(MIB * graph_capture_mib <= graph_capture_tensor_bytes + MIB - 1)

        def calibrated_overhead(
            *,
            tensor_storage_mib: cp_model.IntVar,
            tensor_storage_allocation_count: cp_model.LinearExpr | int,
            dense_capture_count: cp_model.LinearExpr | int,
            moe_capture_count: cp_model.LinearExpr | int,
            lane_count: int,
            branch_count: cp_model.LinearExpr | int,
            joined: int,
        ) -> cp_model.LinearExpr | int:
            if estimator.coefficients is None:
                return 0
            coefficients = estimator.coefficients
            return (
                coefficients.base_bytes
                + coefficients.bytes_per_tensor_storage_mib * tensor_storage_mib
                + coefficients.bytes_per_tensor_storage_allocation * tensor_storage_allocation_count
                + coefficients.bytes_per_dense_graph_capture * dense_capture_count
                + coefficients.bytes_per_moe_graph_capture * moe_capture_count
                + coefficients.bytes_per_executor_lane * lane_count
                + coefficients.bytes_per_compute_branch * branch_count
                + coefficients.dense_implementation_bytes * (dense_present if lane_count else 0)
                + coefficients.moe_implementation_bytes * (moe_present if lane_count else 0)
                + coefficients.joined_ffnagent_bytes * joined
            )

        weight_physical = packed_weight_bytes + packed_weight_allowance
        joined_physical = weight_physical + fabric_join_bytes
        captured_physical = joined_physical + private_pool_bytes + graph_capture_bytes + graph_capture_allowance
        installed_native = execution_runtime_bytes + routing_record_buffer_bytes
        physical_costs = {
            "weight_materialization": weight_physical,
            "fabric_join": joined_physical,
            "graph_capture": captured_physical,
            "execution_installation": captured_physical + installed_native,
            "retained": joined_physical + private_pool_bytes + installed_native,
        }
        capture_correction = calibrated_overhead(
            tensor_storage_mib=graph_capture_mib,
            tensor_storage_allocation_count=packed_weight_allocations
            + graph_capture_allocations
            + resource_allocations,
            dense_capture_count=dense_graph_capture_count,
            moe_capture_count=moe_graph_capture_count,
            lane_count=lanes,
            branch_count=compute_branch_count,
            joined=1,
        )
        calibrated_overheads = {
            "weight_materialization": calibrated_overhead(
                tensor_storage_mib=packed_mib,
                tensor_storage_allocation_count=packed_weight_allocations,
                dense_capture_count=0,
                moe_capture_count=0,
                lane_count=0,
                branch_count=0,
                joined=0,
            ),
            "fabric_join": calibrated_overhead(
                tensor_storage_mib=packed_mib,
                tensor_storage_allocation_count=packed_weight_allocations,
                dense_capture_count=0,
                moe_capture_count=0,
                lane_count=0,
                branch_count=0,
                joined=1,
            ),
            "graph_capture": capture_correction,
            "execution_installation": capture_correction,
            "retained": calibrated_overhead(
                tensor_storage_mib=retained_mib,
                tensor_storage_allocation_count=packed_weight_allocations + resource_allocations,
                dense_capture_count=0,
                moe_capture_count=0,
                lane_count=lanes,
                branch_count=compute_branch_count,
                joined=1,
            ),
        }
        stage_values = {}
        for point, physical_cost in physical_costs.items():
            value = model.new_int_var(0, admission_limit, f"stage_a{ffnagent_index}_{point}")
            model.add(value == physical_cost + calibrated_overheads[point])
            stage_values[point] = value
        peak = model.new_int_var(0, admission_limit, f"peak_a{ffnagent_index}")
        model.add_max_equality(peak, list(stage_values.values()))
        retained = stage_values["retained"]
        available = model.new_int_var(0, admission_limit, f"headroom_a{ffnagent_index}")
        model.add(available == admission_limit - peak)
        retained_bytes.append(retained)
        peak_bytes.append(peak)
        headroom.append(available)

    minimum_headroom = model.new_int_var(0, max(ffnagent_free_memory_bytes), "minimum_headroom")
    model.add_min_equality(minimum_headroom, headroom)
    signature_materialization_count = sum(signature_present.values())
    primary_match_count = sum(primary_layer_count)
    total_peak_bytes = sum(peak_bytes)
    objectives = (
        ("signature materialization", "min", signature_materialization_count),
        ("Primary Group reuse", "max", primary_match_count),
        ("minimum headroom", "max", minimum_headroom),
        ("total peak bytes", "min", total_peak_bytes),
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = config.ffn.placement.parallelism
    solver.parameters.random_seed = 1

    # Fix each lexicographic objective at its optimum
    # before a deterministic fixed search selects one equivalent solution.
    completed_objectives: list[tuple[str, int]] = []
    for objective_index, (name, direction, expression) in enumerate(objectives):
        solver.parameters.max_time_in_seconds = remaining_placement_seconds(deadline, name)
        if direction == "min":
            model.minimize(expression)
        else:
            model.maximize(expression)
        status = solver.solve(model)
        remaining_placement_seconds(deadline, name)
        if status != cp_model.OPTIMAL:
            status_name = solver.status_name(status)
            diagnostic = (
                f"FFN Placement {name} returned {status_name}; completed={completed_objectives}; "
                f"classes={len(class_rank_rows)}; signatures={len(all_signatures)}; "
                f"parallelism={config.ffn.placement.parallelism}"
            )
            if objective_index == 0 and status == cp_model.INFEASIBLE:
                raise XpoolDaemonError("conflict", diagnostic)
            if status == cp_model.MODEL_INVALID or status == cp_model.INFEASIBLE:
                raise RuntimeError(diagnostic)
            raise XpoolDaemonError("not_ready", diagnostic)
        value = int(solver.value(expression))
        completed_objectives.append((name, value))
        model.add(expression == value)

    model.clear_objective()
    model.add_decision_strategy(
        [assignment_count[coordinate] for coordinate in sorted(assignment_count)],
        cp_model.CHOOSE_FIRST,
        cp_model.SELECT_MAX_VALUE,
    )
    model.add_decision_strategy(
        [primary_member[coordinate] for coordinate in sorted(primary_member)],
        cp_model.CHOOSE_FIRST,
        cp_model.SELECT_MAX_VALUE,
    )
    solver.parameters.num_workers = 1
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    # Preserve the declared coordinate order when equal solutions are interchangeable.
    solver.parameters.cp_model_presolve = False
    solver.parameters.max_time_in_seconds = remaining_placement_seconds(deadline, "representative selection")
    status = solver.solve(model)
    remaining_placement_seconds(deadline, "representative selection")
    if status != cp_model.OPTIMAL:
        status_name = solver.status_name(status)
        diagnostic = f"FFN Placement representative returned {status_name}; completed={completed_objectives}"
        if status == cp_model.MODEL_INVALID or status == cp_model.INFEASIBLE:
            raise RuntimeError(diagnostic)
        raise XpoolDaemonError("not_ready", diagnostic)

    # Decompose aggregate assignment counts into concrete
    # ordered TP groups, then prove that projection preserves solver totals.
    groups_by_coordinate: dict[tuple[int, int], tuple[int, ...]] = {}
    for class_index, rank_rows in enumerate(class_rank_rows):
        model_index = class_model_indices[class_index]
        primary_group = tuple(
            next(
                ffnagent_index
                for ffnagent_index in range(ffnagent_count)
                if solver.value(primary_member[model_index, ffnagent_index, tp_rank])
            )
            for tp_rank in range(len(rank_rows))
        )
        primary_count = solver.value(primary_layer_count[class_index])
        layer_indices = class_layer_indices[class_index]
        for layer_ordinal in layer_indices[:primary_count]:
            groups_by_coordinate[model_index, layer_ordinal] = primary_group
        residual_counts = [
            [
                solver.value(assignment_count[class_index, ffnagent_index, tp_rank])
                - solver.value(primary_assignment_count[class_index, ffnagent_index, tp_rank])
                for ffnagent_index in range(ffnagent_count)
            ]
            for tp_rank in range(len(rank_rows))
        ]
        projected = project_assignment_counts(residual_counts, deadline=deadline)
        for layer_ordinal, group in zip(layer_indices[primary_count:], projected, strict=True):
            groups_by_coordinate[model_index, layer_ordinal] = group

    try:
        model_plans = tuple(
            FfnModelPlan(
                model_spec_digest=spec.digest(),
                layers=tuple(
                    place_ffn_layer(
                        layer,
                        groups_by_coordinate[model_index, layer_ordinal],
                        local_intermediate_size=preparation.prepare_ffn_layer(
                            spec,
                            layer_ordinal=layer_ordinal,
                            payload_dtype=estimator.instance_profiles[model_index].payload_dtype,
                            tp_size=len(groups_by_coordinate[model_index, layer_ordinal]),
                            tp_rank=0,
                        ).local_intermediate_size,
                    )
                    for layer_ordinal, layer in enumerate(spec.layers)
                ),
            )
            for model_index, spec in enumerate(model_specs)
        )
    except ValueError as error:
        raise XpoolDaemonError("conflict", str(error)) from error
    for class_index, rank_rows in enumerate(class_rank_rows):
        projected_counts = [[0] * ffnagent_count for _ in rank_rows]
        model_index = class_model_indices[class_index]
        for layer_ordinal in class_layer_indices[class_index]:
            for tp_rank, ffnagent_index in enumerate(groups_by_coordinate[model_index, layer_ordinal]):
                projected_counts[tp_rank][ffnagent_index] += 1
        solved_counts = [
            [
                solver.value(assignment_count[class_index, ffnagent_index, tp_rank])
                for ffnagent_index in range(ffnagent_count)
            ]
            for tp_rank in range(len(rank_rows))
        ]
        if projected_counts != solved_counts:
            raise RuntimeError("FFN Placement projection does not reproduce solved assignment counts")
    estimates = []
    for ffnagent_index in range(ffnagent_count):
        try:
            estimate = estimator.estimate_model_plans(
                model_plans=model_plans,
                ffnagent_index=ffnagent_index,
            )
        except ValueError as error:
            raise XpoolDaemonError("not_ready", str(error)) from error
        if estimate.retained_bytes != solver.value(
            retained_bytes[ffnagent_index]
        ) or estimate.peak_bytes != solver.value(peak_bytes[ffnagent_index]):
            raise RuntimeError(f"FFN Placement memory mismatch for FfnAgent {ffnagent_index}")
        estimates.append(estimate)
    remaining_placement_seconds(deadline, "postsolve validation")
    return FfnPlacementResult(model_plans, tuple(estimates))
