"""Generation-static memory-admitted FFN placement."""

from __future__ import annotations

from time import monotonic

from ortools.graph.python import max_flow
from ortools.sat.python import cp_model

from xpool import ffn
from xpool.fabric import (
    DenseFfnLayerPlan,
    FabricInstancePlan,
    FfnModelPlan,
    MoeFfnLayerPlan,
)
from xpool.memory import MIB, ensure_nonnegative_int64
from xpool.runtime.ffnagent import execution
from xpool.runtime.ffnagent.device_memory import (
    DeviceMemoryEstimator,
    allocator_block_allowance_bytes,
)
from xpool.service.errors import XpoolDaemonError
from xpool.utils import align_up


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
) -> tuple[FfnModelPlan, ...]:
    """Validate one Placement request and solve it before the shared deadline."""

    try:
        estimator = DeviceMemoryEstimator(
            model_specs=model_specs,
            instance_profiles=tuple(instance.ffn_profile for instance in instance_plans),
        )
    except ValueError as error:
        raise XpoolDaemonError("conflict", str(error)) from error
    config = estimator.config
    ffnagent_count = len(config.devices.ffn_cuda_devices)
    if len(instance_plans) != len(config.models) or len(ffnagent_free_memory_bytes) != ffnagent_count:
        raise XpoolDaemonError(
            "conflict",
            "configured Models, Instance Plans, and FfnAgent memory are not co-indexed",
        )
    try:
        ensure_nonnegative_int64(*ffnagent_free_memory_bytes)
        ensure_nonnegative_int64(config.ffn.placement.device_memory_extra_margin_bytes)
    except ValueError as error:
        raise XpoolDaemonError("conflict", str(error)) from error
    if any(value == 0 for value in ffnagent_free_memory_bytes):
        raise XpoolDaemonError("conflict", "FfnAgent free-memory observations must be positive")

    return PlacementSolver(
        model_specs=model_specs,
        instance_plans=instance_plans,
        ffnagent_free_memory_bytes=ffnagent_free_memory_bytes,
        estimator=estimator,
        deadline=monotonic() + config.ffn.placement.optimizer.timeout_seconds,
    ).solve()


class PlacementSolver:
    """Solve one Placement with ordered objectives and deterministic projection.

    Parallel CP-SAT solves fix each objective value in priority order. A final
    single-worker fixed search chooses one reproducible representative before
    aggregate assignment counts are decomposed into concrete TP groups.
    """

    def __init__(
        self,
        *,
        model_specs: tuple[ffn.FfnModelSpec, ...],
        instance_plans: tuple[FabricInstancePlan, ...],
        ffnagent_free_memory_bytes: tuple[int, ...],
        estimator: DeviceMemoryEstimator,
        deadline: float,
    ) -> None:
        """Retain one validated request and its shared monotonic deadline."""

        self.model_specs = model_specs
        self.instance_plans = instance_plans
        self.ffnagent_free_memory_bytes = ffnagent_free_memory_bytes
        self.estimator = estimator
        self.deadline = deadline

    def solve(self) -> tuple[FfnModelPlan, ...]:
        """Return the canonical memory-admitted Placement before the shared deadline."""

        model_specs = self.model_specs
        instance_plans = self.instance_plans
        ffnagent_free_memory_bytes = self.ffnagent_free_memory_bytes
        estimator = self.estimator
        deadline = self.deadline
        config = estimator.config
        ffnagent_count = len(config.devices.ffn_cuda_devices)

        # Layers with identical per-rank resources and Graph
        # signatures are interchangeable to the solver and share one class.
        class_model_indices: list[int] = []
        class_layer_indices: list[tuple[int, ...]] = []
        class_rank_rows: list[tuple[tuple[int, int, int, tuple[execution.ExecutionSignature, ...]], ...]] = []
        for model_index, (model_config, spec, instance_plan) in enumerate(
            zip(config.models, model_specs, instance_plans, strict=True)
        ):
            if spec.model_id != model_config.id:
                raise XpoolDaemonError("conflict", "FfnAgent Model Specs do not follow configured Model order")
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
                        storage_bytes = estimator.packed_weight_storage_bytes(
                            model_index,
                            layer_ordinal,
                            tp_rank,
                        )
                        rows.append(
                            (
                                sum(storage_bytes),
                                len(storage_bytes),
                                allocator_block_allowance_bytes(storage_bytes),
                                execution.required_execution_signatures(
                                    model_spec=spec,
                                    profile=instance_plan.ffn_profile,
                                    layer_ordinal=layer_ordinal,
                                    tp_size=tp_size,
                                    tp_rank=tp_rank,
                                ),
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
                signature
                for rank_rows in class_rank_rows
                for _, _, _, signatures in rank_rows
                for signature in signatures
            )
        )
        signature_index = {signature: index for index, signature in enumerate(all_signatures)}
        capture_weight_pair_keys = tuple(
            dict.fromkeys(execution.capture_weight_pair_key(item) for item in all_signatures)
        )
        capture_weight_pair_index = {key: index for index, key in enumerate(capture_weight_pair_keys)}
        signature_storage_bytes = {
            signature: execution.graph_capture_capacity_storage_bytes(signature) for signature in all_signatures
        }
        capture_weight_pair_storage_bytes = {
            weight_pair_key: execution.control_capture_probe_storage_bytes(weight_pair_key)
            for weight_pair_key in capture_weight_pair_keys
        }
        signature_allocator_allowance_bytes = {
            signature: allocator_block_allowance_bytes(storage_bytes)
            for signature, storage_bytes in signature_storage_bytes.items()
        }
        capture_weight_pair_allocator_allowance_bytes = {
            weight_pair_key: allocator_block_allowance_bytes(storage_bytes)
            for weight_pair_key, storage_bytes in capture_weight_pair_storage_bytes.items()
        }
        remaining_placement_seconds(deadline, "resource calculation")

        maximum_weight_bytes = sum(
            len(class_layer_indices[class_index]) * sum(row[0] for row in rank_rows)
            for class_index, rank_rows in enumerate(class_rank_rows)
        )
        maximum_weight_allocations = sum(
            len(class_layer_indices[class_index]) * sum(row[1] for row in rank_rows)
            for class_index, rank_rows in enumerate(class_rank_rows)
        )
        maximum_weight_allowance = sum(
            len(class_layer_indices[class_index]) * sum(row[2] for row in rank_rows)
            for class_index, rank_rows in enumerate(class_rank_rows)
        )
        maximum_graph_capture_bytes = sum(map(sum, signature_storage_bytes.values())) + sum(
            map(sum, capture_weight_pair_storage_bytes.values())
        )
        maximum_graph_capture_allocations = sum(map(len, signature_storage_bytes.values())) + sum(
            map(len, capture_weight_pair_storage_bytes.values())
        )
        maximum_graph_capture_allowance = sum(signature_allocator_allowance_bytes.values()) + sum(
            capture_weight_pair_allocator_allowance_bytes.values()
        )
        maximum_workspace_bytes = config.scheduler.ffn_concurrency * max(
            (execution.compute_workspace_bytes(item) for item in all_signatures),
            default=0,
        )
        maximum_runtime_bytes = estimator.execution_runtime_bytes(
            local_layer_count=sum(len(item) for item in class_layer_indices),
            local_layer_capacity_count=sum(
                len(class_layer_indices[class_index]) * max(len(row[3]) for row in rank_rows)
                for class_index, rank_rows in enumerate(class_rank_rows)
            ),
            local_signature_count=len(all_signatures),
            coordinator=True,
        )
        maximum_routing_elements = max(
            (
                item.payload_row_capacity * item.effective_topk
                for item in all_signatures
                if isinstance(item, execution.MoeFfnExecutionSignature) and item.router is not None
            ),
            default=0,
        )
        maximum_routing_bytes = estimator.routing_record_buffer_bytes(maximum_routing_elements)
        maximum_exact_bytes = (
            maximum_weight_bytes
            + estimator.fabric_arena_bytes()
            + estimator.fabric_observer_bytes()
            + maximum_graph_capture_bytes
            + maximum_workspace_bytes
            + maximum_runtime_bytes
            + maximum_routing_bytes
        )
        maximum_calibrated_overhead = 0
        if estimator.coefficients is not None:
            maximum_calibrated_overhead = estimator.coefficients.evaluate(
                tensor_storage_bytes=maximum_weight_bytes + maximum_graph_capture_bytes,
                tensor_storage_allocation_count=maximum_weight_allocations + maximum_graph_capture_allocations,
                dense_graph_capture_count=sum(
                    isinstance(item, execution.DenseFfnExecutionSignature) for item in all_signatures
                ),
                moe_graph_capture_count=sum(
                    isinstance(item, execution.MoeFfnExecutionSignature) for item in all_signatures
                ),
                executor_lane_count=config.scheduler.ffn_concurrency,
                compute_branch_count=config.scheduler.ffn_concurrency * len(all_signatures),
                dense_implementation_present=any(
                    isinstance(item, execution.DenseFfnExecutionSignature) for item in all_signatures
                ),
                moe_implementation_present=any(
                    isinstance(item, execution.MoeFfnExecutionSignature) for item in all_signatures
                ),
                joined_ffnagent=True,
            )
        try:
            maximum_stage_overhead = (
                maximum_weight_allowance + maximum_graph_capture_allowance + maximum_calibrated_overhead
            )
            ensure_nonnegative_int64(
                maximum_exact_bytes,
                maximum_stage_overhead,
                maximum_exact_bytes + maximum_stage_overhead + config.ffn.placement.device_memory_extra_margin_bytes,
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
        margin = config.ffn.placement.device_memory_extra_margin_bytes

        # Every FfnAgent must fit the peak of the modeled startup
        # stages, not merely the retained steady-state allocation.
        for ffnagent_index, free_bytes in enumerate(ffnagent_free_memory_bytes):
            admission_limit = free_bytes - margin
            if admission_limit <= 0:
                raise XpoolDaemonError(
                    "conflict",
                    f"FfnAgent {ffnagent_index} has no memory after the configured margin",
                )
            packed_weight_bytes = sum(
                row[0] * assignment_count[class_index, ffnagent_index, tp_rank]
                for class_index, rank_rows in enumerate(class_rank_rows)
                for tp_rank, row in enumerate(rank_rows)
            )
            packed_weight_allocations = sum(
                row[1] * assignment_count[class_index, ffnagent_index, tp_rank]
                for class_index, rank_rows in enumerate(class_rank_rows)
                for tp_rank, row in enumerate(rank_rows)
            )
            packed_weight_allowance = sum(
                row[2] * assignment_count[class_index, ffnagent_index, tp_rank]
                for class_index, rank_rows in enumerate(class_rank_rows)
                for tp_rank, row in enumerate(rank_rows)
            )
            graph_capture_bytes = sum(
                sum(signature_storage_bytes[signature]) * signature_present[ffnagent_index, signature_index[signature]]
                for signature in all_signatures
            ) + sum(
                sum(capture_weight_pair_storage_bytes[weight_pair_key])
                * capture_weight_pair_present[ffnagent_index, capture_weight_pair_index[weight_pair_key]]
                for weight_pair_key in capture_weight_pair_keys
            )
            graph_capture_allocations = sum(
                len(signature_storage_bytes[signature]) * signature_present[ffnagent_index, signature_index[signature]]
                for signature in all_signatures
            ) + sum(
                len(capture_weight_pair_storage_bytes[weight_pair_key])
                * capture_weight_pair_present[ffnagent_index, capture_weight_pair_index[weight_pair_key]]
                for weight_pair_key in capture_weight_pair_keys
            )
            graph_capture_allowance = sum(
                signature_allocator_allowance_bytes[signature]
                * signature_present[ffnagent_index, signature_index[signature]]
                for signature in all_signatures
            ) + sum(
                capture_weight_pair_allocator_allowance_bytes[weight_pair_key]
                * capture_weight_pair_present[ffnagent_index, capture_weight_pair_index[weight_pair_key]]
                for weight_pair_key in capture_weight_pair_keys
            )
            dense_signatures = [
                signature_present[ffnagent_index, signature_index[signature]]
                for signature in all_signatures
                if isinstance(signature, execution.DenseFfnExecutionSignature)
            ]
            moe_signatures = [
                signature_present[ffnagent_index, signature_index[signature]]
                for signature in all_signatures
                if isinstance(signature, execution.MoeFfnExecutionSignature)
            ]
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
            dense_graph_capture_count = sum(dense_signatures)
            moe_graph_capture_count = sum(moe_signatures)
            local_layer_count = sum(
                assignment_count[class_index, ffnagent_index, tp_rank]
                for class_index, rank_rows in enumerate(class_rank_rows)
                for tp_rank in range(len(rank_rows))
            )
            local_layer_capacity_count = sum(
                len(row[3]) * assignment_count[class_index, ffnagent_index, tp_rank]
                for class_index, rank_rows in enumerate(class_rank_rows)
                for tp_rank, row in enumerate(rank_rows)
            )
            local_signature_count = sum(
                signature_present[ffnagent_index, current_signature_index]
                for current_signature_index in range(len(all_signatures))
            )
            compute_branch_count = lanes * local_signature_count
            maximum_workspace = model.new_int_var(0, maximum_workspace_bytes // lanes, f"workspace_a{ffnagent_index}")
            model.add_max_equality(
                maximum_workspace,
                [
                    0,
                    *(
                        execution.compute_workspace_bytes(signature)
                        * signature_present[ffnagent_index, signature_index[signature]]
                        for signature in all_signatures
                    ),
                ],
            )
            lane_workspace_bytes = lanes * maximum_workspace
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
            router_signatures = [
                signature
                for signature in all_signatures
                if isinstance(signature, execution.MoeFfnExecutionSignature) and signature.router is not None
            ]
            if config.debug.ffn_routing_observer.enable and router_signatures:
                routing_elements_upper = max(
                    signature.payload_row_capacity * signature.effective_topk for signature in router_signatures
                )
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
                routing_capacities = sorted(
                    {
                        0,
                        *(signature.payload_row_capacity * signature.effective_topk for signature in router_signatures),
                    }
                )
                routing_bytes_upper = estimator.routing_record_buffer_bytes(routing_capacities[-1])
                routing_record_buffer_bytes = model.new_int_var(
                    0,
                    routing_bytes_upper,
                    f"routing_bytes_a{ffnagent_index}",
                )
                model.add_allowed_assignments(
                    [routing_elements, routing_record_buffer_bytes],
                    [(capacity, estimator.routing_record_buffer_bytes(capacity)) for capacity in routing_capacities],
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
            graph_capture_tensor_bytes = packed_weight_bytes + graph_capture_bytes
            graph_capture_mib = model.new_int_var(
                0,
                align_up(maximum_weight_bytes + maximum_graph_capture_bytes, MIB) // MIB,
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
                    + coefficients.dense_implementation_bytes * dense_present
                    + coefficients.moe_implementation_bytes * moe_present
                    + coefficients.joined_ffnagent_bytes * joined
                )

            retained_exact = (
                packed_weight_bytes
                + fabric_join_bytes
                + lane_workspace_bytes
                + execution_runtime_bytes
                + routing_record_buffer_bytes
            )
            exact_resource_ledgers = (
                packed_weight_bytes,
                packed_weight_bytes + fabric_join_bytes,
                packed_weight_bytes + fabric_join_bytes + graph_capture_bytes,
                retained_exact + graph_capture_bytes,
                retained_exact,
            )
            allocator_allowances = (
                packed_weight_allowance,
                packed_weight_allowance,
                packed_weight_allowance + graph_capture_allowance,
                packed_weight_allowance + graph_capture_allowance,
                packed_weight_allowance,
            )
            calibrated_overheads = (
                calibrated_overhead(
                    tensor_storage_mib=packed_mib,
                    tensor_storage_allocation_count=packed_weight_allocations,
                    dense_capture_count=0,
                    moe_capture_count=0,
                    lane_count=0,
                    branch_count=0,
                    joined=0,
                ),
                calibrated_overhead(
                    tensor_storage_mib=packed_mib,
                    tensor_storage_allocation_count=packed_weight_allocations,
                    dense_capture_count=0,
                    moe_capture_count=0,
                    lane_count=0,
                    branch_count=0,
                    joined=1,
                ),
                calibrated_overhead(
                    tensor_storage_mib=graph_capture_mib,
                    tensor_storage_allocation_count=packed_weight_allocations + graph_capture_allocations,
                    dense_capture_count=dense_graph_capture_count,
                    moe_capture_count=moe_graph_capture_count,
                    lane_count=0,
                    branch_count=0,
                    joined=1,
                ),
                calibrated_overhead(
                    tensor_storage_mib=graph_capture_mib,
                    tensor_storage_allocation_count=packed_weight_allocations + graph_capture_allocations,
                    dense_capture_count=dense_graph_capture_count,
                    moe_capture_count=moe_graph_capture_count,
                    lane_count=lanes,
                    branch_count=compute_branch_count,
                    joined=1,
                ),
                calibrated_overhead(
                    tensor_storage_mib=packed_mib,
                    tensor_storage_allocation_count=packed_weight_allocations,
                    dense_capture_count=0,
                    moe_capture_count=0,
                    lane_count=lanes,
                    branch_count=compute_branch_count,
                    joined=1,
                ),
            )
            stage_values = []
            for point_index, (exact_resource_ledger, allocator_allowance, calibrated) in enumerate(
                zip(exact_resource_ledgers, allocator_allowances, calibrated_overheads, strict=True)
            ):
                value = model.new_int_var(0, admission_limit, f"stage_a{ffnagent_index}_p{point_index}")
                model.add(value == exact_resource_ledger + allocator_allowance + calibrated)
                stage_values.append(value)
            peak = model.new_int_var(0, admission_limit, f"peak_a{ffnagent_index}")
            model.add_max_equality(peak, stage_values)
            retained = stage_values[-1]
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
        solver.parameters.num_workers = config.ffn.placement.optimizer.parallelism
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
                    f"parallelism={config.ffn.placement.optimizer.parallelism}"
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

        model_plans = tuple(
            FfnModelPlan(
                model_spec_digest=spec.digest(),
                layers=tuple(
                    place_layer(layer, groups_by_coordinate[model_index, layer_ordinal])
                    for layer_ordinal, layer in enumerate(spec.layers)
                ),
            )
            for model_index, spec in enumerate(model_specs)
        )
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
        for ffnagent_index in range(ffnagent_count):
            estimate = estimator.estimate_model_plans(
                model_plans=model_plans,
                ffnagent_index=ffnagent_index,
            )
            if estimate.retained_bytes != solver.value(
                retained_bytes[ffnagent_index]
            ) or estimate.peak_bytes != solver.value(peak_bytes[ffnagent_index]):
                raise RuntimeError(f"FFN Placement memory mismatch for FfnAgent {ffnagent_index}")
        remaining_placement_seconds(deadline, "postsolve validation")
        return model_plans


def place_layer(
    layer: ffn.FfnLayerSpec,
    execution_group: tuple[int, ...],
) -> DenseFfnLayerPlan | MoeFfnLayerPlan:
    """Project one intrinsic layer into one true-TP realization."""

    tp_size = len(execution_group)
    if isinstance(layer, ffn.DenseFfnSpec):
        if layer.intermediate_size % tp_size:
            raise XpoolDaemonError(
                "conflict", f"Dense layer {layer.layer_id} intermediate size is not divisible by FFN TP"
            )
        return DenseFfnLayerPlan(
            ffnagent_indices=execution_group,
            local_intermediate_size=layer.intermediate_size // tp_size,
        )

    if layer.expert_intermediate_size % tp_size:
        raise XpoolDaemonError("conflict", f"MoE layer {layer.layer_id} intermediate size is not divisible by FFN TP")
    return MoeFfnLayerPlan(
        ffnagent_indices=execution_group,
        local_intermediate_size=layer.expert_intermediate_size // tp_size,
        effective_topk=layer.routed_topk + layer.shared_expert_count,
    )
