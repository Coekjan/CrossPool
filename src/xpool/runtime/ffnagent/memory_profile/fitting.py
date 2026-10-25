"""Calibrated FFN upper-envelope fitting from attributed measurements."""

from __future__ import annotations

import math
from dataclasses import fields

from ortools.sat.python import cp_model

from xpool.memory import (
    MIB,
    FfnMemoryCalibrationCoefficients,
)
from xpool.runtime.ffnagent.device_memory import DeviceMemoryFeatures
from xpool.runtime.ffnagent.memory_profile.evidence import MemoryProfileWorld
from xpool.utils.integer import SINT64_MAX

COEFFICIENT_NAMES = tuple(field.name for field in fields(FfnMemoryCalibrationCoefficients))


def feature_vector(features: DeviceMemoryFeatures) -> tuple[int, ...]:
    """Project one accepted feature row into stable coefficient order."""

    return (
        1,
        (features.tensor_storage_bytes + MIB - 1) // MIB,
        features.tensor_storage_allocation_count,
        features.dense_graph_capture_count,
        features.moe_graph_capture_count,
        features.executor_lane_count,
        features.compute_branch_count,
        int(features.dense_implementation_present),
        int(features.moe_implementation_present),
        int(features.joined_ffnagent),
    )


def solve_coefficients(
    targets: dict[tuple[int, ...], int],
    observation_quantum_bytes: int,
) -> FfnMemoryCalibrationCoefficients:
    """Solve the quantized cover objective and canonical coefficient tie-break."""

    if not targets:
        raise ValueError("memory-profile fit produced no target rows")
    if observation_quantum_bytes <= 0:
        raise ValueError("memory-profile fit requires a positive observation quantum")
    if any(all(vector[index] == 0 for vector in targets) for index in range(len(COEFFICIENT_NAMES))):
        raise ValueError("memory-profile matrix leaves one coefficient feature unexercised")
    quantized_targets = {
        vector: (target + observation_quantum_bytes - 1) // observation_quantum_bytes
        for vector, target in targets.items()
    }
    maximum_target = max(quantized_targets.values())
    if (
        maximum_target * observation_quantum_bytes > SINT64_MAX
        or maximum_target * sum(sum(vector) for vector in quantized_targets) > SINT64_MAX
    ):
        raise ValueError("memory-profile fit exceeds signed-64-bit arithmetic")

    model = cp_model.CpModel()
    variables = tuple(model.new_int_var(0, maximum_target, name) for name in COEFFICIENT_NAMES)
    predictions = {
        vector: sum(value * variable for value, variable in zip(vector, variables, strict=True))
        for vector in quantized_targets
    }
    for vector, target in quantized_targets.items():
        model.add(predictions[vector] >= target)
    overprediction = sum(predictions[vector] - target for vector, target in quantized_targets.items())
    model.minimize(overprediction)

    def solve_optimal() -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        if solver.solve(model) != cp_model.OPTIMAL:
            raise RuntimeError("memory-profile coefficient fit did not prove OPTIMAL")
        return solver

    solver = solve_optimal()
    model.add(overprediction == int(solver.value(overprediction)))
    values = []
    for variable in variables:
        # The previous optimum stays feasible after fixing its objective value.
        model.clear_hints()
        for candidate in variables:
            model.add_hint(candidate, int(solver.value(candidate)))
        model.minimize(variable)
        solver = solve_optimal()
        value = int(solver.value(variable))
        values.append(value)
        model.add(variable == value)
    return FfnMemoryCalibrationCoefficients(
        **dict(
            zip(
                COEFFICIENT_NAMES,
                (value * observation_quantum_bytes for value in values),
                strict=True,
            )
        )
    )


def fit_worlds(
    fit_worlds: tuple[MemoryProfileWorld, ...],
    held_out_worlds: tuple[MemoryProfileWorld, ...],
) -> tuple[FfnMemoryCalibrationCoefficients, int]:
    """Fit repeated experiments and validate the independent held-out worlds."""

    targets: dict[tuple[int, ...], int] = {}
    positive_observations = []
    # Repeated worlds with the same feature row collapse to their largest
    # residual so fitting cannot hide a high-water observation behind an average.
    for world in fit_worlds:
        for observation in world.observations:
            observed_residual = observation.observed_device_bytes - observation.point.exact_resource_ledger_bytes
            if observed_residual < 0:
                raise ValueError("observed device bytes are below the exact allocation ledger")
            calibration_residual = max(
                0,
                observed_residual - observation.point.allocator_allowance_bytes,
            )
            vector = feature_vector(observation.point.features)
            targets[vector] = max(targets.get(vector, 0), calibration_residual)
            if observation.observed_device_bytes > 0:
                positive_observations.append(observation.observed_device_bytes)
    observation_quantum_bytes = math.gcd(*positive_observations)
    if observation_quantum_bytes <= 0:
        raise ValueError("memory-profile evidence has no positive device-observation quantum")
    # One observed allocation quantum makes the fitted envelope conservative
    # with respect to sampling granularity before integer optimization.
    targets = {vector: target + observation_quantum_bytes for vector, target in targets.items()}
    coefficients = solve_coefficients(targets, observation_quantum_bytes)

    # The held-out matrix is never fitted; retain its minimum headroom as a diagnostic.
    minimum_headroom = SINT64_MAX
    for world in held_out_worlds:
        for observation in world.observations:
            observed_residual = observation.observed_device_bytes - observation.point.exact_resource_ledger_bytes
            if observed_residual < 0:
                raise ValueError("held-out device bytes are below the exact allocation ledger")
            calibration_residual = max(
                0,
                observed_residual - observation.point.allocator_allowance_bytes,
            )
            prediction = sum(
                value * coefficient
                for value, coefficient in zip(
                    feature_vector(observation.point.features), coefficients.values(), strict=True
                )
            )
            if prediction < calibration_residual:
                raise RuntimeError(
                    "memory-profile candidate underpredicts held-out experiment by "
                    f"{calibration_residual - prediction} "
                    f"bytes for models {tuple(str(model.spec.model_id) for model in world.experiment.models)}"
                )
            minimum_headroom = min(
                minimum_headroom,
                observation.point.allocator_allowance_bytes + prediction - observed_residual,
            )
    if minimum_headroom == SINT64_MAX:
        raise ValueError("memory-profile held-out inventory is empty")
    return coefficients, minimum_headroom
