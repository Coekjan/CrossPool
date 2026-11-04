"""Execution-source discovery and logical coverage for retained Lane resources."""

from __future__ import annotations

import xpool.native
from xpool.memory import (
    ExecutionResourceDeclaration,
    ExecutionResourceRequirement,
    FfnExecutionContract,
)
from xpool.runtime.ffnagent import execution


def prepare_execution_sources() -> tuple[FfnExecutionContract, tuple[execution.OwnedExecutionResource, ...]]:
    """Prepare source declarations and retain external owners before the memory baseline.

    Only declarations cross registration. Current stream-private BLAS scratch
    is acquired later in each Lane, so this source has no external owners.
    """

    owner = xpool.native.cuda.OwnedCudaStream.create()
    try:
        capacities = xpool.native.cuda.blas.probe_capacity(owner)
        contract = FfnExecutionContract(
            lane_resources=tuple(
                ExecutionResourceDeclaration(
                    requirement=ExecutionResourceRequirement(
                        key=requirement.key,
                        request_bytes=(requirement.max_bytes,),
                        logical_extent_bound_bytes=requirement.max_bytes,
                        exact_logical_bytes=None,
                        alignment_bytes=requirement.alignment_bytes,
                    ),
                    aliases=tuple(requirement.aliases),
                    uses=("dense", "moe_router"),
                )
                for requirement in capacities
            ),
            shared_resources=(),
        )
        return contract, ()
    finally:
        owner.destroy()


def snapshot_resources(
    stream: xpool.native.cuda.OwnedCudaStream, contract: FfnExecutionContract
) -> tuple[xpool.native.cuda.CaptureAllocation, ...]:
    """Normalize source-owned ranges without exporting library semantics to admission."""

    aliases = {
        alias: declaration.requirement for declaration in contract.lane_resources for alias in declaration.aliases
    }
    backings: dict[str, xpool.native.cuda.CaptureAllocation] = {}
    for allocation in xpool.native.cuda.blas.snapshot_workspaces(stream):
        capacity = aliases.get(allocation.key)
        if capacity is None:
            raise ValueError(f"undeclared execution resource: {allocation.key}")
        if (
            not 0 < allocation.bytes <= min(capacity.logical_extent_bound_bytes, sum(capacity.request_bytes))
            or allocation.address % capacity.alignment_bytes
        ):
            raise ValueError(f"execution resource exceeds its declared capacity or alignment: {capacity.key}")
        previous = backings.get(capacity.key)
        if previous is not None and previous.address != allocation.address:
            raise ValueError(f"execution resource alias relationship changed: {capacity.key}")
        backings[capacity.key] = xpool.native.cuda.CaptureAllocation(
            key=capacity.key,
            address=allocation.address,
            bytes=max(allocation.bytes, 0 if previous is None else previous.bytes),
        )
    return tuple(
        backings[declaration.requirement.key]
        for declaration in contract.lane_resources
        if declaration.requirement.key in backings
    )
