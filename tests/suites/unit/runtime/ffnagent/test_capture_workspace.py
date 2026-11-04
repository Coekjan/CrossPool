from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import pytest

import xpool.native
from xpool.memory import FfnExecutionContract
from xpool.runtime.ffnagent import capture_workspace
from xtest.harness.support.config import lane_resource_declaration


@dataclass(frozen=True)
class CaptureStream:
    cuda_stream: int = 7


@pytest.fixture
def contract() -> FfnExecutionContract:
    return FfnExecutionContract(
        lane_resources=(
            lane_resource_declaration(
                key="blas.cublas", max_bytes=1024, alignment_bytes=256, aliases=("blas.cublas", "blas.cublaslt")
            ),
        ),
        shared_resources=(),
    )


def test_blas_snapshot_counts_shared_backing_once(
    monkeypatch: pytest.MonkeyPatch, contract: FfnExecutionContract
) -> None:
    allocations = (
        xpool.native.cuda.CaptureAllocation("blas.cublas", 0x1000, 1024),
        xpool.native.cuda.CaptureAllocation("blas.cublaslt", 0x1000, 512),
    )
    monkeypatch.setattr(xpool.native.cuda.blas, "snapshot_workspaces", lambda stream: allocations)

    observed = capture_workspace.snapshot_resources(cast(xpool.native.cuda.OwnedCudaStream, CaptureStream()), contract)

    assert [(record.key, record.address, record.bytes) for record in observed] == [("blas.cublas", 0x1000, 1024)]
    assert FfnExecutionContract.model_validate_json(contract.model_dump_json()) == contract


def test_source_range_must_fit_requests_even_when_extent_bound_is_loose(
    monkeypatch: pytest.MonkeyPatch,
    contract: FfnExecutionContract,
) -> None:
    declaration = contract.lane_resources[0]
    declaration = declaration.model_copy(
        update={"requirement": replace(declaration.requirement, logical_extent_bound_bytes=4096)}
    )
    contract = contract.model_copy(update={"lane_resources": (declaration,)})
    monkeypatch.setattr(
        xpool.native.cuda.blas,
        "snapshot_workspaces",
        lambda stream: (xpool.native.cuda.CaptureAllocation("blas.cublas", 0x1000, 2048),),
    )
    with pytest.raises(ValueError, match="declared capacity"):
        capture_workspace.snapshot_resources(cast(xpool.native.cuda.OwnedCudaStream, CaptureStream()), contract)


@pytest.mark.parametrize(
    "records",
    (
        (("unknown", 0x1000, 512),),
        (("blas.cublas", 0x1000, 1025),),
        (("blas.cublas", 0x1001, 1024),),
        (("blas.cublas", 0x1000, 1024), ("blas.cublaslt", 0x2000, 512)),
    ),
)
def test_blas_snapshot_rejects_changed_capacity_alignment_or_aliases(
    monkeypatch: pytest.MonkeyPatch,
    contract: FfnExecutionContract,
    records: tuple[tuple[str, int, int], ...],
) -> None:
    allocations = tuple(xpool.native.cuda.CaptureAllocation(*record) for record in records)
    monkeypatch.setattr(xpool.native.cuda.blas, "snapshot_workspaces", lambda stream: allocations)
    with pytest.raises(ValueError):
        capture_workspace.snapshot_resources(cast(xpool.native.cuda.OwnedCudaStream, CaptureStream()), contract)


@pytest.mark.parametrize("control_address", (0x1000, 0x2000))
def test_resource_pairing_preserves_equal_and_distinct_bases(control_address: int) -> None:
    regions = capture_workspace.pair_resources(
        (xpool.native.cuda.CaptureAllocation("scratch", 0x1000, 512),),
        (xpool.native.cuda.CaptureAllocation("scratch", control_address, 512),),
    )
    assert [(region.key, region.primary_address, region.control_address, region.bytes) for region in regions] == [
        ("scratch", 0x1000, control_address, 512)
    ]
    with pytest.raises(ValueError, match="identities disagree"):
        capture_workspace.pair_resources(
            (xpool.native.cuda.CaptureAllocation("scratch", 0x1000, 512),),
            (xpool.native.cuda.CaptureAllocation("other", control_address, 512),),
        )
