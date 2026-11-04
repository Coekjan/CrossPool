from __future__ import annotations

import gc
import os
import weakref
from collections import Counter
from pathlib import Path

import pytest
import torch

import xpool.native
import xtest
from xpool.fabric import FABRIC_UID_HEX_LENGTH
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import capture_workspace, preparation
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry, LaneCaptureResources
from xtest.harness.native.case import run_native_case
from xtest.harness.native.ffn.qualification import (
    dense_layer_weights,
    execution_fabric_plan,
    execution_model_specs,
    moe_layer_weights,
)


def isolated_blas_capture_source(workspace_config: str | None, unified: str | None) -> None:
    for name in ("CUBLASLT_WORKSPACE_SIZE", "TORCH_BLAS_PREFER_CUBLASLT", "TORCH_CUBLASLT_UNIFIED_WORKSPACE"):
        os.environ.pop(name, None)
    if workspace_config is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace_config
    if unified is not None:
        os.environ["TORCH_CUBLASLT_UNIFIED_WORKSPACE"] = unified
    torch.cuda.set_device(0)
    contract, external_owners = capture_workspace.prepare_execution_sources()
    assert external_owners == ()
    if unified == "0":
        assert len(contract.lane_resources) == 2
        assert all(capacity.aliases == (capacity.requirement.key,) for capacity in contract.lane_resources)

    generator = torch.Generator(device="cpu").manual_seed(17)
    inputs = torch.randn((64, 2880), generator=generator).to(torch.bfloat16).cuda()
    primary_weight = torch.randn((32, 2880), generator=generator).to(torch.bfloat16).cuda()
    control_weight = primary_weight.clone()
    bias = torch.randn((32,), generator=generator).to(torch.bfloat16).cuda()
    output = torch.empty((64, 32), dtype=torch.bfloat16, device="cuda")

    def launch(weight: torch.Tensor) -> None:
        torch.addmm(bias, inputs[:rows], weight.t(), out=output[:rows])

    owner = LaneCaptureResources()
    try:
        owner.initialize()
        assert owner.stream is not None and owner.stream_owner is not None and owner.pool is not None
        for rows in (1, 32, 64):
            paired = owner.capture_pair(
                primary_launch=lambda: launch(primary_weight),
                control_launch=lambda: launch(control_weight),
                local_allocations=(),
                execution_contract=contract,
                external_ranges=(),
            )[2]
            if workspace_config == ":0:0":
                assert contract.lane_resources == ()
                assert paired == ()
            elif rows == 32 and torch.cuda.get_device_capability() == (8, 0):
                assert paired, "the reproducing A100 GEMM must expose its BLAS scratch"
            capacities = {
                capacity.requirement.key: capacity.requirement.logical_extent_bound_bytes
                for capacity in contract.lane_resources
            }
            for region in paired:
                assert region.bytes <= capacities[region.key]
        owner.release_capture_resources()
        assert owner.pool is not None and owner.sealed
        assert xpool.native.cuda.blas.snapshot_workspaces(owner.stream_owner) == []
        retained_bytes = sum(segment["total_size"] for segment in owner.pool.snapshot())
        torch.cuda.empty_cache()
        assert sum(segment["total_size"] for segment in owner.pool.snapshot()) == retained_bytes
    finally:
        owner.close()


@pytest.mark.parametrize(
    ("workspace_config", "unified"),
    ((None, None), (":0:0", None), (None, "0")),
    ids=("default", "zero", "separate-lt"),
)
@xtest.requirements(device_count=1)
def test_blas_capture_source_bounds_pairing_and_cleanup(
    tmp_path: Path, workspace_config: str | None, unified: str | None
) -> None:
    run_native_case(isolated_blas_capture_source, workspace_config, unified, workdir=tmp_path / "case")


def isolated_registry_install_failure_retains_captures(execution_kind: LayerKind, tp_rank: int) -> None:
    torch.cuda.set_device(0)
    contract, external = capture_workspace.prepare_execution_sources()
    tp_size = 1 if execution_kind is LayerKind.DENSE else 2
    plan = execution_fabric_plan(
        uid="a" * FABRIC_UID_HEX_LENGTH,
        atnagent_count=1,
        ffnagent_count=tp_size,
        ffn_tp_size=tp_size,
        executor_lane_count=2,
        instance_count=1,
        execution_kind=execution_kind,
        ffn_execution_contracts=(contract,) * tp_size,
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=4,
        payload_dtype=torch.bfloat16,
    )
    model_specs = execution_model_specs(instance_count=1, execution_kind=execution_kind, layer_count=1)
    factory = dense_layer_weights if execution_kind is LayerKind.DENSE else moe_layer_weights
    weight = factory(ffnagent_index=tp_rank, ffnagent_count=tp_size, device=0, payload_dtype=torch.bfloat16)
    profile = plan.instance_plans[0].ffn_profile
    layout = preparation.prepare_ffn_layer(
        model_specs[0], layer_ordinal=0, payload_dtype=profile.payload_dtype, tp_size=tp_size, tp_rank=tp_rank
    )
    signatures = layout.execution_signatures(profile=profile)
    expected_default_storage = Counter(
        value for signature in signatures for value in signature.capture_placeholder_layout().allocation_bytes()
    )
    weight_ref = weakref.ref(weight.gate_up.values)
    installation = FfnExecutionRegistry(layer_weights=((weight,),), external_resources=external)
    del weight, external
    probe_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def fail_install(descriptor: xpool.native.ffnagent.ExecutionInstallDescriptor) -> None:
        assert isinstance(descriptor, xpool.native.ffnagent.ExecutionInstallDescriptor)
        assert len(installation.capture_lanes) == 2
        assert all(
            lane.capture_graphs and lane.pool is not None and not lane.sealed for lane in installation.capture_lanes
        )
        probe_refs.extend(weakref.ref(probe.gate_up.values) for _, probe in installation.capture_weight_pairs.values())
        for lane in installation.capture_lanes:
            assert lane.pool is not None
            segments = lane.pool.snapshot()
            roots = {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                for tensor in lane.capture_tensors
            }
            actual_default_storage = Counter(
                nbytes
                for address, nbytes in roots.items()
                if not any(
                    segment["address"] <= address and address + nbytes <= segment["address"] + segment["total_size"]
                    for segment in segments
                )
            )
            assert actual_default_storage == expected_default_storage
        raise RuntimeError("native install failed with retained captures")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(xpool.native.ffnagent, "install_execution", fail_install)
        with pytest.raises(RuntimeError, match="native install failed with retained captures"):
            installation.materialize(fabric_plan=plan, model_specs=model_specs, ffnagent_index=tp_rank)
    assert weight_ref() is not None
    assert probe_refs and all(reference() is not None for reference in probe_refs)
    assert all(lane.capture_graphs and not lane.sealed for lane in installation.capture_lanes)
    with pytest.raises(RuntimeError, match="cannot be retried"):
        installation.materialize(fabric_plan=plan, model_specs=model_specs, ffnagent_index=tp_rank)
    # The install substitute acquired no native handles, so capture owners can retire.
    installation.close()
    gc.collect()
    assert weight_ref() is None
    assert all(reference() is None for reference in probe_refs)


@xtest.requirements(device_count=1)
@pytest.mark.parametrize(("execution_kind", "tp_rank"), ((LayerKind.DENSE, 0), (LayerKind.MOE, 0), (LayerKind.MOE, 1)))
def test_registry_install_failure_retains_real_captures(
    tmp_path: Path, execution_kind: LayerKind, tp_rank: int
) -> None:
    run_native_case(
        isolated_registry_install_failure_retains_captures, execution_kind, tp_rank, workdir=tmp_path / "case"
    )
