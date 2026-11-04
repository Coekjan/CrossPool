from __future__ import annotations

import json
import os
import weakref
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
import torch

import xpool.native
import xtest
from xpool import ffn
from xpool.config import XpoolConfig
from xpool.fabric import FabricPlan
from xpool.memory import (
    ExecutionResourceDeclaration,
    ExecutionResourceRequirement,
    FfnExecutionContract,
)
from xpool.native import ABI_VERSION, RuntimeRole
from xpool.native.ffn import ForwardMode, LayerKind, OutputRequirement
from xpool.runtime.ffnagent import capture_workspace, execution, preparation, registry, weights
from xpool.runtime.ffnagent.operators import dense
from xpool.service.wire import FfnAgentRegistration
from xtest.harness.native.fabric import instance, participant, topology
from xtest.harness.native.fabric.bootstrap import fabric_bootstrap
from xtest.harness.native.fabric.protocol import FabricFfnAgentTrace, FabricInstanceSpec, FabricParticipantSpec
from xtest.harness.native.ffn.qualification import execution_model_specs
from xtest.harness.support.config import install_test_config
from xtest.harness.support.native.fabric import assert_fabric_report

pytestmark = pytest.mark.timeout(180)


class SharedTensorSource:
    """Component-only source; the producer relinquishes every strong Tensor reference."""

    def __init__(self, *, applicable: bool) -> None:
        self.applicable = applicable
        self.reference: weakref.ReferenceType[torch.Tensor] | None = None
        self.pending_owners: tuple[execution.OwnedExecutionResource, ...] = ()
        self.contract: FfnExecutionContract | None = None
        self.closed = False
        self.selected_ranges = 0
        self.baseline_free_bytes = 0

    def initialize(self, hidden_size: int, environment: FfnExecutionContract) -> None:
        table = ((torch.arange(hidden_size, device="cuda", dtype=torch.float32) % 7 + 1) / 4).to(torch.bfloat16)
        self.reference = weakref.ref(table)
        nbytes = table.numel() * table.element_size()
        declaration = ExecutionResourceDeclaration(
            requirement=ExecutionResourceRequirement(
                key="fixture.table",
                request_bytes=(table.untyped_storage().nbytes(),),
                logical_extent_bound_bytes=nbytes,
                exact_logical_bytes=nbytes,
                alignment_bytes=256,
            ),
            aliases=("fixture.table",),
            uses=("dense",) if self.applicable else ("moe_router",),
        )
        self.contract = environment.model_copy(
            update={"shared_resources": (*environment.shared_resources, declaration)}
        )
        self.pending_owners = (
            execution.OwnedExecutionResource(
                owner=table,
                region=xpool.native.ffnagent.CapturedResourceRange(
                    "fixture.table", table.data_ptr(), table.data_ptr(), nbytes, True
                ),
            ),
        )

    def prepare(self) -> tuple[FfnExecutionContract, tuple[execution.OwnedExecutionResource, ...]]:
        assert self.contract is not None
        owners = self.pending_owners
        self.pending_owners = ()
        return self.contract, owners


def run_shared_source_participant(
    connection: Connection, spec: FabricParticipantSpec, *, applicable: bool, proof_path: Path
) -> None:
    """Use normal native participants with a real, test-owned external source."""

    if spec.role is RuntimeRole.ATNAGENT:
        participant.run_fabric_participant(connection, spec)
        return
    source = SharedTensorSource(applicable=applicable)
    original_prepare = capture_workspace.prepare_execution_sources
    original_compute = dense.compute_dense_partial

    def prepare_source() -> tuple[FfnExecutionContract, tuple[execution.OwnedExecutionResource, ...]]:
        environment, owners = original_prepare()
        assert owners == ()
        source.initialize(spec.hidden_size, environment)
        assert source.contract is not None
        models = execution_model_specs(
            instance_count=len(spec.forward_modes),
            execution_kind=spec.layer_kind,
            layer_count=spec.execution_layer_count,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
        )
        config = XpoolConfig.from_mapping(
            {
                "scheduler": {"ffn_concurrency": spec.executor_lane_count, "slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": list(range(spec.atnagent_count))},
                "ffn": {"devices": list(range(spec.atnagent_count, spec.atnagent_count + spec.ffnagent_count))},
                "models": [
                    {"id": str(model.model_id), "path": f"/models/source-{index}", "ffn_tp_size": 1}
                    for index, model in enumerate(models)
                ],
            }
        )
        install_test_config(config)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        source.baseline_free_bytes = free
        registration = FfnAgentRegistration(
            device=spec.device,
            pid=os.getpid(),
            abi_version=ABI_VERSION,
            calibration_environment=None,
            device_free_memory_bytes=free,
            device_total_memory_bytes=total,
            execution_contract=source.contract,
            model_specs=models,
        )
        registration = FfnAgentRegistration.model_validate_json(registration.model_dump_json())
        source.contract = registration.execution_contract
        return source.prepare()

    def compute_with_source(
        *,
        hidden_states: torch.Tensor,
        layer_weights: weights.DenseFfnWeights,
        workspace: torch.Tensor,
        output: torch.Tensor,
        activation: ffn.ActivationSpec,
    ) -> torch.Tensor:
        original_compute(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            workspace=workspace,
            output=output,
            activation=activation,
        )
        if source.applicable:
            assert source.reference is not None
            table = source.reference()
            assert table is not None
            output.add_(table)
        # Hold each lease long enough to observe both independently installed Lanes.
        torch.cuda._sleep(10_000_000)
        return output

    class ObservedRegistry(registry.FfnExecutionRegistry):
        def materialize(
            self,
            *,
            fabric_plan: FabricPlan,
            model_specs: tuple[ffn.FfnModelSpec, ...],
            ffnagent_index: int,
        ) -> None:
            contracts = fabric_plan.ffn_execution_contracts
            assert contracts == (source.contract,)
            super().materialize(
                fabric_plan=fabric_plan,
                model_specs=model_specs,
                ffnagent_index=ffnagent_index,
            )
            assert source.reference is not None and source.reference() is not None
            assert not source.pending_owners
            assert all(lane.sealed for lane in self.capture_lanes)

        def external_ranges(
            self, resource_plan: preparation.ExecutionResourcePlan, signature_index: int
        ) -> tuple[xpool.native.ffnagent.CapturedResourceRange, ...]:
            ranges = super().external_ranges(resource_plan, signature_index)
            assert bool(ranges) is source.applicable
            source.selected_ranges += len(ranges)
            return ranges

        def close(self) -> None:
            assert source.reference is not None and source.reference() is not None
            super().close()
            assert source.reference() is None
            source.closed = True

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(participant.capture_workspace, "prepare_execution_sources", prepare_source)
        patch.setattr(participant, "FfnExecutionRegistry", ObservedRegistry)
        patch.setattr(dense, "compute_dense_partial", compute_with_source)
        participant.run_fabric_participant(connection, spec)
    assert source.closed
    assert bool(source.selected_ranges) is applicable
    proof_path.write_text(
        json.dumps(
            {
                "applicable": applicable,
                "selected_ranges": source.selected_ranges,
                "baseline_free_bytes": source.baseline_free_bytes,
                "released_after_retirement": source.closed,
            },
            indent=2,
        )
        + "\n"
    )


def run_shared_source_instance(connection: Connection, spec: FabricInstanceSpec, *, applicable: bool) -> None:
    original_reference = instance.execution_reference
    original_shim = torch.ops.xpool.ffn_shim
    invocation = 0

    def dynamic_input(
        hidden_states: torch.Tensor,
        routing: torch.Tensor | None,
        output: torch.Tensor,
        layer: int,
        mode: int,
        requirement: int,
        epoch: int,
    ) -> None:
        nonlocal invocation
        invocation += 1
        hidden_states.add_(0.125 if invocation % 2 else -0.125)
        original_shim(hidden_states, routing, output, layer, mode, requirement, epoch)

    def expected_with_source(
        hidden_states: torch.Tensor,
        *,
        execution_kind: LayerKind,
        ffn_tp_size: int,
        output_count: int,
        output_rank: int,
        output_requirement: OutputRequirement,
        layer_ordinal: int = 0,
        intermediate_size: int | None = None,
    ) -> torch.Tensor:
        expected = original_reference(
            hidden_states,
            execution_kind=execution_kind,
            ffn_tp_size=ffn_tp_size,
            output_count=output_count,
            output_rank=output_rank,
            output_requirement=output_requirement,
            layer_ordinal=layer_ordinal,
            intermediate_size=intermediate_size,
        )
        if applicable:
            table = (
                (torch.arange(hidden_states.shape[-1], device=hidden_states.device, dtype=torch.float32) % 7 + 1) / 4
            ).to(expected.dtype)
            return expected + table
        return expected

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.ops.xpool, "ffn_shim", dynamic_input)
        patch.setattr(instance, "execution_reference", expected_with_source)
        instance.run_fabric_instance(connection, spec)


@pytest.mark.parametrize("applicable", (False, True), ids=("unused-owner", "shared-read-only"))
@xtest.requirements(device_count=2)
def test_shared_source_survives_capture_trim_replay_and_retirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    applicable: bool,
) -> None:
    proof = tmp_path / "shared-source.json"
    monkeypatch.setattr(
        topology,
        "run_fabric_participant",
        partial(
            run_shared_source_participant,
            applicable=applicable,
            proof_path=proof,
        ),
    )
    monkeypatch.setattr(topology, "run_fabric_instance", partial(run_shared_source_instance, applicable=applicable))
    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = topology.run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=2,
            forward_modes=(ForwardMode.PREFILL, ForwardMode.PREFILL),
            layer_kind=LayerKind.DENSE,
            repetition_count=8,
            prefill_payload_row_capacity=8,
            payload_rows=(4, 8),
            hidden_size=16,
            intermediate_size=32,
        )
    assert_fabric_report(report, expected_repetitions=8)
    (ffnagent,) = (row for row in report.participants if row.role is RuntimeRole.FFNAGENT)
    lanes = {row.executor_lane_index for row in ffnagent.records if isinstance(row, FabricFfnAgentTrace)}
    assert lanes == {0, 1}
    for result in report.instances:
        count = len(result.actual) // result.repetitions_completed
        assert result.actual[:count] != result.actual[count : 2 * count]
    assert json.loads(proof.read_text())["released_after_retirement"] is True
