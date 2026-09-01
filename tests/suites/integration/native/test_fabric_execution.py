"""Explicit final-runtime FFN component qualification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.fabric.bootstrap import fabric_bootstrap
from tests.harness.native.fabric.protocol import FabricCoordinatorTrace, FabricFfnAgentTrace
from tests.harness.native.fabric.topology import run_fabric_topology
from tests.harness.support.native.fabric import assert_fabric_report
from xpool.native import RuntimeRole
from xpool.native.ffn import ForwardMode, LayerKind, OutputRequirement

pytestmark = [
    pytest.mark.requires_cuda(min_devices=2),
    pytest.mark.requires_mps,
    pytest.mark.timeout(180),
]


def test_pre_admission_rejection_preserves_attached_generation(tmp_path: Path) -> None:
    """Reject Arena-incompatible metadata before a following valid request."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.DENSE,
            repetition_count=1,
            pre_admission_rejection=True,
        )

    assert_fabric_report(report, expected_repetitions=1)


def repetition_bytes(values: tuple[float, ...], repetition_count: int, payload_dtype: torch.dtype) -> tuple[bytes, ...]:
    """Recover bit-preserving payload outputs from one pickled Instance report."""

    if not values or repetition_count <= 0 or len(values) % repetition_count != 0:
        raise AssertionError("Fabric replay output cannot be partitioned by repetition")
    element_count = len(values) // repetition_count
    return tuple(
        bytes(
            torch.tensor(values[index * element_count : (index + 1) * element_count], dtype=payload_dtype)
            .view(torch.uint8)
            .tolist()
        )
        for index in range(repetition_count)
    )


@pytest.mark.parametrize(
    "layer_kind",
    [pytest.param(LayerKind.DENSE, id="dense"), pytest.param(LayerKind.MOE, id="moe")],
)
def test_execution_uses_installed_lane_graph(layer_kind: LayerKind, tmp_path: Path) -> None:
    """Run deterministic Dense and MoE requests through installed production execution."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=layer_kind,
            repetition_count=2,
        )

    assert_fabric_report(report, expected_repetitions=2)
    (instance,) = report.instances
    first_output, second_output = repetition_bytes(instance.actual, 2, torch.bfloat16)
    assert first_output == second_output
    (ffnagent,) = (participant for participant in report.participants if participant.role is RuntimeRole.FFNAGENT)
    snapshot = ffnagent.graph_snapshot
    assert snapshot is not None
    assert snapshot.primary_graph_binding_site_counts
    assert all(count > 0 for count in snapshot.primary_graph_binding_site_counts)
    assert snapshot.lane_compute_branch_counts == (len(snapshot.primary_graph_binding_site_counts),)
    assert snapshot.lane_delivery_branch_counts == (2,)
    if layer_kind is LayerKind.MOE:
        assert ffnagent.routing is not None
        assert tuple(record.layer_ordinal for record in ffnagent.routing.records) == (0, 0)
        first_routing, second_routing = ffnagent.routing.records
        assert first_routing.topk_ids_bytes == second_routing.topk_ids_bytes
        assert first_routing.topk_weights_bytes == second_routing.topk_weights_bytes
    else:
        assert ffnagent.routing is None


def test_idle_executor_lane_remains_stable_during_repeated_execution(tmp_path: Path) -> None:
    """Keep one Lane idle while another completes repeated production Graph execution."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=2,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.DENSE,
            repetition_count=1_000,
        )

    assert_fabric_report(report, expected_repetitions=1_000)
    (ffnagent,) = (participant for participant in report.participants if participant.role is RuntimeRole.FFNAGENT)
    snapshot = ffnagent.graph_snapshot
    assert snapshot is not None
    assert snapshot.lane_compute_branch_counts == (len(snapshot.primary_graph_binding_site_counts),) * 2
    records = tuple(record for record in ffnagent.records if isinstance(record, FabricFfnAgentTrace))
    assert len(records) == 1_000
    assert {record.executor_lane_index for record in records} == {0}


@pytest.mark.requires_cuda(min_devices=3)
def test_moe_tp2_installs_router_owner_and_nonrouter_graphs(tmp_path: Path) -> None:
    """Execute one true-TP MoE request across both Primary Graph roles."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=2,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.MOE,
            repetition_count=1,
            execution_tp_size=2,
        )

    assert_fabric_report(report, expected_repetitions=1)
    ffnagents = tuple(participant for participant in report.participants if participant.role is RuntimeRole.FFNAGENT)
    assert len(ffnagents) == 2
    snapshots = tuple(participant.graph_snapshot for participant in ffnagents)
    assert all(snapshot is not None for snapshot in snapshots)
    assert all(snapshot.primary_graph_binding_site_counts for snapshot in snapshots if snapshot is not None)
    router_owner, nonrouter = ffnagents
    assert router_owner.routing is not None
    assert len(router_owner.routing.records) == 1
    assert nonrouter.routing is not None
    assert nonrouter.routing.records == ()


def test_fp16_tp1_uses_installed_lane_graph(tmp_path: Path) -> None:
    """Execute one FP16 request through installed production execution."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.DENSE,
            repetition_count=1,
            payload_dtype=torch.float16,
        )

    assert_fabric_report(report, expected_repetitions=1)
    (ffnagent,) = (participant for participant in report.participants if participant.role is RuntimeRole.FFNAGENT)
    assert ffnagent.graph_snapshot is not None
    assert ffnagent.graph_snapshot.primary_graph_binding_site_counts


@pytest.mark.requires_cuda(min_devices=3)
def test_fp16_tp2_delivers_complete_output(tmp_path: Path) -> None:
    """Execute and deliver one complete FP16 true-TP result."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=2,
            execution_tp_size=2,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.DENSE,
            repetition_count=1,
            payload_dtype=torch.float16,
            output_requirement=OutputRequirement.GROUP_SUM_COMPLETE,
        )

    assert_fabric_report(report, expected_repetitions=1)
    deliveries = {
        record.delivery
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricFfnAgentTrace)
    }
    assert deliveries == {xpool.native.fabric.DeliveryVariant.SINGLE_COMPLETE}


@pytest.mark.requires_cuda(min_devices=3)
def test_unselected_ffnagent_installs_empty_lane_root(tmp_path: Path) -> None:
    """Keep joined FfnAgents outside a Layer Execution Group drainable and silent."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=2,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.DENSE,
            repetition_count=1,
            execution_tp_size=1,
        )

    assert_fabric_report(report, expected_repetitions=1)
    ffnagents = tuple(participant for participant in report.participants if participant.role is RuntimeRole.FFNAGENT)
    assert len(ffnagents) == 2
    selected = ffnagents[0].graph_snapshot
    unselected = ffnagents[1].graph_snapshot
    assert selected is not None and unselected is not None
    assert selected.primary_graph_binding_site_counts
    assert unselected.primary_graph_binding_site_counts == ()
    assert unselected.lane_compute_branch_counts == (0,)


@pytest.mark.requires_cuda(min_devices=6)
@pytest.mark.parametrize(
    (
        "output_requirement",
        "atnagent_count",
        "ffnagent_count",
        "forward_mode",
        "payload_rows",
        "expected_delivery",
    ),
    [
        pytest.param(
            OutputRequirement.GROUP_SUM_COMPLETE,
            4,
            2,
            ForwardMode.DECODE,
            1,
            xpool.native.fabric.DeliveryVariant.DIRECT_PARTIAL,
            id="direct-partial-decode",
        ),
        pytest.param(
            OutputRequirement.GROUP_SUM_COMPLETE,
            4,
            2,
            ForwardMode.PREFILL,
            4,
            xpool.native.fabric.DeliveryVariant.DIRECT_PARTIAL,
            id="direct-partial-prefill",
        ),
        pytest.param(
            OutputRequirement.GROUP_SUM_COMPLETE,
            2,
            4,
            ForwardMode.DECODE,
            1,
            xpool.native.fabric.DeliveryVariant.SINGLE_COMPLETE,
            id="single-complete-decode",
        ),
        pytest.param(
            OutputRequirement.GROUP_SUM_COMPLETE,
            2,
            4,
            ForwardMode.PREFILL,
            4,
            xpool.native.fabric.DeliveryVariant.SINGLE_COMPLETE,
            id="single-complete-prefill",
        ),
        pytest.param(
            OutputRequirement.PER_RANK_COMPLETE,
            2,
            4,
            ForwardMode.DECODE,
            1,
            xpool.native.fabric.DeliveryVariant.REPLICATED_COMPLETE,
            id="replicated-complete-decode",
        ),
        pytest.param(
            OutputRequirement.PER_RANK_COMPLETE,
            2,
            4,
            ForwardMode.PREFILL,
            4,
            xpool.native.fabric.DeliveryVariant.REPLICATED_COMPLETE,
            id="replicated-complete-prefill",
        ),
    ],
)
def test_execution_reuses_1000_leases_across_delivery_shapes(
    output_requirement: OutputRequirement,
    atnagent_count: int,
    ffnagent_count: int,
    forward_mode: ForwardMode,
    payload_rows: int,
    expected_delivery: xpool.native.fabric.DeliveryVariant,
    tmp_path: Path,
) -> None:
    """Run 1,000 real Dense leases for every Delivery and mode cell."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            execution_tp_size=ffnagent_count,
            executor_lane_count=1,
            forward_modes=(forward_mode,),
            layer_kind=LayerKind.DENSE,
            repetition_count=1_000,
            decode_payload_row_capacity=1,
            prefill_payload_row_capacity=4,
            payload_rows=(payload_rows,),
            output_requirement=output_requirement,
        )

    assert_fabric_report(report, expected_repetitions=1_000)
    coordinator = tuple(
        record
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricCoordinatorTrace)
    )
    assert [record.invocation_sequence for record in coordinator] == list(range(1, 1_001))
    assert [record.executor_lease_sequence for record in coordinator] == list(range(1, 1_001))
    observed = {
        record.delivery
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricFfnAgentTrace)
    }
    assert observed == {expected_delivery}


@pytest.mark.parametrize(
    "layer_kind",
    [pytest.param(LayerKind.DENSE, id="dense"), pytest.param(LayerKind.MOE, id="moe")],
)
def test_execution_rebinds_two_layers_a_b_a(layer_kind: LayerKind, tmp_path: Path) -> None:
    """Execute distinct same-Signature weight bindings in A-B-A order."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=layer_kind,
            repetition_count=3,
            execution_layer_count=2,
            payload_rows=(1,),
            layer_ordinals=(0, 1, 0),
        )

    assert_fabric_report(report, expected_repetitions=3)
    (instance,) = report.instances
    first_output, middle_output, final_output = repetition_bytes(instance.actual, 3, torch.bfloat16)
    assert first_output != middle_output
    assert first_output == final_output
    ffnagent_records = tuple(
        record
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricFfnAgentTrace)
    )
    assert tuple(record.layer_ordinal for record in ffnagent_records) == (0, 1, 0)
    (ffnagent,) = (participant for participant in report.participants if participant.role is RuntimeRole.FFNAGENT)
    assert ffnagent.graph_snapshot is not None
    if layer_kind is LayerKind.MOE:
        assert ffnagent.routing is not None
        assert tuple(record.layer_ordinal for record in ffnagent.routing.records) == (0, 1, 0)
        first_routing, _, final_routing = ffnagent.routing.records
        assert first_routing.topk_ids_bytes == final_routing.topk_ids_bytes
        assert first_routing.topk_weights_bytes == final_routing.topk_weights_bytes
    else:
        assert ffnagent.routing is None


@pytest.mark.parametrize(
    "layer_kind",
    [pytest.param(LayerKind.DENSE, id="dense"), pytest.param(LayerKind.MOE, id="moe")],
)
def test_execution_selects_smallest_compatible_capacity(layer_kind: LayerKind, tmp_path: Path) -> None:
    """Exercise both ends of two Capacity-selection intervals on both lanes."""

    rows = (1, 2, 3, 4)
    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=2,
            forward_modes=(ForwardMode.DECODE,) * 2 + (ForwardMode.PREFILL,) * 2,
            layer_kind=layer_kind,
            repetition_count=1,
            decode_payload_row_capacity=2,
            prefill_payload_row_capacity=4,
            payload_rows=rows,
        )

    assert_fabric_report(report, expected_repetitions=1)
    coordinator = tuple(
        record
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricCoordinatorTrace)
    )
    ffnagent = tuple(
        record
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricFfnAgentTrace)
    )
    expected = {(1, 1), (2, 2), (3, 4), (4, 4)}
    assert {(record.payload_rows, record.payload_row_capacity) for record in ffnagent} == expected
    assert {record.executor_lane_index for record in coordinator} == {0, 1}
