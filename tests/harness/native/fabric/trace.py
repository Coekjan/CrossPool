from __future__ import annotations

import torch

import xpool.native
from tests.harness.native.debug import native_debug_options
from tests.harness.native.fabric.protocol import (
    FabricAtnAgentTrace,
    FabricCoordinatorTrace,
    FabricFailureEvidence,
    FabricFfnAgentTrace,
    FabricGraphSnapshotEvidence,
    FabricParticipantReport,
    FabricRoutingRecordEvidence,
    FabricRoutingSnapshotEvidence,
    FabricTrace,
)
from xpool.native import RuntimeRole


def fabric_debug_options(
    *,
    graph_observer: bool = False,
    routing_observer: bool = False,
) -> xpool.native.debug.Options:
    """Return native debug options for traced Fabric execution."""

    return native_debug_options(
        fabric_observer=True,
        graph_observer=graph_observer,
        ffn_routing_observer=routing_observer,
    )


def fabric_trace_report(
    role: RuntimeRole,
    snapshot: xpool.native.devkit.fabric_observer.Snapshot,
    failure: xpool.native.fabric.Failure | None,
    graph_snapshot: xpool.native.devkit.graph_observer.Snapshot | None,
    routing_snapshot: xpool.native.devkit.ffn_routing_observer.Snapshot | None,
) -> FabricParticipantReport:
    """Copy one bound native snapshot into picklable typed trace facts."""

    records: list[FabricTrace] = []
    for record in snapshot.records:
        key = record.key
        if record.kind == xpool.native.devkit.fabric_observer.RecordKind.ATNAGENT:
            event = xpool.native.devkit.fabric_observer.AtnAgentEvent
            records.append(
                FabricAtnAgentTrace(
                    instance_index=key.instance_index,
                    invocation_sequence=key.invocation_sequence,
                    layer_ordinal=record.layer_ordinal,
                    payload_rows=record.payload_rows,
                    executor_lane_index=record.executor_lane_index,
                    executor_lease_sequence=record.executor_lease_sequence,
                    forward_mode=record.forward_mode,
                    submission_prepared_ns=record.timestamp(event.SUBMISSION_PREPARED),
                    output_acknowledgement_published_ns=record.timestamp(event.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED),
                )
            )
        elif record.kind == xpool.native.devkit.fabric_observer.RecordKind.COORDINATOR:
            event = xpool.native.devkit.fabric_observer.CoordinatorEvent
            records.append(
                FabricCoordinatorTrace(
                    instance_index=key.instance_index,
                    invocation_sequence=key.invocation_sequence,
                    layer_ordinal=record.layer_ordinal,
                    payload_rows=record.payload_rows,
                    executor_lane_index=record.executor_lane_index,
                    executor_lease_sequence=record.executor_lease_sequence,
                    ready_ticket=record.ready_ticket,
                    enqueued_ns=record.timestamp(event.ENQUEUED),
                    scheduled_ns=record.timestamp(event.SCHEDULED),
                    lane_released_ns=record.timestamp(event.LANE_RELEASED),
                )
            )
        elif record.kind == xpool.native.devkit.fabric_observer.RecordKind.FFNAGENT:
            event = xpool.native.devkit.fabric_observer.FfnAgentEvent
            records.append(
                FabricFfnAgentTrace(
                    instance_index=key.instance_index,
                    invocation_sequence=key.invocation_sequence,
                    layer_ordinal=record.layer_ordinal,
                    payload_rows=record.payload_rows,
                    executor_lane_index=record.executor_lane_index,
                    executor_lease_sequence=record.executor_lease_sequence,
                    payload_row_capacity=record.payload_row_capacity,
                    delivery=record.delivery,
                    lane_execution_observed_ns=record.timestamp(event.LANE_EXECUTION_OBSERVED),
                    compute_started_ns=record.timestamp(event.COMPUTE_STARTED),
                    compute_completed_ns=record.timestamp(event.COMPUTE_COMPLETED),
                    completion_published_ns=record.timestamp(event.COMPLETION_PUBLISHED),
                )
            )
        else:
            raise RuntimeError(f"native Fabric returned unknown trace kind {record.kind!r}")
    return FabricParticipantReport(
        role=role,
        pe=snapshot.pe,
        sequence=snapshot.sequence,
        dropped=snapshot.dropped,
        records=tuple(records),
        failure=(
            None
            if failure is None
            else FabricFailureEvidence(
                claim=failure.claim,
                publication=failure.publication,
                result_code=failure.payload.result_code,
                origin_pe=failure.payload.origin_pe,
                instance_index=failure.payload.key.instance_index,
                invocation_sequence=failure.payload.key.invocation_sequence,
                layer_ordinal=failure.payload.layer_ordinal,
            )
        ),
        graph_snapshot=(
            None
            if graph_snapshot is None
            else FabricGraphSnapshotEvidence(
                primary_graph_binding_site_counts=tuple(
                    primary_graph.binding_site_count for primary_graph in graph_snapshot.primary_graphs
                ),
                lane_compute_branch_counts=tuple(lane.compute_branch_count for lane in graph_snapshot.lane_graphs),
                lane_delivery_branch_counts=tuple(lane.delivery_branch_count for lane in graph_snapshot.lane_graphs),
            )
        ),
        routing=(
            None
            if routing_snapshot is None
            else FabricRoutingSnapshotEvidence(
                sequence=routing_snapshot.sequence,
                dropped=routing_snapshot.dropped,
                records=tuple(
                    FabricRoutingRecordEvidence(
                        instance_index=record.key.instance_index,
                        invocation_sequence=record.key.invocation_sequence,
                        layer_ordinal=record.layer_ordinal,
                        row_count=record.topk_ids.shape[0],
                        effective_topk=record.topk_ids.shape[1],
                        topk_ids_bytes=bytes(record.topk_ids.view(torch.uint8).flatten().tolist()),
                        topk_weights_bytes=bytes(record.topk_weights.view(torch.uint8).flatten().tolist()),
                    )
                    for record in routing_snapshot.records
                ),
            )
        ),
    )
