"""Typed subprocess harness for native multi-PE Fabric tests."""

from __future__ import annotations

from pathlib import Path

import torch

import xpool.native
from tests.harness.native.fabric.protocol import (
    FabricAtnAgentTrace,
    FabricCoordinatorTrace,
    FabricExecutionTrace,
    FabricFailureEvidence,
    FabricParticipantReport,
    FabricTrace,
)
from xpool.abi import FfnResultCode, TensorDType, XPoolForwardMode
from xpool.config import DebugConfig
from xpool.runtime import RuntimeRole


def fabric_debug_options(*, loopback_enabled: bool = True) -> str:
    """Return native debug options for traced FfnAgent loopback."""

    config = DebugConfig.model_validate(
        {
            "loopback": {
                "enable": loopback_enabled,
                "site": "ffnagent" if loopback_enabled else None,
            },
            "fabric_observer": {
                "enable": True,
                "outdir": Path.cwd(),
            },
        }
    )
    return config.model_dump_json(
        include={
            "loopback": {"enable", "site"},
            "transport_observer": {"enable", "trace_capacity"},
            "fabric_observer": {"enable", "trace_capacity"},
        }
    )


def torch_dtype(dtype: TensorDType) -> torch.dtype:
    """Return the Torch dtype corresponding to one stable xpool dtype."""

    match dtype:
        case TensorDType.BF16:
            return torch.bfloat16
        case TensorDType.FP16:
            return torch.float16
        case TensorDType.FP32:
            return torch.float32


def fabric_trace_report(
    role: RuntimeRole,
    snapshot: xpool.native.FabricTraceSnapshot,
    failure: xpool.native.FabricFailure | None,
) -> FabricParticipantReport:
    """Copy one bound native snapshot into picklable typed trace facts."""

    validate_fabric_trace_event_families(snapshot)
    records: list[FabricTrace] = []
    for record in snapshot.records:
        key = record.key
        if record.kind == xpool.native.FabricTraceKind.ATNAGENT:
            event = xpool.native.AtnAgentTraceEvent
            records.append(
                FabricAtnAgentTrace(
                    model_index=key.model_index,
                    invocation_sequence=key.invocation_sequence,
                    executor_index=record.executor_index,
                    forward_mode=None if record.forward_mode is None else XPoolForwardMode(record.forward_mode),
                    prepared_ns=record.timestamp(event.SUBMISSION_PREPARED),
                    acknowledged_ns=record.timestamp(event.ACKNOWLEDGEMENT_PUBLISHED),
                )
            )
        elif record.kind == xpool.native.FabricTraceKind.COORDINATOR:
            event = xpool.native.CoordinatorTraceEvent
            records.append(
                FabricCoordinatorTrace(
                    model_index=key.model_index,
                    invocation_sequence=key.invocation_sequence,
                    executor_index=record.executor_index,
                    ready_ticket=record.ready_ticket,
                    enqueued_ns=record.timestamp(event.ENQUEUED),
                    scheduled_ns=record.timestamp(event.SCHEDULED),
                    released_ns=record.timestamp(event.SCHEDULER_RELEASED),
                )
            )
        elif record.kind == xpool.native.FabricTraceKind.EXECUTION:
            event = xpool.native.ExecutionTraceEvent
            records.append(
                FabricExecutionTrace(
                    model_index=key.model_index,
                    invocation_sequence=key.invocation_sequence,
                    executor_index=record.executor_index,
                    observed_ns=record.timestamp(event.INVOCATION_OBSERVED),
                    started_ns=record.timestamp(event.EXECUTION_STARTED),
                    completed_ns=record.timestamp(event.EXECUTION_COMPLETED),
                    published_ns=record.timestamp(event.COMPLETION_PUBLISHED),
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
                result_code=FfnResultCode(failure.payload.result_code),
                origin_pe=failure.payload.origin_pe,
                model_index=failure.payload.key.model_index,
                invocation_sequence=failure.payload.key.invocation_sequence,
                layer_ordinal=failure.payload.layer_ordinal,
            )
        ),
    )


def validate_fabric_trace_event_families(snapshot: xpool.native.FabricTraceSnapshot) -> None:
    """Prove mismatched bound event families fail recoverably for real records."""

    families = (
        (xpool.native.FabricTraceKind.ATNAGENT, xpool.native.AtnAgentTraceEvent.SUBMISSION_PREPARED),
        (xpool.native.FabricTraceKind.COORDINATOR, xpool.native.CoordinatorTraceEvent.ENQUEUED),
        (xpool.native.FabricTraceKind.EXECUTION, xpool.native.ExecutionTraceEvent.INVOCATION_OBSERVED),
    )
    for record in snapshot.records:
        for kind, event in families:
            if record.kind == kind:
                continue
            for operation in (record.recorded, record.timestamp):
                try:
                    operation(event)
                except ValueError:
                    continue
                raise AssertionError(f"Fabric trace accepted {event!r} for {record.kind!r}")
