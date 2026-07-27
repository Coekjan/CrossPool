"""Structured output for per-PE native Fabric traces."""

from __future__ import annotations

import json
import logging
from functools import wraps
from pathlib import Path
from threading import Lock

import xpool.native
from xpool.abi import DpPaddingMode, FfnResultHandoff, XPoolForwardMode
from xpool.config import get_global_config
from xpool.fabric import FabricGeneration, FabricParticipantPhase
from xpool.runtime import RuntimeRole
from xpool.runtime.agent import Agent

runtime_roles = frozenset({RuntimeRole.ATNAGENT, RuntimeRole.FFNAGENT})
logger = logging.getLogger(__name__)
install_lock = Lock()
installed = False


def install() -> None:
    """Install per-PE Fabric snapshot recording after local drain completes."""

    outdir = get_global_config().debug.fabric_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool fabric observer requires debug.fabric_observer.outdir")
    global installed
    with install_lock:
        outdir.mkdir(parents=True, exist_ok=True)
        if installed:
            return
        original_advance = Agent.advance_fabric_lifecycle

        @wraps(original_advance)
        def observed_advance(agent: Agent) -> None:
            plan = agent.fabric_plan
            previous_report = agent.participant_report
            previous_phase = previous_report.phase if previous_report is not None else None
            original_advance(agent)
            updated_report = agent.participant_report
            if (
                plan is None
                or previous_phase is FabricParticipantPhase.DRAINED
                or updated_report is None
                or updated_report.phase is not FabricParticipantPhase.DRAINED
            ):
                return
            try:
                snapshot = xpool.native.fabric.read_trace()
                if snapshot is not None:
                    write_fabric_snapshot(plan.generation, snapshot)
            except Exception:
                logger.warning("Failed to record xpool Fabric observer snapshot", exc_info=True)

        setattr(Agent, "advance_fabric_lifecycle", observed_advance)
        installed = True


def forward_mode_name(value: int | None) -> str | None:
    """Project one validated forward mode to its observer spelling."""

    if value is None:
        return None
    match value:
        case XPoolForwardMode.EXTEND:
            return "extend"
        case XPoolForwardMode.DECODE:
            return "decode"
        case XPoolForwardMode.IDLE:
            return "idle"
        case _:
            raise RuntimeError(f"xpool Fabric observer received invalid forward mode {value}")


def execution_mode_name(value: int | None) -> str | None:
    """Project one Fabric execution mode to its observer spelling."""

    match value:
        case None:
            return None
        case 1:
            return "prefill"
        case 2:
            return "decode"
        case _:
            raise RuntimeError(f"xpool Fabric observer received invalid execution mode {value}")


def result_handoff_name(value: int) -> str:
    """Project one validated result-handoff mode to its observer spelling."""

    match value:
        case FfnResultHandoff.REPLICATED_FULL:
            return "replicated_full"
        case FfnResultHandoff.REDUCE_SCATTER_INPUT:
            return "reduce_scatter_input"
        case _:
            raise RuntimeError(f"xpool Fabric observer received invalid result handoff {value}")


def dp_padding_mode_name(value: int) -> str:
    """Project one validated DP-padding mode to its observer spelling."""

    match value:
        case DpPaddingMode.NONE:
            return "none"
        case DpPaddingMode.MAX_LEN:
            return "max_len"
        case DpPaddingMode.SUM_LEN:
            return "sum_len"
        case _:
            raise RuntimeError(f"xpool Fabric observer received invalid DP padding mode {value}")


def result_contribution_name(value: int | None) -> str | None:
    """Project one Fabric result contribution to its observer spelling."""

    match value:
        case None:
            return None
        case 1:
            return "full"
        case 2:
            return "zero"
        case _:
            raise RuntimeError(f"xpool Fabric observer received invalid result contribution {value}")


def durations(
    events_ns: dict[str, int],
    pairs: tuple[tuple[str, str, str], ...],
) -> dict[str, int]:
    """Derive valid same-record durations from explicit event pairs."""

    result: dict[str, int] = {}
    for name, start_name, end_name in pairs:
        start = events_ns[start_name]
        end = events_ns[end_name]
        if start != 0 and end >= start:
            result[name] = end - start
    return result


def atnagent_payload(
    record: xpool.native.FabricTraceRecord,
) -> tuple[dict[str, object], dict[str, int], dict[str, int]]:
    """Project one AtnAgent trace alternative."""

    event = xpool.native.AtnAgentTraceEvent
    events_ns = {
        "submission_prepared": record.timestamp(event.SUBMISSION_PREPARED),
        "decode_input_staged": record.timestamp(event.DECODE_INPUT_STAGED),
        "submission_published": record.timestamp(event.SUBMISSION_PUBLISHED),
        "admission_observed": record.timestamp(event.ADMISSION_OBSERVED),
        "prefill_input_staged": record.timestamp(event.PREFILL_INPUT_STAGED),
        "prefill_input_published": record.timestamp(event.PREFILL_INPUT_PUBLISHED),
        "result_observed": record.timestamp(event.RESULT_OBSERVED),
        "output_prepared": record.timestamp(event.OUTPUT_PREPARED),
        "transport_evaluated_published": record.timestamp(event.TRANSPORT_EVALUATED_PUBLISHED),
        "acknowledgement_published": record.timestamp(event.ACKNOWLEDGEMENT_PUBLISHED),
    }
    facts: dict[str, object] = {
        "submission_payload_rows": record.submission_payload_rows,
        "local_token_count": record.local_token_count,
        "forward_mode": forward_mode_name(record.forward_mode),
        "executor_index": record.executor_index,
        "execution_mode": execution_mode_name(record.execution_mode),
        "result_contribution": result_contribution_name(record.result_contribution),
    }
    projected_durations = durations(
        events_ns,
        (
            ("submission", "submission_prepared", "submission_published"),
            ("decode_input_staging", "submission_prepared", "decode_input_staged"),
            ("admission_wait", "submission_published", "admission_observed"),
            ("prefill_input_staging", "admission_observed", "prefill_input_staged"),
            ("prefill_input_publication", "prefill_input_staged", "prefill_input_published"),
            ("result_wait", "admission_observed", "result_observed"),
            ("output_preparation", "result_observed", "output_prepared"),
            (
                "transport_evaluation_publication",
                "output_prepared",
                "transport_evaluated_published",
            ),
            (
                "acknowledgement_publication",
                "transport_evaluated_published",
                "acknowledgement_published",
            ),
            ("total", "submission_prepared", "acknowledgement_published"),
        ),
    )
    return facts, events_ns, projected_durations


def coordinator_payload(
    record: xpool.native.FabricTraceRecord,
) -> tuple[dict[str, object], dict[str, int], dict[str, int]]:
    """Project one Coordinator trace alternative."""

    event = xpool.native.CoordinatorTraceEvent
    events_ns = {
        "enqueued": record.timestamp(event.ENQUEUED),
        "scheduled": record.timestamp(event.SCHEDULED),
        "admissions_published": record.timestamp(event.ADMISSIONS_PUBLISHED),
        "invocations_published": record.timestamp(event.INVOCATIONS_PUBLISHED),
        "completions_observed": record.timestamp(event.COMPLETIONS_OBSERVED),
        "results_published": record.timestamp(event.RESULTS_PUBLISHED),
        "acknowledgements_observed": record.timestamp(event.ACKNOWLEDGEMENTS_OBSERVED),
        "scheduler_released": record.timestamp(event.SCHEDULER_RELEASED),
    }
    if not record.recorded(event.ENQUEUED):
        scheduler = None
    elif record.ready_ticket is None:
        scheduler = {"policy": "random"}
    else:
        scheduler = {"policy": "fifo", "ready_ticket": record.ready_ticket}
    facts: dict[str, object] = {
        "invocation_payload_rows": record.invocation_payload_rows,
        "input_pe": record.input_pe,
        "execution_mode": execution_mode_name(record.execution_mode),
        "executor_index": record.executor_index,
        "scheduler": scheduler,
    }
    projected_durations = durations(
        events_ns,
        (
            ("scheduling", "enqueued", "scheduled"),
            ("admission_publication", "scheduled", "admissions_published"),
            ("invocation_publication", "scheduled", "invocations_published"),
            ("completion_wait", "invocations_published", "completions_observed"),
            ("result_publication", "completions_observed", "results_published"),
            ("acknowledgement_wait", "results_published", "acknowledgements_observed"),
            ("scheduler_release", "acknowledgements_observed", "scheduler_released"),
            ("active_total", "scheduled", "scheduler_released"),
            ("total", "enqueued", "scheduler_released"),
        ),
    )
    return facts, events_ns, projected_durations


def execution_payload(
    record: xpool.native.FabricTraceRecord,
) -> tuple[dict[str, object], dict[str, int], dict[str, int]]:
    """Project one Execution trace alternative."""

    event = xpool.native.ExecutionTraceEvent
    events_ns = {
        "invocation_observed": record.timestamp(event.INVOCATION_OBSERVED),
        "decode_input_pull_started": record.timestamp(event.DECODE_INPUT_PULL_STARTED),
        "decode_input_pull_completed": record.timestamp(event.DECODE_INPUT_PULL_COMPLETED),
        "prefill_input_ready_observed": record.timestamp(event.PREFILL_INPUT_READY_OBSERVED),
        "execution_started": record.timestamp(event.EXECUTION_STARTED),
        "execution_completed": record.timestamp(event.EXECUTION_COMPLETED),
        "completion_published": record.timestamp(event.COMPLETION_PUBLISHED),
    }
    mode = execution_mode_name(record.execution_mode)
    if mode == "decode":
        mode_pairs = (
            ("input_readiness", "invocation_observed", "decode_input_pull_completed"),
            ("execution_start", "decode_input_pull_completed", "execution_started"),
            ("decode_input_pull", "decode_input_pull_started", "decode_input_pull_completed"),
        )
    elif mode == "prefill":
        mode_pairs = (
            ("input_readiness", "invocation_observed", "prefill_input_ready_observed"),
            ("execution_start", "prefill_input_ready_observed", "execution_started"),
        )
    else:
        raise RuntimeError("xpool Fabric Execution trace has no execution mode")
    facts: dict[str, object] = {
        "invocation_payload_rows": record.invocation_payload_rows,
        "input_pe": record.input_pe,
        "execution_mode": mode,
        "executor_index": record.executor_index,
    }
    projected_durations = durations(
        events_ns,
        (
            *mode_pairs,
            ("execution", "execution_started", "execution_completed"),
            ("completion_publication", "execution_completed", "completion_published"),
            ("total", "invocation_observed", "completion_published"),
        ),
    )
    return facts, events_ns, projected_durations


def record_payload(record: xpool.native.FabricTraceRecord) -> dict[str, object]:
    """Serialize one local Fabric record and derive same-device durations."""

    match record.kind:
        case xpool.native.FabricTraceKind.ATNAGENT:
            kind = "atnagent"
            facts, events_ns, durations_ns = atnagent_payload(record)
        case xpool.native.FabricTraceKind.COORDINATOR:
            kind = "coordinator"
            facts, events_ns, durations_ns = coordinator_payload(record)
        case xpool.native.FabricTraceKind.EXECUTION:
            kind = "execution"
            facts, events_ns, durations_ns = execution_payload(record)
        case unexpected_kind:
            raise RuntimeError(f"xpool Fabric observer received unknown trace kind {unexpected_kind!r}")
    return {
        "local_trace_id": record.local_trace_id,
        "kind": kind,
        "model_index": record.key.model_index,
        "invocation_sequence": record.key.invocation_sequence,
        "layer_ordinal": record.layer_ordinal,
        "layer_id": record.layer_id,
        "result_handoff": result_handoff_name(record.result_handoff),
        "dp_padding_mode": dp_padding_mode_name(record.dp_padding_mode),
        "facts": facts,
        "events_ns": events_ns,
        "durations_ns": durations_ns,
    }


def write_fabric_snapshot(
    generation: FabricGeneration,
    snapshot: xpool.native.FabricTraceSnapshot,
) -> Path:
    """Atomically write one PE's local Fabric trace snapshot as JSON."""

    outdir = get_global_config().debug.fabric_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool fabric observer requires debug.fabric_observer.outdir")
    records = [record_payload(record) for record in snapshot.records if record.local_trace_id != 0]
    records.sort(key=lambda record: int(record["local_trace_id"]))
    generation_text = generation.format()
    payload = {
        "generation": generation_text,
        "pe": snapshot.pe,
        "atnagent_count": snapshot.atnagent_count,
        "ffnagent_count": snapshot.ffnagent_count,
        "model_topologies": [
            {
                "atn_tp_size": topology.atn_tp_size,
                "atn_dp_size": topology.atn_dp_size,
            }
            for topology in snapshot.model_topologies
        ],
        "sequence": snapshot.sequence,
        "dropped": snapshot.dropped,
        "records": records,
    }
    path = outdir / f"xpool.fabric-observer.{generation_text}.{snapshot.pe}.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path
