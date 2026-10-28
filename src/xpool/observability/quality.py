"""Cumulative Producer accounting and explicit evidence gaps."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import Event, Finding, Metadata, QualityCheckpoint

__all__ = ["QualityReport"]


@dataclass(frozen=True)
class QualityReport:
    """Latest cumulative checkpoints plus local findings, without invented totals."""

    findings: tuple[Finding, ...]
    checkpoints: dict[tuple[str, int], QualityCheckpoint]
    # None means no closed exact checkpoint proves final totals.
    final_counters: dict[tuple[str, int], dict[str, int] | None]

    @property
    def codes(self) -> frozenset[str]:
        """Return structured finding categories without parsing diagnostic text."""
        return frozenset(finding.code for finding in self.findings)


def summarize_quality(
    events: Iterable[Event],
    documents: tuple[Metadata, ...],
    checkpoints: Iterable[QualityCheckpoint],
    findings: list[Finding],
) -> QualityReport:
    """Combine retained facts with cumulative checkpoints, preserving finding order."""
    latest: dict[tuple[str, int], QualityCheckpoint] = {}
    revisions: dict[tuple[str, int, int], QualityCheckpoint] = {}
    for checkpoint in checkpoints:
        key = (checkpoint.run_id, checkpoint.producer_id)
        rev_key = (*key, checkpoint.revision)
        if rev_key in revisions and revisions[rev_key] != checkpoint:
            raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "conflicting quality checkpoint", producer=key[1])
        revisions[rev_key] = checkpoint
    for checkpoint in sorted(revisions.values(), key=lambda item: item.revision):
        key = (checkpoint.run_id, checkpoint.producer_id)
        previous_checkpoint = latest.get(key)
        if previous_checkpoint is not None:
            previous_counters = previous_checkpoint.counters.model_dump()
            if (
                checkpoint.host_time_ns < previous_checkpoint.host_time_ns
                or checkpoint.last_attempted_sequence < previous_checkpoint.last_attempted_sequence
            ) or any(value < previous_counters[name] for name, value in checkpoint.counters.model_dump().items()):
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "cumulative checkpoint regressed", producer=key[1])
        latest[key] = checkpoint
    observed: dict[tuple[str, int], list[Event]] = defaultdict(list)
    for event in events:
        observed[(event.source.identity.run_id, event.source.identity.producer)].append(event)
    registered = {(m.run_id, p.id) for m in documents for p in m.producers}
    for key in sorted(registered | latest.keys() | observed.keys()):
        closing_checkpoint = latest.get(key)
        producer_events = sorted(observed[key], key=lambda item: item.record.event_sequence)
        if producer_events:
            sequences = [event.record.event_sequence for event in producer_events]
            if sequences[0] != 1 or any(b != a + 1 for a, b in pairwise(sequences)):
                findings.append(
                    Finding(
                        "event_missing",
                        "attempted sequence gaps; cause not inferred",
                        tuple(event.source for event in producer_events),
                        *key,
                    )
                )
            if closing_checkpoint is not None and closing_checkpoint.has_final_totals:
                if (
                    sequences[-1] > closing_checkpoint.last_attempted_sequence
                    or len(sequences) > closing_checkpoint.counters.exported
                ):
                    raise ObservationError(
                        ErrorCode.IDENTITY_CONFLICT, "closed checkpoint contradicts retained events", producer=key[1]
                    )
                if sequences[-1] < closing_checkpoint.last_attempted_sequence:
                    findings.append(
                        Finding("event_missing", "closing checkpoint proves a missing event suffix", (), *key)
                    )
        elif closing_checkpoint is None:
            findings.append(
                Finding("process_artifact_missing", "no events or checkpoint for registered Producer", (), *key)
            )
        if closing_checkpoint is None or closing_checkpoint.tail != "complete":
            findings.append(Finding("tail_unknown", "no complete closing checkpoint", (), *key))
        if closing_checkpoint is not None:
            if closing_checkpoint.counter_exhausted:
                findings.append(
                    Finding("counter_exhausted", "checkpoint counters are saturated lower bounds", (), *key)
                )
            if not closing_checkpoint.exact_totals:
                findings.append(Finding("totals_inexact", "checkpoint does not prove exact totals", (), *key))
            for counter in (
                "dropped_full",
                "abandoned",
                "post_commit_lost",
                "snapshot_operations_dropped",
                "attachment_capacity_exhausted",
                "attachment_payload_oversize",
                "attachment_not_collected",
                "attachment_publish_failed",
                "sample_failed",
                "context_unavailable",
            ):
                value = getattr(closing_checkpoint.counters, counter)
                if value:
                    findings.append(Finding(counter, f"cumulative checkpoint count {value}", (), *key))
            if closing_checkpoint.stop_reason:
                findings.append(Finding(closing_checkpoint.stop_reason, "collection stopped", (), *key))
    final = {
        key: (checkpoint.counters.model_dump() if checkpoint.has_final_totals else None)
        for key, checkpoint in latest.items()
    }
    for key in registered:
        final.setdefault(key, None)
    return QualityReport(tuple(findings), latest, final)
