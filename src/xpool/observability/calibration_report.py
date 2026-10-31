"""Deterministic calibration evidence, coverage and conditional uncertainty reports."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from xpool.observability.calibration import (
    CalibrationModel,
    CalibrationStatus,
    ClockAssumption,
    DriftEstimate,
    Exact,
    LoadPhase,
    MappedTime,
    MappingStatus,
    estimate_drift,
    map_time,
)
from xpool.observability.calibration_index import (
    CalibrationIndex,
    SamplerDeclaration,
    map_event,
)
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import ClockDomain, Finding, QualityCheckpoint, StrictModel, uint64
from xpool.observability.reconstruction import Reconstruction

__all__ = ["CalibrationReport", "build_calibration_report"]


class Distribution(StrictModel):
    """Exact count, extrema and mean with sorted original integer observations."""

    count: uint64
    minimum_ns: int | None
    maximum_ns: int | None
    mean_ns: Exact | None
    values_ns: tuple[int, ...]

    @classmethod
    def from_values(cls, values: tuple[int, ...]) -> Self:
        return cls(
            count=len(values),
            minimum_ns=min(values, default=None),
            maximum_ns=max(values, default=None),
            mean_ns=Fraction(sum(values), len(values)) if values else None,
            values_ns=tuple(sorted(values)),
        )


class PhaseWindows(StrictModel):
    """Host bracketing widths grouped by the caller's declared load phase."""

    phase: LoadPhase
    widths: Distribution


class SamplerReport(StrictModel):
    """Original declaration, models and measured diagnostics for one sampler/Epoch."""

    declaration: SamplerDeclaration
    models: tuple[CalibrationModel, ...]
    retained_attempts: uint64
    retained_failed_attempts: uint64
    duplicate_sample_count: uint64
    window_widths: Distribution
    phase_windows: tuple[PhaseWindows, ...]
    host_before_intervals: Distribution
    sampling_overhead: Distribution | None


class CoverageSpan(StrictModel):
    """A point or open interval with a constant applicable-model count.

    count=0 is a gap, count=1 supports mapping, count>1 is ambiguous. Points
    preserve inclusive-endpoint ambiguity; they have zero continuous duration.
    """

    lower_ns: uint64
    upper_ns: uint64
    endpoints_inclusive: bool
    model_count: uint64


class MappingCount(StrictModel):
    """Number of retained Device events with one mapping outcome."""

    status: MappingStatus
    count: uint64


class UncertaintySummary(StrictModel):
    """Conditional exact interval widths over mapped retained Device events."""

    count: uint64
    minimum_ns: Exact | None
    maximum_ns: Exact | None
    mean_ns: Exact | None


class DomainReport(StrictModel):
    """Per-Run raw-domain coverage; Device-time and retained-event denominators differ."""

    run_id: str
    domain: ClockDomain
    host_creation_id: str | None
    observation_interval: tuple[uint64, uint64] | None
    support_intervals: tuple[tuple[uint64, uint64], ...]
    spans: tuple[CoverageSpan, ...]
    covered_time_ns: uint64 | None
    observation_time_ns: uint64 | None
    time_coverage: Exact | None
    retained_events: uint64
    mapped_events: uint64
    retained_event_coverage: Exact | None
    mapping_counts: tuple[MappingCount, ...]
    uncertainty: UncertaintySummary


class CalibrationReport(StrictModel):
    """Synthetic/model-consistency evidence, not independent hardware timing truth.

    Quality checkpoints retain their original process/Producer scope. Measured
    overhead is caller-supplied; missing data remains null. Event mappings are
    queried on demand and are not stored in this report.
    """

    contract_version: Literal[4] = 4
    algorithm_version: Literal[1] = 1
    assumption: Literal[ClockAssumption.SEGMENT_AFFINE_BRACKETING] = ClockAssumption.SEGMENT_AFFINE_BRACKETING
    samplers: tuple[SamplerReport, ...]
    domains: tuple[DomainReport, ...]
    drift: tuple[DriftEstimate, ...]
    quality_checkpoints: tuple[QualityCheckpoint, ...]
    quality_findings: tuple[Finding, ...]
    limitations: tuple[str, ...] = (
        "Conditional affine bracketing; held-out validation is not independent hardware ground truth.",
        "Time coverage uses declared raw-domain intervals; event coverage covers retained Device events only.",
        "Sampler diagnostics require caller measurements; process counters are not apportioned to Epochs.",
        "Synthetic agreement does not establish loaded, MPS or multiprocess hardware precision.",
    )

    def to_json(self) -> str:
        """Return stable UTF-8-compatible JSON with exact rational values and a final newline."""
        return (
            json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        )


def sampler_report(index: CalibrationIndex, declaration: SamplerDeclaration) -> SamplerReport:
    models = tuple(
        sorted(
            (model for model in index.models if model.block.sampler == declaration.sampler),
            key=lambda model: model.model_id.block_index,
        )
    )
    # Models preserve duplicate sources; distributions count each attempt once.
    samples = {sample.attempt_sequence: sample for model in models for sample in model.block.samples}
    ordered = tuple(samples[key] for key in sorted(samples))
    widths: list[int] = []
    phases: dict[LoadPhase, list[int]] = defaultdict(list)
    for sample in ordered:
        if sample.host_after_ns is not None:
            width = sample.host_after_ns - sample.host_before_ns
            widths.append(width)
            phase = next(
                (
                    phase.phase
                    for phase in declaration.load_phases
                    if phase.first <= sample.attempt_sequence <= phase.last
                ),
                LoadPhase.UNSPECIFIED,
            )
            phases[phase].append(width)
    overhead = declaration.diagnostics.sampling_overhead_ns
    return SamplerReport(
        declaration=declaration,
        models=models,
        retained_attempts=len(ordered),
        retained_failed_attempts=sum(s.outcome == "failed" for s in ordered),
        duplicate_sample_count=sum(model.duplicate_sample_count for model in models),
        window_widths=Distribution.from_values(tuple(widths)),
        phase_windows=tuple(
            PhaseWindows(phase=phase, widths=Distribution.from_values(tuple(values)))
            for phase, values in sorted(phases.items())
        ),
        host_before_intervals=Distribution.from_values(
            tuple(b.host_before_ns - a.host_before_ns for a, b in pairwise(ordered))
        ),
        sampling_overhead=Distribution.from_values(overhead) if overhead is not None else None,
    )


def coverage_spans(
    intervals: tuple[tuple[int, int], ...],
    observation: tuple[int, int] | None,
) -> tuple[CoverageSpan, ...]:
    """Partition finite support/observation extents, retaining endpoint-only overlaps."""
    points = {value for interval in intervals for value in interval}
    if observation is not None:
        points.update(observation)
    ordered = sorted(points)
    spans: list[CoverageSpan] = []
    for position, point in enumerate(ordered):
        spans.append(
            CoverageSpan(
                lower_ns=point,
                upper_ns=point,
                endpoints_inclusive=True,
                model_count=sum(lo <= point <= hi for lo, hi in intervals),
            )
        )
        if position + 1 < len(ordered):
            end = ordered[position + 1]
            midpoint = Fraction(point + end, 2)
            spans.append(
                CoverageSpan(
                    lower_ns=point,
                    upper_ns=end,
                    endpoints_inclusive=False,
                    model_count=sum(lo <= midpoint <= hi for lo, hi in intervals),
                )
            )
    return tuple(spans)


@dataclass
class EventStatistics:
    """Streaming per-domain counts and uncertainty, without storing event results."""

    counts: Counter[MappingStatus] = field(default_factory=Counter)
    width_count: int = 0
    width_total: Fraction = Fraction()
    width_minimum: Fraction | None = None
    width_maximum: Fraction | None = None

    def observe(self, mapped: MappedTime) -> None:
        self.counts[mapped.status] += 1
        if mapped.status != MappingStatus.MAPPED:
            return
        assert mapped.upper_exact is not None and mapped.lower_exact is not None
        width = mapped.upper_exact - mapped.lower_exact
        self.width_count += 1
        self.width_total += width
        self.width_minimum = min(self.width_minimum, width) if self.width_minimum is not None else width
        self.width_maximum = max(self.width_maximum, width) if self.width_maximum is not None else width

    def uncertainty(self) -> UncertaintySummary:
        return UncertaintySummary(
            count=self.width_count,
            minimum_ns=self.width_minimum,
            maximum_ns=self.width_maximum,
            mean_ns=self.width_total / self.width_count if self.width_count else None,
        )


@dataclass(frozen=True)
class ModelEnvelope:
    """Conservative Host support used only to select cross-domain drift pairs."""

    run_id: str
    host_creation_id: str
    lower: Fraction
    upper: Fraction
    model: CalibrationModel

    @property
    def sort_key(self) -> tuple[str, str, Fraction, Fraction, tuple[str, int, int], int]:
        return (
            self.run_id,
            self.host_creation_id,
            self.lower,
            self.upper,
            self.model.model_id.sampler,
            self.model.model_id.block_index,
        )


def report_drift(models: tuple[CalibrationModel, ...]) -> tuple[DriftEstimate, ...]:
    """Compare adjacent sampler blocks and overlapping cross-domain Host supports.

    Arbitrary pairs remain available through estimate_drift. The summary avoids
    emitting every historical pair in a long run. Cross-domain overlap uses
    conservative mapped support envelopes, not an inferred protocol relation.
    """
    groups: dict[tuple[str, int, int], list[CalibrationModel]] = defaultdict(list)
    envelopes: list[ModelEnvelope] = []
    for model in models:
        if model.status != CalibrationStatus.VALID:
            continue
        groups[model.block.sampler.identity].append(model)
        assert model.valid_G is not None
        lower = map_time(model, model.valid_G[0]).lower_exact
        upper = map_time(model, model.valid_G[1]).upper_exact
        assert lower is not None and upper is not None
        envelopes.append(ModelEnvelope(model.block.sampler.run_id, model.block.host_creation_id, lower, upper, model))
    results = [
        estimate_drift(a, b)
        for _, group in sorted(groups.items())
        for a, b in pairwise(sorted(group, key=lambda model: model.model_id.block_index))
    ]
    active: list[ModelEnvelope] = []
    for envelope in sorted(envelopes, key=lambda item: item.sort_key):
        active = [
            item
            for item in active
            if (item.run_id, item.host_creation_id) == (envelope.run_id, envelope.host_creation_id)
            and item.upper >= envelope.lower
        ]
        for other in active:
            if other.model.block.sampler.device_clock_domain_id != envelope.model.block.sampler.device_clock_domain_id:
                results.append(estimate_drift(other.model, envelope.model))
        active.append(envelope)
    return tuple(results)


def build_calibration_report(index: CalibrationIndex, view: Reconstruction) -> CalibrationReport:
    """Report evidence and coverage without materializing every event mapping.

    Invalid identities/declarations fail immediately. Missing observation ranges
    and measurements remain null. Duplicate events use the canonical reconstruction inventory.
    """
    samplers = tuple(sampler_report(index, declaration) for declaration in index.declarations)
    domains: list[DomainReport] = []
    # Accumulate only per-domain statistics while querying each retained event once.
    statistics: dict[tuple[str, int], EventStatistics] = defaultdict(EventStatistics)
    for event in view.events.values():
        key = event.source.identity.run_id, event.record.clock_domain_id
        reference = index.domain(*key)
        if reference.domain.kind != "device_globaltimer":
            continue
        statistics[key].observe(map_event(index, event))

    # Time coverage measures exactly-one-model spans inside a declared denominator.
    for reference in index.domains:
        if reference.domain.kind != "device_globaltimer":
            continue
        key = reference.run_id, reference.domain.id
        intervals = tuple(
            sorted(
                model.valid_G
                for model in index.models_for(*key)
                if model.status == CalibrationStatus.VALID and model.valid_G is not None
            )
        )
        observations = {
            declaration.observation_interval
            for declaration in index.declarations
            if (declaration.sampler.run_id, declaration.sampler.device_clock_domain_id) == key
            and declaration.observation_interval is not None
        }
        if len(observations) > 1:
            raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "conflicting domain observation intervals")
        observation = next(iter(observations), None)
        spans = coverage_spans(intervals, observation)
        covered = denominator = None
        if observation is not None:
            lo, hi = observation
            denominator = hi - lo
            covered = sum(
                max(0, min(span.upper_ns, hi) - max(span.lower_ns, lo))
                for span in spans
                if span.model_count == 1 and not span.endpoints_inclusive
            )
        stats = statistics[key]
        event_counts = stats.counts
        total_events = sum(event_counts.values())
        mapped_events = event_counts.get(MappingStatus.MAPPED, 0)
        domains.append(
            DomainReport(
                run_id=key[0],
                domain=reference.domain,
                host_creation_id=reference.host_creation_id,
                observation_interval=observation,
                support_intervals=intervals,
                spans=spans,
                covered_time_ns=covered,
                observation_time_ns=denominator,
                time_coverage=Fraction(covered, denominator) if covered is not None and denominator else None,
                retained_events=total_events,
                mapped_events=mapped_events,
                retained_event_coverage=Fraction(mapped_events, total_events) if total_events else None,
                mapping_counts=tuple(
                    MappingCount(status=status, count=event_counts.get(status, 0)) for status in MappingStatus
                ),
                uncertainty=stats.uncertainty(),
            )
        )
    return CalibrationReport(
        samplers=samplers,
        domains=tuple(domains),
        drift=report_drift(index.models),
        quality_checkpoints=tuple(view.quality.checkpoints[key] for key in sorted(view.quality.checkpoints)),
        quality_findings=tuple(
            sorted(
                view.quality.findings,
                key=lambda finding: (
                    finding.run_id or "",
                    finding.producer or 0,
                    finding.code,
                    finding.detail,
                    tuple((source.identity, str(source.path), source.offset) for source in finding.sources),
                ),
            )
        ),
    )
