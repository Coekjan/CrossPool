"""Join explicit sampler declarations to verified offline reconstruction evidence."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType

from xpool.observability.calibration import (
    CalibrationError,
    CalibrationModel,
    CalibrationStatus,
    ClockAssumption,
    LoadPhaseRange,
    MappedTime,
    MappingReason,
    MappingStatus,
    SampleBlock,
    Sampler,
    calibrate,
    map_time,
)
from xpool.observability.codec import validate_event
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import (
    ClockDomain,
    Event,
    PairedClockSample,
    Producer,
    StrictModel,
    positive_uint64,
    uint64,
)
from xpool.observability.provenance import Source
from xpool.observability.reconstruction import Reconstruction

__all__ = ["CalibrationIndex", "SamplerDeclaration", "SamplerDiagnostics", "build_calibration", "map_event"]


class SamplerDiagnostics(StrictModel):
    """Caller-measured diagnostics scoped to this sampler/Epoch, not a process.

    Overhead values are explicit measured nanoseconds with caller provenance;
    Host bracket widths do not implicitly measure sampling overhead.
    build_calibration checks cross-field counts with the declaration's sampler.
    """

    attempted: uint64 | None = None
    failed: uint64 | None = None
    skipped_busy: uint64 | None = None
    sampling_overhead_ns: tuple[uint64, ...] | None = None
    provenance: tuple[str, ...] = ()


class SamplerDeclaration(StrictModel):
    """Caller-owned evidence for one sampler/Epoch and its observation scope.

    Attempt and Device observation ranges are inclusive. Missing attempt ranges
    never imply completeness. Phase ranges may partition one continuous run.
    Construction validates fields; build_calibration rejects contradictory
    ranges, phases and counts through CalibrationError.
    """

    sampler: Sampler
    declared_attempt_range: tuple[positive_uint64, positive_uint64] | None = None
    observation_interval: tuple[uint64, uint64] | None = None
    load_phases: tuple[LoadPhaseRange, ...] = ()
    provenance: tuple[str, ...] = ()
    diagnostics: SamplerDiagnostics = SamplerDiagnostics()

    def validate_semantics(self, *, sources: tuple[Source, ...] = ()) -> None:
        """Reject contradictory ranges/counts with sampler and available sample sources.

        Field types and individual integer bounds are checked during construction.
        build_calibration calls this semantic boundary before resolving registration
        or creating blocks. An invalid declaration raises CalibrationError.
        """
        # Interval ordering belongs to the sampler-aware semantic boundary.
        for interval in (self.declared_attempt_range, self.observation_interval):
            if interval is not None and interval[0] > interval[1]:
                raise CalibrationError(
                    ErrorCode.INVALID_SAMPLE, "reversed declaration range", self.sampler, sources=sources
                )

        # Phase ranges must form disjoint portions of the declared attempted scope.
        phases = sorted(self.load_phases, key=lambda phase: (phase.first, phase.last))
        previous = 0
        for phase in phases:
            if phase.first > phase.last:
                raise CalibrationError(
                    ErrorCode.INVALID_SAMPLE, "reversed load-phase range", self.sampler, sources=sources
                )
            if phase.first <= previous:
                raise CalibrationError(
                    ErrorCode.INVALID_SAMPLE, "overlapping load phases", self.sampler, sources=sources
                )
            if self.declared_attempt_range is not None and not (
                self.declared_attempt_range[0] <= phase.first <= phase.last <= self.declared_attempt_range[1]
            ):
                raise CalibrationError(
                    ErrorCode.INVALID_SAMPLE, "phase outside declared attempts", self.sampler, sources=sources
                )
            previous = phase.last

        # Failed attempts are a subset of this sampler's measured attempts.
        diagnostics = self.diagnostics
        if (
            diagnostics.attempted is not None
            and diagnostics.failed is not None
            and (diagnostics.failed > diagnostics.attempted)
        ):
            raise CalibrationError(
                ErrorCode.INVALID_SAMPLE, "failed samples exceed attempted samples", self.sampler, sources=sources
            )


@dataclass(frozen=True)
class DomainReference:
    """A registered raw domain and its common Host reference when evidenced."""

    run_id: str
    domain: ClockDomain
    host_creation_id: str | None


@dataclass(frozen=True)
class CalibrationIndex:
    """Immutable model inventory with per-domain lookup, without event duplication."""

    models: tuple[CalibrationModel, ...]
    declarations: tuple[SamplerDeclaration, ...]
    domains: tuple[DomainReference, ...]
    domain_lookup: Mapping[tuple[str, int], DomainReference] = field(init=False, repr=False)
    model_lookup: Mapping[tuple[str, int], tuple[CalibrationModel, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        domains: dict[tuple[str, int], DomainReference] = {}
        for reference in self.domains:
            key = reference.run_id, reference.domain.id
            old = domains.get(key)
            if old is not None and old != reference:
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "conflicting clock reference")
            domains[key] = reference
        grouped: dict[tuple[str, int], list[CalibrationModel]] = defaultdict(list)
        seen: set[tuple[tuple[str, int, int], int]] = set()
        for model in self.models:
            identity = model.model_id.sampler, model.model_id.block_index
            if identity in seen:
                raise CalibrationError(ErrorCode.IDENTITY_CONFLICT, "duplicate model identity", model.block.sampler)
            seen.add(identity)
            sampler = model.block.sampler
            registered = domains.get((sampler.run_id, sampler.device_clock_domain_id))
            if (
                registered is None
                or registered.domain.raw_epoch != sampler.raw_epoch
                or (registered.host_creation_id != model.block.host_creation_id)
            ):
                raise CalibrationError(ErrorCode.IDENTITY_CONFLICT, "model clock reference mismatch", sampler)
            grouped[sampler.run_id, sampler.device_clock_domain_id].append(model)
        object.__setattr__(self, "domain_lookup", MappingProxyType(domains))
        object.__setattr__(
            self, "model_lookup", MappingProxyType({key: tuple(value) for key, value in grouped.items()})
        )

    def domain(self, run_id: str, domain_id: int) -> DomainReference:
        """Resolve a registered domain; unknown identities are input errors."""
        try:
            return self.domain_lookup[run_id, domain_id]
        except KeyError as exc:
            raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "unregistered calibration clock domain") from exc

    def models_for(self, run_id: str, domain_id: int) -> tuple[CalibrationModel, ...]:
        """Return this raw domain's models, including unavailable blocks."""
        return self.model_lookup.get((run_id, domain_id), ())

    def map_raw(self, run_id: str, domain_id: int, raw_time_ns: int) -> MappedTime:
        """Select exactly one supported model; never interpolate or extrapolate."""
        reference = self.domain(run_id, domain_id)
        domain = reference.domain
        base = MappedTime(
            run_id=run_id,
            host_creation_id=reference.host_creation_id,
            clock_domain_id=domain_id,
            raw_epoch=domain.raw_epoch,
            raw_time_ns=raw_time_ns,
            status=MappingStatus.UNCALIBRATED,
        )
        if domain.kind == "host_monotonic":
            exact = Fraction(raw_time_ns)
            return base.model_copy(
                update={
                    "status": MappingStatus.MAPPED,
                    "estimate_exact": exact,
                    "lower_exact": exact,
                    "upper_exact": exact,
                    "estimate_ns": raw_time_ns,
                    "lower_ns": raw_time_ns,
                    "upper_ns": raw_time_ns,
                    "assumption": ClockAssumption.HOST_IDENTITY,
                }
            )
        models = self.models_for(run_id, domain_id)
        valid = tuple(model for model in models if model.status == CalibrationStatus.VALID)
        applicable = tuple(
            model
            for model in valid
            if model.valid_G is not None and model.valid_G[0] <= raw_time_ns <= model.valid_G[1]
        )
        if len(applicable) == 1:
            return map_time(applicable[0], raw_time_ns)
        if len(applicable) > 1:
            return base.model_copy(
                update={"status": MappingStatus.AMBIGUOUS_MODEL, "reasons": (MappingReason.AMBIGUOUS_MODEL,)}
            )
        reasons = tuple(sorted({reason for model in models for reason in model.reasons}))
        return base.model_copy(
            update={
                "status": MappingStatus.OUTSIDE_SUPPORT if valid else MappingStatus.UNCALIBRATED,
                "reasons": reasons
                or ((MappingReason.OUTSIDE_SUPPORT,) if valid else (MappingReason.CALIBRATION_MISSING,)),
            }
        )


def build_calibration(view: Reconstruction, declarations: Iterable[SamplerDeclaration]) -> CalibrationIndex:
    """Build fixed blocks from explicit declarations and verified reconstruction samples.

    Every retained sampler must be declared. Conflicts fail the entire call.
    Undeclared Device event domains remain queryable as uncalibrated. A jump
    poisons subsequent blocks of its Epoch until a new registered Epoch begins.
    """
    # Validate declarations with all available physical sources before resolving registrations.
    declarations = tuple(declarations)
    sample_sources: dict[tuple[str, int, int], set[Source]] = defaultdict(set)
    for sample in view.calibration_samples:
        identity = sample.source.identity
        sample_key = identity.run_id, identity.producer, sample.device_clock_domain_id
        sample_sources[sample_key].update(view.event_sources.get(identity, (sample.source,)))
    for declaration in declarations:
        declaration.validate_semantics(
            sources=tuple(
                sorted(
                    sample_sources[declaration.sampler.identity],
                    key=lambda source: (source.identity, str(source.path), source.offset),
                )
            )
        )

    # Resolve authoritative registrations across revisions before joining samples.
    producers: dict[tuple[str, int], Producer] = {}
    domains: dict[tuple[str, int], ClockDomain] = {}
    for metadata in view.metadata:
        for producer in metadata.producers:
            producer_key = metadata.run_id, producer.id
            if producer_key in producers and producers[producer_key] != producer:
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "conflicting Producer registration")
            producers[producer_key] = producer
        for domain in metadata.clock_domains:
            domain_key = metadata.run_id, domain.id
            if domain_key in domains and domains[domain_key] != domain:
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "conflicting clock registration")
            domains[domain_key] = domain
    references: dict[tuple[str, int], DomainReference] = {}
    for (run_id, domain_id), domain in sorted(domains.items()):
        hosts = {
            domains[run_id, p.clock_domain_id].host_creation_id
            for (run, _), p in producers.items()
            if run == run_id and p.source == "host" and p.process_creation_id == domain.process_creation_id
        }
        if domain.kind == "host_monotonic":
            hosts = {domain.host_creation_id}
        if len(hosts) > 1:
            raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "process has conflicting Host references")
        references[run_id, domain_id] = DomainReference(run_id, domain, next(iter(hosts), None))

    # One declaration owns a sampler/Epoch; exact duplicates do not create models twice.
    declared: dict[tuple[str, int, int], SamplerDeclaration] = {}
    for declaration in declarations:
        sampler = declaration.sampler
        key = sampler.identity
        old = declared.get(key)
        if old is not None and old != declaration:
            raise CalibrationError(ErrorCode.IDENTITY_CONFLICT, "conflicting sampler declarations", sampler)
        host = producers.get((sampler.run_id, sampler.host_producer))
        device = references.get((sampler.run_id, sampler.device_clock_domain_id))
        if (
            host is None
            or host.source != "host"
            or device is None
            or (
                device.domain.kind != "device_globaltimer"
                or device.domain.raw_epoch != sampler.raw_epoch
                or device.domain.process_creation_id != host.process_creation_id
                or device.host_creation_id is None
                or domains[sampler.run_id, host.clock_domain_id].host_creation_id != device.host_creation_id
            )
        ):
            raise CalibrationError(ErrorCode.IDENTITY_CONFLICT, "sampler registration mismatch", sampler)
        declared[key] = declaration
    grouped: dict[tuple[str, int, int], list[PairedClockSample]] = defaultdict(list)
    for sample in view.calibration_samples:
        key = sample.source.identity.run_id, sample.source.identity.producer, sample.device_clock_domain_id
        if key not in declared:
            sampler = Sampler(
                run_id=key[0], host_producer=key[1], device_clock_domain_id=key[2], raw_epoch=sample.raw_epoch
            )
            raise CalibrationError(
                ErrorCode.INVALID_SAMPLE,
                "retained sampler has no declaration",
                sampler,
                attempt_sequence=sample.attempt_sequence,
                sources=(sample.source,),
            )
        locations = view.event_sources.get(sample.source.identity, (sample.source,))
        canonical = min(locations, key=lambda source: (str(source.path), source.offset))
        grouped[key].append(replace(sample, source=canonical))

    for key, samples in grouped.items():
        diagnostics = declared[key].diagnostics
        if diagnostics.attempted is not None and any(s.attempt_sequence > diagnostics.attempted for s in samples):
            raise CalibrationError(
                ErrorCode.INVALID_SAMPLE,
                "retained attempt exceeds measured total",
                declared[key].sampler,
                sources=tuple(sample.source for sample in samples),
            )
        failed = {s.attempt_sequence for s in samples if s.outcome == "failed"}
        if diagnostics.failed is not None and len(failed) > diagnostics.failed:
            raise CalibrationError(
                ErrorCode.INVALID_SAMPLE,
                "retained failures exceed measured total",
                declared[key].sampler,
                sources=tuple(sample.source for sample in samples if sample.outcome == "failed"),
            )

    # Preserve attempted numbering and propagate observed jumps across fixed blocks.
    models: list[CalibrationModel] = []
    for key, declaration in sorted(declared.items()):
        reference = references[key[0], key[2]]
        assert reference.host_creation_id is not None
        by_block: dict[int, list[PairedClockSample]] = defaultdict(list)
        for sample in grouped[key]:
            by_block[(sample.attempt_sequence - 1) // 10].append(sample)
        if declaration.declared_attempt_range is None:
            indices = sorted(by_block)
        else:
            lo, hi = declaration.declared_attempt_range
            indices = sorted(set(range((lo - 1) // 10, (hi - 1) // 10 + 1)) | set(by_block))
        previous_device: int | None = None
        jumped = False
        for block_index in indices:
            block = SampleBlock(
                sampler=declaration.sampler,
                host_creation_id=reference.host_creation_id,
                block_index=block_index,
                declared_attempt_range=declaration.declared_attempt_range,
                samples=tuple(sorted(by_block[block_index], key=lambda s: (s.attempt_sequence, s.source.identity))),
                source_locations=tuple(
                    sorted(
                        {
                            source
                            for sample in by_block[block_index]
                            for source in view.event_sources.get(sample.source.identity, (sample.source,))
                        },
                        key=lambda source: (source.identity, str(source.path), source.offset),
                    )
                ),
                load_phases=tuple(
                    sorted(
                        (
                            phase
                            for phase in declaration.load_phases
                            if phase.first <= block_index * 10 + 10 and phase.last > block_index * 10
                        ),
                        key=lambda phase: phase.first,
                    )
                ),
                provenance=declaration.provenance,
            )
            canonical_samples, _ = block.canonical_samples()
            for sample in canonical_samples:
                if sample.device_time_ns is not None:
                    if previous_device is not None and sample.device_time_ns < previous_device:
                        jumped = True
                    previous_device = sample.device_time_ns
            models.append(calibrate(block.model_copy(update={"clock_jump": jumped})))
    return CalibrationIndex(tuple(models), tuple(declared[key] for key in sorted(declared)), tuple(references.values()))


def map_event(index: CalibrationIndex, event: Event) -> MappedTime:
    """Map a verified event while checking its registered clock and retaining its source."""
    validate_event(event)
    source = event.source
    record = event.record
    reference = index.domain(source.identity.run_id, record.clock_domain_id)
    domain = next(item for item in event.metadata.clock_domains if item.id == record.clock_domain_id)
    if (
        domain != reference.domain
        or source.identity.run_id != event.metadata.run_id
        or source.identity.producer != event.metadata.producer_id
        or source.identity.event_sequence != record.event_sequence
        or source.clock_domain_id != record.clock_domain_id
        or source.process_creation_id != event.metadata.process.creation_id
    ):
        raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "event calibration provenance mismatch", sources=(source,))
    return index.map_raw(source.identity.run_id, record.clock_domain_id, record.raw_time_ns).model_copy(
        update={"sources": (source,)}
    )
