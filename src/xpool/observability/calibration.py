"""Exact, conditional single-Host clock calibration over immutable paired samples."""

from enum import StrEnum
from fractions import Fraction
from itertools import combinations, pairwise
from math import ceil, floor
from typing import Annotated, Literal, NamedTuple, Self, TypeGuard, cast
from uuid import UUID

from pydantic import PlainSerializer, model_validator

from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import PairedClockSample, StrictModel, positive_uint64, uint64
from xpool.observability.provenance import Source
from xpool.observability.wire import SampleFailureCode

__all__ = [
    "CalibrationError",
    "CalibrationModel",
    "CalibrationStatus",
    "ClockAssumption",
    "DriftEstimate",
    "DurationEstimate",
    "DurationReason",
    "DurationStatus",
    "LoadPhase",
    "LoadPhaseRange",
    "MappedTime",
    "MappingReason",
    "MappingStatus",
    "ModelId",
    "Ordering",
    "SampleBlock",
    "Sampler",
    "calibrate",
    "estimate_drift",
    "estimate_duration",
    "map_time",
]


class Rational(StrictModel):
    """Lossless JSON representation of a reduced rational number."""

    numerator: str
    denominator: str

    @classmethod
    def from_fraction(cls, value: Fraction) -> Self:
        """Preserve arbitrary integer precision when serializing exact arithmetic."""
        return cls(numerator=str(value.numerator), denominator=str(value.denominator))


type Exact = Annotated[Fraction, PlainSerializer(Rational.from_fraction, return_type=Rational, when_used="json")]


class CalibrationStatus(StrEnum):
    """Block outcomes in numerical-contract precedence order."""

    VALID = "valid"
    ATTEMPT_RANGE_UNKNOWN = "attempt_range_unknown"
    INCOMPLETE_BLOCK = "incomplete_block"
    SAMPLING_GAP = "sampling_gap"
    SAMPLE_FAILED = "sample_failed"
    CLOCK_JUMP = "clock_jump"
    INSUFFICIENT_SPAN = "insufficient_span"
    MODEL_CONFLICT = "model_conflict"
    VALIDATION_OUTSIDE_SUPPORT = "validation_outside_support"
    VALIDATION_FAILED = "validation_failed"


class MappingStatus(StrEnum):
    """Availability of a raw timestamp on the common Host reference."""

    MAPPED = "mapped"
    OUTSIDE_SUPPORT = "outside_support"
    AMBIGUOUS_MODEL = "ambiguous_model"
    UNCALIBRATED = "uncalibrated"


class ClockAssumption(StrEnum):
    """Evidence assumptions under which a Host-reference mapping is reported."""

    HOST_IDENTITY = "host_identity"
    SEGMENT_AFFINE_BRACKETING = "segment_affine_bracketing"


class MappingReason(StrEnum):
    """Mapping-specific limitations, alongside existing block status reasons."""

    OUTSIDE_SUPPORT = "outside_support"
    AMBIGUOUS_MODEL = "ambiguous_model"
    CALIBRATION_MISSING = "calibration_missing"


class DurationReason(StrEnum):
    """Unavailable elapsed-time evidence at one or both mapped endpoints."""

    UNAVAILABLE_ENDPOINT = "unavailable_endpoint"


class Ordering(StrEnum):
    """Temporal relation of end relative to start, independent of protocol edges."""

    AFTER = "after"
    BEFORE = "before"
    SIMULTANEOUS = "simultaneous"
    INDETERMINATE = "indeterminate"


class DurationStatus(StrEnum):
    """Whether both endpoints support an elapsed-time estimate."""

    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class LoadPhase(StrEnum):
    """Declared measurement conditions; absent phase evidence remains unspecified."""

    INITIALIZATION = "initialization"
    IDLE = "idle"
    SERVING = "serving"
    STRESS = "stress"
    FINALIZATION = "finalization"
    SYNTHETIC = "synthetic"
    UNSPECIFIED = "unspecified"


def is_uint64(value: object) -> TypeGuard[int]:
    """Validate raw integer observations, accepting integer subclasses but rejecting bool."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**64 - 1


class Sampler(StrictModel):
    """One Host Producer sampling one process-owned Device clock Epoch."""

    run_id: str
    host_producer: positive_uint64
    device_clock_domain_id: positive_uint64
    raw_epoch: positive_uint64

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        """Reject malformed or noncanonical Run UUIDs before sampler identity is used."""
        if str(UUID(self.run_id)) != self.run_id:
            raise ValueError("Run UUID must be canonical")
        return self

    @property
    def identity(self) -> tuple[str, int, int]:
        """Return the contract's sampler identity; the domain owns its Epoch."""
        return self.run_id, self.host_producer, self.device_clock_domain_id


class CalibrationError(ObservationError):
    """An invalid calibration input with exact sampler and sample context."""

    def __init__(
        self,
        code: ErrorCode,
        detail: str,
        sampler: Sampler,
        *,
        attempt_sequence: int | None = None,
        sources: tuple[Source, ...] = (),
    ) -> None:
        self.sampler = sampler
        self.attempt_sequence = attempt_sequence
        super().__init__(code, detail, producer=sampler.host_producer, sources=sources)


class LoadPhaseRange(StrictModel):
    """Inclusive phase declaration; build_calibration rejects reversed/overlapping ranges."""

    first: positive_uint64
    last: positive_uint64
    phase: LoadPhase


class SampleBlock(StrictModel):
    """One fixed ten-attempt block; declarations never derive from retained samples.

    host_creation_id identifies the common CLOCK_MONOTONIC reference. An unknown
    attempted range makes the block unavailable even if ten samples survived.
    clock_jump records a preceding jump in this Epoch, including block boundaries.
    """

    sampler: Sampler
    host_creation_id: str
    block_index: uint64
    declared_attempt_range: tuple[positive_uint64, positive_uint64] | None
    samples: tuple[PairedClockSample, ...]
    source_locations: tuple[Source, ...] = ()
    load_phases: tuple[LoadPhaseRange, ...] = ()
    provenance: tuple[str, ...] = ()
    clock_jump: bool = False

    def canonical_samples(self) -> tuple[tuple[PairedClockSample, ...], int]:
        """Return attempts in sequence order and the identical-duplicate count.

        Raw PairedClockSample dataclasses are validated here. Conflicting values
        for one attempt raise CalibrationError with both physical sources.
        """
        # Establish the declared scope and common Host provenance.
        sampler = self.sampler
        first = self.block_index * 10 + 1
        if self.declared_attempt_range is not None:
            lo, hi = self.declared_attempt_range
            if lo > hi:
                raise CalibrationError(ErrorCode.INVALID_SAMPLE, "reversed declared range", sampler)
        previous: dict[int, PairedClockSample] = {}
        duplicates = 0
        provenance = {(sample.source.process_creation_id, sample.source.clock_domain_id) for sample in self.samples}
        if len(provenance) > 1:
            raise CalibrationError(
                ErrorCode.INVALID_SAMPLE,
                "mixed Host source provenance",
                sampler,
                sources=tuple(sample.source for sample in self.samples),
            )
        # Check outcome-dependent fields before comparing retained attempts.
        for sample in self.samples:
            identity = sample.source.identity
            values = sample.attempt_sequence, sample.host_before_ns
            valid = (
                all(is_uint64(value) for value in values)
                and first <= sample.attempt_sequence < first + 10
                and (identity.run_id, identity.producer, sample.device_clock_domain_id) == sampler.identity
                and sample.raw_epoch == sampler.raw_epoch
                and all(
                    is_uint64(value) and value > 0
                    for value in (
                        sample.raw_epoch,
                        sample.device_clock_domain_id,
                        sample.source.clock_domain_id,
                        sample.source.identity.event_sequence,
                    )
                )
            )
            if self.declared_attempt_range is not None:
                valid = valid and lo <= sample.attempt_sequence <= hi
            if sample.outcome == "succeeded":
                valid = valid and all(is_uint64(value) for value in (sample.device_time_ns, sample.host_after_ns))
                valid = valid and sample.host_after_ns is not None and sample.host_after_ns >= sample.host_before_ns
                valid = valid and sample.failure_code is None
            elif sample.outcome == "failed":
                valid = valid and sample.device_time_ns is None and sample.host_after_ns is None
                valid = valid and isinstance(sample.failure_code, SampleFailureCode)
            else:
                valid = False
            if not valid:
                raise CalibrationError(
                    ErrorCode.INVALID_SAMPLE,
                    "invalid paired sample",
                    sampler,
                    attempt_sequence=sample.attempt_sequence,
                    sources=(sample.source,),
                )
            # Deduplicate values while preserving conflicts as structured errors.
            old = previous.get(sample.attempt_sequence)
            if old is not None:
                if (
                    old.host_before_ns,
                    old.device_time_ns,
                    old.host_after_ns,
                    old.outcome,
                    old.failure_code,
                    old.source.process_creation_id,
                    old.source.clock_domain_id,
                ) != (
                    sample.host_before_ns,
                    sample.device_time_ns,
                    sample.host_after_ns,
                    sample.outcome,
                    sample.failure_code,
                    sample.source.process_creation_id,
                    sample.source.clock_domain_id,
                ):
                    raise CalibrationError(
                        ErrorCode.SAMPLE_CONFLICT,
                        "contradictory attempt",
                        sampler,
                        attempt_sequence=sample.attempt_sequence,
                        sources=(old.source, sample.source),
                    )
                duplicates += 1
            else:
                previous[sample.attempt_sequence] = sample
        return tuple(previous[key] for key in sorted(previous)), duplicates


class ModelId(StrictModel):
    """Stable model identity within one Run and registered sampler."""

    sampler: tuple[str, positive_uint64, positive_uint64]
    block_index: uint64


class Vertex(StrictModel):
    """One exact feasible slope/intercept vertex in local coordinates."""

    a: Exact
    c: Exact


class CalibrationModel(StrictModel):
    """A validated affine segment or an explicit unavailable block.

    Numerical fields are null on unavailable models. Samples and validation
    diagnostics remain available; bounds are conditional on affine bracketing.
    """

    block: SampleBlock
    model_id: ModelId
    algorithm_version: Literal[1] = 1
    status: CalibrationStatus
    reasons: tuple[CalibrationStatus, ...] = ()
    G0: uint64 | None = None
    h: uint64 | None = None
    vertices: tuple[Vertex, ...] = ()
    a: Exact | None = None
    b: Exact | None = None
    fit_sequences: tuple[positive_uint64, ...] = ()
    validation_sequences: tuple[positive_uint64, ...] = ()
    valid_G: tuple[uint64, uint64] | None = None
    validation_pass: tuple[bool, ...] = ()
    duplicate_sample_count: uint64 = 0
    rate_ppm: Exact | None = None
    rate_lower_ppm: Exact | None = None
    rate_upper_ppm: Exact | None = None

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        """Require complete positive-slope evidence only for valid calibration models."""
        numerical = (
            self.G0,
            self.h,
            self.a,
            self.b,
            self.valid_G,
            self.rate_ppm,
            self.rate_lower_ppm,
            self.rate_upper_ppm,
        )
        if self.status == CalibrationStatus.VALID:
            if any(value is None for value in numerical) or not self.vertices:
                raise ValueError("valid model requires complete numerical evidence")
            if self.a is None or self.a <= 0 or self.valid_G is None or self.valid_G[0] >= self.valid_G[1]:
                raise ValueError("valid model requires positive slope and span")
        elif any(value is not None for value in numerical) or self.vertices:
            raise ValueError("unavailable model must have null numerical parameters")
        return self


class MappedTime(StrictModel):
    """Raw timestamp and its conditional Host-reference interval in nanoseconds."""

    run_id: str
    host_creation_id: str | None
    clock_domain_id: positive_uint64
    raw_time_ns: uint64
    raw_epoch: positive_uint64 | None = None
    status: MappingStatus
    model_id: ModelId | None = None
    estimate_exact: Exact | None = None
    lower_exact: Exact | None = None
    upper_exact: Exact | None = None
    estimate_ns: int | None = None
    lower_ns: int | None = None
    upper_ns: int | None = None
    assumption: ClockAssumption | None = None
    reasons: tuple[CalibrationStatus | MappingReason, ...] = ()
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Require complete ordered bounds and exact outward integer rounding for mapped values."""
        exact = self.estimate_exact, self.lower_exact, self.upper_exact
        integer = self.estimate_ns, self.lower_ns, self.upper_ns
        if self.status == MappingStatus.MAPPED:
            if self.host_creation_id is None:
                raise ValueError("mapped time requires a common Host reference")
            if (
                self.estimate_exact is None
                or self.lower_exact is None
                or self.upper_exact is None
                or any(value is None for value in integer)
            ):
                raise ValueError("mapped values require complete bounds")
            if self.lower_exact > self.estimate_exact or self.estimate_exact > self.upper_exact:
                raise ValueError("invalid mapped interval")
            if integer != (round(self.estimate_exact), floor(self.lower_exact), ceil(self.upper_exact)):
                raise ValueError("inconsistent integer mapping")
        elif any(value is not None for value in (*exact, *integer)):
            raise ValueError("unavailable mapping must have null numerical values")
        return self


class DurationEstimate(StrictModel):
    """End-minus-start with both endpoints and outward-rounded uncertainty."""

    start: MappedTime
    end: MappedTime
    status: DurationStatus
    estimate_exact: Exact | None = None
    lower_exact: Exact | None = None
    upper_exact: Exact | None = None
    estimate_ns: int | None = None
    lower_ns: int | None = None
    upper_ns: int | None = None
    ordering: Ordering | None = None
    reasons: tuple[DurationReason, ...] = ()


class DriftEstimate(StrictModel):
    """Difference in end/start affine rates; bounds need no independence assumption."""

    start: ModelId
    end: ModelId
    estimate_ppm: Exact | None = None
    lower_ppm: Exact | None = None
    upper_ppm: Exact | None = None
    resolved: bool | None = None


class Window(NamedTuple):
    """A successful window after SampleBlock.canonical_samples establishes nonnull integers."""

    sequence: int
    before: int
    device: int
    after: int


def successful_windows(samples: tuple[PairedClockSample, ...]) -> tuple[Window, ...]:
    # SampleBlock.canonical_samples established outcome-dependent nullability at the boundary.
    return tuple(
        Window(s.attempt_sequence, s.host_before_ns, cast(int, s.device_time_ns), cast(int, s.host_after_ns))
        for s in samples
        if s.outcome == "succeeded"
    )


def feasible_vertices(samples: tuple[Window, ...], G0: int, h: int) -> tuple[Vertex, ...]:
    """Enumerate exact intersections of Ax+By<=C boundaries in (a,c)."""
    constraints = [(-1, 0, 0)]
    for sample in samples:
        x = sample.device - G0
        constraints.extend(((x, 1, sample.after - h), (-x, -1, h - sample.before)))
    vertices: set[tuple[Fraction, Fraction]] = set()
    for (A, B, C), (D, E, F) in combinations(constraints, 2):
        determinant = A * E - B * D
        if determinant == 0:
            continue
        a = Fraction(C * E - B * F, determinant)
        c = Fraction(A * F - C * D, determinant)
        if all(A * a + B * c <= C for A, B, C in constraints):
            vertices.add((a, c))
    return tuple(Vertex(a=a, c=c) for a, c in sorted(vertices))


def calibrate(block: SampleBlock) -> CalibrationModel:
    """Fit algorithm version 1; held-out samples never alter feasible vertices.

    Invalid input raises CalibrationError. Missing/failed attempts and failed
    validation return unavailable models with contract-ordered diagnostics.
    """
    samples, duplicates = block.canonical_samples()
    first = block.block_index * 10 + 1
    fit = tuple(sample for sample in samples if (sample.attempt_sequence - 1) % 10 not in (3, 8))
    validation = tuple(sample for sample in samples if (sample.attempt_sequence - 1) % 10 in (3, 8))
    model_id = ModelId(sampler=block.sampler.identity, block_index=block.block_index)

    # Record independent evidence failures before attempting the numerical model.
    reasons: list[CalibrationStatus] = []
    declared = block.declared_attempt_range
    if declared is None:
        reasons.append(CalibrationStatus.ATTEMPT_RANGE_UNKNOWN)
    elif declared[0] > first or declared[1] < first + 9:
        reasons.append(CalibrationStatus.INCOMPLETE_BLOCK)
    if declared is not None and declared[0] <= first and declared[1] >= first + 9 and len(samples) < 10:
        reasons.append(CalibrationStatus.SAMPLING_GAP)
    if any(sample.outcome == "failed" for sample in samples):
        reasons.append(CalibrationStatus.SAMPLE_FAILED)
    successful = successful_windows(samples)
    jump = any(b.device < a.device for a, b in pairwise(successful))
    if block.clock_jump or jump:
        reasons.append(CalibrationStatus.CLOCK_JUMP)

    # Fit only complete fitting evidence; keep every diagnostic already known.
    fit_ok = successful_windows(fit)
    validation_ok = successful_windows(validation)
    times = tuple(s.device for s in fit_ok)
    if len(set(times)) < 2:
        reasons.append(CalibrationStatus.INSUFFICIENT_SPAN)
    vertices: tuple[Vertex, ...] = ()
    passes: tuple[bool, ...] = ()
    a = b = None
    G0 = h = None
    support = None
    if len(fit_ok) == 8 and len(set(times)) >= 2:
        G0, h = fit_ok[0].device, fit_ok[0].before
        vertices = feasible_vertices(fit_ok, G0, h)
        if not vertices or max(vertex.a for vertex in vertices) <= 0:
            reasons.append(CalibrationStatus.MODEL_CONFLICT)
        else:
            a = sum((v.a for v in vertices), Fraction()) / len(vertices)
            c = sum((v.c for v in vertices), Fraction()) / len(vertices)
            b = h + c - a * G0
            support = min(times), max(times)
            if any(s.device < support[0] or s.device > support[1] for s in validation_ok):
                reasons.append(CalibrationStatus.VALIDATION_OUTSIDE_SUPPORT)
            passes = tuple(s.before <= a * s.device + b <= s.after for s in validation_ok)
            if len(passes) == 2 and not all(passes):
                reasons.append(CalibrationStatus.VALIDATION_FAILED)
    base = CalibrationModel(
        block=block,
        model_id=model_id,
        status=reasons[0] if reasons else CalibrationStatus.VALIDATION_FAILED,
        reasons=tuple(reasons),
        validation_pass=passes,
        fit_sequences=tuple(s.attempt_sequence for s in fit),
        validation_sequences=tuple(s.attempt_sequence for s in validation),
        duplicate_sample_count=duplicates,
    )
    if reasons:
        return base
    assert a is not None
    return CalibrationModel(
        block=block,
        model_id=model_id,
        status=CalibrationStatus.VALID,
        fit_sequences=base.fit_sequences,
        validation_sequences=base.validation_sequences,
        duplicate_sample_count=duplicates,
        G0=G0,
        h=h,
        vertices=vertices,
        a=a,
        b=b,
        valid_G=support,
        validation_pass=passes,
        rate_ppm=(a - 1) * 10**6,
        rate_lower_ppm=(min(v.a for v in vertices) - 1) * 10**6,
        rate_upper_ppm=(max(v.a for v in vertices) - 1) * 10**6,
    )


def map_time(model: CalibrationModel, raw_time_ns: int) -> MappedTime:
    """Map only inside a validated segment; unavailable outputs have null values."""
    block = model.block
    base = MappedTime(
        run_id=block.sampler.run_id,
        host_creation_id=block.host_creation_id,
        clock_domain_id=block.sampler.device_clock_domain_id,
        raw_epoch=block.sampler.raw_epoch,
        raw_time_ns=raw_time_ns,
        status=MappingStatus.UNCALIBRATED,
        model_id=model.model_id,
    )
    if model.status != CalibrationStatus.VALID:
        return base.model_copy(update={"reasons": model.reasons})
    # Constructor validation establishes complete parameters for valid models.
    assert model.valid_G is not None and model.a is not None and model.b is not None
    assert model.G0 is not None and model.h is not None
    if not model.valid_G[0] <= raw_time_ns <= model.valid_G[1]:
        return base.model_copy(
            update={"status": MappingStatus.OUTSIDE_SUPPORT, "reasons": (MappingReason.OUTSIDE_SUPPORT,)}
        )
    estimate = model.a * raw_time_ns + model.b
    values = tuple(v.a * (raw_time_ns - model.G0) + v.c + model.h for v in model.vertices)
    lower, upper = min(values), max(values)
    return base.model_copy(
        update={
            "status": MappingStatus.MAPPED,
            "estimate_exact": estimate,
            "lower_exact": lower,
            "upper_exact": upper,
            "estimate_ns": round(estimate),
            "lower_ns": floor(lower),
            "upper_ns": ceil(upper),
            "assumption": ClockAssumption.SEGMENT_AFFINE_BRACKETING,
        }
    )


def estimate_duration(start: MappedTime, end: MappedTime) -> DurationEstimate:
    """Subtract exact endpoint intervals, preserving negatives and unresolved order."""
    if start.run_id != end.run_id or (
        start.host_creation_id is not None
        and end.host_creation_id is not None
        and start.host_creation_id != end.host_creation_id
    ):
        raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "duration requires one Run and common Host reference")
    if start.status != MappingStatus.MAPPED or end.status != MappingStatus.MAPPED:
        return DurationEstimate(
            start=start, end=end, status=DurationStatus.UNAVAILABLE, reasons=(DurationReason.UNAVAILABLE_ENDPOINT,)
        )
    # MappedTime validates complete exact bounds before reaching this branch.
    assert start.estimate_exact is not None and start.lower_exact is not None and start.upper_exact is not None
    assert end.estimate_exact is not None and end.lower_exact is not None and end.upper_exact is not None
    estimate = end.estimate_exact - start.estimate_exact
    lower = end.lower_exact - start.upper_exact
    upper = end.upper_exact - start.lower_exact
    if lower > 0:
        ordering = Ordering.AFTER
    elif upper < 0:
        ordering = Ordering.BEFORE
    elif lower == upper == 0:
        ordering = Ordering.SIMULTANEOUS
    else:
        ordering = Ordering.INDETERMINATE
    return DurationEstimate(
        start=start,
        end=end,
        status=DurationStatus.ESTIMATED,
        estimate_exact=estimate,
        lower_exact=lower,
        upper_exact=upper,
        estimate_ns=round(estimate),
        lower_ns=floor(lower),
        upper_ns=ceil(upper),
        ordering=ordering,
    )


def estimate_drift(start: CalibrationModel, end: CalibrationModel) -> DriftEstimate:
    """Compare end-minus-start rates in ppm through the same Host reference."""
    if (start.block.sampler.run_id, start.block.host_creation_id) != (
        end.block.sampler.run_id,
        end.block.host_creation_id,
    ):
        raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "drift requires one Run and common Host reference")
    if start.status != CalibrationStatus.VALID or end.status != CalibrationStatus.VALID:
        return DriftEstimate(start=start.model_id, end=end.model_id)
    assert start.rate_ppm is not None and start.rate_lower_ppm is not None and start.rate_upper_ppm is not None
    assert end.rate_ppm is not None and end.rate_lower_ppm is not None and end.rate_upper_ppm is not None
    lower = end.rate_lower_ppm - start.rate_upper_ppm
    upper = end.rate_upper_ppm - start.rate_lower_ppm
    return DriftEstimate(
        start=start.model_id,
        end=end.model_id,
        estimate_ppm=end.rate_ppm - start.rate_ppm,
        lower_ppm=lower,
        upper_ppm=upper,
        resolved=not lower <= 0 <= upper,
    )
