"""Client-observed timing, token-weighted distributions and throughput buckets."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, FiniteFloat, JsonValue

from xbench.harness.serving.case import (
    BenchCase,
    BenchValue,
    HostedLaunchDeclaration,
    ResolvedDeployment,
)
from xbench.harness.serving.workload import PreparedWorkload, ScheduledRequest, content_digest
from xkit.case import CaseId
from xkit.config import ToolConfigRecord
from xkit.results import file_sha256, iter_jsonl

REPETITION_DIRECTORY_PATTERN = re.compile(r"repetition-([0-9]{4,})(?:\.attempt-([0-9]{4,}))?")


def retained_file(directory: Path, reference: str) -> Path:
    """Resolve a retained file within its owning directory, rejecting escapes."""
    path = (directory / reference).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"benchmark artifact is missing or escapes its owner: {reference}")
    return path


class BenchRunManifest(BenchValue):
    """Invocation selection and runner-finalized outcome, independent of reports."""

    run_id: str
    tool_config: ToolConfigRecord
    selected_cases: tuple[CaseId, ...]
    case_directories: tuple[str, ...] = ()
    finished: bool = Field(default=False, description="All admitted case execution and finalization has ended.")
    result_code: int | None = Field(
        default=None, ge=0, description="Current invocation outcome; None until finalization."
    )
    infrastructure_error: str | None = None


class BenchCaseManifest(BenchValue):
    """Case declaration, replay identities, deployment and effective attempt references."""

    case: BenchCase
    prompt_sha256: str
    trace_sha256: str
    warmup_sha256: str
    deployment: ResolvedDeployment | None = None
    hosted_launch: HostedLaunchDeclaration | None = None
    repetitions: tuple[str, ...]

    def effective_attempts(self, directory: Path) -> dict[int, Path]:
        """Resolve one physical attempt per logical repetition inside its case."""
        attempts = {}
        for reference in self.repetitions:
            match = REPETITION_DIRECTORY_PATTERN.fullmatch(reference)
            path = (directory / reference).resolve()
            if (
                match is None
                or int(match[1]) < 1
                or match[2] is None
                or int(match[2]) < 1
                or path.parent != directory.resolve()
                or path.name != reference
            ):
                raise ValueError("benchmark repetition reference escapes its case or is not a physical attempt")
            number = int(match[1])
            if number in attempts:
                raise ValueError("benchmark case declares multiple effective attempts for one repetition")
            attempts[number] = path
        return attempts

    def load_workload(self, directory: Path) -> PreparedWorkload:
        """Read the original replay and check its case-owned identities and timing."""
        workload = PreparedWorkload.load(directory, case=self.case)
        for name, rows, expected in (
            ("prompt", workload.prompts, self.prompt_sha256),
            ("trace", workload.requests, self.trace_sha256),
            ("warmup", workload.warmup, self.warmup_sha256),
        ):
            if content_digest(rows) != expected:
                raise ValueError(f"retained {name} content digest does not match the case")
        if workload.bucket_seconds != self.case.bucket_seconds or (
            self.case.arrivals.duration_seconds is not None
            and workload.arrival_horizon_seconds != self.case.arrivals.duration_seconds
        ):
            raise ValueError("retained workload declarations disagree with the case")
        return workload


class RepetitionManifest(BenchValue):
    """One attempt's timing and outcome, sealed by the runner after domain drain."""

    repetition: int = Field(gt=0)
    window_end_seconds: FiniteFloat | None = Field(
        default=None, ge=0, description="Common window end relative to T0; None when a usable window is unavailable."
    )
    window_kind: Literal["complete", "interrupted", "observed_prefix"] | None = Field(
        default=None, description="Measurement-owner end or recovered observation prefix; None without a usable window."
    )
    finished: bool = Field(default=False, description="Runner finalization has ended; this is not a success verdict.")
    result_code: int | None = Field(
        default=None, ge=0, description="Original repetition outcome; None until finalization."
    )
    cleanup_verified: bool | None = Field(
        default=None, description="Outer runner's descendant-drain proof; None until verified or rejected."
    )
    raw_evidence_complete: bool | None = Field(
        default=None, description="Raw recording and terminal request accounting are intact; None before finalization."
    )
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    infrastructure_error: str | None = None

    @property
    def sealed(self) -> bool:
        """Whether this attempt has its own final outcome and cleanup observation."""
        return self.finished and self.result_code is not None and self.cleanup_verified is not None

    @classmethod
    def from_directory(cls, directory: Path) -> RepetitionManifest:
        """Read an attempt checkpoint whose logical number matches its path."""
        match = REPETITION_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None or match[2] is None or int(match[2]) < 1:
            raise ValueError(f"invalid benchmark attempt directory: {directory.name}")
        manifest = cls.model_validate_json(retained_file(directory, "repetition.json").read_bytes())
        if manifest.repetition != int(match[1]):
            raise ValueError("benchmark attempt number disagrees with its directory")
        return manifest


class MeasurementOrigin(BenchValue):
    """Clock readings when measurement starts.

    The monotonic clock measures elapsed time. The wall clock dates the run.
    """

    monotonic_origin_seconds: FiniteFloat = Field(ge=0)
    wall_clock_origin_seconds: FiniteFloat = Field(ge=0)


@dataclass(frozen=True, slots=True)
class MeasurementEvidence:
    """Validated terminal records, stream histories and attempt completeness."""

    requests: Sequence[RequestRecord]
    events_by_request: dict[str, list[StreamEvent]]
    execution_complete: bool
    evidence_complete: bool


def load_measurement(
    directory: Path, workload: PreparedWorkload, *, checkpoint: RepetitionManifest
) -> MeasurementEvidence:
    """Validate saved attempt measurements independently of the parent run's seal.

    Continuation can reuse a successful sealed attempt in an interrupted parent.
    Reporting applies the parent publication boundary separately.
    """
    required = {"events.jsonl", "requests.jsonl"}
    if (
        checkpoint.window_end_seconds is not None
        or "measurement.json" in checkpoint.artifact_sha256
        or (directory / "measurement.json").is_file()
    ):
        required.add("measurement.json")
    if not required <= checkpoint.artifact_sha256.keys():
        raise ValueError("benchmark repetition lacks required evidence digests")
    for reference in required:
        artifact = retained_file(directory, reference)
        if file_sha256(artifact) != checkpoint.artifact_sha256[reference]:
            raise ValueError(f"retained benchmark artifact digest mismatch: {reference}")
    if "measurement.json" in required:
        MeasurementOrigin.model_validate_json(retained_file(directory, "measurement.json").read_bytes())
    requests = tuple(iter_jsonl(directory / "requests.jsonl", RequestRecord))
    events = tuple(iter_jsonl(directory / "events.jsonl", StreamEvent))
    evidence = validate_measurement(
        workload,
        requests,
        events,
        window_end_seconds=checkpoint.window_end_seconds,
        window_kind=checkpoint.window_kind,
    )
    if checkpoint.result_code == 0 and (
        not evidence.execution_complete
        or not requests
        or any(request.outcome != "success" for request in requests)
        or not checkpoint.cleanup_verified
        or checkpoint.raw_evidence_complete is False
        or checkpoint.infrastructure_error is not None
    ):
        raise ValueError("successful benchmark checkpoint contradicts its completeness or outcomes")
    return replace(
        evidence,
        execution_complete=evidence.execution_complete and checkpoint.raw_evidence_complete is not False,
        evidence_complete=evidence.evidence_complete and checkpoint.sealed and checkpoint.raw_evidence_complete is True,
    )


class StreamEvent(BenchValue):
    """Timestamped lifecycle or protocol observation, including rejected frames."""

    request_id: str
    sequence: int = Field(ge=0)
    observed_at_seconds: FiniteFloat = Field(ge=0)
    kind: Literal["enqueue", "http_start", "data", "usage", "done", "error"]
    completion_tokens: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    accepted: bool = True
    reported_meta: JsonValue = None
    error_kind: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class StreamProgress:
    """Validate the same Completions transitions live and during replay."""

    completion_tokens: int | None = None
    finish_reason: str | None = None
    usage_received: bool = False
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    first_token_at_seconds: float | None = None
    last_token_at_seconds: float | None = None
    first_completion_tokens: int | None = None

    def accept(self, event: StreamEvent) -> None:
        match event.kind:
            case "data":
                if self.finish_reason is not None:
                    raise ValueError("generation frame after the finish reason")
                if event.completion_tokens is None:
                    raise ValueError("generation frame lacks cumulative completion tokens")
                if event.completion_tokens < (self.completion_tokens or 0):
                    raise ValueError("cumulative completion count regressed")
                if event.finish_reason not in {None, "stop", "length"}:
                    raise ValueError("completion finish reason must be stop or length")
                if event.completion_tokens > (self.completion_tokens or 0):
                    if self.first_token_at_seconds is None:
                        self.first_token_at_seconds = event.observed_at_seconds
                        self.first_completion_tokens = event.completion_tokens
                    self.last_token_at_seconds = event.observed_at_seconds
                self.completion_tokens = event.completion_tokens
                self.finish_reason = event.finish_reason
            case "usage":
                if self.finish_reason is None or self.usage_received:
                    raise ValueError("final usage must follow the finish reason exactly once")
                if event.prompt_tokens is None or event.completion_tokens != self.completion_tokens:
                    raise ValueError("final usage must match the last cumulative completion count")
                if event.finish_reason is not None:
                    raise ValueError("final usage frame includes a finish reason")
                if event.cached_tokens is not None and event.cached_tokens > event.prompt_tokens:
                    raise ValueError("cached tokens exceed logical prompt usage")
                self.usage_received = True
            case "done":
                if not self.usage_received:
                    raise ValueError("DONE requires final usage")
                return
            case _:
                return
        if event.prompt_tokens is not None:
            self.prompt_tokens = event.prompt_tokens
        if event.cached_tokens is not None:
            self.cached_tokens = event.cached_tokens


class RequestMetrics(BenchValue):
    """Per-request projections; unavailable timings and token intervals stay None."""

    queue_wait_seconds: FiniteFloat | None = Field(default=None, ge=0)
    arrival_lateness_seconds: FiniteFloat | None = Field(default=None, ge=0)
    http_ttft_seconds: FiniteFloat | None = Field(default=None, ge=0)
    arrival_ttft_seconds: FiniteFloat | None = Field(default=None, ge=0)
    http_latency_seconds: FiniteFloat | None = Field(default=None, ge=0)
    arrival_latency_seconds: FiniteFloat | None = Field(default=None, ge=0)
    mean_itl_seconds: FiniteFloat | None = Field(default=None, ge=0)
    tpot_seconds: FiniteFloat | None = Field(default=None, ge=0)
    itl_coverage: FiniteFloat | None = Field(default=None, ge=0)
    itl_sample_count: int = 0
    observed_itl_sample_count: int = 0
    estimated_itl_sample_count: int = 0


class RequestRecord(ScheduledRequest):
    """Terminal request facts checked against replay and the retained event prefix."""

    outcome: Literal["success", "failed", "cancelled", "not_sent"]
    enqueued_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    http_started_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    first_token_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    last_token_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    ended_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    http_status: int | None = None
    error_kind: str | None = None
    error_message: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    first_completion_tokens: int | None = Field(default=None, gt=0)

    def validate_evidence(self, events: Sequence[StreamEvent]) -> None:
        """Check the request result against its recorded events.

        Measurement validation checks event order separately. Requests recovered
        with missing evidence keep their timestamps unset.
        """

        timestamps = (
            self.enqueued_at_seconds,
            self.http_started_at_seconds,
            self.first_token_at_seconds,
            self.last_token_at_seconds,
            self.ended_at_seconds,
        )
        if (self.outcome == "not_sent" or self.error_kind == "evidence_missing") and any(
            value is not None
            for value in (
                self.http_status,
                self.finish_reason,
                self.prompt_tokens,
                self.cached_tokens,
                self.completion_tokens,
                self.first_completion_tokens,
            )
        ):
            raise ValueError("not-sent/evidence-missing request cannot claim HTTP state or token usage")
        if self.error_kind == "evidence_missing":
            if self.outcome != "failed" or any(value is not None for value in timestamps):
                raise ValueError("evidence-missing request cannot claim known termination or timestamps")
            return
        enqueued = events[0].observed_at_seconds if events else None
        started = events[1].observed_at_seconds if len(events) > 1 else None
        if self.enqueued_at_seconds != enqueued or self.http_started_at_seconds != started:
            raise ValueError("request dispatch timestamps differ from lifecycle events")
        if self.outcome == "not_sent":
            if len(events) > 1 or any(value is not None for value in timestamps[1:]):
                raise ValueError("not-sent request cannot have HTTP or termination observations")
        elif self.outcome == "success":
            if (
                not events
                or events[-1].kind != "done"
                or not events[-1].accepted
                or self.http_status != 200
                or self.finish_reason not in {"stop", "length"}
                or self.prompt_tokens is None
                or self.completion_tokens is None
                or self.error_kind is not None
            ):
                raise ValueError("success requires HTTP 200, normal final usage and accepted DONE evidence")
        elif not events or events[-1].kind != "error":
            raise ValueError("failed/cancelled request requires a terminal error observation")
        if self.outcome != "not_sent" and self.ended_at_seconds != events[-1].observed_at_seconds:
            raise ValueError("request termination time differs from its terminal event")
        progress = StreamProgress()
        for event in events:
            if event.accepted:
                progress.accept(event)
        if (
            self.prompt_tokens != progress.prompt_tokens
            or self.cached_tokens != progress.cached_tokens
            or self.completion_tokens != progress.completion_tokens
            or self.finish_reason != progress.finish_reason
            or self.first_token_at_seconds != progress.first_token_at_seconds
            or self.last_token_at_seconds != progress.last_token_at_seconds
            or self.first_completion_tokens != progress.first_completion_tokens
        ):
            raise ValueError("request usage or token timestamps differ from accepted stream evidence")

    def compute_metrics(self, intervals: Sequence[TokenInterval]) -> RequestMetrics:
        """Project raw observations to nullable metrics, retaining partial evidence."""

        def difference(end: float | None, start: float | None) -> float | None:
            return None if end is None or start is None else end - start

        count = sum(interval.weight for interval in intervals)
        first = self.first_token_at_seconds
        completed = self.completion_tokens
        first_count = self.first_completion_tokens
        return RequestMetrics(
            queue_wait_seconds=difference(self.http_started_at_seconds, self.enqueued_at_seconds),
            arrival_lateness_seconds=difference(self.enqueued_at_seconds, self.arrival_seconds),
            http_ttft_seconds=difference(first, self.http_started_at_seconds),
            arrival_ttft_seconds=difference(first, self.enqueued_at_seconds),
            http_latency_seconds=difference(self.ended_at_seconds, self.http_started_at_seconds),
            arrival_latency_seconds=difference(self.ended_at_seconds, self.enqueued_at_seconds),
            mean_itl_seconds=math.fsum(interval.value_seconds * interval.weight for interval in intervals) / count
            if count
            else None,
            tpot_seconds=(self.ended_at_seconds - first) / (completed - 1)
            if first is not None and self.ended_at_seconds is not None and completed is not None and completed > 1
            else None,
            itl_coverage=(completed - first_count) / (completed - 1)
            if completed is not None and completed > 1 and first_count is not None
            else None,
            itl_sample_count=count,
            observed_itl_sample_count=sum(interval.weight for interval in intervals if interval.kind == "observed"),
            estimated_itl_sample_count=sum(interval.weight for interval in intervals if interval.kind == "estimated"),
        )


@dataclass(frozen=True, slots=True)
class TokenInterval:
    value_seconds: float
    weight: int
    kind: Literal["observed", "estimated"]


def token_intervals(events: Sequence[StreamEvent]) -> tuple[TokenInterval, ...]:
    """Calculate intervals from accepted counts and chronological observations."""

    intervals = []
    previous_count = 0
    previous_time = None
    for event in events:
        if event.kind != "data" or not event.accepted or event.completion_tokens is None:
            continue
        count = event.completion_tokens
        delta = count - previous_count
        if delta == 0:
            continue
        if previous_time is not None:
            gap = event.observed_at_seconds - previous_time
            intervals.append(TokenInterval(gap / delta, delta, "observed" if delta == 1 else "estimated"))
        previous_count = count
        previous_time = event.observed_at_seconds
    return tuple(intervals)


class Distribution(BenchValue):
    """Weighted population statistics; an empty population has nullable values."""

    sample_count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None


def distribution(samples: Sequence[tuple[float, int]]) -> Distribution:
    """Aggregate validated finite samples with positive token or request weights."""

    if not samples:
        return Distribution(sample_count=0)
    ordered = sorted(samples)
    weights = []
    cumulative = 0
    for _, weight in ordered:
        cumulative += weight
        weights.append(cumulative)
    mean = math.fsum(value * weight for value, weight in ordered) / cumulative

    def percentile(p: float) -> float:
        rank = (cumulative - 1) * p / 100
        low, high = math.floor(rank), math.ceil(rank)
        left = ordered[bisect_right(weights, low)][0]
        right = ordered[bisect_right(weights, high)][0]
        return left + (right - left) * (rank - low)

    return Distribution(
        sample_count=cumulative,
        mean=mean,
        std=math.sqrt(math.fsum((value - mean) ** 2 * weight for value, weight in ordered) / cumulative),
        min=ordered[0][0],
        max=ordered[-1][0],
        median=percentile(50),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
    )


class CdfPoint(BenchValue):
    """One empirical step with request or token-interval weight."""

    target_id: str = Field(description="Population label: full Model ID or aggregate.")
    metric: str
    sample_kind: str
    value_seconds: FiniteFloat = Field(ge=0)
    cumulative_probability: float
    weight: int


class ThroughputBucket(BenchValue):
    """Aligned common-window counts; the final bucket uses its actual width."""

    target_id: str = Field(description="Population label: full Model ID or aggregate.")
    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(ge=0)
    width_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    partial_output_tokens: int = 0
    input_tokens_per_second: float = 0.0
    output_tokens_per_second: float = 0.0
    partial_output_tokens_per_second: float = 0.0


class TargetSummary(BenchValue):
    outcomes: dict[str, int]
    distributions: dict[str, Distribution]
    input_tokens: int
    output_tokens: int
    partial_output_tokens: int
    input_tokens_per_second: float | None
    output_tokens_per_second: float | None


class BenchSummary(BenchValue):
    """Recomputed measurement populations with separate execution and evidence facts."""

    latency_units: Literal["seconds"] = "seconds"
    case_id: CaseId
    execution_complete: bool
    measurement_available: bool
    cleanup_verified: bool | None = None
    evidence_complete: bool
    arrival_horizon_seconds: FiniteFloat = Field(ge=0)
    window_end_seconds: FiniteFloat | None = Field(
        ge=0, description="Common window end relative to T0; None when a usable window is unavailable."
    )
    window_kind: Literal["complete", "interrupted", "observed_prefix"] | None
    provisional_requests: int
    outcomes: dict[str, int]
    targets: dict[str, TargetSummary]
    cdf: tuple[CdfPoint, ...]
    throughput: tuple[ThroughputBucket, ...]
    infrastructure_error: str | None = None


def validate_request_identity(
    workload: PreparedWorkload, requests: Sequence[RequestRecord]
) -> dict[str, ScheduledRequest]:
    """Match unique retained identities and requested policy to their replay authority."""

    expected = {request.request_id: request for request in workload.requests}
    records = {request.request_id: request for request in requests}
    if len(records) != len(requests) or set(records) - expected.keys():
        raise ValueError("request records contain duplicate or unplanned IDs")
    for request in requests:
        if request.model_dump(include=set(ScheduledRequest.model_fields)) != expected[request.request_id].model_dump():
            raise ValueError("retained request identity differs from the prepared workload")
    return expected


def validate_measurement(
    workload: PreparedWorkload,
    requests: Sequence[RequestRecord],
    events: Sequence[StreamEvent],
    *,
    window_end_seconds: float | None,
    window_kind: Literal["complete", "interrupted", "observed_prefix"] | None,
) -> MeasurementEvidence:
    """Validate replay and raw lifecycle evidence without building statistics.

    An unavailable window supports only not-sent or unknown owner-loss outcomes.
    Timed observations must follow the declared arrival and fit the window.
    """

    expected = validate_request_identity(workload, requests)
    evidence_complete = len(requests) == len(expected) and all(
        request.error_kind != "evidence_missing" for request in requests
    )
    if (window_end_seconds is None) != (window_kind is None):
        raise ValueError("measurement window end and kind must be available together")
    if window_end_seconds is None:
        if events or any(
            request.outcome not in {"not_sent", "failed"}
            or (request.outcome == "failed" and request.error_kind != "evidence_missing")
            for request in requests
        ):
            raise ValueError("unavailable measurement window cannot support dispatched observations")
        for request in requests:
            request.validate_evidence(())
        return MeasurementEvidence(requests, {}, execution_complete=not expected, evidence_complete=evidence_complete)
    if not math.isfinite(window_end_seconds) or window_end_seconds <= 0:
        raise ValueError("timed measurement window must be positive and finite")
    by_request: dict[str, list[StreamEvent]] = defaultdict(list)
    for event in events:
        if event.request_id not in expected:
            raise ValueError("stream event refers to an unplanned request")
        by_request[event.request_id].append(event)
    for request_id, history in by_request.items():
        previous_time = expected[request_id].arrival_seconds
        for sequence, event in enumerate(history):
            if event.sequence != sequence:
                raise ValueError("stream event sequence must be contiguous from zero")
            if not previous_time <= event.observed_at_seconds <= window_end_seconds:
                raise ValueError("stream event timestamps must be chronological within the measurement window")
            previous_time = event.observed_at_seconds
            if (
                (sequence == 0 and event.kind != "enqueue")
                or (sequence == 1 and event.kind != "http_start")
                or (sequence > 1 and event.kind in {"enqueue", "http_start"})
                or (not event.accepted and event.kind not in {"data", "usage", "done"})
            ):
                raise ValueError("stream event lifecycle must start with enqueue then HTTP dispatch")
            if sequence:
                previous = history[sequence - 1]
                if (
                    previous.kind == "error"
                    or (previous.kind == "done" and previous.accepted)
                    or (not previous.accepted and event.kind != "error")
                ):
                    raise ValueError("stream event follows terminal or rejected Completions evidence")
    for request in requests:
        request.validate_evidence(by_request[request.request_id])
    if window_kind == "complete" and window_end_seconds < workload.arrival_horizon_seconds:
        raise ValueError("completed measurement window must cover the arrival horizon")
    execution_complete = (
        window_kind == "complete"
        and evidence_complete
        and all(request.outcome in {"success", "failed"} for request in requests)
    )
    return MeasurementEvidence(requests, by_request, execution_complete, evidence_complete)


def summarize(
    workload: PreparedWorkload,
    evidence: MeasurementEvidence,
    *,
    checkpoint: RepetitionManifest,
) -> BenchSummary:
    """Aggregate successful samples and separate failed partial output.

    The common monotonic window includes arrival idle time and queue drain.
    Output increments use observed chunk times; input includes logical cache-hit
    tokens and is attributed at successful request termination. Normalized
    intervals carry token weights, never artificially spaced token timestamps.
    Cleanup verification belongs to the outer runner.
    """

    requests = evidence.requests
    window_end_seconds = checkpoint.window_end_seconds
    if window_end_seconds is None:
        outcomes = Counter(request.outcome for request in requests)
        return BenchSummary(
            case_id=workload.case_id,
            execution_complete=evidence.execution_complete,
            measurement_available=False,
            cleanup_verified=checkpoint.cleanup_verified,
            evidence_complete=evidence.evidence_complete,
            arrival_horizon_seconds=workload.arrival_horizon_seconds,
            window_end_seconds=None,
            window_kind=None,
            provisional_requests=len(workload.requests) - len(requests),
            outcomes={kind: outcomes[kind] for kind in ("success", "failed", "cancelled", "not_sent")},
            targets={},
            cdf=(),
            throughput=(),
            infrastructure_error=checkpoint.infrastructure_error,
        )
    samples: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    cdf: list[CdfPoint] = []
    buckets: dict[str, list[list[int]]] = {
        id: [[0, 0, 0] for _ in range(math.ceil(window_end_seconds / workload.bucket_seconds))]
        for id in (*(str(model_id) for model_id in workload.model_ids), "aggregate")
    }

    def attribute(id: str, when: float, slot: int, tokens: int) -> None:
        index = min(int(when / workload.bucket_seconds), len(buckets[id]) - 1)
        for key in (id, "aggregate"):
            buckets[key][index][slot] += tokens

    for request in requests:
        target = str(request.model_id)
        request_events = evidence.events_by_request[request.request_id]
        success = request.outcome == "success"
        if success:
            intervals = token_intervals(request_events)
            metrics = request.compute_metrics(intervals)
            for name, value in metrics.model_dump().items():
                if name.endswith("_seconds") or name == "itl_coverage":
                    if value is not None:
                        for key in (target, "aggregate"):
                            samples[key][name].append((value, 1))
            for interval in intervals:
                for key in (target, "aggregate"):
                    samples[key][f"itl_{interval.kind}"].append((interval.value_seconds, interval.weight))
                    samples[key]["itl_combined"].append((interval.value_seconds, interval.weight))
            attribute(target, cast(float, request.ended_at_seconds), 0, cast(int, request.prompt_tokens))
        previous = 0
        for event in request_events:
            if event.kind == "data" and event.accepted and event.completion_tokens is not None:
                increment = event.completion_tokens - previous
                attribute(target, event.observed_at_seconds, 1 if success else 2, increment)
                previous = event.completion_tokens
    throughput: list[ThroughputBucket] = []
    targets: dict[str, TargetSummary] = {}
    metric_names = (*RequestMetrics.model_fields.keys(), "itl_observed", "itl_estimated", "itl_combined")
    metric_names = tuple(name for name in metric_names if not name.endswith("sample_count"))
    for id, values in buckets.items():
        for index, (input_tokens, output_tokens, partial) in enumerate(values):
            start = index * workload.bucket_seconds
            end = min(start + workload.bucket_seconds, window_end_seconds)
            width = end - start
            throughput.append(
                ThroughputBucket(
                    target_id=id,
                    start_seconds=start,
                    end_seconds=end,
                    width_seconds=width,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    partial_output_tokens=partial,
                    input_tokens_per_second=input_tokens / width,
                    output_tokens_per_second=output_tokens / width,
                    partial_output_tokens_per_second=partial / width,
                )
            )
        target_requests = tuple(request for request in requests if id == "aggregate" or str(request.model_id) == id)
        outcomes = Counter(request.outcome for request in target_requests)
        distributions = {name: distribution(samples[id][name]) for name in metric_names}
        for metric, population in samples[id].items():
            if metric == "itl_coverage":
                continue
            counts: dict[float, int] = defaultdict(int)
            for value, weight in population:
                counts[value] += weight
            total = sum(counts.values())
            cumulative = 0
            for value, weight in sorted(counts.items()):
                cumulative += weight
                cdf.append(
                    CdfPoint(
                        target_id=id,
                        metric=metric,
                        sample_kind=metric.removeprefix("itl_") if metric.startswith("itl_") else "request",
                        value_seconds=value,
                        cumulative_probability=cumulative / total,
                        weight=weight,
                    )
                )
        input_total, output_total, partial_total = (sum(value[slot] for value in values) for slot in range(3))
        targets[id] = TargetSummary(
            outcomes={kind: outcomes[kind] for kind in ("success", "failed", "cancelled", "not_sent")},
            distributions=distributions,
            input_tokens=input_total,
            output_tokens=output_total,
            partial_output_tokens=partial_total,
            input_tokens_per_second=input_total / window_end_seconds,
            output_tokens_per_second=output_total / window_end_seconds,
        )
    return BenchSummary(
        case_id=workload.case_id,
        execution_complete=evidence.execution_complete,
        measurement_available=targets["aggregate"].outcomes["success"] > 0,
        cleanup_verified=checkpoint.cleanup_verified,
        evidence_complete=evidence.evidence_complete,
        arrival_horizon_seconds=workload.arrival_horizon_seconds,
        window_end_seconds=window_end_seconds,
        window_kind=checkpoint.window_kind,
        provisional_requests=len(workload.requests) - len(requests),
        outcomes=targets["aggregate"].outcomes,
        targets=targets,
        cdf=tuple(cdf),
        throughput=tuple(throughput),
        infrastructure_error=checkpoint.infrastructure_error,
    )
